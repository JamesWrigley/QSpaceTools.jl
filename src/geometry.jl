"""
Geometry layer for reciprocal-space conversion.

Mirrors xrayutilities' `QConversion.area` + `init_area` for the HXRD case
(`ang2q_conversion_area` in `src/qconversion.c`). UB is identity; the
sample/detector rotation chain and a per-pixel direction are enough to map
every pixel to a momentum-transfer vector
`q = M_s^{-1} (M_d r̂_d - r̂_i) · 2π/λ` in the sample frame.

`Geometry` stores `r̂_d` for every pixel eagerly, as a `(3, npix)` array of
unit vectors in the lab frame. Nothing downstream assumes those pixels form a
lattice, so multi-module detectors work too. The regular-grid constructor
derives the directions from `pixel_size`/`center`/`distance`; the `positions`
constructor takes them from anywhere (see the PythonCall extension for the
EXtra-geom path).

Normalization depends only on the detector, not on the angles, so it happens
once at construction and the per-frame kernel is a bare mat-vec.
"""

const Vec3 = SVector{3, Float64}
const Mat3 = SMatrix{3, 3, Float64, 9}

# h·c in eV·Å, matching the constant xrayutilities' `en2lam` uses (CODATA h, c,
# qe).
const _HC_EV_ANGSTROM = 12398.419843320026

@inline energy2wavelength(energy_eV::Real) = _HC_EV_ANGSTROM / Float64(energy_eV)
@inline wavelength2energy(wavelength_Å::Real) = _HC_EV_ANGSTROM / Float64(wavelength_Å)

const AXIS_VECS = Dict(
    "x+" => Vec3(1.0, 0.0, 0.0),  "x-" => Vec3(-1.0,  0.0,  0.0),
    "y+" => Vec3(0.0, 1.0, 0.0),  "y-" => Vec3( 0.0, -1.0,  0.0),
    "z+" => Vec3(0.0, 0.0, 1.0),  "z-" => Vec3( 0.0,  0.0, -1.0),
)

parse_axis(v::AbstractVector) = Vec3(v)
parse_axis(v::Tuple) = Vec3(v)
function parse_axis(s::AbstractString)
    if !haskey(AXIS_VECS, s)
        throw(ArgumentError("Invalid axis string: $(repr(s))"))
    end

    AXIS_VECS[s]
end

# Right-handed rotation by `θ_deg` degrees around unit vector `e` (Rodrigues' formula).
# Axes like "y-" are passed as (0,-1,0); the sign is folded into the vector.
@inline function rotation_arb(θ_deg::Real, e::Vec3)
    s, c = sincosd(θ_deg)
    c1 = 1 - c
    ex, ey, ez = e

    @SMatrix [
        c + ex*ex*c1      ex*ey*c1 - ez*s   ex*ez*c1 + ey*s
        ey*ex*c1 + ez*s   c + ey*ey*c1      ey*ez*c1 - ex*s
        ez*ex*c1 - ey*s   ez*ey*c1 + ex*s   c + ez*ez*c1
    ]
end

"""
    Geometry

Geometry of an area-detector experiment, in xrayutilities conventions.

Holds the sample/detector rotation chains, the incident beam, and one unit
vector per pixel pointing from the sample to that pixel in the lab frame.
`data_shape` is the shape frames must have in their leading dimensions; the
columns of `directions` follow the column-major linear ordering of a
`data_shape`-shaped array.

`image_axes`, `pixel_size`, `center` and `distance` describe the detector in
image space. For a multi-module detector that is the *assembled* 2D image,
independent of the 3D array the data is stored in: `image_axes` are the
lab-frame directions of its two axes and `center` is the beam position in its
pixel units. The regular-grid constructor builds the pixel directions out of
them; everywhere else they are descriptive only.

Build one with either of the two constructors below rather than directly.

`sample_normal` and `sample_faceup` are stored for completeness but not used
by the current q kernel, which assumes UB = I (the default of
`HXRD.Ang2Q.area`).
"""
struct Geometry{N}
    sample_axes::Vector{Vec3}
    detector_axes::Vector{Vec3}
    image_axes::NTuple{2, Vec3}
    beam_direction::Vec3
    sample_normal::Vec3
    sample_faceup::Vec3
    directions::Matrix{Float64}   # (3, npix) unit vectors, lab frame
    data_shape::Dims{N}
    pixel_size::NTuple{2, Float64}
    center::NTuple{2, Float64}
    distance::Float64
    wavelength::Float64
end

npixels(g::Geometry) = size(g.directions, 2)
data_shape(g::Geometry) = g.data_shape

_show_vec(v::Vec3) = string("(", join(v, ", "), ")")
_show_axes(axes) = join((_show_vec(a) for a in axes), " ")

function Base.show(io::IO, ::MIME"text/plain", g::Geometry)
    println(io, "Geometry: ", join(data_shape(g), "×"), " (", npixels(g), " pixels)")
    println(io, "  sample axes    ", _show_axes(g.sample_axes))
    println(io, "  detector axes  ", _show_axes(g.detector_axes))
    println(io, "  image axes     ", _show_axes(g.image_axes))
    println(io, "  beam           ", _show_vec(g.beam_direction))
    println(io, "  pixel size     ", g.pixel_size)
    println(io, "  center         ", g.center)
    println(io, "  distance       ", g.distance)
    print(io,   "  wavelength     ", g.wavelength)
end

function Base.show(io::IO, g::Geometry)
    print(io, "Geometry(", join(data_shape(g), "×"), ", λ=", g.wavelength, ")")
end

"""
    Geometry(positions, data_shape; sample_axes, detector_axes, image_axes,
             beam_direction, sample_normal, sample_faceup, pixel_size, center,
             distance, wavelength)

Build a geometry from explicit per-pixel `positions`: a `(3, npix)` array of
sample-to-pixel vectors in the lab frame, whose columns follow the
column-major linear ordering of a `data_shape`-shaped frame.

The vectors are normalized here, so only the direction of each column matters
and `positions` may be in any length unit. `image_axes`, `pixel_size`, `center`
and `distance` are recorded as metadata and do not have to reproduce
`positions`.

This is the constructor multi-module detectors use; see the PythonCall
extension for building one from an EXtra-geom geometry.
"""
function Geometry(positions::AbstractMatrix{<:Real}, data_shape::NTuple{N, Integer};
                  sample_axes,
                  detector_axes,
                  image_axes,
                  beam_direction,
                  sample_normal,
                  sample_faceup,
                  pixel_size,
                  center,
                  distance,
                  wavelength) where {N}
    shape = Dims{N}(data_shape)
    npix = prod(shape)
    if size(positions) != (3, npix)
        throw(DimensionMismatch("positions must be (3, $npix) to match data_shape $shape; got $(size(positions))"))
    end

    directions = Matrix{Float64}(undef, 3, npix)
    @inbounds for k in 1:npix
        u = normalize(Vec3(positions[1, k], positions[2, k], positions[3, k]))
        directions[1, k] = u[1]
        directions[2, k] = u[2]
        directions[3, k] = u[3]
    end

    return Geometry{N}(
        [parse_axis(a) for a in sample_axes],
        [parse_axis(a) for a in detector_axes],
        (parse_axis(image_axes[1]), parse_axis(image_axes[2])),
        parse_axis(beam_direction),
        parse_axis(sample_normal),
        parse_axis(sample_faceup),
        directions,
        shape,
        (Float64(pixel_size[1]), Float64(pixel_size[2])),
        (Float64(center[1]), Float64(center[2])),
        Float64(distance),
        Float64(wavelength),
    )
end

"""
    Geometry(; sample_axes, detector_axes, image_axes, beam_direction,
               sample_normal, sample_faceup, pixel_size, center, shape,
               distance, wavelength)

Build a geometry for a single detector laid out on a regular grid, in
xrayutilities conventions.

Axis fields accept either `"y-"`-style strings or 3-tuples / vectors.
`pixel_size` and `distance` must share a unit; nothing else depends on which.
`shape` is `(Nch1, Nch2)` — the same `(rows, cols)` order xrayutilities uses
for `init_area` — and `center` is `(cch1, cch2)`.
"""
function Geometry(;
        sample_axes,
        detector_axes,
        image_axes,
        beam_direction,
        sample_normal,
        sample_faceup,
        pixel_size,
        center,
        shape,
        distance,
        wavelength,
    )
    Nch1, Nch2 = Int(shape[1]), Int(shape[2])

    rpixel1 = Float64(pixel_size[1]) * parse_axis(image_axes[1])
    rpixel2 = Float64(pixel_size[2]) * parse_axis(image_axes[2])
    r_i_unit = normalize(parse_axis(beam_direction))
    # Pixel (0, 0) in the lab frame: detector center along the beam, walked
    # back to the corner by the center offset.
    r0 = Float64(distance) * r_i_unit -
         (Float64(center[1]) * rpixel1 + Float64(center[2]) * rpixel2)

    positions = Matrix{Float64}(undef, 3, Nch1 * Nch2)
    @inbounds for j2 in 0:(Nch2 - 1), j1 in 0:(Nch1 - 1)
        p = j1 * rpixel1 + j2 * rpixel2 + r0
        k = j2 * Nch1 + j1 + 1
        positions[1, k] = p[1]
        positions[2, k] = p[2]
        positions[3, k] = p[3]
    end

    return Geometry(positions, (Nch1, Nch2); sample_axes, detector_axes,
                    image_axes, beam_direction, sample_normal, sample_faceup,
                    pixel_size, center, distance, wavelength)
end

# Precomputed per-frame geometry: everything that depends on the angles and
# nothing that depends on the pixel. Built once per
# `(sample_angles, detector_angles)` tuple.
#
# Algebraic collapse of the per-pixel formula:
#     q = ms · (f · (md · rd_unit − r_i_unit))
#       = (f·ms·md) · rd_unit  +  (−f·ms·r_i_unit)
# so the whole per-frame transform is one matrix and one offset, and the
# per-pixel work is a single mat-vec plus a vec-add.
struct FrameTransform
    m_combined::Mat3    # f · ms · md
    q_offset::Vec3      # −f · ms · r_i_unit
end

function FrameTransform(g::Geometry, sample_angles, detector_angles)
    if length(sample_angles) != length(g.sample_axes)
        throw(ArgumentError("expected $(length(g.sample_axes)) sample angles, got $(length(sample_angles))"))
    end
    if length(detector_angles) != length(g.detector_axes)
        throw(ArgumentError("expected $(length(g.detector_axes)) detector angles, got $(length(detector_angles))"))
    end

    # sample chain: M_s = R_0 · R_1 · ... · R_{Ns-1}, ms = M_s^{-1}
    ms_fwd = one(Mat3)
    for (ax, a) in zip(g.sample_axes, sample_angles)
        ms_fwd = ms_fwd * rotation_arb(Float64(a), ax)
    end
    ms = inv(ms_fwd)

    # detector chain
    md = one(Mat3)
    for (ax, a) in zip(g.detector_axes, detector_angles)
        md = md * rotation_arb(Float64(a), ax)
    end

    f = 2π / g.wavelength
    r_i_unit = normalize(g.beam_direction)

    return FrameTransform(f * (ms * md), -f * (ms * r_i_unit))
end

# Per-pixel q in the sample frame, from a lab-frame unit direction.
@inline function direction_to_q(rd_unit::Vec3, ft::FrameTransform)
    return ft.m_combined * rd_unit + ft.q_offset
end

# Fill `q_storage` (shape `(D, npix)`) with the D selected components
# `q[indices[1]], …, q[indices[D]]` of the per-pixel q-vector. Projects the
# per-frame transform to a D×3 matrix once ahead of the pixel scan, saving
# `(3-D)/3` of the per-pixel mat-vec flops when D < 3. Pixels are independent,
# so the flat pixel range is chunked across tasks and each task writes a
# disjoint slice.
function _materialize_q!(q_storage::AbstractMatrix{Float64},
                         g::Geometry, ft::FrameTransform,
                         indices::NTuple{D, Int}; ntasks::Integer=6) where {D}
    npix = npixels(g)
    if size(q_storage) != (D, npix)
        throw(DimensionMismatch("workspace q_storage $(size(q_storage)) != ($D, $npix)"))
    end
    M = ft.m_combined
    # Column-major walk: k = (col-1)*D + row, so row = mod1(k, D), col = cld(k, D).
    m_proj = SMatrix{D, 3, Float64, D * 3}(
        ntuple(k -> @inbounds(M[indices[mod1(k, D)], cld(k, D)]), Val(D * 3))
    )
    qo = SVector{D, Float64}(ntuple(k -> ft.q_offset[indices[k]], Val(D)))
    dirs = g.directions

    # Tasks take a whole index range rather than letting `@tasks` chunk
    # `1:npix` itself: `@tasks` applies the body once per element via
    # `tforeach`, and that call boundary stops LLVM vectorizing the loop. Worth
    # 2x here, for identical output.
    @tasks for rng in index_chunks(1:npix; n=ntasks)
        @set scheduler = :static

        @inbounds for k in rng
            d = Vec3(dirs[1, k], dirs[2, k], dirs[3, k])
            v = m_proj * d + qo
            ntuple(t -> (q_storage[t, k] = v[t]; nothing), Val(D))
        end
    end
    return q_storage
end

# Eager `(3, data_shape...)` array of per-pixel q-vectors: rows 1/2/3 along the
# leading axis are the qx/qy/qz components.
# `sample_angles` and `detector_angles` are specified in degrees.
function pixel_q_array(g::Geometry, sample_angles, detector_angles)
    ft = FrameTransform(g, sample_angles, detector_angles)
    storage = Matrix{Float64}(undef, 3, npixels(g))
    _materialize_q!(storage, g, ft, (1, 2, 3))
    return reshape(storage, 3, data_shape(g)...)
end
