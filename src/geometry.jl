"""
Geometry layer for reciprocal-space conversion.

Mirrors xrayutilities' `QConversion.area` + `init_area` for the HXRD case
(`ang2q_conversion_area` in `src/qconversion.c`). UB is identity; the
sample/detector rotation chain and an area-detector parameterization are
enough to map every pixel `(i, j)` to a momentum-transfer vector
`q = M_s^{-1} (M_d r̂_d - r̂_i) · 2π/λ` in the sample frame.
"""

const Vec3 = SVector{3, Float64}
const Mat3 = SMatrix{3, 3, Float64, 9}

# h·c expressed in eV·Å. Matches the constant xrayutilities' `en2lam` uses
# (CODATA h, c, qe), so `energy2wavelength` here agrees with `xu.en2lam` to
# roundoff.
const _HC_EV_ANGSTROM = 12398.419843320026

"""
    energy2wavelength(energy_eV) -> wavelength_Å

Convert a photon energy in eV to its wavelength in Å. Matches
`xrayutilities.en2lam`.
"""
@inline energy2wavelength(energy_eV::Real) = _HC_EV_ANGSTROM / Float64(energy_eV)

"""
    wavelength2energy(wavelength_Å) -> energy_eV

Convert a photon wavelength in Å to its energy in eV. Inverse of
[`energy2wavelength`](@ref).
"""
@inline wavelength2energy(wavelength_Å::Real) = _HC_EV_ANGSTROM / Float64(wavelength_Å)

const AXIS_VECS = Dict(
    "x+" => Vec3( 1.0,  0.0,  0.0),  "x-" => Vec3(-1.0, 0.0, 0.0),
    "y+" => Vec3( 0.0,  1.0,  0.0),  "y-" => Vec3( 0.0, -1.0, 0.0),
    "z+" => Vec3( 0.0,  0.0,  1.0),  "z-" => Vec3( 0.0, 0.0, -1.0),
)

parse_axis(v::AbstractVector) = Vec3(v)
parse_axis(v::Tuple) = Vec3(v)
function parse_axis(s::AbstractString)
    if !haskey(AXIS_VECS, s)
        throw(ArgumentError("Invalid axis string: $(repr(s))"))
    end

    AXIS_VECS[s]
end

# Right-handed rotation by `θ` around unit vector `e` (Rodrigues' formula).
# Axes like "y-" are passed as (0,-1,0); the sign is folded into the vector.
@inline function rotation_arb(θ::Real, e::Vec3)
    s, c = sincos(θ)
    c1 = 1 - c
    ex, ey, ez = e

    @SMatrix [
        c + ex*ex*c1      ex*ey*c1 - ez*s   ex*ez*c1 + ey*s
        ey*ex*c1 + ez*s   c + ey*ey*c1      ey*ez*c1 - ex*s
        ez*ex*c1 - ey*s   ez*ey*c1 + ex*s   c + ez*ez*c1
    ]
end

"""
    Geometry(; sample_axes, detector_axes, image_axes, beam_direction,
               sample_normal, sample_faceup, pixel_size, center, shape,
               distance, wavelength)

Geometry of an area-detector experiment, in xrayutilities conventions.

Axis fields accept either `"y-"`-style strings or 3-tuples / vectors. All
spatial units must match (`pixel_size`, `distance` typically in mm or m).
`shape` is `(Nch1, Nch2)` — same `(rows, cols)` order xrayutilities uses for
`init_area`. `center` is `(cch1, cch2)`.

`sample_normal` and `sample_faceup` are stored for completeness but not used
by the current `pixel_to_q` kernel (which assumes UB = I, the default of
`HXRD.Ang2Q.area`). They will become load-bearing for RSM-side machinery
that consults the experiment frame.
"""
struct Geometry
    sample_axes::Vector{Vec3}
    detector_axes::Vector{Vec3}
    image_axes::NTuple{2, Vec3}
    beam_direction::Vec3
    sample_normal::Vec3
    sample_faceup::Vec3
    pixel_size::NTuple{2, Float64}
    center::NTuple{2, Float64}
    shape::NTuple{2, Int}
    distance::Float64
    wavelength::Float64
end

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
    return Geometry(
        [parse_axis(a) for a in sample_axes],
        [parse_axis(a) for a in detector_axes],
        (parse_axis(image_axes[1]), parse_axis(image_axes[2])),
        parse_axis(beam_direction),
        parse_axis(sample_normal),
        parse_axis(sample_faceup),
        (Float64(pixel_size[1]), Float64(pixel_size[2])),
        (Float64(center[1]), Float64(center[2])),
        (Int(shape[1]), Int(shape[2])),
        Float64(distance),
        Float64(wavelength),
    )
end

# Precomputed per-frame geometry: matrices that depend on angles only, not
# on pixel. Built once per `(sample_angles, detector_angles)` and closed
# over by the lazy q-array.
#
# Algebraic collapse of the per-pixel formula:
#     q = ms · (f · (md · rd_unit − r_i_unit))
#       = (f·ms·md) · rd_unit  +  (−f·ms·r_i_unit)
# so we store only the combined matrix `m_combined` and offset `q_offset`,
# plus the pixel-step vectors and a fused `r0 = rcch_lab − rcchp` origin.
# Per pixel: 2 vec scales + 2 vec adds + `normalize` + 1 mat-vec + 1 vec add
# (was: 2 vec scales + 3 vec adds + `normalize` + 2 mat-vecs + 1 scalar-vec
# mul + 1 vec sub).
struct FrameTransform
    rpixel1::Vec3       # pixel-step vector along image axis 1 (length = pwidth1)
    rpixel2::Vec3       # pixel-step vector along image axis 2 (length = pwidth2)
    r0::Vec3            # rcch_lab − rcchp: pixel (0, 0) lab position
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

    rpixel1 = g.pixel_size[1] * g.image_axes[1]
    rpixel2 = g.pixel_size[2] * g.image_axes[2]
    r_i_unit = normalize(g.beam_direction)
    rcch_lab = g.distance * r_i_unit
    rcchp = g.center[1] * rpixel1 + g.center[2] * rpixel2
    f = 2π / g.wavelength

    r0 = rcch_lab - rcchp
    m_combined = f * (ms * md)
    q_offset = -f * (ms * r_i_unit)

    return FrameTransform(rpixel1, rpixel2, r0, m_combined, q_offset)
end

"""
    pixel_to_q(j1, j2, ft::FrameTransform) -> SVector{3, Float64}

Per-pixel q in the sample frame. `j1`/`j2` are 0-based pixel indices along
image axes 1 and 2 (matching xrayutilities; subtract 1 when calling from
Julia's 1-based loops).
"""
@inline function pixel_to_q(j1::Real, j2::Real, ft::FrameTransform)
    rd_lab = j1 * ft.rpixel1 + j2 * ft.rpixel2 + ft.r0
    rd_unit = normalize(rd_lab)
    return ft.m_combined * rd_unit + ft.q_offset
end

# Fill `q_storage` (shape `(D, Nch1, Nch2)`) with the D selected components
# `q[indices[1]], …, q[indices[D]]` of the per-pixel q-vector. Projects the
# per-frame transform to a D×3 matrix once ahead of the pixel scan, saving
# `(3-D)/3` of the per-pixel mat-vec flops when D < 3.
#
# Recurrence `rd_lab(j1+1, j2) = rd_lab(j1, j2) + ft.rpixel1` lets the inner
# loop step by one vec-add instead of recomputing the affine combo. Math
# stays in Float64 so per-pixel q precision (~1e-15) is independent of the
# caller's storage eltype. Parallelizes over columns: each j2 writes to a
# disjoint slice of q_storage, so no contention; the within-column recurrence
# is preserved per task.
function _materialize_q!(q_storage::AbstractArray{Float64, 3},
                         g::Geometry, ft::FrameTransform,
                         indices::NTuple{D, Int}; ntasks::Integer=6) where {D}
    if size(q_storage) != (D, g.shape[1], g.shape[2])
        throw(DimensionMismatch("workspace q_storage $(size(q_storage)) != ($D, $(g.shape[1]), $(g.shape[2]))"))
    end
    M = ft.m_combined
    # Column-major walk: k = (col-1)*D + row, so row = mod1(k, D), col = cld(k, D).
    m_proj = SMatrix{D, 3, Float64, D * 3}(
        ntuple(k -> @inbounds(M[indices[mod1(k, D)], cld(k, D)]), Val(D * 3))
    )
    qo = SVector{D, Float64}(ntuple(k -> ft.q_offset[indices[k]], Val(D)))

    Nch1, Nch2 = g.shape
    @tasks for j2 in 0:(Nch2 - 1)
        @set begin
            scheduler = :static
            ntasks = ntasks
        end

        rd_lab = j2 * ft.rpixel2 + ft.r0
        @inbounds for j1_idx in 1:Nch1
            rd_unit = normalize(rd_lab)
            xy = m_proj * rd_unit + qo
            ntuple(k -> (q_storage[k, j1_idx, j2 + 1] = xy[k]; nothing), Val(D))
            rd_lab += ft.rpixel1
        end
    end
    return q_storage
end

"""
    pixel_q_array(g::Geometry, sample_angles, detector_angles) -> Array{Float64, 3}

Eager `(3, Nch1, Nch2)` array of per-pixel q-vectors: rows 1/2/3 are the
qx/qy/qz components. Pixel indices `(j1, j2)` are 1-based externally; the
kernel uses 0-based indices internally to match xrayutilities.
"""
function pixel_q_array(g::Geometry, sample_angles, detector_angles)
    ft = FrameTransform(g, sample_angles, detector_angles)
    storage = Array{Float64, 3}(undef, 3, g.shape[1], g.shape[2])
    _materialize_q!(storage, g, ft, (1, 2, 3))
    return storage
end
