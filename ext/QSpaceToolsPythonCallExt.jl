module QSpaceToolsPythonCallExt

using QSpaceTools: QSpaceTools
using LinearAlgebra: normalize
using PythonCall: Py, PyArray, pyconvert, pyimport, pyisinstance, pybuiltins, pytype

# Build a BakedIntegrator straight from the Python `baked` dict returned by
# `bake_for_batch`, skipping the HDF5 layer entirely. The +1 index shift and
# the (H,W)→(W,H) flip live in `QSpaceTools._baked_from`; here we only supply
# the per-field accessor (every field is a plain dict entry, so one rule).
function QSpaceTools.load_baked(baked::Py)
    if !pyisinstance(baked, pybuiltins.dict)
        throw(ArgumentError("load_baked(::Py) expected the dict returned by bake_for_batch, got a $(pytype(baked))"))
    end

    QSpaceTools._baked_from((T, key) -> pyconvert(T, baked[key]))
end

# Function barrier: `pos` wraps a numpy array zero-copy, so its type is only
# known here.
function _lab_positions(pos::AbstractArray, R::QSpaceTools.Mat3,
                        offset::QSpaceTools.Vec3, npix::Int)
    src = reshape(pos, 3, npix)
    positions = Matrix{Float64}(undef, 3, npix)
    @inbounds for k in 1:npix
        p = R * (QSpaceTools.Vec3(src[1, k], src[2, k], src[3, k]) + offset)
        positions[1, k] = p[1]
        positions[2, k] = p[2]
        positions[3, k] = p[3]
    end
    return positions
end

"""
    Geometry(geom; distance, sample_axes, detector_axes, beam_direction,
             sample_normal, sample_faceup, wavelength,
             extra_geom_axes=("y+", "z+", "x+"), center=nothing)

Build a [`Geometry`](@ref) from an EXtra-geom detector geometry, taking one
position per pixel so that multi-module detectors need no assembly.

`extra_geom_axes` says where EXtra-geom's own `(x, y, z)` axes point in the lab
frame the rest of the geometry uses, in the same `"y-"`-style notation as the
other axis arguments (explicit vectors work too, for a detector that isn't
axis-aligned). EXtra-geom's `z` is its beam axis, so the third entry must agree
with `beam_direction`.

`distance` is the sample-detector distance **in metres**, matching EXtra-geom's
units: its `z` coordinates carry only relative module offsets, so the distance
is added along EXtra-geom's beam axis before the mapping into the lab frame.

`data_shape` comes out reversed from EXtra-geom's `(nmodules, ss, fs)`, i.e.
`(fs, ss, nmodules)` — numpy stores that shape row-major and Julia stores the
reverse column-major, so the two agree pixel for pixel. **Frames passed to
`rss`/`rsm` must already be transposed** into that order. `center` and
`image_axes` are likewise reported in reversed, assembled-image order.

`center` is the beam position in assembled-image pixels, defaulting to the
middle of the snapped detector image.
"""
function QSpaceTools.Geometry(geom::Py;
                              distance,
                              sample_axes,
                              detector_axes,
                              beam_direction,
                              sample_normal,
                              sample_faceup,
                              wavelength,
                              extra_geom_axes=("y+", "z+", "x+"),
                              center=nothing)
    base = pyimport("extra_geom.base")
    if !pyisinstance(geom, base.DetectorGeometryBase)
        throw(ArgumentError("Geometry(::Py) expected an EXtra-geom DetectorGeometryBase, got a $(pytype(geom))"))
    end

    # Columns are the lab-frame images of EXtra-geom's x, y and z.
    ax = map(QSpaceTools.parse_axis, extra_geom_axes)
    R = QSpaceTools.Mat3(hcat(ax...))
    beam = normalize(QSpaceTools.parse_axis(beam_direction))
    if !isapprox(ax[3], beam; atol=1e-12)
        throw(ArgumentError("extra_geom_axes[3] $(ax[3]) is EXtra-geom's beam axis and must match beam_direction $beam"))
    end

    d = Float64(distance)

    # EXtra-geom hands back (nmodules, ss, fs, 3) in C order, so `.T` is a
    # (3, fs, ss, nmodules) Fortran-ordered view — already the Julia layout,
    # and the flat pixel order matches a reversed data array with no permute.
    pos = PyArray(geom.get_pixel_positions().T)
    shape = reverse(pyconvert(NTuple{3, Int}, geom.expected_data_shape))
    npix = prod(shape)
    positions = _lab_positions(pos, R, d * QSpaceTools.AXIS_VECS["z+"], npix)

    px = pyconvert(Float64, geom.pixel_size)

    # `GeometryFragment.snap` reindexes (x, y) to (y, x) without changing sign,
    # so a +1 step along the assembled image's rows runs along physical +y and
    # along its columns +x. Reversed here like everything else.
    image_axes = (R * QSpaceTools.AXIS_VECS["x+"], R * QSpaceTools.AXIS_VECS["y+"])

    c = if isnothing(center)
        reverse(pyconvert(NTuple{2, Int}, geom._snapped().size_yx)) ./ 2
    else
        (Float64(center[1]), Float64(center[2]))
    end

    return QSpaceTools.Geometry(positions, shape;
                                sample_axes, detector_axes, image_axes,
                                beam_direction, sample_normal, sample_faceup,
                                pixel_size=(px, px), center=c, distance=d,
                                wavelength)
end

end # module QSpaceToolsPythonCallExt
