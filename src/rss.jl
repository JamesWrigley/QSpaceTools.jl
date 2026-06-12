"""
Reciprocal Space Slicing (RSS) — one detector frame → 2D q-space image.

Composes `Geometry` + `pixel_q_array` (per-pixel q in the sample frame)
with `fuzzygridder2d!` (fractional 2D binning). The output is a
`DimArray` indexed by two of `(qx, qy, qz)`; the third q-component is
collapsed (projected away) — see the session notes for why this is a
projection rather than a true slice.
"""

const Qx = Dim{:qx}
const Qy = Dim{:qy}
const Qz = Dim{:qz}
const Q_DIMS = (Qx, Qy, Qz)

# Translate a projection spec to a pair of component indices into the
# 3-vector returned by `pixel_to_q`.
const _Q_AXIS_IDX = (qx=1, qy=2, qz=3)
_proj_indices(p::Tuple{Symbol, Symbol}) = (_Q_AXIS_IDX[p[1]], _Q_AXIS_IDX[p[2]])
_proj_indices(p::Tuple{Integer, Integer}) = (Int(p[1]), Int(p[2]))

"""
    RSSWorkspace(geom, gridder_size)

Preallocated scratch space for `rss` / `rss!`. Holds the gridder's
`norm` denominator buffer and a `(2, Nch1, Nch2)` buffer for the
projected per-pixel q-components. Reuse one workspace across many frames
at the same geometry/grid size to avoid allocations in the hot loop.
"""
struct RSSWorkspace
    norm::Matrix{Float64}
    q_storage::Array{Float64, 3}    # (2, Nch1, Nch2): row 1 = projected x, row 2 = projected y
end

function RSSWorkspace(geom::Geometry, gridder_size::Tuple{Integer, Integer})
    nx, ny = Int(gridder_size[1]), Int(gridder_size[2])
    norm = Matrix{Float64}(undef, nx, ny)
    q_storage = Array{Float64, 3}(undef, 2, geom.shape[1], geom.shape[2])
    return RSSWorkspace(norm, q_storage)
end

"""
    allocate_output(geom, gridder_size) -> Matrix{Float64}

Allocate an `(nx, ny)` output buffer suitable for `rss!`.
"""
function allocate_output(::Geometry, gridder_size::Tuple{Integer, Integer})
    return Matrix{Float64}(undef, Int(gridder_size[1]), Int(gridder_size[2]))
end

@inline _combine_bounds(a, b) = (min(a[1], b[1]), max(a[2], b[2]),
                                 min(a[3], b[3]), max(a[4], b[4]))

# Per-component (xmin, xmax, ymin, ymax) over a `(2, N)` `points` matrix.
function _bounds_xy(points::AbstractMatrix, ntasks)
    tmapreduce(_combine_bounds, axes(points, 2); ntasks, scheduler=:static) do p
        xv = @inbounds points[1, p]
        yv = @inbounds points[2, p]
        (xv, xv, yv, yv)
    end
end

"""
    rss(frame, geom; sample_angles, detector_angles,
        gridder_size=(500, 500), projection=(:qx, :qz),
        bounds=nothing, fuzzy_width=nothing,
        workspace=nothing) -> DimArray

Bin a single detector `frame` into a 2D q-space image, allocating the output.
Equivalent to allocating an output with [`allocate_output`](@ref) and a
workspace with [`RSSWorkspace`](@ref), then calling [`rss!`](@ref).
`sample_angles` and `detector_angles` are specified in degrees.
"""
function rss(frame::AbstractMatrix, geom::Geometry;
             gridder_size::Tuple{Integer, Integer}=(500, 500),
             workspace::Union{Nothing, RSSWorkspace}=nothing,
             kwargs...)
    if isnothing(workspace)
        workspace = RSSWorkspace(geom, gridder_size)
    end

    image = allocate_output(geom, gridder_size)
    return rss!(image, frame, geom; workspace, kwargs...)
end

"""
    rss!(image, frame, geom; sample_angles, detector_angles,
         projection=(:qx, :qz), bounds=nothing, fuzzy_width=nothing,
         workspace=nothing) -> DimArray

In-place RSS: write the 2D q-space image into `image` and return a
`DimArray` that wraps it. The gridder size is `size(image)`. `workspace`
defaults to a freshly allocated [`RSSWorkspace`](@ref); pass one explicitly
to reuse buffers across frames.

`image` must be an `(nx, ny)` `AbstractMatrix{Float64}`. See [`rss`](@ref)
for the meaning of the remaining keyword arguments. `sample_angles` and
`detector_angles` are specified in degrees.
"""
function rss!(image::AbstractMatrix{Float64}, frame::AbstractMatrix, geom::Geometry;
              sample_angles,
              detector_angles,
              projection=(:qx, :qz),
              bounds::Union{Nothing, NTuple{4, Real}}=nothing,
              fuzzy_width::Union{Nothing, NTuple{2, Real}}=nothing,
              workspace::Union{Nothing, RSSWorkspace}=nothing,
              ntasks::Integer=4)
    if size(frame) != geom.shape
        throw(ArgumentError("frame size $(size(frame)) does not match geom.shape $(geom.shape)"))
    end
    nx, ny = size(image)

    ws = isnothing(workspace) ? RSSWorkspace(geom, (nx, ny)) : workspace
    if size(ws.norm) != (nx, ny)
        throw(ArgumentError("workspace.norm size $(size(ws.norm)) does not match gridder_size $((nx, ny))"))
    end
    if size(ws.q_storage) != (2, geom.shape[1], geom.shape[2])
        throw(ArgumentError("workspace.q_storage size $(size(ws.q_storage)) does not match (2, $(geom.shape[1]), $(geom.shape[2]))"))
    end

    ia, ib = _proj_indices(projection)
    ft = FrameTransform(geom, sample_angles, detector_angles)
    _materialize_q!(ws.q_storage, geom, ft, (ia, ib); ntasks)

    # Flat (2, npix) view into the projected components — same memory, just
    # collapses the trailing pixel dims so the gridder can index by one pixel
    # number.
    points = reshape(ws.q_storage, 2, length(frame))

    xmin, xmax, ymin, ymax = if isnothing(bounds)
        _bounds_xy(points, ntasks)
    else
        bounds
    end

    wx, wy = isnothing(fuzzy_width) ? (nothing, nothing) : fuzzy_width

    # Output partitioning: split along the slow (y) axis into disjoint column
    # slabs and run one gridder call per slab in parallel. Each task reads
    # `points`/`frame` in full but only zeros/writes its slab of image/norm,
    # so there's no contention and no reduction. The post-pass `image /= norm`
    # is per-slab inside the call.
    nslabs = clamp(Int(ntasks), 1, ny)
    chunk = cld(ny, nslabs)
    @tasks for i in 1:nslabs
        @set ntasks = nslabs
        @set scheduler = :static

        slab = ((i - 1) * chunk + 1):min(i * chunk, ny)
        fuzzygridder2d!(image, ws.norm, points, frame,
                        xmin, xmax, ymin, ymax;
                        wx, wy, y_range=slab)
    end

    dim_a = Q_DIMS[ia](_axis(xmin, xmax, nx))
    dim_b = Q_DIMS[ib](_axis(ymin, ymax, ny))
    return DimArray(image, (dim_a, dim_b))
end
