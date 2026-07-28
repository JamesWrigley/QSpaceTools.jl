"""
Fractional-overlap N-dimensional histogramming.

Port of xrayutilities `fuzzygridder2d`/`fuzzygridder3d` (`src/gridder2d.c`,
`src/gridder3d.c`), generalized to any dimensionality. Each input point
is treated as a finite box (default width = half a bin); contribution is
split fractionally across every output bin the box overlaps. Per-bin
numerator (`data`) and denominator (`norm`) are accumulated, divided at the
end. NaN data values and out-of-range points are dropped.

Bin centers follow xrayutilities' convention: with `n` bins on `[xmin, xmax]`,
the step is `dx = (xmax - xmin) / (n - 1)` and the bin index for value `x`
is `round((x - xmin) / dx)`. That is, `xmin` and `xmax` are the *centers* of
the first and last bins, not the edges.
"""

# These match the xrayutilities helpers verbatim, except `_bin_index` uses
# `unsafe_trunc(Int, t + 0.5)` instead of `round(Int, t)` to avoid the branchy
# banker's-rounding path. The gridder's `x < xmin || x > xmax` guard ensures
# t ≥ 0, so the `unsafe_` precondition holds.
@inline _bin_delta(xmin, xmax, n) = (xmax - xmin) / (n - 1)
@inline _bin_index(x, xmin, dx) = unsafe_trunc(Int, (x - xmin) / dx + 0.5)

# Allocation-free coord axis for the output DimArray.
@inline _axis(xmin, xmax, n) = range(Float64(xmin), Float64(xmax); length=Int(n))

# Scratch the gridder writes into but the caller owns: the `norm` denominator
# over the output grid, and the `_slab_index!` table over the points. Reuse one
# across calls at the same grid size. `_slab_index!` sizes the index on demand,
# so calls owning the whole last dimension never allocate it.
struct GridderWorkspace{T, D}
    norm::Array{T, D}
    lo_last::Vector{Int32}
    hi_last::Vector{Int32}
end

function GridderWorkspace(gridder_size::NTuple{D, Integer};
                          norm_type::Type{T}=Float64) where {T, D}
    return GridderWorkspace{T, D}(Array{T, D}(undef, Int.(gridder_size)...),
                                  Int32[], Int32[])
end

# Per-axis bin counts, minima, maxima, bin spacings, physical box widths and
# widths in bin units, derived once per call from the output size and the
# requested bounds.
#
# `use_slab_index` says whether accumulation should consult the workspace slab
# index. It is false when a call owns the whole last dimension: no point can be
# rejected then, so building the index would be wasted work.
struct GridParams{D, T}
    mins::NTuple{D, T}
    maxs::NTuple{D, T}
    deltas::NTuple{D, T}
    widths::NTuple{D, T}
    dwidths::NTuple{D, T}
    sizes::NTuple{D, Int}
    use_slab_index::Bool
end

# Fraction of an input box along dimension `d` that falls in bin `idx`, given
# the box's bin-index range `[lo, hi]` and its center `val`. Interior bins get
# `1/dw` (`dw` = box width in bin-widths), first/last touched bins the partial
# overlap with the bin edge, single-bin boxes 1.0.
@inline function _overlap(params::GridParams, d::Integer, idx, lo, hi, val)
    if lo == hi
        return 1.0
    end
    w_half = params.widths[d] / 2
    vmin, dv, dw = params.mins[d], params.deltas[d], params.dwidths[d]
    if idx == lo
        return (idx - (val - w_half - vmin + dv/2) / dv) / dw
    elseif idx == hi
        return ((val + w_half - vmin + dv/2) / dv - idx + 1) / dw
    else
        return 1.0 / dw
    end
end

# The per-point prologue both accumulation kernels share: drop NaN data and
# out-of-range points, then compute the bin-index range of the point's box per
# dimension.
@inline _point_coords(points::AbstractMatrix, p::Integer, ::GridParams{D}) where {D} =
    ntuple(d -> points[d, p], Val(D))

@inline function _point_inside(coords::NTuple{D}, params::GridParams{D}) where {D}
    mins, maxs = params.mins, params.maxs
    return !any(ntuple(d -> coords[d] < mins[d] || coords[d] > maxs[d], Val(D)))
end

@inline function _point_bins(coords::NTuple{D}, params::GridParams{D}) where {D}
    mins, deltas, widths, sizes = params.mins, params.deltas, params.widths, params.sizes
    lo = ntuple(Val(D)) do d
        lower = coords[d] - widths[d] / 2
        lower <= mins[d] ? 1 : _bin_index(lower, mins[d], deltas[d]) + 1
    end
    hi = ntuple(Val(D)) do d
        upper = coords[d] + widths[d] / 2
        min(_bin_index(upper, mins[d], deltas[d]) + 1, sizes[d])
    end
    return (lo, hi)
end

function _check_gridder_inputs(image::AbstractArray{T, N}, ws::GridderWorkspace,
                               points::AbstractMatrix, data, bounds::Tuple,
                               widths::Union{Nothing, Tuple},
                               last_range::Union{Nothing, UnitRange{Int}}) where {T, N}
    if size(image) != size(ws.norm)
        throw(DimensionMismatch("image $(size(image)) and norm $(size(ws.norm)) must match"))
    end
    if size(points, 1) != N
        throw(DimensionMismatch("points must have $N rows; got $(size(points, 1))"))
    end
    if size(points, 2) != length(data)
        throw(DimensionMismatch("points has $(size(points, 2)) columns but data has $(length(data)) entries"))
    end
    if length(bounds) != N
        throw(DimensionMismatch("bounds must have $N entries; got $(length(bounds))"))
    end
    if !isnothing(widths) && length(widths) != N
        throw(DimensionMismatch("widths must have $N entries; got $(length(widths))"))
    end

    if isnothing(last_range)
        return 1:size(image, N)
    else
        if first(last_range) < 1 || last(last_range) > size(image, N)
            throw(ArgumentError("last_range $last_range escapes 1:$(size(image, N))"))
        end
        return last_range
    end
end

# Split out from `fuzzygridder!` so that callers accumulating many point sets
# into one grid (RSM) can drive the three phases themselves. `sizes` is the bin
# count per dimension — of the output grid, or, for the projections of
# `_project_accumulate!`, per q-component.
function _grid_params(sizes::NTuple{N, Integer}, bounds::Tuple,
                      widths::Union{Nothing, Tuple},
                      use_slab_index::Bool) where {N}
    n = map(Int, sizes)
    mins = ntuple(d -> bounds[d][1], Val(N))
    maxs = ntuple(d -> bounds[d][2], Val(N))
    deltas = ntuple(d -> _bin_delta(mins[d], maxs[d], n[d]), Val(N))
    width_values = ntuple(Val(N)) do d
        isnothing(widths) || isnothing(widths[d]) ? deltas[d] / 2 : widths[d]
    end
    dwidths = ntuple(d -> width_values[d] / deltas[d], Val(N))
    return GridParams(mins, maxs, deltas, width_values, dwidths, n, use_slab_index)
end

function _clear_grid!(image::AbstractArray{T, N}, norm::AbstractArray,
                      last_range::UnitRange{Int}) where {T, N}
    ranges = ntuple(d -> d == N ? last_range : axes(image, d), Val(N))
    image_zero = zero(eltype(image))
    norm_zero = zero(eltype(norm))
    @inbounds for I in CartesianIndices(ranges)
        image[I] = image_zero
        norm[I] = norm_zero
    end
    return nothing
end

# Record every point's bin range in the last dimension, which decides the
# output slabs it can write to. Slab-parallel accumulation hands every task all
# the points, so without this table each task pays the full per-point prologue
# for the points it discards. Points out of range get an empty range
# (`lo > hi`).
function _slab_index!(ws::GridderWorkspace, points::AbstractMatrix,
                      params::GridParams{D}; ntasks::Integer=1) where {D}
    lo_last, hi_last = ws.lo_last, ws.hi_last
    npoints = size(points, 2)
    if length(lo_last) != npoints
        resize!(lo_last, npoints)
        resize!(hi_last, npoints)
    end
    mins, maxs, deltas, widths = params.mins, params.maxs, params.deltas, params.widths
    nlast = params.sizes[D]
    nt = ntasks
    @tasks for p in axes(points, 2)
        @set ntasks = nt
        @set scheduler = :static
        @inbounds begin
            c = points[D, p]
            if c < mins[D] || c > maxs[D]
                lo_last[p], hi_last[p] = Int32(1), Int32(0)   # empty: always skipped
            else
                lower = c - widths[D] / 2
                l = lower <= mins[D] ? 1 : _bin_index(lower, mins[D], deltas[D]) + 1
                h = min(_bin_index(c + widths[D] / 2, mins[D], deltas[D]) + 1, nlast)
                lo_last[p], hi_last[p] = Int32(l), Int32(h)
            end
        end
    end
    return nothing
end

# When `params.use_slab_index`, the workspace index (filled by `_slab_index!`
# over these same points and `params`) replaces the post-prologue slab test.
function _fuzzygridder_accumulate!(image::AbstractArray{T, N}, ws::GridderWorkspace,
                                   points::AbstractMatrix, data,
                                   params::GridParams,
                                   last_range::UnitRange{Int}) where {T, N}
    norm = ws.norm
    lo_last, hi_last = ws.lo_last, ws.hi_last
    use_slab_index = params.use_slab_index
    slab_lo, slab_hi = first(last_range), last(last_range)

    data_lin = vec(data)
    @inbounds for p in axes(points, 2)
        if use_slab_index && (hi_last[p] < slab_lo || lo_last[p] > slab_hi)
            continue
        end

        v = data_lin[p]
        coords = _point_coords(points, p, params)
        if isnan(v) || !_point_inside(coords, params)
            continue
        end
        lo, hi = _point_bins(coords, params)

        # Fast path: when the input box sits entirely inside one output bin
        # (common with the default half-bin width), all overlap factors are
        # 1.0 and the Cartesian loop becomes a single update.
        if all(ntuple(d -> lo[d] == hi[d], Val(N)))
            I = CartesianIndex(lo)
            image[I] += v
            norm[I] += one(eltype(norm))
            continue
        end

        # Clamp only the last dimension to this slab. Overlap weights use the
        # original lo/hi so slabbed and unslabbed calls remain equivalent.
        ranges = ntuple(Val(N)) do d
            d == N ? (max(lo[d], slab_lo):min(hi[d], slab_hi)) : (lo[d]:hi[d])
        end
        # Keep dimension 1 as an ordinary inner loop so writes are contiguous
        # and the dimensions 2:N weight is computed once per inner run.
        for J in CartesianIndices(Base.tail(ranges))
            outer_weight = prod(ntuple(Val(N - 1)) do k
                d = k + 1
                _overlap(params, d, J[k], lo[d], hi[d], coords[d])
            end)
            for i in ranges[1]
                w = outer_weight * _overlap(params, 1, i, lo[1], hi[1], coords[1])
                image[i, J] += v * w
                norm[i, J] += w
            end
        end
    end
end

# One point's contribution to one 2D grid over q-components `a` and `b`, given
# the bin ranges `lo`/`hi` the shared prologue already computed.
@inline function _accumulate_pair!(grid::AbstractMatrix, norm::AbstractMatrix,
                                   v, coords, lo, hi, a::Int, b::Int,
                                   params::GridParams)
    @inbounds begin
        la, ha, lb, hb = lo[a], hi[a], lo[b], hi[b]
        if la == ha && lb == hb
            grid[la, lb] += v
            norm[la, lb] += one(eltype(norm))
            return nothing
        end

        for j in lb:hb
            wj = _overlap(params, b, j, lb, hb, coords[b])
            for i in la:ha
                w = wj * _overlap(params, a, i, la, ha, coords[a])
                grid[i, j] += v * w
                norm[i, j] += w
            end
        end
    end
    return nothing
end

# Accumulate the points in `prange` into all `G` 2D grids at once, `grids[k]`
# binning the q-component pair `pairs[k]`. `params` is per q-component, not per
# grid: a component's bounds, bin spacing and box width are the same wherever
# it appears, so the per-point prologue is shared across the grids.
#
# Since all `D` components are tested up front, every grid keeps exactly the
# points inside the full q-box, including along the component it collapses.
#
# Parallelism here is over points, not output slabs: the grids share no axis to
# slab consistently. Callers hand disjoint `prange`s to tasks writing into
# their own `grids`/`norms`, and sum them afterwards.
function _project_accumulate!(grids::NTuple{G, AbstractMatrix},
                              norms::NTuple{G, AbstractMatrix},
                              pairs::NTuple{G, NTuple{2, Int}},
                              points::AbstractMatrix, data,
                              params::GridParams{D}, prange::UnitRange{Int}) where {G, D}
    data_lin = vec(data)
    @inbounds for p in prange
        v = data_lin[p]
        coords = _point_coords(points, p, params)
        if isnan(v) || !_point_inside(coords, params)
            continue
        end
        lo, hi = _point_bins(coords, params)

        map(grids, norms, pairs) do grid, norm, pair
            _accumulate_pair!(grid, norm, v, coords, lo, hi, pair[1], pair[2], params)
        end
    end
    return nothing
end

# `out` may alias `num`, as a one-shot `fuzzygridder!` call does; an
# accumulator keeps `num` intact and normalizes into a separate buffer.
function _normalize_grid!(out::AbstractArray{<:Any, N}, num::AbstractArray,
                          norm::AbstractArray, last_range::UnitRange{Int}) where {N}
    ranges = ntuple(d -> d == N ? last_range : axes(out, d), Val(N))
    out_zero = zero(eltype(out))
    # 1e-16 is a magic number copied from xrayutilities' gridder2d.c
    @inbounds for I in CartesianIndices(ranges)
        n = norm[I]
        out[I] = ifelse(n > 1e-16, num[I] / n, out_zero)
    end
end

"""
    fuzzygridder!(image, workspace, points, data, bounds;
                  widths=nothing, last_range=nothing) -> image

Fractionally grid points in `N` dimensions, where `N == ndims(image)`.
`points` has shape `(N, npoints)`, and `bounds[d]` is the `(min, max)` pair
for dimension `d`. `widths` optionally supplies the full box width in each
dimension; `nothing` entries default to half the corresponding bin spacing.

`workspace` is a `GridderWorkspace` matching `size(image)`, holding the `norm`
denominator and the gridder's scratch.

The output buffers are cleared before accumulation and normalized afterward.
`last_range` restricts all writes to a slab of the last output dimension,
allowing concurrent calls on disjoint slabs.

Entries of `data` that are NaN are skipped.
"""
function fuzzygridder!(image::AbstractArray{T, N}, ws::GridderWorkspace,
                       points::AbstractMatrix, data, bounds::Tuple;
                       widths::Union{Nothing, Tuple}=nothing,
                       last_range::Union{Nothing, UnitRange{Int}}=nothing) where {T, N}
    slab = _check_gridder_inputs(image, ws, points, data, bounds, widths, last_range)
    params = _grid_params(size(image), bounds, widths, length(slab) != size(image, N))

    if params.use_slab_index
        _slab_index!(ws, points, params)
    end
    _clear_grid!(image, ws.norm, slab)
    _fuzzygridder_accumulate!(image, ws, points, data, params, slab)
    _normalize_grid!(image, image, ws.norm, slab)

    return image
end
