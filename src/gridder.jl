"""
FuzzyGridder2D — fractional-overlap 2D histogramming.

Port of xrayutilities `fuzzygridder2d` (`src/gridder2d.c`). Each input point
is treated as a finite box (default width = half a bin); contribution is
split fractionally across every output bin the box overlaps. Per-bin
numerator (`data`) and denominator (`norm`) are accumulated, divided at the
end. NaN inputs and out-of-range points are dropped.

Bin centers follow xrayutilities' convention: with `n` bins on `[xmin, xmax]`,
the step is `dx = (xmax - xmin) / (n - 1)` and the bin index for value `x`
is `round((x - xmin) / dx)`. That is, `xmin` and `xmax` are the *centers* of
the first and last bins, not the edges.
"""

# delta and gindex match the xrayutilities helpers verbatim, except `_bin_index`
# uses `unsafe_trunc(Int, t + 0.5)` instead of `round(Int, t)` — single
# `cvttsd2si` rather than the branchy banker's-rounding path. The gridder's
# `isnan(x) || x < xmin || x > xmax` guards ensure t ≥ 0 / non-NaN before we
# get here, so the `unsafe_` precondition holds.
@inline _bin_delta(xmin, xmax, n) = (xmax - xmin) / (n - 1)
@inline _bin_index(x, xmin, dx) = unsafe_trunc(Int, (x - xmin) / dx + 0.5)

# Fraction of an input box that falls in bin `idx`, when the box's bin-index
# range is `[lo, hi]`. `val` is the box center, `w_half = w/2`, `vmin` the
# axis origin, `dv` the bin width, `dw = w/dv` the box width in bin-widths.
# Interior fully-covered bins get `1/dw`; first/last touched bins get partial
# overlap with the bin edge; single-bin boxes get 1.0.
@inline function _overlap(idx, lo, hi, val, w_half, vmin, dv, dw)
    lo == hi  && return 1.0
    idx == lo && return (idx - (val - w_half - vmin + dv/2) / dv) / dw
    idx == hi && return ((val + w_half - vmin + dv/2) / dv - idx + 1) / dw
    return 1.0 / dw
end

# A `LinRange` value: same numeric content as the eager vector form, but
# allocation-free — useful as the coord axis attached to the output DimArray.
@inline _axis(xmin, xmax, n) = range(Float64(xmin), Float64(xmax); length=Int(n))

"""
    fuzzygridder2d!(image, norm, points, data, xmin, xmax, ymin, ymax;
                    wx=nothing, wy=nothing, y_range=nothing) -> image

Non-allocating fuzzy gridding. `image` and `norm` are the output and
accumulator buffers — both `(nx, ny)` `Matrix{Float64}` — and are zeroed
before accumulation.

`points` is a `(2, N)` `AbstractMatrix{Float64}` whose first row holds the
x-coordinates and second row the y-coordinates. `data` has `N` entries
(any shape, iterated by linear index). The `(2, N)` packing keeps each
pixel's `(x, y)` pair in adjacent memory — friendlier to the cache than
two parallel arrays.

`y_range::UnitRange{Int}`, if given, restricts both the zero-fill and the
write-back to columns `image[:, y_range]` / `norm[:, y_range]`. Pixels
whose box doesn't overlap `y_range` are skipped. Used to give each task a
disjoint output slab when called concurrently.
"""
function fuzzygridder2d!(image::AbstractMatrix{Float64},
                         norm::AbstractMatrix{Float64},
                         points::AbstractMatrix{Float64}, data,
                         xmin::Real, xmax::Real, ymin::Real, ymax::Real;
                         wx::Union{Real, Nothing}=nothing,
                         wy::Union{Real, Nothing}=nothing,
                         y_range::Union{Nothing, UnitRange{Int}}=nothing)
    if size(image) != size(norm)
        throw(DimensionMismatch("image $(size(image)) and norm $(size(norm)) must match"))
    end
    if size(points, 1) != 2
        throw(DimensionMismatch("points must have 2 rows (x, y); got $(size(points, 1))"))
    end
    if size(points, 2) != length(data)
        throw(DimensionMismatch("points has $(size(points, 2)) columns but data has $(length(data)) entries"))
    end
    nx, ny = size(image)

    slab_lo, slab_hi = if isnothing(y_range)
        (1, ny)
    else
        if first(y_range) < 1 || last(y_range) > ny
            throw(ArgumentError("y_range $y_range escapes 1:$ny"))
        end
        (first(y_range), last(y_range))
    end

    dx = _bin_delta(xmin, xmax, nx)
    dy = _bin_delta(ymin, ymax, ny)
    wxv = isnothing(wx) ? dx / 2 : Float64(wx)
    wyv = isnothing(wy) ? dy / 2 : Float64(wy)
    # Box width measured in bin-widths. Dividing each per-bin overlap by
    # `dwx*dwy` makes the per-point fractions sum to ≈1 across all bins it
    # touches, so the contribution conserves total weight.
    dwx = wxv / dx
    dwy = wyv / dy

    @views fill!(image[:, slab_lo:slab_hi], 0.0)
    @views fill!(norm[:,  slab_lo:slab_hi], 0.0)

    data_lin = vec(data)
    @inbounds for p in axes(points, 2)
        x = points[1, p]
        y = points[2, p]
        v = data_lin[p]
        if isnan(v) || isnan(x) || isnan(y) || x < xmin || x > xmax || y < ymin || y > ymax
            continue
        end

        # Bin-index range the box spans (1-based). The lower-bound clamp is
        # needed because the box can extend below `xmin` even when the point
        # itself is in range, which would give `_bin_index` a result < 1.
        ox1 = (x - wxv/2) <= xmin ? 1 : _bin_index(x - wxv/2, xmin, dx) + 1
        ox2 = min(_bin_index(x + wxv/2, xmin, dx) + 1, nx)
        oy1 = (y - wyv/2) <= ymin ? 1 : _bin_index(y - wyv/2, ymin, dy) + 1
        oy2 = min(_bin_index(y + wyv/2, ymin, dy) + 1, ny)

        # Skip the pixel entirely if its box doesn't touch this slab.
        if oy2 < slab_lo || oy1 > slab_hi
            continue
        end

        # Fast path: when the input box sits entirely inside one output bin
        # (common with the default `wxv = dx/2` whenever the pixel isn't near
        # a bin edge), the fy/fx branches collapse to 1.0 each and the inner
        # double loop becomes a single bin update. The overlap check above
        # guarantees oy1 ∈ [slab_lo, slab_hi] when oy1 == oy2.
        if ox1 == ox2 && oy1 == oy2
            image[ox1, oy1] += v
            norm[ox1, oy1]  += 1.0
            continue
        end

        # Clamp the column range to this slab — fy still uses the unclamped
        # oy1/oy2 because the per-bin fraction depends on the box's true span.
        oy1_eff = max(oy1, slab_lo)
        oy2_eff = min(oy2, slab_hi)

        # Outer loop over `k` (slow axis), inner over `j` (fast axis) so the
        # `image[j, k]` writes stride contiguously in column-major memory.
        for k in oy1_eff:oy2_eff
            fy = _overlap(k, oy1, oy2, y, wyv/2, ymin, dy, dwy)
            for j in ox1:ox2
                fx = _overlap(j, ox1, ox2, x, wxv/2, xmin, dx, dwx)
                w = fx * fy
                image[j, k] += v * w
                norm[j, k]  += w
            end
        end
    end

    # 1e-16 is a magic number copied from xrayutilities' gridder2d.c
    @inbounds for k in slab_lo:slab_hi
        @simd for j in 1:nx
            n = norm[j, k]
            image[j, k] = ifelse(n > 1e-16, image[j, k] / n, 0.0)
        end
    end

    return image
end
