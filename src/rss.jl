const Qx = Dim{:qx}
const Qy = Dim{:qy}
const Qz = Dim{:qz}
const Q_DIMS = (Qx, Qy, Qz)

# Which two q-components each projection grids.
const _PROJECTION_PAIRS = ((1, 2), (1, 3), (2, 3))

# Translate a projection spec to a pair of component indices into the
# 3-vector returned by `pixel_to_q`.
const _Q_AXIS_IDX = (qx=1, qy=2, qz=3)
_proj_indices(p::Tuple{Symbol, Symbol}) = (_Q_AXIS_IDX[p[1]], _Q_AXIS_IDX[p[2]])
_proj_indices(p::Tuple{Integer, Integer}) = (Int(p[1]), Int(p[2]))

"""
    QProjections(qxqy, qxqz, qyqz)

The three axis-pair projections of a reciprocal space map, as returned by
`rsm(...; output=:projections)` and [`allocate_output`](@ref).
"""
struct QProjections{XY <: AbstractMatrix, XZ <: AbstractMatrix, YZ <: AbstractMatrix}
    qxqy::XY
    qxqz::XZ
    qyqz::YZ
end

# One entry per output grid: its `size` and the rows of the materialized q
# matrix its dimensions bin. A projection takes two of the three rows; the
# single grid of `rss` and of `output=:volume` takes all of them, in order.
struct GridSpec{D}
    size::NTuple{D, Int}
    rows::NTuple{D, Int}
end

# One `GridSpec` per output grid, plus `component_sizes`, the bin count per
# q-component. The number of grids `G` encodes the output mode: one for `rss`
# and `output=:volume`, three for `output=:projections`.
struct GridPlan{D, G, S <: NTuple{G, GridSpec}}
    specs::S
    component_sizes::NTuple{D, Int}
end

_plan(specs::NTuple{G, GridSpec}, sizes::NTuple{D, Int}) where {G, D} =
    GridPlan{D, G, typeof(specs)}(specs, sizes)

# `rss`: one grid, over both components it materializes.
GridPlan(gridder_size::NTuple{2, Integer}) =
    _plan((GridSpec(Int.(gridder_size), (1, 2)),), Int.(gridder_size))

# `rsm`: `gridder_size` is the bin count per q-axis in both modes, so the
# projection over axes `(a, b)` is `(gridder_size[a], gridder_size[b])` either
# way.
function GridPlan(gridder_size::NTuple{3, Integer}, output::Symbol)
    n = Int.(gridder_size)
    if output === :volume
        return _plan((GridSpec(n, (1, 2, 3)),), n)
    elseif output === :projections
        specs = ntuple(Val(3)) do i
            a, b = _PROJECTION_PAIRS[i]
            GridSpec((n[a], n[b]), (a, b))
        end
        return _plan(specs, n)
    else
        throw(ArgumentError("output must be :projections or :volume; got $(repr(output))"))
    end
end

# The plan a set of existing output buffers implies. Each projection shares an
# axis with both others, so the per-q-axis bin counts are recoverable, and must
# agree.
GridPlan(volume::AbstractArray{<:Any, 3}) = GridPlan(size(volume), :volume)

function GridPlan(p::QProjections)
    nx, ny = size(p.qxqy)
    nx2, nz = size(p.qxqz)
    ny2, nz2 = size(p.qyqz)
    if (nx2, ny2, nz2) != (nx, ny, nz)
        throw(ArgumentError("projection sizes disagree: qxqy $(size(p.qxqy)), " *
                            "qxqz $(size(p.qxqz)), qyqz $(size(p.qyqz))"))
    end
    return GridPlan((nx, ny, nz), :projections)
end

_allocate(spec::GridSpec{D}) where {D} = Array{Float64, D}(undef, spec.size...)

# Output grids travel through the core as a tuple, one per GridSpec.
_wrap(::GridPlan{D, 1}, grids::Tuple) where {D} = grids[1]
_wrap(::GridPlan{3, 3}, grids::Tuple) = QProjections(grids...)

_output_grids(image::AbstractMatrix) = (image,)
_output_grids(volume::AbstractArray{<:Any, 3}) = (volume,)
_output_grids(p::QProjections) = (p.qxqy, p.qxqz, p.qyqz)

Base.:(==)(a::QProjections, b::QProjections) =
    all(map(==, _output_grids(a), _output_grids(b)))
Base.isapprox(a::QProjections, b::QProjections; kwargs...) =
    all(map((x, y) -> isapprox(x, y; kwargs...), _output_grids(a), _output_grids(b)))

# Scratch for the fused projections path: a private copy of every output grid
# and its norm, per task. The three grids share no axis, so no single output
# dimension can be slabbed across all of them; parallelism is over points
# instead and each task accumulates into its own replica, summed by
# `_reduce_replicas!` at the end. The grids are small enough for this to be
# affordable (~786 KB per replica at 128 bins/axis); the volume keeps slabbing.
struct ProjectionScratch{G}
    grids::Vector{NTuple{G, Matrix{Float64}}}    # [task][grid]
    norms::Vector{NTuple{G, Matrix{Float64}}}
end

# At least one replica always exists, so the grid sizes can be read back off it.
function ProjectionScratch(specs::NTuple{G, GridSpec{2}}, ntasks::Integer) where {G}
    scratch = ProjectionScratch{G}(Vector{NTuple{G, Matrix{Float64}}}(),
                                   Vector{NTuple{G, Matrix{Float64}}}())
    _grow_replicas!(scratch, map(spec -> spec.size, specs), max(ntasks, 1))
    return scratch
end

_scratch_sizes(scratch::ProjectionScratch) = map(size, first(scratch.grids))

# A call may ask for more tasks than the workspace was built for; replicas are
# added once and kept.
function _grow_replicas!(scratch::ProjectionScratch, sizes::Tuple, n::Integer)
    while length(scratch.grids) < n
        push!(scratch.grids, map(sz -> zeros(Float64, sz), sizes))
        push!(scratch.norms, map(sz -> zeros(Float64, sz), sizes))
    end
    return nothing
end

# Scratch for the slabbed path. The numerator lives here rather than in the
# caller's output buffer so that summing the map leaves it intact.
struct SlabWorkspace{D}
    grid::Array{Float64, D}
    gridder::GridderWorkspace{Float64, D}
end

SlabWorkspace(size::NTuple{D, Int}) where {D} =
    SlabWorkspace{D}(Array{Float64, D}(undef, size...), GridderWorkspace(size))

# One numerator over the single grid when it is filled by slabs; per-task
# replicas when several grids are filled together in one pass over the points.
_grid_scratch(specs::Tuple{GridSpec}, ::Integer) = SlabWorkspace(specs[1].size)
_grid_scratch(specs::NTuple{3, GridSpec{2}}, ntasks::Integer) =
    ProjectionScratch(specs, ntasks)

"""
    RSMWorkspace(geom, gridder_size; output=:projections, ntasks=4)

Preallocated scratch space for
[`rss`](@ref)/[`rss!`](@ref)/[`rsm`](@ref)/[`rsm!`](@ref), sized for a
`geom`-shaped frame gridded onto `gridder_size` (a 2-tuple for RSS, a 3-tuple
for RSM). It holds every buffer the accumulation needs, including the running
unnormalized grids; only the normalized output lives outside it. `output` picks
which grids an RSM workspace serves, as it does for [`rsm`](@ref). Pass
`ntasks` if you already know how many tasks will be used; the internal fields
are resized if needed.
"""
struct RSMWorkspace{D, S}
    scratch::S                    # see `_grid_scratch`
    q_storage::Matrix{Float64}    # (D, npix): rows are the kept q-components
end

function RSMWorkspace(geom::Geometry, plan::GridPlan{D}; ntasks::Integer=4) where {D}
    scratch = _grid_scratch(plan.specs, ntasks)
    q_storage = Matrix{Float64}(undef, D, npixels(geom))
    return RSMWorkspace{D, typeof(scratch)}(scratch, q_storage)
end

RSMWorkspace(geom::Geometry, gridder_size::NTuple{2, Integer}; ntasks::Integer=4) =
    RSMWorkspace(geom, GridPlan(gridder_size); ntasks)

RSMWorkspace(geom::Geometry, gridder_size::NTuple{3, Integer};
             output::Symbol=:projections, ntasks::Integer=4) =
    RSMWorkspace(geom, GridPlan(gridder_size, output); ntasks)

"""
    allocate_output(geom, gridder_size; output=:projections)

Allocate the output buffer(s) for `rss!` (a 2-tuple `gridder_size`) or `rsm!`
(a 3-tuple). `output=:volume` gives one `gridder_size`-shaped array,
`output=:projections` gives a [`QProjections`](@ref) of the three axis-pair
matrices.
"""
allocate_output(plan::GridPlan) = _wrap(plan, map(_allocate, plan.specs))

allocate_output(::Geometry, gridder_size::NTuple{2, Integer}) =
    allocate_output(GridPlan(gridder_size))

allocate_output(::Geometry, gridder_size::NTuple{3, Integer};
                output::Symbol=:projections) =
    allocate_output(GridPlan(gridder_size, output))

# Bounds are carried as a flat (min₁, max₁, min₂, max₂, …) tuple: odd slots
# are minima, even slots maxima.
@inline function _combine_bounds(a::NTuple{M}, b::NTuple{M}) where {M}
    ntuple(k -> isodd(k) ? min(a[k], b[k]) : max(a[k], b[k]), Val(M))
end

# Per-component extrema over a `(D, N)` `points` matrix.
function _point_bounds(points::AbstractMatrix, ::Val{D}, ntasks) where {D}
    tmapreduce(_combine_bounds, axes(points, 2); ntasks, scheduler=:static) do p
        ntuple(k -> @inbounds(points[cld(k, 2), p]), Val(2D))
    end
end

# Angles are either one tuple shared by every frame or a per-frame collection.
# A bare `Real` covers the common single-axis chain.
_angles_at(angles::Tuple{Vararg{Real}}, ::Integer, ::Integer) = angles
_angles_at(angles::Real, ::Integer, ::Integer) = (angles,)
function _angles_at(angles, i::Integer, nframes::Integer)
    if length(angles) != nframes
        throw(ArgumentError("expected $nframes per-frame angles, got $(length(angles))"))
    end
    a = angles[begin + i - 1]
    return a isa Real ? (a,) : Tuple(a)
end

# Fill `q` with frame `f`'s per-pixel q-components.
function _frame_q!(q::AbstractMatrix, geom::Geometry, indices::NTuple{D, Int},
                   sample_angles, detector_angles, f, nframes, ntasks) where {D}
    ft = FrameTransform(geom, _angles_at(sample_angles, f, nframes),
                        _angles_at(detector_angles, f, nframes))
    _materialize_q!(q, geom, ft, indices; ntasks)
    return q
end

# Streaming pass: reduce the global q-extrema one frame at a time, so only one
# frame's q-vectors are ever resident. Depends only on the geometry and the
# angles, never on the frame data.
function _scan_q_bounds(q::AbstractMatrix, geom::Geometry, indices::NTuple{D, Int},
                        nframes, sample_angles, detector_angles, ntasks) where {D}
    acc = ntuple(k -> isodd(k) ? Inf : -Inf, Val(2D))
    for f in 1:nframes
        points = _frame_q!(q, geom, indices, sample_angles, detector_angles,
                           f, nframes, ntasks)
        acc = _combine_bounds(acc, _point_bounds(points, Val(D), ntasks))
    end
    return acc
end

# How many frames a pair of angle arguments describes: `nothing` for one tuple
# shared by every frame, a length for a per-frame collection.
_angle_length(::Tuple{Vararg{Real}}) = nothing
_angle_length(::Real) = nothing
_angle_length(angles) = length(angles)

function _angle_nframes(sample_angles, detector_angles)
    ns = _angle_length(sample_angles)
    nd = _angle_length(detector_angles)
    if !isnothing(ns) && !isnothing(nd) && ns != nd
        throw(ArgumentError("$ns sample angles but $nd detector angles"))
    end
    return something(ns, nd, 1)
end

"""
    q_bounds(geom; sample_angles, detector_angles, projection=nothing,
             nframes=nothing, ntasks=4)

The q-extent a scan will cover, as a flat `(qxmin, qxmax, qymin, qymax, qzmin,
qzmax)` tuple suitable for the `bounds` argument of [`rsm`](@ref) and
[`RSMAccumulator`](@ref).

Only the geometry and the angles are needed, no frame data, so a streaming
caller can fix the grid before reading a single image. `sample_angles` and
`detector_angles` are given in degrees, either as one angle tuple shared by
every frame or as per-frame collections; `nframes` is only needed when both are
shared. Pass `projection` to get the 4-tuple for an [`rss`](@ref) axis pair
instead.
"""
function q_bounds(geom::Geometry; sample_angles, detector_angles,
                  projection::Union{Nothing, Tuple}=nothing,
                  nframes::Union{Nothing, Integer}=nothing, ntasks::Integer=4)
    indices = isnothing(projection) ? (1, 2, 3) : _proj_indices(projection)
    n = isnothing(nframes) ? _angle_nframes(sample_angles, detector_angles) : Int(nframes)
    q = Matrix{Float64}(undef, length(indices), npixels(geom))
    return _scan_q_bounds(q, geom, indices, n, sample_angles, detector_angles, ntasks)
end

function _check_scratch(scratch::SlabWorkspace, specs::Tuple)
    if size(scratch.grid) != specs[1].size
        throw(ArgumentError("workspace grid size $(size(scratch.grid)) does not match gridder_size $(specs[1].size)"))
    end
    return nothing
end

function _check_scratch(scratch::ProjectionScratch, specs::Tuple)
    sizes = map(spec -> spec.size, specs)
    got = _scratch_sizes(scratch)
    if got != sizes
        throw(ArgumentError("workspace grid sizes $got do not match $sizes"))
    end
    return nothing
end

function _check_workspace(ws::RSMWorkspace{D}, geom::Geometry, plan::GridPlan{D}) where {D}
    if size(ws.q_storage) != (D, npixels(geom))
        throw(ArgumentError("workspace.q_storage size $(size(ws.q_storage)) does not match ($D, $(npixels(geom)))"))
    end
    _check_scratch(ws.scratch, plan.specs)
    return nothing
end

# How accumulation is divided up, fixed for an accumulator's lifetime: the grid
# parameters plus the ranges the tasks own. One type per strategy, see the two
# `_accumulate_frame!` methods.
struct SlabState{D, T, S}
    params::GridParams{D, T}
    slabs::S
end

struct ChunkState{D, T, G, C}
    params::GridParams{D, T}
    pairs::NTuple{G, Tuple{Int, Int}}
    chunks::C
end

# One grid, filled by output slab: each task owns a slab of the last dimension
# and reads every point, so no task writes where another does and nothing has
# to be summed afterwards. The slab index is only worth building when there is
# more than one slab to reject points from.
function _begin_accumulate!(scratch::SlabWorkspace, ::Tuple, sizes::NTuple{D, Int},
                            bnds::Tuple, widths::Tuple, ::Integer,
                            ntasks::Integer) where {D}
    slabs = index_chunks(1:sizes[D]; n=ntasks)
    state = SlabState(_grid_params(sizes, bnds, widths, length(slabs) > 1), slabs)
    _clear!(state, scratch)
    return state
end

function _clear!(state::SlabState, scratch::SlabWorkspace)
    @tasks for slab in state.slabs
        @set scheduler = :static
        _clear_grid!(scratch.grid, scratch.gridder.norm, slab)
    end
    return nothing
end

function _accumulate_frame!(state::SlabState, scratch::SlabWorkspace, points,
                            frame, ntasks::Integer)
    if state.params.use_slab_index
        _slab_index!(scratch.gridder, points, state.params; ntasks)
    end

    @tasks for slab in state.slabs
        @set scheduler = :static
        _fuzzygridder_accumulate!(scratch.grid, scratch.gridder, points, frame,
                                  state.params, slab)
    end
    return nothing
end

function _normalize_into!(state::SlabState, scratch::SlabWorkspace, grids::Tuple,
                          ::Integer)
    @tasks for slab in state.slabs
        @set scheduler = :static
        _normalize_grid!(grids[1], scratch.grid, scratch.gridder.norm, slab)
    end
    return nothing
end

# Several grids at once, filled by point chunk: one pass over the points fills
# every grid, each task into its own replicas because the grids share no axis
# to slab. `_reduce_replicas!` sums and normalizes them at the end.
function _begin_accumulate!(scratch::ProjectionScratch{G}, specs::NTuple{G, GridSpec{2}},
                            sizes::NTuple{3, Int}, bnds::Tuple, widths::Tuple,
                            npoints::Integer, ntasks::Integer) where {G}
    chunks = index_chunks(1:npoints; n=ntasks)
    _grow_replicas!(scratch, map(spec -> spec.size, specs), length(chunks))
    state = ChunkState(_grid_params(sizes, bnds, widths, false),
                       map(spec -> (spec.rows[1], spec.rows[2]), specs), chunks)
    _clear!(state, scratch)
    return state
end

function _clear!(state::ChunkState, scratch::ProjectionScratch{G}) where {G}
    @tasks for t in 1:length(state.chunks)
        @set scheduler = :static
        for k in 1:G
            fill!(scratch.grids[t][k], 0.0)
            fill!(scratch.norms[t][k], 0.0)
        end
    end
    return nothing
end

function _accumulate_frame!(state::ChunkState, scratch::ProjectionScratch, points,
                            frame, ::Integer)
    @tasks for t in 1:length(state.chunks)
        @set scheduler = :static
        _project_accumulate!(scratch.grids[t], scratch.norms[t], state.pairs, points,
                             frame, state.params, state.chunks[t])
    end
    return nothing
end

function _normalize_into!(state::ChunkState, scratch::ProjectionScratch{G},
                          grids::NTuple{G}, ntasks::Integer) where {G}
    for k in 1:G
        _reduce_replicas!(grids[k], scratch, k, length(state.chunks), ntasks)
    end
    return nothing
end

# Sum every task's replica of grid `k` into `grid` and normalize in the same
# pass. The 1e-16 floor matches `_normalize_grid!`.
function _reduce_replicas!(grid, scratch::ProjectionScratch, k::Integer,
                           nreplicas::Integer, ntasks::Integer)
    nums = [scratch.grids[t][k] for t in 1:nreplicas]
    dens = [scratch.norms[t][k] for t in 1:nreplicas]

    @tasks for rng in index_chunks(1:length(grid); n=ntasks)
        @set scheduler = :static

        @inbounds for i in rng
            num = 0.0
            den = 0.0
            for t in eachindex(nums)
                num += nums[t][i]
                den += dens[t][i]
            end
            grid[i] = ifelse(den > 1e-16, num / den, 0.0)
        end
    end
    return nothing
end

"""
    RSMAccumulator(geom; bounds, gridder_size=(200, 200, 200), output=:projections,
                   fuzzy_width=nothing, ntasks=4)
    RSMAccumulator(geom; bounds, gridder_size=(500, 500), projection=(:qx, :qz), …)

A reciprocal space map built one frame at a time. Add frames with
[`push!`](@ref) or [`append!`](@ref), and read the map out with `sum(acc)` (or
`sum!(outputs, acc)` into your own buffers) at any point. Summing does not
disturb the accumulator, so frames can keep arriving afterwards.

This is what [`rsm!`](@ref) uses internally, and is the interface to use when
the frames do not all fit in memory.

`bounds` is required and fixes the grid up front: a flat `(qxmin, qxmax, qymin,
qymax, qzmin, qzmax)` tuple, or the 4-tuple of the chosen axis pair when
`gridder_size` is a 2-tuple. [`q_bounds`](@ref) computes it from the geometry
and the scan angles alone, without reading any frames. `output`, `gridder_size`
and `fuzzy_width` mean what they do for [`rsm`](@ref) and [`rss`](@ref); every
intermediate buffer is allocated and owned by the accumulator.

```julia
bounds = q_bounds(geom; sample_angles=thetas, detector_angles=(γ, δ))
acc = RSMAccumulator(geom; bounds, gridder_size=(200, 200, 200))
for (frame, θ) in image_stream
    push!(acc, frame; sample_angles=θ, detector_angles=(γ, δ))
end
projections = sum(acc)
```
"""
mutable struct RSMAccumulator{D, G, GE <: Geometry, P <: GridPlan{D, G},
                              W <: RSMWorkspace{D}, S, B <: Tuple}
    const geom::GE
    const plan::P
    const workspace::W
    const state::S
    const indices::NTuple{D, Int}
    const bounds::B          # flat (min₁, max₁, min₂, max₂, …)
    const ntasks::Int
    nframes::Int
end

function RSMAccumulator(geom::Geometry, plan::GridPlan{D, G}, indices::NTuple{D, Int},
                        bounds::NTuple{L, Real};
                        fuzzy_width::Union{Nothing, Tuple}=nothing,
                        ntasks::Integer=4,
                        workspace::Union{Nothing, RSMWorkspace}=nothing) where {D, G, L}
    if L != 2 * D
        throw(ArgumentError("expected $(2 * D) bounds for a $D-component map, got $L"))
    end
    ws = isnothing(workspace) ? RSMWorkspace(geom, plan; ntasks) : workspace
    _check_workspace(ws, geom, plan)

    # Float64 throughout, so a caller's integer `bounds` can't leave the
    # GridParams fields at mixed types.
    b = map(Float64, bounds)
    bnds = ntuple(d -> (b[2d - 1], b[2d]), Val(D))
    widths = isnothing(fuzzy_width) ? ntuple(_ -> nothing, Val(D)) : fuzzy_width
    state = _begin_accumulate!(ws.scratch, plan.specs, plan.component_sizes, bnds,
                               widths, npixels(geom), ntasks)

    return RSMAccumulator{D, G, typeof(geom), typeof(plan), typeof(ws), typeof(state),
                          typeof(b)}(geom, plan, ws, state, indices, b, ntasks, 0)
end

_plan_and_indices(gridder_size::NTuple{2, Integer}, ::Symbol, projection) =
    (GridPlan(gridder_size), _proj_indices(projection))
_plan_and_indices(gridder_size::NTuple{3, Integer}, output::Symbol, _) =
    (GridPlan(gridder_size, output), (1, 2, 3))

function RSMAccumulator(geom::Geometry; bounds::Tuple,
                        gridder_size::Tuple{Vararg{Integer}}=(200, 200, 200),
                        output::Symbol=:projections,
                        projection=(:qx, :qz), kwargs...)
    plan, indices = _plan_and_indices(gridder_size, output, projection)
    return RSMAccumulator(geom, plan, indices, bounds; kwargs...)
end

# The number of frames `frames` holds. It must match `data_shape(geom)` in its
# leading dimensions; everything after those is the frame axis, so a bare frame
# is a one-frame stack.
function _check_frames(geom::Geometry{N}, frames::AbstractArray) where {N}
    shape = data_shape(geom)
    if ndims(frames) < N || size(frames)[1:N] != shape
        throw(ArgumentError("frame size $(size(frames)) does not start with data_shape(geom) $shape"))
    end
    return length(frames) ÷ npixels(geom)
end

# Accumulate one frame against the q-vectors already in the workspace.
function _push_q!(acc::RSMAccumulator, frame::AbstractVector)
    _accumulate_frame!(acc.state, acc.workspace.scratch, acc.workspace.q_storage,
                       frame, acc.ntasks)
    acc.nframes += 1
    return acc
end

"""
    push!(acc::RSMAccumulator, frame; sample_angles, detector_angles)

Add one detector `frame`, taken at the given angles (in degrees), to the map.
`frame` must match `data_shape(geom)`.
"""
function Base.push!(acc::RSMAccumulator, frame::AbstractArray;
                    sample_angles, detector_angles)
    if _check_frames(acc.geom, frame) != 1
        throw(ArgumentError("push! takes a single frame; use append! for a stack"))
    end

    _frame_q!(acc.workspace.q_storage, acc.geom, acc.indices, sample_angles,
              detector_angles, 1, 1, acc.ntasks)
    return _push_q!(acc, reshape(frame, npixels(acc.geom)))
end

"""
    append!(acc::RSMAccumulator, frames; sample_angles, detector_angles)

Add a `(data_shape(geom)..., Nframes)` stack of frames to the map. The angles
are given in degrees, either as one angle tuple shared by every frame or as a
per-frame collection of length `Nframes`.
"""
function Base.append!(acc::RSMAccumulator, frames::AbstractArray;
                      sample_angles, detector_angles)
    nframes = _check_frames(acc.geom, frames)
    frames_flat = reshape(frames, npixels(acc.geom), nframes)

    for i in 1:nframes
        _frame_q!(acc.workspace.q_storage, acc.geom, acc.indices, sample_angles,
                  detector_angles, i, nframes, acc.ntasks)
        _push_q!(acc, @view frames_flat[:, i])
    end
    return acc
end

"""
    empty!(acc::RSMAccumulator)

Discard every frame accumulated so far, leaving the geometry and grid intact.
"""
function Base.empty!(acc::RSMAccumulator)
    _clear!(acc.state, acc.workspace.scratch)
    acc.nframes = 0
    return acc
end

Base.length(acc::RSMAccumulator) = acc.nframes

"""
    sum(acc::RSMAccumulator)
    sum!(outputs, acc::RSMAccumulator)

The map accumulated so far, normalized per bin, as `DimArray`s: one for
`rss`-shaped and `output=:volume` accumulators, a [`QProjections`](@ref) of
three for `output=:projections`.

`sum` allocates the output; `sum!` writes into `outputs`, which must have the
shape [`allocate_output`](@ref) would give. Neither disturbs the accumulator,
so more frames can be added afterwards.
"""
Base.sum(acc::RSMAccumulator) = sum!(allocate_output(acc.plan), acc)

function Base.sum!(outputs::Union{AbstractArray{Float64}, QProjections},
                   acc::RSMAccumulator)
    grids = _output_grids(outputs)
    map(grids, acc.plan.specs) do grid, spec
        if size(grid) != spec.size
            throw(ArgumentError("output size $(size(grid)) does not match gridder_size $(spec.size)"))
        end
    end

    _normalize_into!(acc.state, acc.workspace.scratch, grids, acc.ntasks)
    wrapped = map((grid, spec) -> DimArray(grid, _grid_dims(spec, acc.indices, acc.bounds)),
                  grids, acc.plan.specs)
    return _wrap(acc.plan, wrapped)
end

# Shared core of `rss!`/`rsm!`: accumulate every frame of `frames`, keeping the
# q-components named by `indices`, and hand back the accumulator for the caller
# to sum.
function _accumulate_all!(plan::GridPlan{D}, indices::NTuple{D, Int},
                          frames::AbstractArray, geom::Geometry,
                          ws::RSMWorkspace{D}, sample_angles, detector_angles,
                          bounds, fuzzy_width, ntasks::Integer) where {D}
    nframes = _check_frames(geom, frames)
    frames_flat = reshape(frames, npixels(geom), nframes)

    b = if isnothing(bounds)
        _scan_q_bounds(ws.q_storage, geom, indices, nframes, sample_angles,
                       detector_angles, ntasks)
    else
        map(Float64, bounds)
    end
    acc = RSMAccumulator(geom, plan, indices, b; fuzzy_width, ntasks, workspace=ws)

    # A single-frame scan leaves that frame's q-vectors in the buffer, so the
    # accumulation pass can use them as-is instead of recomputing.
    reuse_q = isnothing(bounds) && nframes == 1
    for i in 1:nframes
        if !(reuse_q && i == 1)
            _frame_q!(ws.q_storage, geom, indices, sample_angles, detector_angles,
                      i, nframes, ntasks)
        end
        _push_q!(acc, @view frames_flat[:, i])
    end

    return acc
end

# Output axes for one grid: the q-component each dimension resolves, over the
# bounds actually used. `indices` maps a q-matrix row back to its component.
function _grid_dims(spec::GridSpec{D}, indices::NTuple, b) where {D}
    return ntuple(Val(D)) do d
        r = spec.rows[d]
        Q_DIMS[indices[r]](_axis(b[2r - 1], b[2r], spec.size[d]))
    end
end

"""
    rss(frame, geom; sample_angles, detector_angles,
        gridder_size=(500, 500), projection=(:qx, :qz),
        bounds=nothing, fuzzy_width=nothing,
        workspace=nothing) -> DimArray

Bin a single detector `frame` into a 2D q-space image, allocating the output.
Equivalent to allocating an output with [`allocate_output`](@ref) and a
workspace with [`RSMWorkspace`](@ref), then calling [`rss!`](@ref).
`sample_angles` and `detector_angles` are specified in degrees.
"""
function rss(frame::AbstractArray, geom::Geometry;
             gridder_size::NTuple{2, Integer}=(500, 500),
             workspace::Union{Nothing, RSMWorkspace{2}}=nothing,
             kwargs...)
    plan = GridPlan(gridder_size)
    if isnothing(workspace)
        workspace = RSMWorkspace(geom, plan)
    end

    return rss!(allocate_output(plan), frame, geom; workspace, kwargs...)
end

"""
    rss!(image, frame, geom; sample_angles, detector_angles,
         projection=(:qx, :qz), bounds=nothing, fuzzy_width=nothing,
         workspace=nothing) -> DimArray

In-place RSS: write the 2D q-space image into `image` and return a
`DimArray` that wraps it. The gridder size is `size(image)`. `workspace`
defaults to a freshly allocated [`RSMWorkspace`](@ref); pass one explicitly
to reuse buffers across frames.

`image` must be an `(nx, ny)` `AbstractMatrix{Float64}`, and `frame` must match
`data_shape(geom)`. See [`rss`](@ref) for the meaning of the remaining keyword
arguments. `sample_angles` and `detector_angles` are specified in degrees.
"""
function rss!(image::AbstractMatrix{Float64}, frame::AbstractArray, geom::Geometry;
              sample_angles,
              detector_angles,
              projection=(:qx, :qz),
              bounds::Union{Nothing, NTuple{4, Real}}=nothing,
              fuzzy_width::Union{Nothing, NTuple{2, Real}}=nothing,
              workspace::Union{Nothing, RSMWorkspace{2}}=nothing,
              ntasks::Integer=4)
    plan = GridPlan(size(image))
    ws = isnothing(workspace) ? RSMWorkspace(geom, plan; ntasks) : workspace
    indices = _proj_indices(projection)

    acc = _accumulate_all!(plan, indices, frame, geom, ws, sample_angles,
                           detector_angles, bounds, fuzzy_width, ntasks)

    return sum!(image, acc)
end

"""
    rsm(frames, geom; sample_angles, detector_angles,
        gridder_size=(200, 200, 200), output=:projections, bounds=nothing,
        fuzzy_width=nothing, workspace=nothing)

Bin a stack of detector `frames` into q-space, allocating the output.
Equivalent to allocating it with [`allocate_output`](@ref) and a workspace with
[`RSMWorkspace`](@ref), then calling [`rsm!`](@ref). `sample_angles` and
`detector_angles` are specified in degrees.

`output=:projections` (the default) returns a [`QProjections`](@ref) of the
three axis-pair projections as `DimArray`s, computing them directly without
ever materializing the volume. `output=:volume` returns the full 3D volume as
one `DimArray`. `gridder_size` is the bin count per q-axis either way, so a
projection over `(qx, qz)` is `(gridder_size[1], gridder_size[3])` bins.

A projection is *not* a sum over the volume's third axis: as in [`rss`](@ref)
the collapsed component is averaged away by the per-bin normalization. It does
cover the same points as the volume: a point outside the q-range along the
collapsed axis is dropped from the projection too.
"""
function rsm(frames::AbstractArray, geom::Geometry;
             gridder_size::NTuple{3, Integer}=(200, 200, 200),
             output::Symbol=:projections,
             workspace::Union{Nothing, RSMWorkspace{3}}=nothing,
             kwargs...)
    plan = GridPlan(gridder_size, output)
    if isnothing(workspace)
        workspace = RSMWorkspace(geom, plan; ntasks=get(kwargs, :ntasks, 4))
    end

    return rsm!(allocate_output(plan), frames, geom; workspace, kwargs...)
end

"""
    rsm!(outputs, frames, geom; sample_angles, detector_angles, bounds=nothing,
         fuzzy_width=nothing, workspace=nothing)

In-place RSM: accumulate every frame into `outputs` and return `DimArray`s
wrapping it. `outputs` is either a 3D array (the full volume, indexed by
`(qx, qy, qz)`) or a [`QProjections`](@ref) of three matrices, and picks
which of the two `rsm` modes runs; the gridder size follows from its size(s).

`frames` is `(data_shape(geom)..., Nframes)`. `sample_angles` and
`detector_angles` are given in degrees, either as one angle tuple shared by
every frame or as a per-frame collection of length `Nframes`.

`bounds` is a flat `(qxmin, qxmax, qymin, qymax, qzmin, qzmax)` tuple; when
omitted the q-extent is computed in a first streaming pass over the frames,
which materializes each frame's q-vectors twice. See [`rss!`](@ref) for the
remaining keyword arguments.
"""
function rsm!(outputs::Union{AbstractArray{Float64, 3}, QProjections},
              frames::AbstractArray, geom::Geometry;
              sample_angles,
              detector_angles,
              bounds::Union{Nothing, NTuple{6, Real}}=nothing,
              fuzzy_width::Union{Nothing, NTuple{3, Real}}=nothing,
              workspace::Union{Nothing, RSMWorkspace{3}}=nothing,
              ntasks::Integer=4)
    plan = GridPlan(outputs)
    ws = isnothing(workspace) ? RSMWorkspace(geom, plan; ntasks) : workspace

    acc = _accumulate_all!(plan, (1, 2, 3), frames, geom, ws, sample_angles,
                           detector_angles, bounds, fuzzy_width, ntasks)

    return sum!(outputs, acc)
end
