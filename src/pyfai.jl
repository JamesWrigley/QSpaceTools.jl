"""
PyFAI baked-integrator consumer.

Loads an HDF5 file from `bake_for_batch.write_hdf5(...)` and applies it
to a frame or stack of frames. Returns a `DimArray` with axes from
`bin_centers0` / `bin_centers1` and unit strings in `metadata`.

Per-frame math (derived in the exporter docstring):

    I[bin] = (Σ raw_w[p]·X[pix(p)]) / (Σ corr_w[p]·1) over finite pixels

`_fused_spmv!()` walks each bin's NNZ once, accumulating S and N together with
on-the-fly `isfinite` masking.

The sparse layout is `CSC(Aᵀ) ≡ CSR(A)`: pyFAI's `(indptr, indices)` map
1:1 onto `(colptr, rowval)` of the transposed CSC after a 1-shift.

Input is an N-dim array whose leading two axes are the (W, H) frame; any
further axes are treated as scan dimensions and looped over. `DimArray`
inputs propagate their trailing dims; plain arrays get auto-named `Frame`
(for one extra dim) or `Dim{:dim_3}, …`.

Output shape:
    single frame  (W,H)            →  (nbins0,)                 1D
                                      (nbins1, nbins0)          2D
    K extra dims  (W,H,d1,…,dK)    →  (nbins0, d1,…,dK)         1D
                                      (nbins1, nbins0, d1,…,dK) 2D

For 2D, the leading axis is `Azimuthal` because the CSR row layout is
`row = bin_rad * nbins_azim + bin_azim` (radial-major, C order); reshaping
that flat layout into Julia's column-major puts azimuth on the fast axis.
"""

const Radial    = Dim{:radial}
const Azimuthal = Dim{:azimuthal}
const Frame     = Dim{:frame}

struct BakedIntegrator
    # CSR of A (≡ CSC of Aᵀ), shape (nbins, npix). `colptr[bin]:colptr[bin+1]-1`
    # gives each bin's NNZ slice; `rowval` holds 1-based pixel indices into
    # `vec(frame)`; `raw_nz` and `corr_nz` are the parallel weight arrays.
    colptr::Vector{Int32}
    rowval::Vector{Int32}
    raw_nz::Vector{Float32}
    corr_nz::Vector{Float32}

    bin_centers0::Vector{Float32}            # radial axis (display units)
    bin_centers1::Vector{Float32}            # azimuthal axis (empty for 1D)
    shape::Tuple{Vararg{Int}}                # frame shape, reverse(python_shape)
    unit0::String
    unit1::String                            # "" for 1D
    split::String
    npt0::Int                                # radial bins
    npt1::Int                                # azimuthal bins (0 for 1D)
    ndim::Int                                # 1 or 2
end

function Base.show(io::IO, b::BakedIntegrator)
    if b.ndim == 1
        print(io, BakedIntegrator, "(1D, $(b.shape), npt=$(b.npt0))")
    else
        print(io, BakedIntegrator, "(2D, $(b.shape), npt=($(b.npt0), $(b.npt1)))")
    end
end

function Base.:(==)(a::BakedIntegrator, b::BakedIntegrator)
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(BakedIntegrator))
end

function Base.hash(b::BakedIntegrator, h::UInt)
    for f in fieldnames(BakedIntegrator)
        h = hash(getfield(b, f), h)
    end

    h
end

# Shared assembly for both `load_baked` methods (HDF5 file here, Python `baked`
# dict in the PythonCall extension). `get(T, key)::T` pulls one field from the
# backing store; everything that must stay in lockstep across backends lives
# here: the +1 index shift, the (H,W)→(W,H) flip, and the 1D/2D field branch.
function _baked_from(get)
    ndim    = get(Int, "ndim")
    shape_c = get(Vector{Int}, "shape")
    length(shape_c) == 2 ||
        error("only 2D detector shapes are supported here, got $shape_c")
    H, W = shape_c

    bin_centers1, unit1, npt1 = if ndim == 2
        get(Vector{Float32}, "bin_centers1"), get(String, "unit1"), get(Int, "npt1")
    else
        Float32[], "", 0
    end

    # Reversed (W, H) so column-major `vec` matches pyFAI's C-order flat index
    # without a remap.
    BakedIntegrator(get(Vector{Int32}, "indptr")  .+ Int32(1),
                    get(Vector{Int32}, "indices") .+ Int32(1),
                    get(Vector{Float32}, "data_raw"),
                    get(Vector{Float32}, "data_corr"),
                    get(Vector{Float32}, "bin_centers0"),
                    bin_centers1, (W, H),
                    get(String, "unit0"), unit1, get(String, "split"),
                    get(Int, "npt0"), npt1, ndim)
end

# pyFAI scalars/strings are stored as HDF5 attributes; the CSR arrays and bin
# centers as datasets. `_baked_from`'s accessor dispatches on this split.
const _BAKED_ATTRS = ("shape", "ndim", "unit0", "unit1", "split", "npt0", "npt1")

function load_baked(path::AbstractString)
    h5open(path, "r") do f
        get(::Type{T}, key) where {T} =
            if key in _BAKED_ATTRS
                raw = read_attribute(f, key)
                raw isa T ? raw : T(raw)
            else
                read(f[key])::T
            end

        b = _baked_from(get)

        # We don't apply pyFAI's per-frame |raw - dummy| <= delta_dummy mask;
        # warn if it's set so the user isn't surprised by drift from
        # `ai.integrate1d`.
        dummy       = Float32(read_attribute(f, "dummy"))
        delta_dummy = Float32(read_attribute(f, "delta_dummy"))
        if (isfinite(dummy) && dummy != 0) || (isfinite(delta_dummy) && delta_dummy != 0)
            @warn "baked integrator has nonzero pyFAI dummy/delta_dummy; \
                   these are NOT applied here, so I(q) may differ from \
                   ai.integrate1d on pixels matching the dummy sentinel" dummy delta_dummy
        end

        b
    end
end

function _meta(b::BakedIntegrator)
    md = Dict("unit0" => b.unit0, "split" => b.split)
    if b.ndim == 2
        md["unit1"] = b.unit1
    end

    return md
end

# Wrap a flat output into a DimArray. 2D reshape is zero-copy: row index
# `bin_rad * npt1 + bin_azim` + column-major puts azimuth fast. Trailing
# dims (for K ≥ 1 extra post-frame dims) come from a DimArray input via
# `otherdims`; plain arrays get `Frame` (K==1) or `Dim{:dim_3}, …` (K≥2).
function _wrap(b::BakedIntegrator, out::AbstractArray{Float32}, extra_shape::Tuple, frames)
    nd_frame = length(b.shape)
    trail_dims = if frames isa AbstractDimArray && !isempty(extra_shape)
        otherdims(frames, ntuple(identity, nd_frame))
    elseif isempty(extra_shape)
        ()
    elseif length(extra_shape) == 1
        (Frame(1:extra_shape[1]),)
    else
        ntuple(i -> Dim{Symbol(:dim_, i + nd_frame)}(1:extra_shape[i]),
               length(extra_shape))
    end

    if b.ndim == 1
        DimArray(reshape(out, b.npt0, extra_shape...),
                 (Radial(b.bin_centers0), trail_dims...);
                 name=:intensity, metadata=_meta(b))
    else
        DimArray(reshape(out, b.npt1, b.npt0, extra_shape...),
                 (Azimuthal(b.bin_centers1), Radial(b.bin_centers0),
                  trail_dims...);
                 name=:intensity, metadata=_meta(b))
    end
end

"""
    output_size(b::BakedIntegrator, frames::AbstractArray)

Calculate the size of the output array needed for `integrate!(out, b, frames)`.
"""
output_size(b::BakedIntegrator) = b.ndim == 2 ? (b.npt0, b.npt1) : (b.npt0,)

function output_size(b::BakedIntegrator, frames::AbstractArray)
    nd_frame = length(b.shape)
    extra_size = size(frames)[nd_frame + 1:end]
    (output_size(b)..., extra_size...)
end

# Fused single-frame kernel. Each bin's NNZ is walked once; S and N
# accumulate in scalar registers with on-the-fly `isfinite` masking.
function _fused_spmv!(I::AbstractVector{Float32}, b::BakedIntegrator, x::AbstractVector)
    colptr = b.colptr
    rowval = b.rowval
    raw_nz = b.raw_nz
    corr_nz = b.corr_nz

    @inbounds for bin in eachindex(I)
        s = 0.0
        n = 0.0

        for p in colptr[bin]:colptr[bin+1]-1
            pix = rowval[p]
            v   = x[pix]
            ok  = isfinite(v)

            s = muladd(raw_nz[p],  ifelse(ok, v,   0.0), s)
            n = muladd(corr_nz[p], ifelse(ok, 1.0, 0.0), n)
        end

        I[bin] = s / n
    end
end

"""
    allocate_output(b::BakedIntegrator, frame::AbstractMatrix)
    allocate_output(b::BakedIntegrator, frames::AbstractArray)

Allocate an output `Array` to be used with `integrate!()`.
"""
function allocate_output(b::BakedIntegrator, frames::AbstractArray)
    Array{Float32}(undef, output_size(b, frames))
end

"""
    integrate!(out::AbstractVector{Float32}, b, frame::AbstractMatrix)

Integrate a single frame into the caller-provided `out`. Returns a `DimArray`
view wrapping `out`.
"""
function integrate!(out::AbstractVector{Float32}, b::BakedIntegrator, frame::AbstractMatrix;
                    scheduler=nothing, chunksize=nothing)
    if size(frame) != b.shape
        throw(DimensionMismatch("frame size $(size(frame)) ≠ baked shape $(b.shape)"))
    elseif size(out) != output_size(b, frame)
        throw(DimensionMismatch("out length $(length(out)) ≠ nbins = $(b.npt0)"))
    end

    _fused_spmv!(out, b, vec(frame))
    _wrap(b, out, (), frame)
end

"""
    integrate!(out::AbstractArray{Float32}, b, frames::AbstractArray;
               scheduler=:static, chunksize=nothing)

Integrate `frames` of shape `(b.shape..., d_1, …, d_K)` into `out` of
shape `(nbins, d_1, …, d_K)`. Loops the single-frame kernel over the
extra-dim cartesian indices in parallel via OhMyThreads.
"""
function integrate!(out::AbstractArray{Float32}, b::BakedIntegrator, frames::AbstractArray;
                    scheduler::Symbol=:static, chunksize=nothing)
    nd_frame = length(b.shape)
    input_frame_size = size(frames)[1:nd_frame]
    extra_shape = size(frames)[nd_frame + 1:end]

    if ndims(frames) == 1
        throw(ArgumentError("A vector was passed as `frame`, but it needs to be a 2D array with shape $(b.shape)"))
    elseif input_frame_size != b.shape
        throw(DimensionMismatch("frame size $(size(frames)[1:nd_frame]) ≠ baked shape $(b.shape)"))
    elseif size(out) != output_size(b, frames)
        throw(DimensionMismatch("out size $(size(out)) ≠ (nbins, extra...) = $(output_size(b, frames))"))
    end

    npix = prod(b.shape)
    src  = frames isa AbstractDimArray ? parent(frames) : frames
    X    = reshape(src, npix, :)
    Y    = reshape(out, prod(output_size(b)), :)

    @tasks for k in axes(X, 2)
        @set begin
            scheduler = scheduler
            chunksize = chunksize
        end

        _fused_spmv!(@view(Y[:, k]), b, @view(X[:, k]))
    end

    _wrap(b, out, extra_shape, frames)
end

"""
    integrate(b, frames::AbstractArray; scheduler=:static, chunksize=nothing)

Allocate a `(nbins, d_1, …, d_K)` output array (where `d_*` are the extra
dims of `frames` past the leading `(W, H)` frame axes) and forward to
`integrate!`. See `integrate!` for the meaning of `scheduler`/`chunksize`.
"""
function integrate(b::BakedIntegrator, frames::AbstractArray;
                   scheduler::Symbol=:static, chunksize=nothing)
    out = allocate_output(b, frames)
    integrate!(out, b, frames; scheduler, chunksize)
end
