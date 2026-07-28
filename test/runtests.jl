# Copy CondaPkg.toml to the test project so that it gets found by CondaPkg
# during the tests. If this was instead in the project directory it would also
# be used by CondaPkg outside of the tests, which we don't want.
cp(joinpath(@__DIR__, "CondaPkg.toml"), joinpath(dirname(Base.active_project()), "CondaPkg.toml"); force=true)

ENV["JULIA_CONDAPKG_ENV"] = "@qspacetools-tests"
ENV["JULIA_CONDAPKG_VERBOSITY"] = -1

# If you're running the tests locally you could uncomment the two environment
# variables below. This will be a bit faster since it stops CondaPkg from
# re-resolving the environment each time (but you do need to run it at least
# once locally to initialize the environment).
# ENV["JULIA_PYTHONCALL_EXE"] = joinpath(Base.DEPOT_PATH[1], "conda_environments", "qspacetools-tests", "bin", "python")
# ENV["JULIA_CONDAPKG_BACKEND"] = "Null"


using Test
using PythonCall
using DimensionalData: dims, name
using QSpaceTools: QSpaceTools as QST

@py import sys
sys.path.append(dirname(@__DIR__))

@py begin
    import numpy as np
    import pyFAI
    import pyFAI.test.utilstest: create_fake_data
    import bake_for_batch: bake_for_batch, write_hdf5
    import xrayutilities as xu
end

# A fresh fake detector image + configured AzimuthalIntegrator, as the
# Python tests build in setUpClass. The image is cast to float32 and pyFAI's
# empty-bin sentinel is set to NaN to match the exporter.
function fake_data()
    image, ai = create_fake_data(poissonian=false)
    image = image.astype(np.float32)
    # pyFAI snapshots its empty-bin sentinel from ai._empty at engine-creation
    # time; the exporter always emits NaN for empty bins, so align pyFAI here.
    ai._empty = np.nan
    image, ai
end

# pyFAI/numpy frames are C-order (H, W); the Julia consumer wants (W, H) so
# that column-major `vec` reproduces pyFAI's C-order flat index. Transposing
# the numpy array hands PythonCall exactly that.
jl_frame(img) = pyconvert(Matrix, img.T)

# A 3-frame batch derived from `frame`: itself, a scaled copy, and a noisy one.
function make_batch(frame)
    cat(frame, frame .* 0.5f0, frame .+ rand(Float32, size(frame)); dims=3)
end

# Compare a Julia integration result against a reference array, checking the
# NaN (empty-bin) pattern exactly and the finite values within tolerance.
function compare_result(got, ref; rtol, atol)
    mask = .!isnan.(ref)
    @test isnan.(got) == isnan.(ref)
    @test got[mask] ≈ ref[mask] rtol=rtol atol=atol
end

function test_integrator(; shape, npt0, npt1, ndim)
    nbins = ndim == 2 ? npt0 * npt1 : npt0
    QST.BakedIntegrator(
        ones(Int32, nbins + 1), Int32[], Float32[], Float32[],
        Float32[], Float32[],
        shape, "", "", "", npt0, npt1, ndim,
    )
end

@testset "allocate_output()" begin
    b1 = test_integrator(shape=(8, 6), npt0=10, npt1=0, ndim=1)
    b2 = test_integrator(shape=(8, 6), npt0=10, npt1=4, ndim=2)

    @test size(QST.allocate_output(b1, zeros(8, 6))) == (10,)
    @test size(QST.allocate_output(b2, zeros(8, 6))) == (10, 4)
    @test size(QST.allocate_output(b1, zeros(8, 6, 3, 5))) == (10, 3, 5)
    @test size(QST.allocate_output(b2, zeros(8, 6, 3))) == (10, 4, 3)
end

# Run one pyFAI reference integration (1D or 2D, selected by `ndim`) and
# compare the baked Julia result against it. The only per-dimensionality bits
# are the pyFAI entry point, the `npt` argument, and the result rank;
# everything else — fake data, polarization, bin-center checks, comparison —
# is shared.
function check_split(split; ndim, with_pol=false, azimuth_range=nothing)
    image, ai = fake_data()
    unit = "q_A^-1"
    pol = with_pol ? 0.97 : nothing
    npt = ndim == 1 ? 800 : (500, 180)          # (npt_rad, npt_azim) for 2D

    ai.reset()
    ref = if ndim == 1
        ai.integrate1d(image; npt, unit, method=(split, "csr", "cython"),
                       correctSolidAngle=true, dummy=np.nan,
                       polarization_factor=pol, azimuth_range)
    else
        npt_rad, npt_azim = npt
        ai.integrate2d(image; npt_rad, npt_azim, unit,
                       method=(split, "csr", "cython"), correctSolidAngle=true,
                       dummy=np.nan, polarization_factor=pol, azimuth_range)
    end

    # 1D intensity is a vector; 2D is (npt_azim, npt_rad), which lines up with
    # the Julia consumer's (npt1, npt0) output directly.
    ref_q = pyconvert(Vector, ref.radial)
    ref_I = ndim == 1 ? pyconvert(Vector, ref.intensity) :
                        pyconvert(Matrix, ref.intensity)

    baked = bake_for_batch(ai, npt; unit, split, solidangle=true,
                           polarization_factor=pol, azimuth_range)
    @test pyconvert(Int, baked["ndim"]) == ndim
    @test pyconvert(Vector, baked["bin_centers0"]) ≈ ref_q rtol=1e-6
    if ndim == 2
        @test pyconvert(Vector, baked["bin_centers1"]) ≈
              pyconvert(Vector, ref.azimuthal) rtol=1e-5
    end

    got = parent(QST.integrate(QST.load_baked(baked), jl_frame(image)))
    if ndim == 2
        @test size(got) == reverse(npt)
    end

    compare_result(got, ref_I; rtol=1e-4, atol=1e-4)
end

# Integrate a 3-frame batch and check each slice equals the per-frame result.
# Shape-agnostic: `selectdim` grabs the i-th output along the trailing batch
# axis whether the per-frame output is a vector (1D) or matrix (2D).
function check_batch(npt, seed)
    image, ai = fake_data()
    b = QST.load_baked(bake_for_batch(ai, npt; unit="2th_deg", split="bbox"))
    batch = make_batch(jl_frame(image))
    got_batch = parent(QST.integrate(b, batch))
    @test size(got_batch, ndims(got_batch)) == size(batch, 3)

    for i in axes(batch, 3)
        got_one   = parent(QST.integrate(b, batch[:, :, i]))
        got_slice = selectdim(got_batch, ndims(got_batch), i)
        @test size(got_slice) == size(got_one)
        mask = .!isnan.(got_one)
        @test got_slice[mask] ≈ got_one[mask] rtol=1e-6 atol=1e-6
    end
end

@testset "$(ndim)D correctness" for ndim in (1, 2)
    @testset "split=$split" for split in ("no", "bbox", "full")
        check_split(split; ndim)
    end

    @testset "with polarization (bbox)" begin
        check_split("bbox"; ndim, with_pol=true)
    end

    if ndim == 1
        @testset "1D with azimuth_range (bbox)" begin
            check_split("bbox"; ndim=1, azimuth_range=(-45.0, 45.0))
        end
    end
end

@testset "$(ndim)D batch matches per-frame loop" for (ndim, npt, seed) in
        ((1, 800, 7), (2, (500, 180), 13))
    check_batch(npt, seed)
end

@testset "NaN pixels match pyFAI" begin
    image, ai = fake_data()
    npt, unit = 800, "2th_deg"

    poisoned = image.copy()
    idx = np.random.choice(poisoned.size, size=50, replace=false)
    poisoned.reshape(-1)[idx] = np.float32(np.nan)

    ai.reset()
    ref = ai.integrate1d(poisoned; npt, unit, method=("bbox", "csr", "cython"),
                         correctSolidAngle=true, dummy=np.nan)
    ref_I = pyconvert(Vector, ref.intensity)

    b = QST.load_baked(bake_for_batch(ai, npt; unit, split="bbox"))
    got = parent(QST.integrate(b, jl_frame(poisoned)))
    compare_result(got, ref_I; rtol=1e-4, atol=1e-4)
end

@testset "load_baked()" begin
    # Test that load_baked() implementations match
    for npt in (800, (500, 180))
        image, ai = fake_data()
        baked = bake_for_batch(ai, npt; unit="2th_deg", split="bbox")

        mktempdir() do td
            path = joinpath(td, "setup.h5")
            write_hdf5(baked, path)
            @test QST.load_baked(path) == QST.load_baked(baked)
        end
    end

    # Check that load_baked(::Py) throws on non-dict inputs
    @test_throws ArgumentError QST.load_baked(np.zeros(3))
end

# Small detector + simple goniometer chain, built identically on both sides.
# Returns the xrayutilities HXRD and the matching Julia Geometry.
function hxrd_geometry(; nch1, nch2, pixel_size=0.2, sdd=500.0, photon_energy=8000.0)
    sample_axes = ("y-", "x+", "z+")
    detector_axes = ("y-",)
    image_axes = ("y-", "z-")
    beam_direction = (1.0, 0.0, 0.0)
    sample_normal = (0.0, 0.0, 1.0)
    sample_faceup = "z+"

    qconv = xu.experiment.QConversion(
        sampleAxis=pylist(sample_axes),
        detectorAxis=pylist(detector_axes),
        r_i=pylist(beam_direction),
        en=photon_energy,
    )
    hxrd = xu.HXRD(
        idir=pylist(beam_direction), ndir=pylist(sample_normal),
        sampleor=sample_faceup, qconv=qconv, en=photon_energy,
    )
    hxrd.Ang2Q.init_area(
        detectorDir1=image_axes[1], detectorDir2=image_axes[2],
        cch1=nch1 ÷ 2, cch2=nch2 ÷ 2, Nch1=nch1, Nch2=nch2,
        pwidth1=pixel_size, pwidth2=pixel_size, distance=sdd,
    )

    geom = QST.Geometry(;
        sample_axes, detector_axes, image_axes,
        beam_direction, sample_normal, sample_faceup,
        pixel_size=(pixel_size, pixel_size),
        center=(nch1 ÷ 2, nch2 ÷ 2),
        shape=(nch1, nch2),
        distance=sdd, wavelength=QST.energy2wavelength(photon_energy),
    )

    return hxrd, geom
end

@testset "rss matches xrayutilities" begin
    nch1, nch2 = 64, 80
    photon_energy = 8000.0  # eV
    wl = QST.energy2wavelength(photon_energy)
    @test wl ≈ pyconvert(Float64, xu.en2lam(photon_energy)) rtol=1e-12
    @test QST.wavelength2energy(wl) ≈ photon_energy rtol=1e-12
    nx, ny = 96, 96

    hxrd, geom = hxrd_geometry(; nch1, nch2, photon_energy)

    # Angles in degrees, matching xrayutilities' default convention.
    theta, chi, phi, twotheta = rad2deg.((0.30, 0.05, -0.10, 0.62))

    qx_py, qy_py, qz_py = hxrd.Ang2Q.area(
        theta, chi, phi, twotheta; deg=true,
    )
    qx_arr = pyconvert(Matrix{Float64}, qx_py)
    qz_arr = pyconvert(Matrix{Float64}, qz_py)

    # Frame: smooth synthetic gradient — easy to localize errors. xrayutilities
    # detector indexing is C-order (Nch1 rows, Nch2 cols), so we make the
    # Python and Julia frames consistent through `.T` like the pyFAI path.
    image_py = np.random.rand(nch1, nch2).astype(np.float64)
    frame_jl = pyconvert(Matrix, image_py)  # (nch1, nch2)

    # Cross-check the geometry kernel alone: per-pixel q must agree pointwise.
    q_jl = QST.pixel_q_array(geom, (theta, chi, phi), (twotheta,))
    qx_jl = @view q_jl[1, :, :]
    qz_jl = @view q_jl[3, :, :]
    @test qx_jl ≈ qx_arr rtol=1e-10 atol=1e-12
    @test qz_jl ≈ qz_arr rtol=1e-10 atol=1e-12

    # Don't combine the geometry kernel and the gridder into one end-to-end
    # test: the Julia matvecs use FMA (via StaticArrays) while xrayutilities
    # does plain scalar sums, producing sub-ULP differences in the per-pixel
    # q values. The vast majority of bins are unaffected, but the
    # `< xmin` / `> xmax` boundary check in fuzzygridder! will reject a
    # pixel on one side and accept it on the other. Instead, feed the gridder
    # the SAME (Python-derived) q-array on both sides to isolate it from the
    # geometry layer.
    xmin = pyconvert(Float64, qx_py.min())
    xmax = pyconvert(Float64, qx_py.max())
    ymin = pyconvert(Float64, qz_py.min())
    ymax = pyconvert(Float64, qz_py.max())

    gridder = xu.FuzzyGridder2D(nx, ny)
    gridder.dataRange(xmin, xmax, ymin, ymax, true)
    gridder(qx_py.flatten(), qz_py.flatten(), image_py.flatten())
    ref = pyconvert(Matrix{Float64}, gridder.data)

    image = Matrix{Float64}(undef, nx, ny)
    points = Matrix{Float64}(undef, 2, length(frame_jl))
    points[1, :] .= vec(qx_arr)
    points[2, :] .= vec(qz_arr)
    ws = QST.GridderWorkspace((nx, ny))
    QST.fuzzygridder!(image, ws, points, frame_jl,
                      ((xmin, xmax), (ymin, ymax)))
    @test size(image) == (nx, ny)
    @test image ≈ ref rtol=1e-10 atol=1e-12

    # Smoke test: rss runs end-to-end, returns a DimArray with the expected
    # shape and axes, and at least some bins receive contributions.
    got = QST.rss(frame_jl, geom;
                  sample_angles=(theta, chi, phi),
                  detector_angles=(twotheta,),
                  gridder_size=(nx, ny),
                  projection=(:qx, :qz))
    @test size(parent(got)) == (nx, ny)
    @test name.(dims(got)) == (:qx, :qz)

    # Same call with an alternate projection — exercises _proj_indices and
    # confirms the returned DimArray axes track the chosen pair.
    got_yz = QST.rss(frame_jl, geom;
                     sample_angles=(theta, chi, phi),
                     detector_angles=(twotheta,),
                     gridder_size=(nx, ny),
                     projection=(:qy, :qz))
    @test size(parent(got_yz)) == (nx, ny)
    @test name.(dims(got_yz)) == (:qy, :qz)
end

# The geometry and the 3D gridder are each checked against xrayutilities
# elsewhere, so this covers only what `rsm` adds on top of them: the frame
# loop, per-frame angles, and accumulating into one volume normalized once.
@testset "rsm" begin
    nch1, nch2 = 32, 40
    nframes = 5
    nx, ny, nz = 24, 26, 28

    _, geom = hxrd_geometry(; nch1, nch2)

    # A rocking scan: theta steps per frame, the rest of the chain is fixed.
    thetas = range(16.0, 18.0; length=nframes)
    chi, phi, twotheta = 0.05, -0.10, 35.5
    sample_angles = [(t, chi, phi) for t in thetas]
    frames = rand(nch1, nch2, nframes)

    # Reference: every frame's q-vectors concatenated into one point set and
    # gridded in a single call — which is what the streaming frame loop must
    # reproduce.
    points = reduce(hcat, (reshape(QST.pixel_q_array(geom, a, (twotheta,)), 3, :)
                           for a in sample_angles))
    bounds = ntuple(6) do k
        row = @view points[cld(k, 2), :]
        isodd(k) ? minimum(row) : maximum(row)
    end
    ref = Array{Float64, 3}(undef, nx, ny, nz)
    ref_ws = QST.GridderWorkspace((nx, ny, nz))
    QST.fuzzygridder!(ref, ref_ws, points, vec(frames),
                      ((bounds[1], bounds[2]), (bounds[3], bounds[4]),
                       (bounds[5], bounds[6])))

    got = QST.rsm(frames, geom; sample_angles, detector_angles=(twotheta,),
                  gridder_size=(nx, ny, nz), output=:volume)
    @test size(parent(got)) == (nx, ny, nz)
    @test name.(dims(got)) == (:qx, :qy, :qz)
    @test parent(got) ≈ ref rtol=1e-14
    @test any(!iszero, got)

    # A bare Real stands in for a single-axis chain.
    @test parent(QST.rsm(frames, geom; sample_angles, detector_angles=twotheta,
                         gridder_size=(nx, ny, nz), output=:volume)) == parent(got)

    # Explicit bounds skip the scan pass; a reused workspace must not drift.
    ws = QST.RSMWorkspace(geom, (nx, ny, nz); output=:volume)
    out = QST.allocate_output(geom, (nx, ny, nz); output=:volume)
    for _ in 1:2
        QST.rsm!(out, frames, geom; sample_angles, detector_angles=(twotheta,),
                 bounds, workspace=ws)
        @test out ≈ ref rtol=1e-14
    end

    # The same workspace type serves RSS at D = 2.
    ws2 = QST.RSMWorkspace(geom, (nx, ny))
    img = QST.allocate_output(geom, (nx, ny))
    QST.rss!(img, view(frames, :, :, 1), geom; sample_angles=sample_angles[1],
             detector_angles=(twotheta,), workspace=ws2)
    @test any(!iszero, img)
end

# `output=:projections` is the default and never builds the volume, so what it
# adds over the volume path is the per-grid row selection and the guard on the
# collapsed component. The reference grids the same points over the two rows
# directly, dropping guarded-out points by NaN-ing their intensity.
@testset "rsm projections" begin
    nch1, nch2, nframes = 32, 40, 5
    grid = (24, 26, 28)
    _, geom = hxrd_geometry(; nch1, nch2)

    twotheta = 35.5
    sample_angles = [(t, 0.05, -0.10) for t in range(16.0, 18.0; length=nframes)]
    detector_angles = (twotheta,)
    frames = rand(nch1, nch2, nframes)

    points = reduce(hcat, (reshape(QST.pixel_q_array(geom, a, detector_angles), 3, :)
                           for a in sample_angles))
    full = ntuple(6) do k
        row = @view points[cld(k, 2), :]
        isodd(k) ? minimum(row) : maximum(row)
    end
    # Tight enough that the collapsed component rejects points: unguarded, each
    # projection would keep pixels the volume drops.
    tight = ntuple(6) do k
        lo, hi = full[2cld(k, 2) - 1], full[2cld(k, 2)]
        isodd(k) ? lo + 0.3 * (hi - lo) : hi - 0.3 * (hi - lo)
    end

    # `a`, `b` are the q-components this projection grids and `g` the third one
    # it collapses — the components are 1, 2, 3, so the missing one is
    # `6 - a - b`. `bounds` is flat, two slots per component, so component `c`
    # spans `bounds[2c - 1]` (min) to `bounds[2c]` (max).
    function reference((a, b), bounds)
        g = 6 - a - b
        data = [(bounds[2g - 1] <= points[g, p] <= bounds[2g]) ? vec(frames)[p] : NaN
                for p in axes(points, 2)]
        img = Matrix{Float64}(undef, grid[a], grid[b])
        ws = QST.GridderWorkspace(size(img))
        QST.fuzzygridder!(img, ws, points[[a, b], :], data,
                          ((bounds[2a - 1], bounds[2a]), (bounds[2b - 1], bounds[2b])))
        return img
    end

    for bounds in (nothing, tight)
        got = QST.rsm(frames, geom; sample_angles, detector_angles, gridder_size=grid,
                      bounds)
        @test got isa QST.QProjections
        for (i, pair) in enumerate(QST._PROJECTION_PAIRS)
            g = QST._output_grids(got)[i]
            @test name.(dims(g)) == map(c -> (:qx, :qy, :qz)[c], pair)
            @test parent(g) ≈ reference(pair, something(bounds, full)) rtol=1e-14
        end
    end

    # Projections and volume are gridded on the same axes.
    vol = QST.rsm(frames, geom; sample_angles, detector_angles, gridder_size=grid,
                  output=:volume)
    got = QST.rsm(frames, geom; sample_angles, detector_angles, gridder_size=grid)
    @test dims(got.qxqz) == (dims(vol, :qx), dims(vol, :qz))

    # In-place into caller-owned projections, with a reused workspace.
    ws = QST.RSMWorkspace(geom, grid)
    out = QST.allocate_output(geom, grid)
    @test out isa QST.QProjections
    for _ in 1:2
        QST.rsm!(out, frames, geom; sample_angles, detector_angles, workspace=ws)
        @test out ≈ got
    end

    # The projections share their axes pairwise, so inconsistent sizes describe
    # no grid.
    bad = QST.QProjections(zeros(grid[1], grid[2]), zeros(grid[1], grid[3]),
                           zeros(grid[2], grid[3] + 1))
    @test_throws ArgumentError QST.rsm!(bad, frames, geom; sample_angles, detector_angles)
    @test_throws ArgumentError QST.rsm(frames, geom; sample_angles, detector_angles,
                                       output=:slices)
end

# `rsm!` is a bounds scan followed by the streaming interface, so what the
# accumulator has to show is that driving it by hand reproduces the batch call
# exactly — and that summing it leaves it accumulating.
@testset "RSMAccumulator" begin
    nch1, nch2, nframes = 32, 40, 5
    grid = (24, 26, 28)
    _, geom = hxrd_geometry(; nch1, nch2)

    detector_angles = (35.5,)
    sample_angles = [(t, 0.05, -0.10) for t in range(16.0, 18.0; length=nframes)]
    frames = rand(nch1, nch2, nframes)

    # The scan reads no frame data, so it can fix the grid before any image is
    # loaded.
    bounds = QST.q_bounds(geom; sample_angles, detector_angles)

    for output in (:projections, :volume)
        ref = QST.rsm(frames, geom; sample_angles, detector_angles,
                      gridder_size=grid, output, bounds)

        acc = QST.RSMAccumulator(geom; bounds, gridder_size=grid, output)
        for i in 1:nframes
            push!(acc, view(frames, :, :, i); sample_angles=sample_angles[i],
                  detector_angles)
        end
        @test sum(acc) == ref

        # Summing is non-destructive: repeatable, and frames may still follow.
        @test sum(acc) == ref
        push!(acc, view(frames, :, :, 1); sample_angles=sample_angles[1],
              detector_angles)
        @test sum(acc) != ref

        empty!(acc)
        append!(acc, frames; sample_angles, detector_angles)
        @test sum(acc) == ref
    end
end

# The regular-grid constructor delegates to the positions one, so the
# xrayutilities comparison above already covers it. What's left is the 3D
# `data_shape`: the same positions declared as a module stack must reproduce
# the flat result exactly, lined up by the column order of `positions`.
@testset "multi-module data_shape" begin
    nch1, nch2, nmod, nframes = 32, 40, 5, 4
    grid = (16, 18, 20)

    common = (; sample_axes=("y-", "x+", "z+"), detector_axes=("y-",),
                image_axes=("y-", "z-"), beam_direction=(1.0, 0.0, 0.0),
                sample_normal=(0.0, 0.0, 1.0), sample_faceup="z+",
                pixel_size=(0.2, 0.2), center=(nch1 ÷ 2, nch2 ÷ 2),
                distance=500.0, wavelength=QST.energy2wavelength(8000.0))

    # Any (3, npix) array is valid here; a real detector's own directions keep
    # the gridded output physically sensible.
    positions = QST.Geometry(; shape=(nch1, nch2), common...).directions
    flat = QST.Geometry(positions, (nch1, nch2); common...)
    split = QST.Geometry(positions, (nch1, nch2 ÷ nmod, nmod); common...)

    @test QST.npixels(split) == nch1 * nch2
    @test QST.data_shape(split) == (nch1, nch2 ÷ nmod, nmod)

    sample_angles = [(t, 0.05, -0.10) for t in range(16.0, 18.0; length=nframes)]
    detector_angles = (35.5,)
    frames = rand(nch1, nch2, nframes)
    split_frames = reshape(frames, nch1, nch2 ÷ nmod, nmod, nframes)

    @test reshape(QST.pixel_q_array(split, sample_angles[1], detector_angles), 3, nch1, nch2) ==
          QST.pixel_q_array(flat, sample_angles[1], detector_angles)

    @test QST.rsm(split_frames, split; sample_angles, detector_angles, gridder_size=grid) ==
          QST.rsm(frames, flat; sample_angles, detector_angles, gridder_size=grid)

    @test QST.rss(view(split_frames, :, :, :, 1), split;
                  sample_angles=sample_angles[1], detector_angles) ==
          QST.rss(view(frames, :, :, 1), flat;
                  sample_angles=sample_angles[1], detector_angles)

    @test_throws ArgumentError QST.rsm(rand(nch1, nch2 + 1, nframes), flat;
                                       sample_angles, detector_angles)
    @test_throws DimensionMismatch QST.Geometry(positions, (nch1, nch2 + 1); common...)
end

@testset "fuzzygridder! matches xu.FuzzyGridder3D" begin
    # Python grids (nx, ny, nz) row-major; Julia grids (nz, ny, nx)
    # column-major, so both languages iterate the same fast axis. Julia
    # bounds/points are therefore in (z, y, x) order throughout, and numpy
    # results are converted through `.T` so the layouts line up. Three
    # distinct extents keep an axis mix-up from passing unnoticed.
    nx, ny, nz = 13, 11, 9
    xrange = (-1.0, 1.0)
    yrange = (0.0, 5.0)
    zrange = (-3.0, -1.0)
    bounds = (zrange, yrange, xrange)

    # Sample each axis a little wider than its grid range so out-of-bounds
    # points (which both implementations must drop) occur naturally.
    npts = 4000
    pad(lo, hi) = (lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
    coords = reduce(vcat, map(bounds) do (lo, hi)
        pyconvert(Matrix{Float64}, np.random.uniform(pad(lo, hi)..., (1, npts)))
    end)
    values = pyconvert(Vector{Float64}, np.random.rand(npts))

    # Points sitting exactly on the grid limits: the `<= min` / `>= max` bin
    # clamps in both kernels are only exercised here.
    corners = [zrange[1] zrange[2] zrange[1] zrange[2] zrange[1]
               yrange[1] yrange[2] yrange[2] yrange[1] yrange[2]
               xrange[1] xrange[2] xrange[1] xrange[2] sum(xrange)/2]
    coords = hcat(coords, corners)
    values = vcat(values, fill(0.75, size(corners, 2)))

    # NaN intensities are dropped by both sides.
    values[3:97:end] .= NaN

    zs, ys, xs = coords[1, :], coords[2, :], coords[3, :]

    # Run xrayutilities on the same coordinates. `data` is the normalized
    # volume, `_gnorm` the raw denominator — the Julia `norm` buffer is
    # likewise left un-normalized.
    xs_py, ys_py, zs_py = np.asarray(xs), np.asarray(ys), np.asarray(zs)
    values_py = np.asarray(values)

    function reference(width)
        g = xu.FuzzyGridder3D(nx, ny, nz)
        g.dataRange(xrange..., yrange..., zrange..., true)
        if isnothing(width)
            g(xs_py, ys_py, zs_py, values_py)
        else
            g(xs_py, ys_py, zs_py, values_py; width=pylist(width))
        end
        return (pyconvert(Array{Float64, 3}, g.data.T),
                pyconvert(Array{Float64, 3}, g._gnorm.T))
    end

    dz = QST._bin_delta(zrange..., nz)
    dy = QST._bin_delta(yrange..., ny)
    dx = QST._bin_delta(xrange..., nx)

    @testset "widths=$label" for (label, widths, py_width) in (
        ("default", nothing, nothing),
        # Equal widths on every axis, spanning several bins so the multi-bin
        # Cartesian path and the partial end-bin overlaps are hit hard.
        ("scalar", (0.35, 0.35, 0.35), [0.35, 0.35, 0.35]),
        # Per-axis widths, each a different multiple of its own bin spacing.
        # `py_width` is in (x, y, z) order to match xrayutilities' signature.
        ("per-axis", (4.4 * dz, 2.7 * dy, 1.2 * dx), [1.2 * dx, 2.7 * dy, 4.4 * dz]),
    )
        ref_image, ref_norm = reference(py_width)

        image = Array{Float64, 3}(undef, nz, ny, nx)
        ws = QST.GridderWorkspace((nz, ny, nx))
        QST.fuzzygridder!(image, ws, coords, values, bounds; widths)
        norm = ws.norm
        @test size(image) == (nz, ny, nx)
        @test any(>(0), norm)
        @test norm ≈ ref_norm rtol=1e-12 atol=1e-14
        @test image ≈ ref_image rtol=1e-12 atol=1e-14

        # Disjoint slabs of the last dimension covering 1:nx must reassemble
        # into exactly the full-volume result.
        slabbed = Array{Float64, 3}(undef, nz, ny, nx)
        slab_ws = QST.GridderWorkspace((nz, ny, nx))
        for slab in (1:2, 3:3, 4:nx)
            QST.fuzzygridder!(slabbed, slab_ws, coords, values, bounds;
                              widths, last_range=slab)
        end
        @test slabbed == image
        @test slab_ws.norm == norm
    end
end
