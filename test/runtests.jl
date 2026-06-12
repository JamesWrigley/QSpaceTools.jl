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

@testset "rss matches xrayutilities" begin
    # Small detector + simple goniometer chain; modest grid keeps runtime down.
    nch1, nch2 = 64, 80
    pixel_size = 0.2
    sdd = 500.0
    photon_energy = 8000.0  # eV
    wl = QST.energy2wavelength(photon_energy)
    @test wl ≈ pyconvert(Float64, xu.en2lam(photon_energy)) rtol=1e-12
    @test QST.wavelength2energy(wl) ≈ photon_energy rtol=1e-12
    nx, ny = 96, 96

    sample_axes = ("y-", "x+", "z+")
    detector_axes = ("y-",)
    image_axes = ("y-", "z-")
    beam_direction = (1.0, 0.0, 0.0)
    sample_normal = (0.0, 0.0, 1.0)
    sample_faceup = "z+"

    # Angles in degrees, matching xrayutilities' default convention.
    theta, chi, phi, twotheta = rad2deg.((0.30, 0.05, -0.10, 0.62))

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

    geom = QST.Geometry(;
        sample_axes, detector_axes, image_axes,
        beam_direction, sample_normal, sample_faceup,
        pixel_size=(pixel_size, pixel_size),
        center=(nch1 ÷ 2, nch2 ÷ 2),
        shape=(nch1, nch2),
        distance=sdd, wavelength=wl,
    )

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
    # `< xmin` / `> xmax` boundary check in fuzzygridder2d will reject a
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
    norm  = Matrix{Float64}(undef, nx, ny)
    points = Matrix{Float64}(undef, 2, length(frame_jl))
    points[1, :] .= vec(qx_arr)
    points[2, :] .= vec(qz_arr)
    QST.fuzzygridder2d!(image, norm, points, frame_jl,
                        xmin, xmax, ymin, ymax)
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
