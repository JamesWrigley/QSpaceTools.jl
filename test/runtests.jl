# Copy CondaPkg.toml to the test project so that it gets found by CondaPkg
# during the tests. If this was instead in the project directory it would also
# be used by CondaPkg outside of the tests, which we don't want.
cp(joinpath(@__DIR__, "CondaPkg.toml"), joinpath(dirname(Base.active_project()), "CondaPkg.toml"))

ENV["JULIA_CONDAPKG_ENV"] = "@qspacetools-tests"
ENV["JULIA_CONDAPKG_VERBOSITY"] = -1

# If you're running the tests locally you could uncomment the two environment
# variables below. This will be a bit faster since it stops CondaPkg from
# re-resolving the environment each time (but you do need to run it at least
# once locally to initialize the environment).
# ENV["JULIA_PYTHONCALL_EXE"] = joinpath(Base.DEPOT_PATH[1], "conda_environments", "qspacetools-tests", "bin", "python")
# ENV["JULIA_CONDAPKG_BACKEND"] = "Null"


using Test
using QSpaceTools: QSpaceTools as QST
using CondaPkg: CondaPkg

function test_integrator(; shape, npt0, npt1, ndim)
    nbins = ndim == 2 ? npt0 * npt1 : npt0
    QST.BakedIntegrator(
        ones(Int32, nbins + 1), Int32[], Float32[], Float32[],
        Float32[], Float32[],
        shape, "", "", "", npt0, npt1, ndim,
    )
end

@testset "allocate_output" begin
    b1 = test_integrator(shape=(8, 6), npt0=10, npt1=0, ndim=1)
    b2 = test_integrator(shape=(8, 6), npt0=10, npt1=4, ndim=2)

    @test size(QST.allocate_output(b1, zeros(8, 6))) == (10,)
    @test size(QST.allocate_output(b2, zeros(8, 6))) == (10, 4)
    @test size(QST.allocate_output(b1, zeros(8, 6, 3, 5))) == (10, 3, 5)
    @test size(QST.allocate_output(b2, zeros(8, 6, 3))) == (10, 4, 3)
end

@testset "PyFAI correctness tests" begin
    CondaPkg.withenv() do
        test_file = joinpath(@__DIR__, "test_bake_for_batch.py")

        run(addenv(`python $test_file -f`, "PYFAI_JULIA_PROJECT" => Base.active_project()))
    end
end
