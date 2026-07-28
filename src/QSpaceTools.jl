module QSpaceTools

using HDF5: h5open, read_attribute
using DimensionalData: Dim, DimArray, AbstractDimArray, otherdims
using LinearAlgebra: normalize
using OhMyThreads: @tasks, @set, tmapreduce, index_chunks
using StaticArrays: SVector, SMatrix, @SMatrix

include("azimuthal_integration.jl")
include("geometry.jl")
include("gridder.jl")
include("rss.jl")

export Geometry, rss, rss!, rsm, rsm!, RSMWorkspace, RSMAccumulator, QProjections,
       allocate_output, q_bounds

end # module QSpaceTools
