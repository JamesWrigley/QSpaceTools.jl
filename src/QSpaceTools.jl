module QSpaceTools

using HDF5: h5open, read_attribute
using DimensionalData: Dim, DimArray, AbstractDimArray, otherdims
using LinearAlgebra: normalize
using OhMyThreads: @tasks, @set, tmapreduce
using StaticArrays: SVector, SMatrix, @SMatrix

include("pyfai.jl")
include("geometry.jl")
include("gridder.jl")
include("rss.jl")

export Geometry, rss, rss!, RSSWorkspace, allocate_output

end # module QSpaceTools
