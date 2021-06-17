module NNlibCPU

using NNlib, LoopVectorization, Polyester, StrideArrays, ChainRulesCore, ForwardDiff
using StrideArrays: AbstractStrideArray
# using Octavian

# `dense.jl` is waiting on Octavian support
include("dense.jl")
include("conv.jl")

end
