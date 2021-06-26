module NNlibCPU

using NNlib, LoopVectorization, Polyester, StrideArrays, ChainRulesCore, ForwardDiff
using CloseOpenIntervals
using StrideArrays: AbstractStrideArray
using Static: Zero, One

using NNlib: input_size, output_size, kernel_size, channels_out, padding, dilation,
  calc_padding_regions


# `dense.jl` is waiting on Octavian support
include("dense.jl")
include("conv.jl")
include("maxpool.jl")

end
