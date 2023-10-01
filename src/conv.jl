

function kernaxes(::DenseConvDims{2,K,C_in, C_out}) where {K,C_in, C_out} # LoopVectorization can take advantage of static size information
  K₁ =  StaticInt(1):StaticInt(K[1])
  K₂ =  StaticInt(1):StaticInt(K[2])
  Cᵢₙ =  StaticInt(1):StaticInt(C_in)
  Cₒᵤₜ = StaticInt(1):StaticInt(C_out)
  (K₁, K₂, Cᵢₙ, Cₒᵤₜ)
end
function kernaxes(::DenseConvDims{3,K,C_in, C_out}) where {K,C_in, C_out} # LoopVectorization can take advantage of static size information
  K₁ =  StaticInt(1):StaticInt(K[1])
  K₂ =  StaticInt(1):StaticInt(K[2])
  K₃ =  StaticInt(1):StaticInt(K[2])
  Cᵢₙ =  StaticInt(1):StaticInt(C_in)
  Cₒᵤₜ = StaticInt(1):StaticInt(C_out)
  (K₁, K₂, K₃, Cᵢₙ, Cₒᵤₜ)
end

function convlayer!(
    out::AbstractArray{<:Any,4}, img, kern,
    dcd::DenseConvDims{2, <:Any, <:Any, <:Any, (1, 1), (0, 0, 0, 0), (1, 1), true}
)
  @batch for d ∈ axes(out,4)
    (K₁, K₂, Cᵢₙ, Cₒᵤₜ) = kernaxes(dcd)
    for o ∈ Cₒᵤₜ
      @turbo for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2)
        s = zero(eltype(out))
        for k₁ ∈ K₁, k₂ ∈ K₂, i ∈ Cᵢₙ
          s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * kern[k₁, k₂, i, o]
        end
        out[j₁, j₂, o, d] = s
      end
    end
  end
  out
end

function convlayer!(
  out::AbstractArray{<:Any,5}, img::AbstractArray{<:Any,5}, kern::AbstractArray{<:Any,5},
  dcd::DenseConvDims{3, <:Any, <:Any, <:Any, (1, 1, 1), (0, 0, 0, 0, 0, 0), (1, 1, 1), true}
  )
  @batch for d ∈ axes(out,4)
    (K₁, K₂, K₃, Cᵢₙ, Cₒᵤₜ) = kernaxes(dcd)
    for o ∈ Cₒᵤₜ
      @turbo for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), j₃ ∈ axes(out,3)
        s = zero(eltype(out))
        for k₁ ∈ K₁, k₂ ∈ K₂, k₃ ∈ K₃, i ∈ Cᵢₙ
          s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, j₃ + k₃ - 1, i, d] * kern[k₁, k₂, k₃, i, o]
        end
        out[j₁, j₂, j₃, o, d] = s
      end
    end
  end
end



function convlayeravx!(out::AbstractArray{<:Any,4}, img, kern)
  @batch for d ∈ axes(out,4)
    @turbo for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), o ∈ axes(kern,4)
      s = zero(eltype(out))
      for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), i ∈ axes(kern,3)
        s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * kern[k₁, k₂, i, o]
      end
      out[j₁, j₂, o, d] = s
    end
  end
  out
end

function convlayeradjkern!(kernadj::AbstractArray{<:Any,4}, img, outadj)
  @tturbo for k₁ ∈ axes(kernadj,1), k₂ ∈ axes(kernadj,2), i ∈ axes(kernadj,3), o ∈ axes(kernadj,4)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1), j₂ ∈ axes(outadj,2), d ∈ axes(outadj,4)
      s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * outadj[j₁, j₂, o, d]
    end
    kernadj[k₁, k₂, i, o] = s
  end
  kernadj
end

@generated function convlayeradjimg!(imgadj, kern::AbstractStrideArray{Tuple{StaticInt{K₁},StaticInt{K₂},StaticInt{I},StaticInt{O}},T,4}, outadj) where {K₁,K₂,I,O,T}
  quote
    @batch for d ∈ axes(outadj,4)
      @turbo for j₁ ∈ axes(imgadj,1), j₂ ∈ axes(imgadj,2), i ∈ axes(kern,3)
        s = zero($T)
        for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), o ∈ axes(kern,4)
          s += kern[k₁, k₂, i, o] * outadj[j₁ - k₁ + $K₁, j₂ - k₂ + $K₂, o, d]
        end
        imgadj[j₁, j₂, i, d] = s
      end
    end
    imgadj
  end
end

function convlayeradjkern!(kernadj::AbstractArray{<:Any,4}, img, outadj)
  @tturbo for k₁ ∈ axes(kernadj,1), k₂ ∈ axes(kernadj,2), k₃ ∈ axes(kernadj,3), i ∈ axes(kernadj,4), o ∈ axes(kernadj,5)
    s = zero(eltype(kernadj))
    for j₁ ∈ axes(outadj,1), j₂ ∈ axes(outadj,2), j₃ ∈ axes(outadj,3), d ∈ axes(outadj,5)
      s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, j₃ + k₃ - 1, i, d] * outadj[j₁, j₂, j₃, o, d]
    end
    kernadj[k₁, k₂, k₃, i, o] = s
  end
  kernadj
end

@generated function convlayeradjimg!(imgadj, kern::AbstractStrideArray{Tuple{StaticInt{K₁},StaticInt{K₂},StaticInt{K₃},StaticInt{I},StaticInt{O}},T,5}, outadj) where {K₁,K₂,I,O,T}
  quote
    @batch for d ∈ axes(outadj,5)
      for i ∈ axes(kern,4)
        @turbo for j₁ ∈ axes(imgadj,1), j₂ ∈ axes(imgadj,2), j₃ ∈ axes(imgadj,3)
          s = zero($T)
          for k₁ ∈ axes(kern,1), k₂ ∈ axes(kern,2), k₃ ∈ axes(kern,3), o ∈ axes(kern,5)
            s += kern[k₁, k₂, k₃, i, o] * outadj[j₁ - k₁ + $K₁, j₂ - k₂ + $K₂, j₃ - k₃ + $K₃, o, d]
          end
          imgadj[j₁, j₂, j₃, i, d] = s
        end
      end
    end
    imgadj
  end
end



