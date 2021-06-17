

function kernaxes(::DenseConvDims{2,K,C_in, C_out}) where {K,C_in, C_out} # LoopVectorization can take advantage of static size information
  K₁ =  StaticInt(1):StaticInt(K[1])
  K₂ =  StaticInt(1):StaticInt(K[2])
  Cᵢₙ =  StaticInt(1):StaticInt(C_in)
  Cₒᵤₜ = StaticInt(1):StaticInt(C_out)
  (K₁, K₂, Cᵢₙ, Cₒᵤₜ)
end

function convlayer!(
    out::AbstractArray{<:Any,4}, img, kern,
    dcd::DenseConvDims{2, <:Any, <:Any, <:Any, (1, 1), (0, 0, 0, 0), (1, 1), true}
)
  @batch for d ∈ axes(out,4)
    (K₁, K₂, Cᵢₙ, Cₒᵤₜ) = kernaxes(dcd)
    @turbo for j₁ ∈ axes(out,1), j₂ ∈ axes(out,2), o ∈ Cₒᵤₜ
      s = zero(eltype(out))
      for k₁ ∈ K₁, k₂ ∈ K₂, i ∈ Cᵢₙ
        s += img[j₁ + k₁ - 1, j₂ + k₂ - 1, i, d] * kern[k₁, k₂, i, o]
      end
      out[j₁, j₂, o, d] = s
    end
  end
  out
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

@generated function convlayeradjimg!(imgadj, kern::AbstractStrideArray{Tuple{K₁,K₂,I,O},T,4}, outadj) where {K₁,K₂,I,O,T}
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



