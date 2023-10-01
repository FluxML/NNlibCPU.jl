
# is it fast to fuse `f` with the dense layer evaluation?
fast_fuse(::typeof(relu)) = True()
fast_fuse(::typeof(abs)) = True()
fast_fuse(::typeof(abs2)) = True()
fast_fuse(::typeof(identity)) = True()
fast_fuse(_) = False()
dense!(f::F, out, W, X, b) where {F} = dense!(f, out, W, X, b, fase_fuse(f))

function dense!(f::F, out, W, X, b, ::True) where {F}
  batch((size(out,2),num_cores()), f, out, W, X, b) do (f, out, W, X, b), start, stop
    @turbo for n ∈ start:stop, m ∈ indices((out,W,b), 1)
      outₘₙ = zero(eltype(out))
      for k ∈ indices((W,X),(2,1))
        outₘₙ += W[m,k] * X[k,n]
      end
      out[m,n] = f(outₘₙ + b[m])
    end
  end
  return out
end
function dense!(f::F, out, W, X, b, ::False) where {F}
  batch((size(out,2),num_cores()), f, out, W, X, b) do (f, out, W, X, b), start, stop
    @turbo for n ∈ start:stop, m ∈ indices((out,W,b), 1)
      outₘₙ = zero(eltype(out))
      for k ∈ indices((W,X),(2,1))
        outₘₙ += W[m,k] * X[k,n]
      end
      out[m,n] = outₘₙ + b[m]
    end
    @turbo for n ∈ start:stop, m ∈ indices((out,W,b), 1)
      out[m,n] = f(out[m,n])
    end
  end
  return out
end


