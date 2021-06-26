function pool_project(idx::Symbol, stride::Int, pad::Int, idk::Symbol, dilation::Int)
  offset = 1 - pad - stride
  :($idx*$stride + $idk*$dilation + $offset)
end

@generated function _pool_turbo!(
  ::Val{name},
  y::AbstractArray{T,M},
  x::AbstractArray{T,M},
  pdims::PoolDims{N,K,S,P,D};
  alpha=One(), beta = Zero()
) where {name,T,M,N,K,S,P,D}
  # @assert beta == T(0) "beta not supported yet"

  kernel_w, kernel_h, kernel_d = K
  pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = P
  dil_w, dil_h, dil_d = D
  stride_w, stride_h, stride_d = S

  # We use calc_padding_regions to split outselves up into separate regions that may or
  # may not need to worry about padding:
 
  # Each loop, we initialize `m` to something, set that here.
  if $(name === :max)
    # m_init = T <: AbstractFloat ? nextfloat(typemin(T)) : typemin(T)
    m_init = typemin(T)
    f = :max
  elseif $(name === :mean)
    m_init = T(0)
    f = :(+)
  else
    error("Unimplemented codegen path")
  end
  
  input_kd = pool_project(:d, stride_d, pad_d_lo, :kd, dil_d)
  input_kh = pool_project(:h, stride_h, pad_h_lo, :kh, dil_h)
  input_kw = pool_project(:w, stride_w, pad_w_lo, :kw * dil_w)

  # beta ≠ 0 not supported
  # beta * y[w, h, d, c, batch_idx]
  # If we're doing mean pooling, we represent division by kernel size by rolling it
  # into the `alpha` multiplier.
  assignment = if alpha === One
    name === :mean ? :($(inv(prod(K))) * m) : :m
  else
    name === :mean ? :($(inv(prod(K))) * alpha * m) : :(alpha * m)
  end
  quote
    check_dims(size(x), size(y), pdims)

    width, height, depth = input_size(pdims)
    out_c = channels_out(pdims)
    out_width, out_height, out_depth = output_size(pdims)

    padded_regions, central_region = calc_padding_regions(pdims)

    # Start with the central region
    w_region, h_region, d_region = central_region
    @batch for batch_idx in 1:size(x,$M)
      @turbo for c in 1:out_c,
        d in d_region,
        h in h_region,
        w in w_region
        
        # Initialize `m` to `0.0`, or `typemin(T)` or whatever.
        m = $m_init

        for kd in CloseOpen(StaticInt($kernel_d)),
          kh in CloseOpen(StaticInt($kernel_h)),
          kw in CloseOpen(StaticInt($kernel_w))

          m = $f(m, x[$input_kw, $input_kh, $input_kd, c, batch_idx])
        end
        y[w, h, d, c, batch_idx] = $assignment
      end
      
      # Next, the padded regions
      @inbounds @fastmath for (w_region, h_region, d_region) in padded_regions
        for c in 1:out_c,
          d in d_region,
          h in h_region,
          w in w_region

          # In these loops, we have to check that we're not reaching off the edge, we
          # do so by putting in a bunch of conditionals.  :/
          m = m_init
          for kd in CloseOpen(StaticInt($kernel_d))
            input_kd = $input_kd
            ((input_kd <= 0) | (input_kd > depth)) && continue
            for kh in CloseOpen(StaticInt($kernel_h))
              input_kh = $input_kh
              ((input_kh <= 0) | (input_kh > height)) && continue
              for kw in CloseOpen(StaticInt($kernel_w))
                input_kw = $input_kw
                ((input_kw <= 0) | (input_kw > width)) && continue
                m = $f(m, x[input_kw, input_kh, input_kd, c, batch_idx])
              end
            end
          end
          y[w, h, d, c, batch_idx] = $assignment
        end
      end
    end

    # Return `y`
    return y
  end
end

# Same story for gradients, and although this is very similar to the forward pass,
# it's unfortunately different enough that I think we need a separate function.  :(
@generated function _∇pool_turbo!(
  ::Val{name},
  dx::AbstractArray{T,M}, dy::AbstractArray{T,M},
  y::AbstractArray{T,M}, x::AbstractArray{T,M},
  pdims::PoolDims{N,K,S,P,D};
  alpha=One(), beta = Zero()
) where {name,T,M,N,K,S,P,D}


  kernel_w, kernel_h, kernel_d = K
  pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = P
  dil_w, dil_h, dil_d = D
  stride_w, stride_h, stride_d = S

  
  check_dims(size(x), size(dy), pdims)

  width, height, depth = input_size(pdims)
  kernel_w, kernel_h, kernel_d = kernel_size(pdims)
  out_c = channels_out(pdims)
  pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(pdims)
  dil_w, dil_h, dil_d = dilation(pdims)
  stride_w, stride_h, stride_d = stride(pdims)
  out_width, out_height, out_depth = output_size(pdims)

  # We use calc_padding_regions to split outselves up into separate regions that
  # may or may not need to worry about padding:
  padded_regions, central_region = calc_padding_regions(pdims)

  input_kd = pool_project(:d, stride_d, pad_d_lo, :kd, dil_d)
  input_kh = pool_project(:h, stride_h, pad_h_lo, :kh, dil_h)
  input_kw = pool_project(:w, stride_w, pad_w_lo, :kw * dil_w)

  # If we're doing mean pooling, we represent division by kernel size by rolling
  # it into the `alpha` multiplier.
  assignment = if alpha === One
    name === :mean ? :($(inv(prod(K))) * m) : :m
  else
    name === :mean ? :($(inv(prod(K))) * alpha * m) : :(alpha * m)
  end
  quote
    # Start with the central region
    w_region, h_region, d_region = central_region
    @batch for batch_idx in 1:size(x, $M)
      for c in 1:out_c,
        d in d_region,
        h in h_region,
        w in w_region

        # Grab the output at this index for future use
        y_idx = y[w, h, d, c, batch_idx]
        dy_idx = dy[w, h, d, c, batch_idx]
        maxpool_already_chose = false

        for kd in 1:kernel_d,
          kh in 1:kernel_h,
          kw in 1:kernel_w

          input_kd = project(d, stride_d, pad_d_lo) + (kd - 1) * dil_d
          input_kh = project(h, stride_h, pad_h_lo) + (kh - 1) * dil_h
          input_kw = project(w, stride_w, pad_w_lo) + (kw - 1) * dil_w

          # This conditional will be optimized away at compile time,
          # or my name isn't shengdan jingyu
          if $(name == :max)
            # If it's equal; this is the one we chose. We only choose one per
            # kernel window, all other elements of dx must be zero.
            # Uncomment line below if using with non-precise output (e.g. by NNPACK)
            # if abs(y_idx - x[x_idxs...]) < 1e-5 && !maxpool_already_chose
            if y_idx ≈ x[$input_kw, $input_kh, $input_kd, c, batch_idx] && !maxpool_already_chose
              dx[$input_kw, $input_kh, $input_kd, c, batch_idx] += dy_idx * alpha + beta * dx[$input_kw, $input_kh, $input_kd, c, batch_idx]
              maxpool_already_chose = true
              # Maxpooling does not support `beta` right now.  :(
            end
          elseif $(name == :mean)
            # Either does meanpool :(
            dx[$input_kw, $input_kh, $input_kd, c, batch_idx] = dy_idx * alpha + dx[$input_kw, $input_kh, $input_kd, c, batch_idx]
          else
            error("Unimplemented codegen path")
          end
        end
      end

      # Next, the padded regions
      @inbounds for (w_region, h_region, d_region) in padded_regions
        for c in 1:out_c,
          d in d_region,
          h in h_region,
          w in w_region

          # Grab the incoming gradient at this index for future use
          y_idx = y[w, h, d, c, batch_idx]
          dy_idx = dy[w, h, d, c, batch_idx]
          maxpool_already_chose = false

          # In these loops, we have to check that we're not reaching off the edge,
          # we do so by putting in a bunch of conditionals.  :/
          for kd in 1:kernel_d
            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1) * dil_d
            if input_kd <= 0 || input_kd > depth
              continue
            end

            for kh in 1:kernel_h
              input_kh = project(h, stride_h, pad_h_lo) + (kh - 1) * dil_h
              if input_kh <= 0 || input_kh > height
                continue
              end

              for kw in 1:kernel_w
                input_kw = project(w, stride_w, pad_w_lo) + (kw - 1) * dil_w
                if input_kw <= 0 || input_kw > width
                  continue
                end

                # Same as above
                if $(name == :max)
                  # Uncomment line below if using with non-precise output
                  # if abs(y_idx - x[x_idxs...]) < 1e-5 && !maxpool_already_chose
                  if y_idx ≈ x[$input_kw, $input_kh, $input_kd, c, batch_idx] && !maxpool_already_chose
                    dx[$input_kw, $input_kh, $input_kd, c, batch_idx] += dy_idx * alpha + beta * dx[$input_kw, $input_kh, $input_kd, c, batch_idx]
                    maxpool_already_chose = true
                  end
                elseif $(name == :mean)
                  dx[$input_kw, $input_kh, $input_kd, c, batch_idx] += dy_idx * alpha + beta * dx[$input_kw, $input_kh, $input_kd, c, batch_idx]
                else
                  error("Unimplemented codegen path")
                end
              end
            end
          end
        end
      end
    end
    # Return `dx`
    return dx
  end
end
