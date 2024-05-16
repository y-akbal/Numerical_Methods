using Zygote

## Derivative with respecto to union.
mutable struct a
    θ::AbstractFloat
    η::AbstractFloat
end

β::AbstractFloat = 5.0
m = a(randn(), randn())

t = Zygote.gradient(m,β) do m, β
    m.θ^2 +(m.η)^2+β
end



x = [1 2 3; 4 5 6]; y = [7, 8]; z = [1, 10, 100];
g = gradient(Params([x, y])) do
    sum(x .* y .* z')
  end
y = randn(100)
x = randn(100)
A = randn(100,100)


function mymul!(y, A, x)
   @inbounds for j in 1:length(x)
        @simd for i in 1:length(y)
                    y[i] = A[i,j] * x[j] 
        end
    end
end

@btime mymul!(y, A, x)

