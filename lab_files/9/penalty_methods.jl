using LinearAlgebra
using Zygote
using ForwardDiff
using Base

## Let's implement a something very similar, we shall than jump to more advanced things
f(x::AbstractVector) = x[1]*sin(x[2]) + x[2]*sin(x[1]) + x[1]^2 + x[2]^2


function optimizer(f::Function, 
    x_init::Vector{T};
    lr::Float64 = 0.001, 
    n_iter::Int64 = 100, 
    λ::Float64 = 0.01) where T<:Real
    for i in 1:n_iter
        value, gradient = Zygote.withgradient(x_init) do x
            f(x) + λ*norm(x)
        end 
    x_init -= lr*gradient[1]
    end

    return x_init, norm(x_init)
end

optimizer(x -> f(x), randn(2); n_iter = 10000, λ = 5.0)### Experiment with some λ values see why this method is called penalty method
### Question, how do you make sure that norm(x) ∼ 1??? 
