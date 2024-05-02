using LinearAlgebra
using Zygote
using ForwardDiff
using Base
using Random
## Let's implement a something very similar, we shall than jump to more advanced things
f(x::AbstractVector) = x[1]*sin(x[2]) + x[2]*sin(x[1]) + x[1]^2 + x[2]^2

## The problem is to minimize f with respect to norm(x) <= 1
### lagrangian  = f(x)+ λ*(norm(x)-1), we need to insist on λ >= 0.
## the partial derivative of lagrangian wrt λ is norm(x) - 1

function optimizer_lag(f::Function, 
    x_init::Vector{T};
    lr::Float64 = 0.001, 
    n_iter::Int64 = 100, 
    λ::Float64 = 0.01) where T<:Real

    gradient_ = similar(x_init)

    for i in 1:n_iter
        value, gradient = Zygote.withgradient(x_init) do x
            f(x) + λ*(norm(x)-1)
        end 
    ## Update the gradients of the lagrangian
    x_init -= lr*gradient[1]
    gradient_ = gradient[1]
    λ = λ + max(lr*(norm(x_init)-1),0) ##This is the part we insist on λ>= 0
    end

    return x_init, norm(x_init), λ, gradient_
end

Random.seed!(0)
optimizer_lag(x -> f(x), 20*randn(2), lr = 0.09; n_iter = 10000, λ = 10.0)
### Question, how do you make sure that norm(x) ∼ 1??? 
