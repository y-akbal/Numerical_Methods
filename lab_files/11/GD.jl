using Zygote
using LinearAlgebra

abstract type AbstractOptimiser end


mutable struct Descent <: AbstractOptimiser
    α ::AbstractFloat
end 

function apply!(o::Descent, ∇)
    @fastmath ∇ .*= o.α
    return ∇
end

function step!(opt::AbstractOptimiser, x::AbstractArray, ∇)
    @fastmath x .-= apply!(opt, ∇)
    return x
end

function optimize(f::Function, x::AbstractArray, opt::AbstractOptimiser; max_iter = 100, stopping_criterion = 1e-10)
    @inline for i in 1:max_iter
        grad = Zygote.gradient(t->f(t), x)[1]
        x = step!(opt, x, grad)
        if norm(grad) < stopping_criterion
            return x
        end
    end
    return x
end


f(x) = (x[1]-4)^2 +(x[2]-2)^2 + 100*sin(x[1])
opt = Descent(0.1)
fit!(t->f(t), randn(2), opt)
