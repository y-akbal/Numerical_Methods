using Zygote
using LinearAlgebra

abstract type AbstractOptimiser end


mutable struct Descent <: AbstractOptimiser
    α ::AbstractFloat
end 

function apply!(o::Descent, ∇)
    ∇ .*= o.α
    return ∇
end

function step!(opt::AbstractOptimiser, x::AbstractArray, ∇)
    x .-= apply!(opt, ∇)
    return x
end


function optimize(f::Function, 
    x::AbstractArray, 
    opt::AbstractOptimiser; 
    max_iter = 100, 
    stopping_criterion = 1e-10)
    for i in 1:max_iter
        grad = Zygote.gradient(t->f(t), x)[1]
        x = step!(opt, x, grad)
        if norm(grad) < stopping_criterion
            @info "$i steps needed for convergence"
            return x
        end
    end
    return x
end

begin
    Random.seed!(0)
    x = randn(2)
end
    

g(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1]+x[1]*x[2]^2)^2 + (2.625 -x[1]+ x[1]*x[2]^3)^2

opt = Descent(0.001)
optimize(t->g(t), x, opt, max_iter = 100000,stopping_criterion = 1e-10)
