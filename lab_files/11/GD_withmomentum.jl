using Zygote
using LinearAlgebra
using Random
abstract type AbstractOptimiser end
mutable struct Momentum <: AbstractOptimiser
    eta::AbstractFloat ## learning rate
    rho::AbstractFloat ## β -- 
    velocity::IdDict ## Momentum term!!!
end

Momentum(η = 0.01, ρ = 0.9) = Momentum(η, ρ, IdDict())

function apply!(o::Momentum, x, Δ)
    η, ρ = o.eta, o.rho
    v = get!(() -> zero(x), o.velocity, x)
    @. v = ρ * v - η * Δ
    @. Δ = -v
end


function step!(opt::AbstractOptimiser, x::AbstractArray, Δ)
    @fastmath x .-= apply!(opt,x, Δ)
    return x
end 

function optimize(f::Function, x::AbstractArray, opt::AbstractOptimiser; max_iter = 2, stopping_criterion = 1e-10)
    for i in 1:max_iter
        grad = Zygote.gradient(t->f(t), x)[1]
        x = step!(opt, x, grad)
        if norm(grad) < stopping_criterion
            @info "ok in $(i) steps"
            return x
        end
    end
    @info "No convergence buddy!!!"
    return x
end

begin
Random.seed!(0)
x = randn(2)
end
g(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1]+x[1]*x[2]^2)^2 + (2.625 -x[1]+ x[1]*x[2]^3)^2

opt = Momentum(0.001, 0.99, IdDict())
### Experiment with some velocity values to see what you got!!!
grad = Zygote.gradient(t->g(t), x)[1]
optimize(t->g(t), x, opt; max_iter = 100000)
