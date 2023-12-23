using Zygote
using Random, LinearAlgebra

const EPS = 1e-8
abstract type AbstractOptimiser end


mutable struct AdaGrad <: AbstractOptimiser
    eta::Float64
    epsilon::Float64
    acc::IdDict
  end
  
AdaGrad(η::Real = 0.1, ϵ::Real = EPS) = AdaGrad(η, ϵ, IdDict())
AdaGrad(η::Real, state::IdDict) = AdaGrad(η, EPS, state)
  
function apply!(o::AdaGrad, x, Δ)
    η = o.eta
    acc = get!(() -> fill!(similar(x), o.epsilon), o.acc, x)::typeof(x)
    @. acc += Δ * conj(Δ)
    @. Δ *= η / (√acc + o.epsilon)
  end

  function step!(opt::AbstractOptimiser, x::AbstractArray, ∇)
    @fastmath x .-= apply!(opt, x, ∇)
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

opt = AdaGrad()
### Experiment with some velocity values to see what you got!!!
optimize(t->g(t), randn(2), opt; max_iter = 100000)

