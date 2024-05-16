using Zygote
using Random, LinearAlgebra

const EPS = 1e-8
abstract type AbstractOptimiser end

mutable struct Adam <: AbstractOptimiser
    eta::Float64
    beta::Tuple{Float64,Float64}
    epsilon::Float64
    state::IdDict{Any, Any}
  end

Adam(η::Real = 0.001, β::Tuple = (0.9, 0.999), ϵ::Real = EPS) = Adam(η, β, ϵ, IdDict())
Adam(η::Real, β::Tuple, state::IdDict) = Adam(η, β, EPS, state)
  
  function apply!(o::Adam, x, Δ)
    η, β = o.eta, o.beta
  
    mt, vt, βp = get!(o.state, x) do
        (zero(x), zero(x), Float64[β[1], β[2]])
    end :: Tuple{typeof(x),typeof(x),Vector{Float64}}
  
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ * conj(Δ)
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + o.epsilon) * η
    βp .= βp .* β
  
    return Δ
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

opt = Adam(0.001)
### Experiment with some velocity values to see what you got!!!
optimize(t->g(t), x, opt; max_iter = 100000)

