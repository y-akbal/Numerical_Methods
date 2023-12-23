using LinearAlgebra
using Zygote
using ForwardDiff
using Random

## Let's implement a something very similar, we shall than jump to more advanced things
f(x::AbstractVector) = x[1]*sin(x[2]) + x[2]*sin(x[1]) + x[1]^2 + x[2]^2
##our task is to minimize f over the x^2 + y^2 = 1
projection(x::AbstractVector) = x/norm(x)


function optimizer_projection(f::Function, projection::Function, x_init::Vector{T};
    lr::Float64 = 0.001, n_iter::Int64 = 100) where T<:Real
    for i in 1:n_iter
        value, gradient = Zygote.withgradient(x_init) do x
            f(x)
        end
    ## This dude takes care of projection!!!   
    x_init -= lr*gradient[1]
    x_init = projection(x_init)
    end
    return x_init, norm(x_init)
end

optimizer_projection(x -> f(x), y->projection(y), randn(2); n_iter = 100, lr = 0.01)

### Question, how do you make sure that x∈ [0,1]×[0,1]??? 
q(x) = max(0, min(1,x))

q(-0.1)

optimizer_projection(x -> norm(x), y->q.(y), randn(2); n_iter = 100, lr = 0.01)
## oki doki