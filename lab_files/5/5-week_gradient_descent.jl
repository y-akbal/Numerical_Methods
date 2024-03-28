### Gradient Descent example
using ForwardDiff, Zygote, LinearAlgebra
using Plots

## See for test functions https://en.wikipedia.org/wiki/Test_functions_for_optimization
## We shall use Beale function 
f(x) = (1.5 - x[1] + x[1]*x[2])^2+ (2.25-x[1]+ x[1]*x[2]^2)^2+(2.625-x[1]+x[1]*x[2]^3)^2
f(x::AbstractFloat,y::AbstractFloat) = f((x,y))
## See here to 
let 
    x = 1:0.01:3
    y = -1:0.01:1
    z =  @. f(x', y)
    contour(x,y,z, levels = 100,color=:turbo, clabels=true, cbar=false, lw=2, fill = true)   
end

## Mind the type of the gradients returned by Zygote
Zygote.gradient(x->f(x), randn(2))[1]
## Syntactic sugar for getting grads 
Zygote.gradient(randn(2)) do x
    f(x)
end
### 

function optimize(f::Function, x_init::AbstractVector; lr::AbstractFloat = 0.001, 
    max_iter::Integer = 100,
    stopping_criterion::Float64 = 1e-2)
    for i in 1:max_iter
        ## Find the gradient
        grad = Zygote.gradient(x->f(x), x_init)[1]
        ## update x_init
        x_init -= lr*grad
        if norm(grad) < stopping_criterion
            @info "OK Boomer!!! you are done in $(i) steps"
            return x_init
        end
    end
    @info "Convergence criterion not met!"
    return x_init
end

optimize(x->f(x), randn(2); lr = 0.001, max_iter = 1230, stopping_criterion = 0.01)
