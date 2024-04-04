using Zygote
using LinearAlgebra
using Plots
using ForwardDiff
using Roots
using Random


Q = [1 0 0;0 3 0; 0 0 9]
b = randn(3)


f(x::AbstractVector) = transpose(x)*Q*x -transpose(b)*x
x = rand(3) # initial_point

function grad_descent(f::Function, 
                    x::AbstractVector;
                    n_iter::Int = 100, 
                    lr::Float64 = 0.004, 
                    stop_crit::Float64 =1e-5)
    for i in 1:n_iter
        grad = Zygote.gradient(f, x)[1]
        x -= lr*grad
        if norm(grad) < stop_crit
            @info "OK Boomer, you got it in $i steps!!!"
            return x
        end
    end
    return x
end

begin
    ## A local scope is good!!
    Random.seed!(0)
    grad_descent(x->f(x), randn(3); n_iter = 100000, lr = 0.1)
end

### Let's now jump to steepest descent!!! ### 
x = randn(3)
grad = Zygote.gradient(t->f(t), x)[1]
g(alpha) = f(x-alpha*grad)
g_deriv(alpha) = Zygote.gradient(t->g(t), alpha)[1]
### Some plots ### 
g_deriv(0.001)
plot(0:0.0001:.4, g.(0:0.0001:.4), label ="function")
plot!(0:0.0001:.4, g_deriv.(0:0.0001:.4), label = "its derivative")

### Steepest descent now on the scene ### 
function fit!(f::Function, x::AbstractVector; n_iter::Int = 100, ϵ::Float64 = 1e-5)
    ## Starting learning rate is pretty small!!
    α::Float64 = 0.002*rand()
    for i in 1:n_iter
        grad = Zygote.gradient(t -> f(t), x)[1]
        phi(α) = f(x - α*grad)
        ## -- we are using derivative -- ##
        D(α) = Zygote.gradient(t -> phi(t), α)[1]
        ## Here comes the learning rate!!!
        α_ = find_zero(α -> D(α), (0,1))
        ## -- end of derivative -- ##
        x -= α_*grad
        if norm(grad) < ϵ
            @info "Ok Boomer you are done in $(i) steps!!!"
            return x
        end
    end
    return x
end


begin
    ## A local scope is good!!
    Random.seed!(0)
    fit!(x->f(x), randn(3); n_iter = 100000, ϵ = 1e-5)
end



"""
## Let's give a try to rosenbrock function!!!using Plots;
g(x) = (3x[1] + x[2]^2) * abs(sin(x[1]) + cos(x[2]))
g(x,y) = g([x,y])

begin
    ## A local scope is good!!
    Random.seed!(0)
    x, Q = fit!(x->g(x), randn(2); n_iter = 100000, ϵ = 1e-5)
end



x = -5:0.1:5
y = -5:0.1:5
z = @. g(x', y)
contour(x, y, z, fill = true)

"""
