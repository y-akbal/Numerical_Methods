using Zygote
using ForwardDiff
using Plots
using LinearAlgebra

function get_grad(x::AbstractVector, f::Function)
    return Zygote.gradient(t -> f(t), x)[1]
end

function get_hessian(x::AbstractVector, f::Function)
    return Zygote.hessian(t -> f(t) ,x)
end
Q = [1 0 0;0 3 0; 0 0 9]
b = randn(3)

k(x::AbstractVector) = (1/2)*transpose(x)*Q*x -transpose(b)*x
x = rand(3) # initial_point

get_hessian(x, x->k(x))

function fit!(k::Function, x::AbstractVecOrMat; max_iter = 100, stopping_criterion::Float64 = 1e-10)
    for i in 1:max_iter
        grad = get_grad(x,k)
        hessian = get_hessian(x,k)
        eigen_values = eigvals(hessian)
        if isapprox(det(hessian), 0)
            @info "Hessian is not invertible!!!"
            break
        end
        x -= (hessian^(-1))*grad
        if norm(grad) < stopping_criterion
            if all(eigen_values .> 0)
                @info "Trainin' is done!!! $i steps needed for convergence, we now have a local minimum"
                return x
            elseif all(eigen_values.< 0)
                @info "Trainin' is done!!! $i steps needed for convergence, we now have a local max"
                return x
            elseif prod(eigen_values) < 0
                @info "Trainin' is done!!! $i steps needed for convergence, we now have a saddle point"
                return x
            end
        end
    end
    @info "Algorithm did not converge!!!"    
    return x
end

g(x) = (1.5 - x[1] + x[1]*x[2])^2 + (2.25 - x[1]+x[1]*x[2]^2)^2 + (2.625 -x[1]+ x[1]*x[2]^3)^2
q(x) = (1-x[1])^2 + 100*(x[2] - x[1]^2)^2

x = randn(2)

fit!(t->g(t),x; max_iter = 100)
