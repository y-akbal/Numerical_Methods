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

Q = 2*randn(3,3) 
Q = transpose(Q)*Q
b = randn(3) 
k(x::AbstractVector) = transpose(x)*Q*x -transpose(b)*x-1/2
x = randn(3) 

get_hessian(x, x->k(x))


function fit!(k::Function, x::AbstractVecOrMat; max_iter = 100, stopping_criterion::Float64 = 1e-10)
    for i in 1:max_iter
        grad = get_grad(x,k)
        hessian = get_hessian(x,k)
        if isapprox(det(hessian), 0)
            @info "Hessian is not invertible!!!"
            break
        end
        x -= (hessian^(-1))*grad
        if norm(grad) < stopping_criterion
            if det(hessian) > 0
                @info "Trainin' is done!!! $i steps needed for convergence, we now have a local minimum"
            else
                @info "Trainin' is done!!! $i steps needed for convergence, we now have a local max"
            end
                return x
        end
    end
    @info "Algorithm did not converge!!!"    
    return x
end

x = 1000*randn(3)   
x = fit!(t->k(t),x; max_iter = 2000)

println(x)
