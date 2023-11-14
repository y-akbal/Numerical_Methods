using Zygote
using ForwardDiff
using Plots
using LinearAlgebra
using Random

function get_grad(x::Vector{Float64}, f::Function)
    return Zygote.gradient(t -> f(t), x)[1]
end

function get_hessian(x::AbstractVector, f::Function)
    return Zygote.hessian(t -> f(t) ,x)
end

Q = 2*randn(3,3)
Q = transpose(Q)*Q
b = randn(3)
k(x::AbstractVector) = transpose(x)*Q*x -transpose(b)*x-1/2
x = rand(3)

function fit!(k::Function, x::AbstractVecOrMat; max_iter = 100, stopping_criterion::Float64 = 1e-10)
    m, _ = get_hessian(x,k) |> size
    for i in 1:max_iter
        grad = get_grad(x,k)
        hessian = get_hessian(x,k)
        smallest_eigval = eigmin(hessian) |> abs
        x -= ((get_hessian(x,k)+smallest_eigval*I(m))^(-1))*grad
        #= If statements to check!!!  =#
        if norm(grad) < stopping_criterion
            @info "Training is done!!!"
            return x
        end
    end
    @info "Algorithm did not converge!!!"    
    return x
end


Random.seed!(0)
fit!(x->k(x),10*randn(3); max_iter = 200)

