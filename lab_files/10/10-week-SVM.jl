using LinearAlgebra
using DataFrames
using CSV
using Zygote
using Plots
using StatsBase

## Here we prepare the dataset
include("10-tools.jl")
X_train, X_test, y_train, y_test = return_dataset()
## --- end of preparation  --- ### 

## Roof as always ##
abstract type SVM end
mutable struct SVMClassifier <: SVM
    w::AbstractVector
    b::Real
    ξ::AbstractVector
    SVMClassifier(n::Int64, m::Int64) = n > 0 ? new(0.05*randn(n), 0.5*randn(), zero(randn(m))) : error("ya kiddin bro?")
end

function sep(q::AbstractFloat)
    return q >= 1 ? 1 : q <= -1 ? -1 : 0
end

function (svm::SVMClassifier)(x::AbstractVector)
    q = transpose(x)*svm.w + svm.b
    return sep(q)
end


function (svm::SVMClassifier)(x::AbstractMatrix)
    return  sep.(x*svm.w .+ svm.b)
end

function fit!(svm::SVMClassifier, 
    X::AbstractMatrix, 
    y::AbstractVector; 
    C::Float64 = 10.0, 
    max_iter::Int64 = 100, lr::Float64 = 0.00000001)
    ### There is a mistake here ###
    m, n = X |>size
    
    for i in 1:max_iter
        val, grad = Zygote.withgradient(svm) do svm
           (1/2)*norm(svm.w)^2 + C*sum(max.(1 .- y.*(X*svm.w .+ svm.b), svm.ξ))
        end
        ## Update the gradients!!!
        begin
            svm.w -= lr*(grad[1].w)
            svm.b -= lr*(grad[1].b)
            svm.ξ -= lr*(grad[1].ξ)
            svm.ξ = max.(svm.ξ, 0)
        end        
        if i % 20 == 0
        println("the loss is $(val)")
        end
    end
end

begin 
    Random.seed!(0)
    svm = SVMClassifier(30, 455)
end

X = X_train
y = y_train
svm = SVMClassifier(30)

val, grad = Zygote.withgradient(svm) do svm
    (1/2)*norm(svm.w)^2 + sum(max.(1 .- y.*(X*svm.w .+ svm.b), 0))
end



fit!(svm, X_train, y_train; max_iter = 100000, C = 100.0)
mean(svm(X_test) .== y_test)