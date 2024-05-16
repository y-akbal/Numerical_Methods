using LinearAlgebra
using Zygote
using StatsBase
using Random

abstract type SVM end ## An abstract class for SVMClassifier

mutable struct SVMClassifier <: SVM
    w::AbstractVector
    b::AbstractFloat
    SVMClassifier(n::Int64; type::Type = Float64) = new(randn(n) .|> type, randn() |> type)
end

svm_ = SVMClassifier(10) ##  This guy does the classification in 10 dimensional space!!!

function sep(q::AbstractFloat)::Int
    ## This is a helper function, to be used inside svm prediction function
    return ifelse(q >= zero(q), one(q), -one(q))
end

function (svm::SVMClassifier)(x::AbstractVector) ##This is for vectors. Let's do it for arrays
    return transpose(svm.w)*x + svm.b |> sep
end

function (svm::SVMClassifier)(x::AbstractMatrix)
    return x*svm.w .+ svm.b .|> sep
end
###  Data is of the form (m, n) where m is the number of samples and n is the number of features
### [[1,2,3,4], [-1,2,3,4]]*(svm.w <--- column vector!!!) .+ b

function fit!(svm::SVMClassifier, 
    X::AbstractMatrix, 
    y::AbstractVector; 
    C::AbstractFloat = 10.0,
    lr::AbstractFloat = 1e-6,
    max_iter::Integer = 100)
    ## Things to do!!
    ## 1) Introduce the slack variables!!!
    ## 2) Update the parameters (w, b) via gradient descent

    m, _ = X |> size
    ξ = 0.0001*randn(m) .|> abs ##Slack variables!!!
    val::AbstractFloat = 0.0
    for i in 1:max_iter
        val, grad_svm = Zygote.withgradient(svm, ξ) do svm, ξ
            prod = X*svm.w
            0.5*norm(svm.w)^2+C*mean(@. max(1 - y*(prod + svm.b), ξ))
        end
        ##We grabbed the gradients!!!
        ## Need to update w, b, and ξ  
        svm.w -= lr*grad_svm[1].w
        svm.b -= lr*grad_svm[1].b
        ξ -= lr*grad_svm[2]
        ξ = max.(ξ, 0) ## We are projecting it back to zero in the case that it is negative!!!

        if i % 20 == 0
            println(val)
        end
    end
    @info "Training is done!"
end


## Here we prepare the dataset
include("10-tools.jl")
X_train, X_test, y_train, y_test = return_dataset()
## --- end of preparation  --- ### 

begin
    Random.seed!(0) ## This is for reproducibility
    svm = SVMClassifier(30)
    fit!(svm, X_train, y_train; max_iter = 10000, lr = 1e-7)
end

(svm(X_test) .== y_test .|> Int64) |> mean

### If we train it more than we need, then the model overfits!!!!
### 


