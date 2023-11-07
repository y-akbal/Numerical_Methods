using LinearAlgebra, Zygote, ForwardDiff, Printf
using CSV, DataFrames
using StatsBase: mean
using Parameters
using Distributions
using Random

### The following for the loss configuration ###
function sigmoid(x::Real)
    ## https://www.youtube.com/watch?v=bn1YCClRF-g
    ## Activation function
    return 1/(1+exp(-x))
end

function local_loss(x::Real,y::Real)
    return -(x*log(y) + (1-x)*log(1-y))
end

function loss(y_pred::AbstractVector{T}, y_true::AbstractVector{T}) where T <: Real
    return mean(local_loss.(y_true, y_pred)) ##broadcasting rocks!!!
end

### the above dude is the loss function
loss([1,1,1].-1e-2, [0,0,0].+1e-2) ## large
loss([0,0,0].+1e-2, [1,1,1].+1e-2) ## large
loss([0,0,0].+1e-2, [0,0,0].+1e-2) ## small
### ----  ###



abstract type LogisticClassifier end ### create a roof for the logistic regression LogisticClassifier
mutable struct logistic_regression <: LogisticClassifier
    θ::AbstractVector{Real}
end

function logistic_regression(n::T)  where T<:Integer### init
    ## Here n is the input dimension: the number of futures
    ## We initialize it very low variance to avoid divergence!!!
    return logistic_regression(0.005*randn(n+1))
end

function (lr::logistic_regression)(X::Matrix{T}) where T<:Real
    ## This dude is the forward pass function!!!
    x_, _ = X |> size
    augmented = vcat([X ones(x_)])
    return sigmoid.(augmented*lr.θ)
end

## Let's create fake data to see how we are doing ##
X = randn(10000, 5)
y = sample([0,1], 10000) .|> Float64

## -- ##
lr = logistic_regression(5)
loss(lr(X), y)
## Let's give a test ### 
#val, grad = Zygote.withgradient(lr->loss(lr(X), y), lr)


function predict_(lr::logistic_regression, X::AbstractMatrix, y::AbstractVector)
    round = x -> x > 1/2 ? 1 : 0
    ## round up if probability is greater than 1/2 else 0
    y_pred = round.(lr(X))
    return mean(y_pred .== y .|> Int)
end

## we shall now implement fit! method!!!
function fit_!(lr::logistic_regression, X::Matrix{T}, y::Vector{T}; learning_rate::Float64 = 0.00001, max_iter::Integer = 5) where T<:Real
    learning_rate = learning_rate |> T
    for i in 1:max_iter
        val, grad = Zygote.withgradient(lr) do lr 
            y_pred = lr(X)
            return loss(y_pred, y)
        end
        ## Update the gradients!!!
        lr.θ -=  learning_rate*grad[1].θ
        if i % 100 == 0
            println("The loss is $(val), $(predict_(lr, X,y)), and the gradient norm is $(norm(grad[1].θ))")
        end
    end
    return nothing
end

## -- ##
fit_!(lr, X, y; learning_rate = 0.00001, max_iter = 10000)
### Things seem to work fine!!!
### Let's now jumpt to a real life application....

cd(@__DIR__) ##Change the dir to your current dir!!

csv = CSV.read("breast-cancer.csv", DataFrame)
### Inspect the data!!!!  See if there is something that you can do???
function one_hot(v)
    unique_labels = unique(v)
    dict::Dict = Dict(label => int_ - 1 for (int_, label) in enumerate(unique_labels))
    inverse_dict::Dict = Dict(int_ - 1 => label for (int_, label) in enumerate(unique_labels))
    vec::Vector{Int64} = ones(Int64, size(v)[1])
    for (i,tag) in enumerate(v)
        vec[i] = dict[tag]
    end
    return dict, inverse_dict, vec
end

dict, inverse_dict, y = csv[:, 2] |> one_hot 

X = csv[:, 3:end] |> Matrix{Float64}
y = y .|> Float64
 
function split_data(X::AbstractMatrix,y::AbstractVector; split_ratio::Float64 = 0.8, shuffle = true)
    ## We use this dude to split the data into train and test
    sample_size_x, _ = size(X)
    sample_size_y = size(y)[1]
    @assert sample_size_x == sample_size_y "The vectors should be of the same size!!! one has $(sample_size_x) while the other $(sample_size_y)"
    indexes = collect(1:sample_size_x)
    if shuffle
        shuffle!(indexes)
    end
    N = split_ratio*sample_size_x |>floor |> Int
    X_train, X_test = X[1:N, :], X[N:end,:]
    y_train, y_test = y[1:N], y[N:end]

    return X_train, X_test, y_train, y_test
end


X_train, X_test, y_train, y_test = split_data(X, y)

lr = logistic_regression(30)

###Initialize network parameters with high variance, what happens?
## Why do you think that you have come accross such a scenerio?

fit_!(lr, X_train, y_train; learning_rate = 0.0000001, max_iter = 10000)
predict_(lr, X_test,y_test)

### Yep we are good now!!!


