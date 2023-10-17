using Pkg
using DataFrames
using CSV
using Statistics
using Plots
using LinearAlgebra


cd(@__DIR__) ### changes directory to your working directory


csv_reader = CSV.File("Fish_multiple_linear.csv")
data = DataFrame(csv_reader)  #### created a data frame here

X = data[:, 3:end] |> Matrix{Float64}
y = data[:, 2] |> Vector{Float64}

mutable struct LR
    coeff:: Vector{Float64}
    fitted::Bool 
end

function LR(n::Int)
    ## Here n stands for the number of futures
    ## to be used!!!
    coeff = randn(n+1) ## including bias
    lr = LR(coeff, false)
    return lr
end

lr = LR(5)

function (lr::LR)(x::Vector{Float64})::Float64  ### This is for single example 
    coeff = lr.coeff[1:end-1]
    bias = lr.coeff[end]
    return transpose(coeff)*x + bias
end


function (lr::LR)(x::Matrix{Float64})::Vector{Float64} ##Forward pass for entire dataset
    return x*lr.coeff[1:end-1] .+ lr.coeff[end]
end

lr(ones(100, 5) .|> Float64).-sum(lr.coeff)


X = randn(100, 10)
lr = LR(10)
y = lr(X)


function fit!(lr::LR, X::Matrix{Float64}, y::Vector{Float64})
    lr.fitted ? throw("The model is already fitted!") : nothing
    n_sample , _ = X |> size
    X_concatted = hcat(X, ones(n_sample, 1))
    X_ = transpose(X_concatted)*X_concatted
    if isapprox(det(X_) ,0, atol = 1e-2) 
        throw("Some features are highly correlated with each other, this may cause numerical instability!!!")
    end
    coeff = (X_)^(-1)*transpose(X_concatted)*y
    lr.coeff = coeff
    print("Model fitted succesfully!!!")
    lr.fitted = true
end

fit!(lr, X, y)


###Arrright comrades, let's now get started!!!
lr = LR(5)
fit!(lr, X[1:100,:], y[1:100])

function R2(y_pred::Vector{Float64}, y::Vector{Float64})
    return 1 - sum((y_pred - y).^2)/sum((y .- mean(y)).^2)
end

y_pred = lr(X[101:end,:])
y_true = y[101:end]

R2(y_pred, y_true)  ## approximately 0.73 yeaah ok!! not that bad.

## How about train error? 
y_pred_train = lr(X[1:100,:])
y_true_train = y[1:100]
R2(y_pred_train, y_true_train)  
## approximately 0.93 yeaah ok!! not that bad.

### Now let's shuffle the dataset see if we can get something better???
using Random
index = collect(1:159)
shuffle!(index)

lr = LR(5)
X_train, y_train = X[index[1:100],:], y[index[1:100]]
X_test, y_test = X[index[101:end],:], y[index[101:end]]
fit!(lr, X_train, y_train)

y_pred = lr(X_test)
y_true = y_test
R2(y_pred, y_true)  ##  A bit increased!!! we owe this due to shuffling!!!

## What ya say bro?