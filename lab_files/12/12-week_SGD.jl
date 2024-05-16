using Zygote
using ForwardDiff
using Plots
using LinearAlgebra
using Distributions
using ProgressBars
using Printf
using Random
abstract type Regression end
mutable struct linear_regression  <: Regression
    W::AbstractVector
    linear_regression(n::Int64) = new(0.1*randn(n+1))
end

function (lr::linear_regression)(X::AbstractMatrix)
    m, _ = size(X)
    X = vcat([X ones(m)])
    return X*lr.W
end


function fit!(lr::linear_regression, 
    X::AbstractMatrix, 
    y::AbstractVector; 
    epochs ::Int64 = 10, 
    batch_size::Int64 = 32)
    
    size_x, _ = X |> size
    end_index  = size_x
    index_set = 1:size_x |> collect
    
    batches = Int(size_x/batch_size |> round)
    ## --- ##
    iter = ProgressBar(1:epochs)
    for _ in iter
        ## --- ##
        shuffle!(index_set) #### shuffle it
        loss::Float64 = 0.0
        for k in 1:batches  #### batch
            indexes = index_set[batch_size*(k-1)+1:min(end_index, batch_size*k)]
            X_, y_ = X[indexes, :], y[indexes]
            val, grad = Zygote.withgradient(lr) do lr
                mean((lr(X_) - y_).^2)
            end
            lr.W -= 0.0001*grad[1].W
            loss += val
        end
        set_postfix(iter, Loss=  @sprintf("%.4f", loss/batches))
    end
end

lr = linear_regression(200)
fit!(lr, randn(1000,200), randn(10000); batch_size = 256, epochs = 10000)