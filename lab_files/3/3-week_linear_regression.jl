#=
Today's Dish:
1) Linear Regression 
=#
using Pkg
using DataFrames
using CSV
using Statistics
using Plots
using LinearAlgebra

cd(@__DIR__) ### changes directory to your working directory

csv_reader = CSV.File("Fish_multiple_linear.csv")
data = DataFrame(csv_reader)  #### created a data frame here

X = data[:, 4] |> Vector{Float64}
y = data[: ,2] |> Vector{Float64}

#= Let's implement our own linear regression thing =#
Base.@kwdef mutable struct LR
    a:: Float64 = randn()
    b:: Float64 = randn()
end

lr = LR() ##intitialize the linear regression object!
(lr::LR)(x::Real) = lr.a * x + lr.b  ## evaluation procedure is here now!
(lr::LR)(x::Array) = lr.(x)  ### it is for forward pass everything

function fit!(lr::LR, X::Vector{Float64}, y::Vector{Float64})::Nothing
    mx = mean(X)
    my = mean(y)
    x_0 = X .- mx
    y_0 = y .- my
    x_0_s = sum((x_0).^2)
    lr.a = transpose(x_0)*y_0/x_0_s
    lr.b = my - lr.a*mx
    println("Model fitted succesfully!")
end

fit!(lr, X, y)

###Let's sketch the predictions
scatter(X, y, label = "Real")
plot!(X, lr(X), label = "Predicted")


function mse(lr::LR, X::Vector{Float64}, y::Vector{Float64})
    return sum((lr(X) - y).^2)
end

mse(lr, X, y)


function R2(lr::LR, X::Vector{Float64}, y::Vector{Float64})
    return 1 - sum((lr(X) - y).^2)/sum((y .- mean(y)).^2)
end

R2(lr, X, y)

###Let's visualize the residuals
scatter(X, (lr(X) - y), label = "As you see, high variance!")
### Remember that this is on the train set! 
### How do you pass to test set?