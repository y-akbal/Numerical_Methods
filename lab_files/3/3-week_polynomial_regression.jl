using LinearAlgebra
using Statistics
using Plots
using DataFrames
using CSV

### we use permutedims for finding transpose of a matrix
### we create some syntetic data, we will then go back to real data
X = -1.5:0.1:1.5 |> collect
f(x) = (x-0.5)*(x-0.3)*(x-0.8)*(x-1.2)*(x-1.3)*(x+1.3)
y = f.(X)
###  end of syntetic data ###

scatter(X, y, label = "simple_polynomial", markersize = 2, markershape = :circle, markercolor = "red")

## create an instance that will store the coefficients of the polynomial
Base.@kwdef mutable struct pol
    degree::Int
    coeff::Vector{Real}
    fitted::Bool = false ### this is important because we do not want to override this later.
end


function pol(n::Int64) ## enjoy multiple dispatch, we are constructing the pol object here
    degree = n
    coeff = randn(degree+1)
    return pol(degree = degree, coeff = coeff)
end

function (p::pol)(x::AbstractFloat)::AbstractFloat  ###the object works like a function now (forwad pass)
    s = p.coeff[1]
    for (i,j) in enumerate(p.coeff[2:end])
        s += j*x^i  #Syntactic sugar :)
    end
    return s
end

function fit!(p::pol, X::Vector{Float64}, y::Vector{Float64})  ###if we are to change p, convention is to use fit! instead of fit
        p.fitted ? throw("The model is alrady fitted!") : p.fitted = true
        ##
        B_p = reduce(hcat, [X.^i for (i,_) in enumerate(p.coeff[2:end])])
        ##
        B = hcat(ones(length(X)), B_p)
        transposed_B = transpose(B)
        A = transposed_B*B
        y_ = transposed_B*y
        coeff_ = A^(-1)*(y_)
        p.coeff = coeff_
        error = (1/length(X))*sum((p.(X)-y).^2)
        println("Fitting is done, the error is $error")
end   


p = pol(5)
fit!(p, X, y)

using Plots
scatter(X, y, label = "data") ### the data looks pretty scary!
plot!(X, p.(X), label = "fitted_curve")
## create an instance that will store the coefficients of the polynomial




cd(@__DIR__)
#= It is time for real data =#
csv_reader = CSV.File("salary_polynomial.csv")
data = DataFrame(csv_reader)  #### created a data frame here
X,y = data[:,1], data[:, 2]
scatter(X,y) ### the data looks pretty scary!
xlabel!("Experience")
ylabel!("Mayış")




p = pol(8)
fit!(p, X, y)

using Plots
scatter(X, y, label = "data") ### the data looks pretty scary!
plot!(X, p.(X), label = "fitted_curve")
xlabel!("Experience")
ylabel!("Mayış")
## create an instance that will store the coefficients of the polynomial


function R2(p::pol, X::Vector{Float64}, y::Vector{Float64})
    return 1 - sum((p.(X) - y).^2)/sum((y .- mean(y)).^2)
end

R2(p, X, y)

### Let's find the best degree
L = ones(15)
for i in 1:15
    p = pol(i)
    fit!(p, X, y)
    L[i] = R2(p, X, y)
    println(i)
end


p = pol(argmax(L))
fit!(p, X, y)
p(8.5)  ##dolares!

## These all happen in the train set BTW therefore we are just 
## playin' in the sandpool.

