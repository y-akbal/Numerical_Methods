## Assume that you are given a linear system
## Ax = y, where A and y is given as follows
## Using gradient descent method find a vector x so that 
## ||Ax-y|| is minimum.

using Distributions
using Random

A,y = begin
    Random.seed!(0)
    randn(10,10), randn(10,1)
end

function fit(X::Matrix{Float64}, 
    y::Vector{Float64}; 
    lr::Float64 = 0.001, 
    max_iter::Int64 = 1000,
    stopping_criterion::Float64 = 1e-2, 
    seed::Int64 = 10)::Vector{Float64}
    ## Your code here
    return x
end


function unit_test()
    
    try
        @assert isapprox(A\y, x, atol = 1e-1)
    catch AssertionError
        @info "You gotto do it again Pal!!, adjust the learning rate and watch the convergence!!!"
        throw("Buddy wrong time")
    end
    @info "Great Success!!!"
    return 1
end

