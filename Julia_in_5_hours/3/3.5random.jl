
### getting random arrays
### drawing samples
### distributions

using Distributions
using Random
using StatsBase
## see here https://juliastats.org/Distributions.jl/stable/starting/
@show names(Distributions)

##We need some functions from this library
randn(Float32, 10, 10) ##Matrix of float32 10,10
randn(10) ##Vector 
##These are from normal distributions

## How about other distributions?
dist = Chi(5)
rand(dist, 10) 
##
dist = Normal(0, 2)
rand(dist, 10) 
## How about fixing seed, to get the same results always?
begin
    Random.seed!(1223)
    dist = Chi(5)
    rand(dist, 10)
end 
## Weighted sample
using Plots
a = [rand() for _ in 1:10]
w = a./sum(a) 
histogram(wsample(collect(1:10), w, 100000), bins = 20, density = true)
## 
wsample(1:10, w, 10; replace = false)