using ProgressMeter
using Flux
using Zygote
using MLDatasets
using MLUtils
using Plots
#using Images
using NNlib  ### activation funtions
using StatsBase
## using CUDA

Network = Chain(Dense(10, 5),sigmoid, Dense(5, 1))
mean(Network(randn(10,100)), dims = 2) 

Network(randn(10, 20))    ##  Unlike python S X F the convention in Julia is F x S 

## How to train neural networks???  Remember that this is a regression problem!!!!!!!
X = randn(10, 20000) .|> Float32
y = randn(20000) .|> Float32
### We want our network to fulfill network(X) ≍ y
###So we need to adjust the weights so that the error (network(X) -y)^2 is minimal.
### This leads to an optimization problem minimize (network(X) -y)^2   using 
### any of the first order gradient descent methods.
### Basically what we do here is that weights = weights - α*gradient

data = DataLoader((X,y), batchsize = 32)

opt_state = Flux.setup(Adam(), Network)


#Network = Network |> gpu ## This is needed if you like to train on GPU
for i in 1:1000
    temp_val::Float32 = 0.f0
    temp_size::Int32 = 0
    for (x,y) in data
        ## x = x|> gpu
        ## y = y|> gpu
        val, grads = Zygote.withgradient(Network) do Network
            mean(Network(x) - transpose(y)).^2  
        end
        temp_val += val
        temp_size += size(x)[end]
        Flux.update!(opt_state, Network, grads[1])
    end
    if i %20 == 0
        println("The loss is $(temp_val/temp_size)")
    end
end


