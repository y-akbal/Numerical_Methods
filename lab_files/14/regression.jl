using Flux
using Zygote
using MLDatasets
using MLUtils
using Plots
using NNlib  ### activation funtions
using StatsBase
using DataFrames
using Random
using PyCall ##This dude is used for calling functions from python

## Google boston housing dataset
data_x, data_y = BostonHousing()[:][1], BostonHousing()[:][2]
data_x = data_x |> Matrix{Float32} |> transpose
data_y = data_y |> Array{Float32}

## We normalize the outputs so that the network converges quickly!!!1
data_x = (data_x .- mean(data_x, dims = 2))./std(data_x, dims = 2)
data_y = (data_y .- mean(data_y))./std(data_y)


Network =  let 
    ## Here we fix the seed for reproduciblity!!!!
    ## Google dropout layers
    Random.seed!(0)
    Chain(Dense(13, 10, relu),  Dropout(0.5),
        Dense(10, 10, relu), Dropout(0.5),
        Dense(10, 10, relu), Dropout(0.5),
        Dense(10, 5, relu), Dropout(0.5),
        Dense(5, 1))
end
opt_state = Flux.setup(Flux.Optimise.Momentum(0.01), Network) ## Optimizer state --> we used SGD with momentum

## We do train test split!!!
indexes = collect(1:506)
let
    Random.seed!(0)
    shuffle!(indexes)
end

train_indexes, val_indexes = indexes[101:end], indexes[1:100]
X_train, X_val, y_train, y_val = data_x[:, train_indexes], data_x[:, val_indexes], data_y[train_indexes], data_y[val_indexes]

train_data = DataLoader((X_train, y_train), batchsize = 16)
val_data = DataLoader((X_val, y_val), batchsize = 8)


#Network = Network |> gpu
for i in 1:1000
    temp_val::Float32 = 0.f0
    temp_size::Int32 = 0
    trainmode!(Network) ##Dropout layers active when in train mode
    for (x,y) in train_data ##grab the data from dataloader
        ## x = x|> gpu
        ## y = y|> gpu
        val, grads = Zygote.withgradient(Network) do Network
            mean(Network(x) - transpose(y) .|> abs)   ## take the gradient here!!
        end 
        temp_val += val
        temp_size += size(x)[end]
        Flux.update!(opt_state, Network, grads[1]) ##update the weights
    end
    temp_validation_val::Float32 = 0.f0
    temp_validation_size::Int32 = 0
    testmode!(Network)
    for (x,y) in val_data
        temp_validation_val += mean(Network(x) - transpose(y)).^2  
        temp_validation_size += size(x)[end]
    end
    println("The train loss is $(temp_val/temp_size), validation loss is $(temp_validation_val/temp_validation_size)")

end
## This part could be done in Julia, but let's use PyCall library,
### for r2 function see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

@pyimport sklearn.metrics as m
## We use here our beloved friend scikitlearn
m.r2_score(y_val, Network(X_val) |> transpose)
## This dude should be around 0.50 which is a bit bad!!! Tweak the hyperparameters for better r2 values
