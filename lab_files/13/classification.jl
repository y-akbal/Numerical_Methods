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
import Flux: onecold, onehot, onehotbatch
## Google boston housing dataset
function return_data_set()
    train_x, train_y = MNIST(:train)[:][1], MNIST(:train)[:][2]
    test_x, test_y = MNIST(:test)[:][1], MNIST(:test)[:][2]
    ## We normalize the outputs so that the network converges quickly!!!1
    return train_x/255.0 .|> Float32, train_y, test_x/255.0 .|> Float32, test_y
end
x
Network =  let 
    ## Here we fix the seed for reproduciblity!!!!
    ## Google dropout layers
    Random.seed!(0)
    Chain(Flux.flatten,
        Dense(28*28, 128, relu),  Dropout(0.1),
        Dense(128, 10))
end

opt_state = Flux.setup(Flux.Optimise.Adam(0.01), Network) ## Optimizer state --> we used SGD with momentum

function return_training_data()
    X_train, y_train, X_test, y_test =  return_data_set()   
    train_data = DataLoader((X_train, y_train), batchsize = 64, shuffle = true)
    test_data = DataLoader((X_test, y_test), batchsize = 64, shuffle = false)
    return train_data, test_data
end


function train_network!(Network,epochs::Int = 10)
    for i in 1:epochs
        temp_train_loss::Float32 = 0.f0
        counter::Int = 0

        trainmode!(Network) ##Dropout layers active when in train mode
        for (x,y) in train_data ##grab the data from dataloader
            y_onehot = Flux.onehotbatch(y, 0:9)
            val, grads = Zygote.withgradient(Network) do Network
                y_hat = Network(x)
                Flux.Losses.logitcrossentropy(y_hat, y_onehot)
            end 
            Flux.update!(opt_state, Network, grads[1]) ##update the weights
            temp_train_loss += val
            counter += 1 #increment counter
        end

        temp_val_acc::Float32 = 0.f0
        total_batch_size::Int = 0
        testmode!(Network) ##Dropout layers active when in train mode

        for (x,y) in test_data ##grab the data from dataloader
                y_pred = onecold(Network(x), collect(0:9))
                temp_val_acc += sum(1*(y_pred .== y))
                total_batch_size += size(y)[1]  #increment counter
        end
    @info "$i epoch passed the loss is $(temp_train_loss/counter), test acc is $(temp_val_acc/total_batch_size)"
    end

end

## Watcha agi is coming for you!!!
train_network!(Network)
