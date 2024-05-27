### Conv2d
### Con1d 
### batchnorm
### Residual
### Global average
### Output_dense
### Dropout

using Flux, CUDA
## using Wandb
using StatsBase
using cuDNN
using NNlib
using MLUtils
using MLDatasets
using ProgressBars

CUDA.devices()
CUDA.device!(1)

function mixer(embedding_dim::Int64 = 512, kernel_size::Int64 = 4)
    first_half = Chain(Conv((kernel_size,kernel_size), embedding_dim=>embedding_dim, groups = embedding_dim; pad = SamePad()), gelu, BatchNorm(embedding_dim))
    first_half_ = SkipConnection(first_half, +)            
    second_half_ = Chain(Conv((1,1), embedding_dim=>embedding_dim), gelu, BatchNorm(embedding_dim))
    return Chain(first_half_, second_half_)
end


function classifier_head(output_dim::Int64 = 10, embedding_dim::Int64 = 512)
    return Chain(AdaptiveMeanPool((1,1,)), Flux.flatten, Dense(embedding_dim, output_dim))
end


function network(embedding_dim::Int64 = 768, 
    patch_size::Int64 = 8, 
    kernel_size::Int64 = 7, 
    depth::Int64 = 16, 
    num_class::Int64 = 10)
    ## 
    patcher = Chain(Conv((patch_size, patch_size), 3=>embedding_dim, stride = patch_size), gelu, BatchNorm(embedding_dim))
    blocks = Chain([mixer(embedding_dim, kernel_size) for i in 1:depth])
    classifier_head_ = classifier_head(num_class, embedding_dim)
    return Chain(patcher, blocks, classifier_head_)
end

function return_dataset(batch_size = 32)
    X_train, y_train = CIFAR100(:train)[:]
    y_train = y_train[:fine]
    X_test, y_test = CIFAR100(:test)[:]
    y_test = y_test[:fine]

    mean_, std_ = mean(X_train, dims =[1,2, 4]), std(X_train, dims =[1,2, 4])

    X_train = @. (X_train - mean_)/std_
    X_test = @. (X_test - mean_)/std_
    ## 
    train_data = DataLoader((X_train, y_train); batchsize = batch_size, buffer = true, parallel = true, shuffle = true)
    test_data = DataLoader((X_test, y_test); batchsize = batch_size, buffer = true, parallel = true, shuffle = false)
    return train_data, test_data
end


### training loop
## 0) Fix optimizer and stuff --- weight decay, GradientClipping, Learning Scheduler, 
## 1) fix loss --- LogitCrossEntropy  --- onehotbatch, label smoothing
## 2) find gradients
## 3) Let the optimizer take a step

function train(model, train_data, test_data; epoch::Int64 = 5)
    opt = Flux.Optimise.Momentum(0.01, 0.95)
    optimizer = Flux.Optimiser(ClipValue(5.0), WeightDecay(0.004), opt) |> gpu
    
    model = model |> gpu
    opt_state = Flux.setup(optimizer, model)

    temp_acc = 0
    temp_counter = 0



    for epoch_ in 1:epoch


        
        i::Int64 = 0
        for (i, (x,y)) in enumerate(train_data)
            y = Flux.onehotbatch(y, 0:99)
            x, y = map(gpu, [x,y])
            val, grads = Flux.withgradient(model) do m
                result = m(x)
                NNlib.logsoftmax()(result, y)
            end
        
            Flux.update!(opt_state, model, grads[1])
        end

        if i % 1 == 0
            println("Epoch $i passed!!!")
        end
        @info "Validation started"
        for (i, (x,y)) in enumerate(test_data)

            x, y = map(gpu, [x,y])
            temp_acc += sum(Flux.onecold(model(x)) .== y)
            temp_counter += size(x)[4]
        end
        @info "your accuracy is $(temp_acc/temp_counter)" 
    end

end
    


model = network(256, 2, 7, 32, 100)
train_data, test_data = return_dataset()

train(model, train_data, test_data)
