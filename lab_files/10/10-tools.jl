using CSV
using DataFrames
using Random

cd(@__DIR__)

function one_hot(v)
    unique_labels = unique(v)
    dict::Dict = Dict(label => int_ - 1 for (int_, label) in enumerate(unique_labels))
    inverse_dict::Dict = Dict(int_ - 1 => label for (int_, label) in enumerate(unique_labels))
    vec::Vector{Int64} = ones(Int64, size(v)[1])
    for (i,tag) in enumerate(v)
        vec[i] = dict[tag]
    end
    return dict, inverse_dict, vec
end

function split_data(X::AbstractMatrix,y::AbstractVector; split_ratio::Float64 = 0.8, shuffle = true)
    ## We use this dude to split the data into train and test
    sample_size_x, _ = size(X)
    sample_size_y = size(y)[1]
    y = 2*y .- 1
    @assert sample_size_x == sample_size_y "The vectors should be of the same size!!! one has $(sample_size_x) while the other $(sample_size_y)"
    indexes = collect(1:sample_size_x)
    if shuffle
        shuffle!(indexes)
    end
    N = split_ratio*sample_size_x |>floor |> Int
    train_index = indexes[1:N]
    test_index = indexes[N:end]
    X_train, X_test = X[train_index, :], X[test_index,:]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test
end




function return_dataset()
    csv = CSV.read("breast-cancer.csv", DataFrame)
    dict, inverse_dict, y = csv[:, 2] |> one_hot 
    X = csv[:, 3:end] |> Matrix{Float64}
    y = y .|> Float64
    @info "The dataset is ready one hot is here $(dict)"
    return split_data(X, y)
end