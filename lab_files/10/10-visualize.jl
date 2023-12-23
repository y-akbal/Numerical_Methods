using Plots
using Distributions
using Random
import Base: split


dist = Distributions.Uniform(-4,4)
sample = rand(dist, (450,2))

scatter(sample[:, 1], sample[:, 2], label = "Standard_sample")
xlims!((-5,5))
ylims!((-5,5))


function split(x::AbstractVector; ϵ = 0.9)
    t = copy(x)
    if abs(x[1] - x[2]) < ϵ
            t[1] = 1
            t[2] = 0
    end
    return t
end
   


function split(x::AbstractMatrix; ϵ = 1e-10)
    m,t = x|>size
    X = ones(m,t)
    for i in 1:m
        X[i, :] = split(x[i,:])
    end

    return X
end        

sample_ = split(5*rand(1000,2).-1/2)
index_ = sample_[:,1] .> sample_[:,2] 
index__ = 1 .- index_ |> BitArray
scatter(sample_[index_, 1], sample_[index_, 2], label = "Blue Balls", color = :blue)
scatter!(sample_[index__, 1], sample_[index__, 2], label = "Orange Balls", color = :orange)
xlims!((-5,5))
ylims!((-5,5))
