##Vectorized operations
using BenchmarkTools
x = 1:0.01:1000

function f(x::T) where T<:AbstractFloat
    return x^2 
end


f(x) ## will not work

##Below will work 
@btime f.(x)

@btime map(f, x)

## How about vectors?
t = randn(100, 100)
## We want to square every element of x
## one way
@benchmark t.*t

@. f(rand(10,10))

@benchmark f.(t)

function f(x::Matrix{Float64})
    return x.*x
end

@benchmark f(t)



### Some further tips and tricks
## Assume that you have the following function
function g(f::Function, x::Number)
    return f(x) + x
end

g(2) do x
    return x^2
end

### Local scopes
x = begin w = 2
    q = 3
    q+w
end


## No pollution to outside!!!
m = let 
    l = 4
    g = 2
    g+l
end

l
g
    