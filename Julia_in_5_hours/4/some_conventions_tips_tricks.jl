##Vectorized operations
using BenchmarkTools
x = 1:0.01:1000

function f(x::T) where T<:AbstractFloat
    return x^2 
end


f(x) ## will not work

##Below will work 
@time f.(x)

@time map(f, x)

## How about vectors?
t = randn(100, 100)
## We want to square every element of x
## one way
@benchmark t.*t

@benchmark @. f(t)

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
x = begin y = 2
    t = 3
    y+t
end

## No pollution to outside!!!
m = let 
    q = 2
    j = 2
    q+j
end


    