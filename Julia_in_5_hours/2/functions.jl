##Julia functions
##There are two ways to define functions
## 1

function g(x)
    ## do something Here
    return x^2 
end


g(2)

g([[1,2] [3,4]])

## 2
f(x) = x^2 ##like lambda functions in Python


## 3
f = x -> x^2 ### generic function (there is a difference between)

## you can fix the types if you like for instance, no need always but good habit to have it!!!!
function g(x::Int64)::Int64
    return x^4
end

function g(x::Float64)::Float64
    return x^2
end

function Q(x::Number)
    return x^2
end

Q(randn(10,10))
Q(2)
Q(2.f0) ## These dudes are different!''

f(x) = x^2
l(x) = x+10
t(x) = x^4

f(l(t(2)))

##instead we have 
2 |> t |> l |> f

##Functions with optional args
function f(x; y = 2)
    return x+y
end

## functions with kwargs
function m(l; kwargs::Dict)
    return kwargs[:t]
    
 end
m(2;)
Dict("t"=>2)