
### functions with different types and how they behave
##Unresonable effectiveness of multiple dispatch
## The same as function overloading but a bit more
## any object can be turned into a function such as


function d(x::T) where T<:Real
    return 2*x
end

d(10)

function d(x::Matrix{T}) where T<:Float32 
    return x*2+x
end

d(ones((2,2)))


