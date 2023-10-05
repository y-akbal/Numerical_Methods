
### Structs
### Mutable Structs
### kwdef structs

##What is a struct?
# You can think of it as a container for different types...
struct A
    a ##here a, and b can be anything 
    b ## b is a called a property here.
end

#Let's create an instance 
l = A(2,3)
l.a
## To get property names!!!
propertynames(l)
## Once you create a struct you can not alter it.
## because Immutable (located in -- fast) 
l.b = 2


mutable struct B
    a::Int
    b::Float32
end

b_ = B(2, randn())
b__ = B(2, randn(Float32))

B() ## No matching method, throws an error, to avoid this we have two options!!! 

@kwdef mutable struct C
    a::Integer = 5
    b::AbstractFloat = randn()
end
C()  ##we keyword def

## There is another way of doing it
using Parameters
@with_kw mutable struct D
    a::Function = x->x^2
    t::Matrix{T} where T<:AbstractFloat = randn(10,10) 
end

## We shall use structs a lot!!!