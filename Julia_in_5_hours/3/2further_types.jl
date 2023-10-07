
## Floats, Subtypes, SuperTypes, AbstractTypes, primitive types. Julia has dynamic types

## In Julia, unlike in Python, types matter a lot
(1+2)::AbstractFloat ## This will thrown an exception


(1.0+2.0)::AbstractFloat ## This will not thrown an exception


(1+2.0)::AbstractFloat ## This will not thrown an exception

## See here for types https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/lecture_06/compositetypes/

x = 2
typeof(x)
t = Float32(x)


abstract type abstract <:AbstractFloat end  ## This dude is a sub type of AbstractFloat

supertype(Float32)
supertypes(Float32)

#= For instance we have the following:

abstract type Number end
abstract type Real          <: Number end
abstract type AbstractFloat <: Real end
abstract type Integer       <: Real end
abstract type Signed        <: Integer end
abstract type Unsigned      <: Integer end
=#

randn(32,32) ## Matrix{Float64}

randn(32) ## Vector{Float64}
[1,2,3] ##  Vector{Int64}

### Type changing 
#Float32 -> Float64
typeof(Float32(randn()))


isa(1, Integer) 