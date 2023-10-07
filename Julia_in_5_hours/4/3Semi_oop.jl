## Julia's paradigm is Multiple dispatch
#= We can overload functions by means of changing their roughly 
=#
# However we can turn structs into functions as well (make them callable things)
import Base:getindex, setindex!

a = [1,2,3]
getindex(a, 2)
a[2]
@kwdef struct func
    t::Float64 = randn()
end

f_ = func()

function (f_::func)(x::Real)
    return f_.t + x
end

@show f_(0)

function getindex(f_::func, i::Int64)
    return f_.t + i 
end

f_[4.0]

getindex(a, 2)


propertynames(f_) ## we have only t

