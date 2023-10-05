## Julia's paradigm is Multiple dispatch
#= We can overload functions by means of changing their roughly 
=#
# However we can turn structs into functions as well (make them callable things)
import Base:getindex, setindex!

@kwdef struct func
    t::Float64 = randn()
end

f_ = func()

function (f_::func)(x::Real)
    return f_.t + x
end

@show f_(0)

function getindex(f_::func, i::Int64)
    return f_.t 
end

f_[1]

getindex(a, 2)


propertynames(f_) ## we have only t

