using ForwardDiff, Zygote
using BenchmarkTools


"""
Numerical Derivative Part
"""
#### One sided derivatives

function one_sided_p(f::Function, x::T; h::Float64 = 1e-10) where T <: Real
        return (f(x+h) - f(x))/h
end
function one_sided_n(f::Function, x::T; h::Float64 = 1e-10) where T <: Real
    return -(f(x-h) - f(x))/h        
end


function double_sided(f::Function, x::T; h::Float64 = 1e-10) where T <: Real
    return (f(x+h) - f(x-h))/(2h)
end

f(x) = (x^3)*sin(x) + (x^10)*cos(x)

double_sided(f, 5)
one_sided_p(f, 5)
one_sided_n(f,5)

ForwardDiff.derivative(f, 5.0)  #### actual values (the one as close as possible)




function local_ext_search(f::Function;n_iterations = 100::Int64)
    x = randn()
    f_d(x) = ForwardDiff.derivative(t->f(t),x)
    f_dd(x) = ForwardDiff.derivative(t -> f_d(t), x)
    for _ in 1:n_iterations
        x -= (f_d(x)+1e-100)/(f_dd(x)+1e-10)   ####Syntactic sugar :)
    end
    return x
end

f(x) = sin(x)/(1+x^2)

local_ext_search(f;n_iterations = 10000000)


