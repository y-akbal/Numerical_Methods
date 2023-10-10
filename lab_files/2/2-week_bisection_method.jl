### Given a function $f$, you can approximately find its derivatives as follows
function derivative(f::Function, x::Real; h::Float64 = 1e-10)
    return (f(x+h) - f(x))/(h)
end


## You know write bisection method in Julia