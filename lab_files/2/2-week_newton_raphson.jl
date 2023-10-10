
## The below dude is for 
function derivative(f::Function, x; h::Float64 = 1e-4)
    return (f(x+h) -f(x-h))/(2h)
end


function newton_raphson(f::Function; x_0::Union{String, T} = "random", 
                        max_iter::Int64 = 100, stop_criteria::Float64 = 1e-4) where T<:Real
    if x_0 == "random"
        x_0 = randn()
    end
    for i in 1:max_iter
        second_deriv = derivative(t->derivative(f, t), x_0)
        first_deriv = derivative(f, x_0)
        x_0 -= first_deriv/(second_deriv)
    if abs(derivative(f, x_0)) < stop_criteria
        if second_deriv < 0
            println("It took $(i) iterations to reach the goal!!, we hitted a local maximum!!!")
        elseif second_deriv > 0
            println("It took $(i) iterations to reach the goal!!, we hitted a local minimum!!!")
        end
        return  x_0, f(x_0), first_deriv, second_deriv 
    end
        
    end
    println("Algorithm did not convertege in the given time period!!!")
end





newton_raphson(x->sin(x)*x^3+3*exp(-x^2); max_iter = 10000, stop_criteria = 1e-10)
