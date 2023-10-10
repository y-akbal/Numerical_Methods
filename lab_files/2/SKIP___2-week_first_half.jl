using Printf



function root_find(f::Function, a::Float64, b::Float64, ϵ= 1e-5::Float64)
    @assert f(a)*f(b) < 0  ### we assert here whether f has at least one zero in (a,b)
    c::Float64 = 0.0  ### because of the last return c, without this an error is thrown.
    while abs(b-a) > ϵ
        c = (a+b)/2
        if f(c) == 0
            println("Hit it for the first time mate!!!!")
            return c
        elseif f(a)*f(c) < 0
            b = c
        elseif f(c)*f(b) < 0
            a = c
        end
    end
    return c
end

a = root_find(x -> exp(x)-1, -1, 1, 1e-10)
@printf("%0.5f", a) 

f(x) = x^2 -2
t = root_find(f, 0, 2, 1e-10)
@printf("%0.5f", a) 


using ForwardDiff 
"""
For the sake of brevity we use automatic differentiation library, otherwise we gotta evaluate the derivative on our own
"""

function newton_raphson(f::Function; n_its::Int = 100, initial_point::Union{Float64, String} = "random")
    if initial_point == "random"
        x::Float64 = randn() # initialize a random point
    else
        x = initial_point
    end
    for _ in 1:n_its
        x = x - f(x)/ForwardDiff.derivative(f,x)
    end
    return x
end


a = newton_raphson(x -> x^2 - 2;  initial_point = "random")  ###this macro allows one to see time to run the function
println("You hit the right spot mate, here it is $a !!!")
@printf("%s:%lf", "you hit the right spot mate", a)


function secant_method(f::Function; n_its::Int64 = 100,  initial_point::Union{Float64, String} = "random")
    if initial_point == "random"
        x::Float64 = randn()
    else
        x = initial_point
    end
    x_ = x + 0.05*randn() 
    for _ in 1:n_its
        x,x_ = x - (x-x_)*f(x)/(f(x)- f(x_)+1e-10), x   ####The correction factor dude is introduced here
    end
    return x
end

a = secant_method(x->sin(x); n_its = 1000, initial_point = 3.14)
println("You hit the right spot mate on using secand method, here it is $a !!!")

