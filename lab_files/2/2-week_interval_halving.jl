### 
## Assume that we have some unimodal function
using Plots
function f(x::Float64)
    return x^4+x^2+x
end

begin
t = -5:0.1:5
y = map(f, t)
plot(t, y, xlims = (-5, 5), ylims = (-10, 10))
title!("A simple dude!")
xlabel!("x")
ylabel!("y")
end

function find_minimum(f::Function, a::Real, b::Real; max_iter::Int = 100, ϵ::Float64 = 1e-3)
    α::Float64 = (a+b)/2
    for k in 1:max_iter
        L = b - a
        x_1, x_2 = a + L/4, b - L/4
        f_1, f_2, f_α = f(x_1), f(x_2), f(α)
        if f_1 < f_α
            b, α, f_α = α, x_1, f_1
        elseif f_2 < f_α
                a, α, f_α = α, x_2, f_2
        else
                a, b = x_1, x_2
        end
        
        if abs(L) < ϵ
            println("Iteration Stopped in $(k) steps")
            return α, f(α)
        end

    end
    @info "Algorithm did not converge correctly!!!"
    return α, f(α)
end

find_minimum(x -> x^5-x, -1, 1; ϵ = 1e-100)





