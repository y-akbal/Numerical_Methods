using Plots
## See below  for more info
# https://docs.juliaplots.org/latest/
### Plots
### Histogram
### Surface Plots
### Contour heatmap

## Assume that we have some function
q = x->x^2
x = -1:.001:1

plot(x, map(q, x), label = "real_q")
plot!(x, map(x->x^3, x), label = "squared_q")

### Sketch some histograms!!!
R_1 = randn(10000)
R_2 = randn(10000) .+ 1
histogram(R_1, label = "normal_mean_0", bins = 100, density = true)
histogram!(R_2, lanel = "normal_mean_1", bins = 100, densiry = true)
 

### Sketch some contour pilots
## see here https://docs.juliaplots.org/latest/series_types/contour/
x = -1:0.01:1
y = -1:0.01:1

function f(x,y)
    return x^2 + 3*y^2
end

z = @. f(x',y)
contour(x,y, z)
## UU vii 
contour(x,y,z, fill = true)
## OHH maaan
using LaTeXStrings
contourf(x, y, z, levels=20, color=:turbo)
title!(L"x^2 +y^2")
xlabel!(L"x")
ylabel!(L"y")

##Let's sketch some surfaces

xs = collect(0.1:0.05: 2.0)
ys = collect(0.2:0.1:2.0)
X = [x for x in xs for _ in ys]
Y = [y for _ in xs for y in ys]
Z = ((x, y)->begin
            1 / x + y * x ^ 2
        end)
surface(X, Y, Z.(X, Y), xlabel = "longer xlabel", ylabel = "longer ylabel", zlabel = "longer zlabel")

## one more example
X = collect(-2:0.1:2)
Y = collect(-2:0.1:2)
Z = x-> x[1]^2+x[2]^2
surface(X, Y, Iterators.product(X,Y) |> collect .|> Z)

