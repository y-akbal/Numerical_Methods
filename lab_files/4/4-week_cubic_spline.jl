using Interpolations, Plots


x = 1.0:2:20.0
y = @. cos(x^2 / 0.5)  +  4*sin(x/100)

interp_cubic = cubic_spline_interpolation(x, y, extrapolation_bc= Reflect())
f_cubic(x) = interp_cubic(x)

interp_linear = linear_interpolation(x,y)
f_linear(x) = interp_linear(x)


scatter(x, y, markersize=5, label="Data points", color = "red")
plot!(f_linear, x,w=1, linestyle=:dash,label="linear_interpolation")
plot!(f_cubic, 1.0:0.1:20.0, w=2, label="cubic_spline_interpolation")
plot!(size = (800, 600))

## How about extrapolation??
