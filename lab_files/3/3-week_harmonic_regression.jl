using DataFrames
using CSV
using Plots
using Plots
using Statistics


#= First Let's work with fake data =# 
x = 0:0.1:20
f = x-> (1+cos(2*pi*x/3)-3*sin(2*pi*x/5)) + randn()
r = map(f, x);
plot(x, r)


function create_x(r::Vector{Float64}, interval::Vector{Float64}; period::Int64 = 10, n::Int64 = 1)
    cos_::Vector{Function} = [x->cos((2*pi*i*x)/period) for i in 1:n]
    sin_::Vector{Function} = [x->sin((2*pi*i*x)/period) for i in 1:n]
    size_r::Int = size(r)[1]
    array::Matrix{Float64} = ones(size_r, 2*n+1)
    for i in 1:n
        array[:,i] = cos_[i].(interval)
        array[:,i+n] = sin_[i].(interval)
    end
    return array
end

A = create_x(r, collect(x); period = 15, n = 16);


struct harmonic_pol
    period::Float64
    coeff::Vector{Float64}
end 

pol = harmonic_pol(15, A\r)



function (pol::harmonic_pol)(x::Real)
    n = size(pol.coeff)[1]
    half::Int64 = Int((n-1)/2)
    period = pol.period
    t::Float64 = 0.0
    for (i,coeff_) in enumerate(pol.coeff)
        if i <= half
            t += coeff_*cos(2*pi*i*x/period)
        elseif half<i<n
            t += coeff_*sin(2*pi*(i-half)*x/period)
        else
            t += coeff_
        end
    end
    return t
end

##  
plot(0:0.1:30, pol.(0:0.1:30), label = "Approximated")
plot!(x, r, label = "real_data")


cd(@__DIR__)
csv = CSV.read("solar_power_gen.csv", DataFrame)
l = groupby(csv, "DATE_TIME") 
l = combine(l, "DC_POWER" .=> mean) 

dataset = l[:, 2]

A = create_x(dataset, collect(1:3259) .|> Float64; period = 100, n = 4000);
pol = harmonic_pol(100, A\dataset)

plot(1:1500, pol.(1:1500), label = "Approximated")
plot!(1:1500, dataset[1:1500], label = "real_data")