struct Dual{T <: Real}
    v::T
    ∂::T
    end


Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.∂ + b.∂)
Base.:-(a::Dual, b::Dual) = Dual(a.v - b.v, a.∂ - b.∂)
Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.v*b.∂ + b.v*a.∂)
Base.:*(c::Real, b::Dual) = Dual(c*b.v, c*b.∂) 
#Base.:/(a::Dual, b::Dual) = Dual()
#Base.sin
Base.exp(a::Dual) = exp(a.v)*(Dual(1,a.∂))

#Base.sqrt
Base.log(a::Dual) = Dual(log(a.v), a.∂/a.v)


function Base.:^(a::Dual, i::Int)
    if i == 0
        return 1
    elseif  i == 1
        return a
    else
    return a.v^i +ia.v^(i-1)a.∂
    end
end

function Base.max(a::Dual, b::Dual)
    v = max(a.v, b.v)
    ∂ = a.v > b.v ? a.∂ : a.v < b.v ? b.∂ : NaN
    return Dual(v, ∂)
end

function Base.max(a::Dual, b::Int)
    v = max(a.v, b)
    ∂ = a.v > b ? a.∂ : a.v < b ? 0 : NaN
    return Dual(v, ∂)
end

x = Dual(1.23,1.0)
f(x) = x^10+x^70
f(x).∂
f_(x) = 10x^9+70*x^69
(f(1.23+1e-5) - f(1.23))/1e-5
f_(1.23)
f(x)

### Another example
f(x) = x*exp(x)
(f(3+1e-2) -f(3))/1e-2
f_(x) = (x+1)*exp(x)
f_(3)
f(Dual(3,1))
### end of another example


### Let's show off a bit!!!
Base.show(io::IO, x::Dual) = print(io, "$(x.v)+"*"$(x.∂)"*"ϵ")
(Dual(2,2)+Dual(3,4))^2-Dual(1,2)


@__DIR__