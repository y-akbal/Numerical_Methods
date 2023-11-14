using Zygote

f(x) = x[1]^2*x[2] + x[2]^2*x[1] + x[2]^2*x[1]^2

x = [1.0, 2.0] 

Zygote.hessian(x->f(x),x)