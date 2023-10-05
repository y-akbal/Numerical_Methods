using LinearAlgebra ## This library is pretty reach!!!

A = randn(10, 10) 
B = randn(10,10)
C = A*B  ## this is starndard matrix multiplication
C_ = A.*B ### this is componentwise matrix multiplication

A'
##Transpose of a matrix
#= Assume that you want to solve the system Ax = b, just hit A\b (least squares solution may not be exact) =#
##Al you need to do
A = randn(10,10)
y = randn(10,1) 
q = A\y ## this is the same as A^{-1}*y, sometimes A may not have an inverse
A*q
