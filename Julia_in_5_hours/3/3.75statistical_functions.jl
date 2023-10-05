## cor 
## mean
## std

using Statistics

### let's pick a random array
const a = randn(100, 100)
@show names(Statistics)

cor(randn(10), randn(10)) ## 
mean(a, dims = 1) ##
mean(a, dims = 2)
std(a, dims = 1) ##
std(a, dims = 2)
##write ?std to see what happens
