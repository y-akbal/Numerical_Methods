## Try except blocks
## Error types
## Finally blocks

## Simplest part of the course
const x::Int64 = 2
## No y defined only x
try
    println(y)
catch Exception
    println(Exception)
    throw(DomainError(x, "yo yo "))
    ## For the error lists see
    ##  https://www.geeksforgeeks.org/exception-handling-in-julia/
finally
    println("YYep yep ye")
end

## Another example
f(x) = x > 0 ? true : error("Number should be positive mate!!!")
f(-1) ### This will throw an error!!!
f(2) ### Nothing's wrong here!!!