
### Concat strings
### replace strings

a::String = "Hoppa"
b::String = "Zoppa"
a*b == "HoppaZoppa"  ## concat is done by *
replace(a, "H"=>"T")


## string formatting
i::Int = 10
str_ = "Do it $(i) times mate!!!"
println(str_)

for i in 1:10
    println("Yep yep yep $i")
end