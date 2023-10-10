
### read file
### open file
### write file
### pickle something to a file


pwd() ## current directory

cd() ## change the directory

readdir() ## See what is insidergRead

a = pwd() ##reads the current directory

path_needed = joinpath(pwd(), "Tedu_Numerical_Analysis", "Julia_in_6_hours")
cd(path_needed)  


##walkdir(dir; topdown=true, follow_symlinks=false, onerror=throw)
for l in walkdir(".")
    println(l)
end
## 
## Split the directory splitdir(pwd())

## Let's create a strings
a = "This is a string -- I an going to write it in a file!!!!\n"
for i in 1:10
    a *= "$(i)"*a
end


file = open("testo_.txt", "w")
write(file, a)
close(file)

q = open("testo_.txt", "r") do l
    readlines(l)
end;

# opening the file wrapped in do block
open("hoppa.txt") do file
    # do something with file
end
