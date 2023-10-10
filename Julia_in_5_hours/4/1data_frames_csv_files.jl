

### read csv
### write csv
### dataframes

using CSV
using DataFrames



@show names(DataFrames)
A = randn(10, 3)
df = DataFrame(A, ["a", "b", "c"])


@show names(CSV, all = true)

## read csv files !!
csv = CSV.read("test.csv", DataFrame)
df = DataFrame(csv)

df[:, 1] = randn(10) ### change the first column
## write csv files!!
CSV.write("test.csv", df)    
## Tonny montana style okkay!!!
##

