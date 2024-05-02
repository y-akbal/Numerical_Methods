using ScikitLearn
using Printf
using Statistics
using ScikitLearn: fit!, predict


## Here we prepare the dataset
include("10-tools.jl")
X_train, X_test, y_train, y_test = return_dataset()
## --- end of preparation  --- ### 
@sk_import svm: LinearSVC
svm = LinearSVC(C = 10, max_iter = 10000, loss = "hinge")
ScikitLearn.fit!(svm, X_train, (y_train.+1)/2 .|> Int)
(predict(svm, X_test) .== (y_test.+1)/2 .|> Int) |> mean