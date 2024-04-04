using DataFrames
using ScikitLearn, Random
using ScikitLearn: fit!, predict


## The real dude is here!!!! Check out!!!
@sk_import linear_model: LogisticRegression
log_reg = fit!(LogisticRegression(), X_train, y_train);
mean(predict(log_reg, X_test) .== y_test)