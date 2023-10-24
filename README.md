# Numerical Analysis with Julia
This course is offered at TED University, and is mainly about optimization methods used in data science from practical point of view. We do not promise a complete optimization course, rather we shall talk abour practical applications of optimization methods. The main goal of this course is to find out what happens under the hood. Things are done from scratch as much as possible (apart from automatic differentiation). As I pace, I will upload the documents. 
The programming language used for labs is particularly chosen Julia, as it rocks for numerical computations. Please see Julia_in_5_Hours directory for a refresher.  

## Curriculum
- Get started with Julia (Easy peasy introduction, This will take may be up to 3 weeks including going over simple old school numerical analysis implementations),
- Curve Fitting - Polynomial Regression - Harmonic Regression,
- Numerical Differentiation vs Automatic Differentiation,
- Gradient Descent (Application: Linear and Logistic Regression from scratch),
- Steepest Descent,
- Second Order Methods,
- Conjugate Direction Methods,
- Constrained Optimization: Duality, Projection Methods, Penalty Methods
- Implementation of Support Vector Machines From Scratch,
- Some First Order Methods in Deep Learning,
- Applications to Neural Networks (with Flux),

````julia
using Flux, Zygote, ForwardDiff
````
