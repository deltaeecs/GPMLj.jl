export GPR

abstract type GPModel <: Model end


py_GPR = py_gpflow.models.GPR
mutable struct GPR{T1,T2} <: GPModel
    likelihood::Likelihood
    X::T1
    Y::T2
    num_latent::Int
    PyGPR::PyObject
end


function GPR(X, Y, kern, mean_function=None, name=None)
end

