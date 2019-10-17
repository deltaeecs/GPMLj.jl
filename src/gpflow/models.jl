__precompile__()
module models

using ..gpflow

export 
GPR,
compile!

abstract type GPModel <: Model end


mutable struct GPR <: GPModel
    X
    Y
    kern::Union{Kernel,Nothing}
    mean_function::Union{MeanFunction,Nothing}
    name::Union{String,Nothing}
    likelihood::Union{Likelihood,Nothing}
    o::Union{PyObject,Nothing}
end

function GPR(X, Y, kern::Kernel; mean_function::Union{MeanFunction,Nothing}=nothing, name::Union{String,Nothing}=nothing)
    GPR(X, Y, kern, mean_function, name, nothing)
end

function compile!(o::Union{GPR,Nothing})
    if o === nothing return nothing end
    kern_ = compile!(o.kern)
    mean_function_ = nothing #compile!(o.mean_function)
    # likelihood = compile!(o.likelihood)
    o.o = py_gpflow.models.GPR(o.X, o.Y, kern_, mean_function_, o.name)
    o.likelihood = likelihoods.Gaussian()
    o.likelihood.o = o.o.likelihood
    return o.o
end

end