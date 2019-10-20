__precompile__()
module models

using ..gpflow
import ..gpflow: compile!, predict_f, predict_f_samples
export 
GPR,
compile!,
predict_f,
predict_f_samples

abstract type GPModel <: Model end

function predict_f(m::GPModel, Xnew)
    return m.o.predict_f(Xnew)
end
function predict_f_samples(m::GPModel, Xnew, num_samples)
    return m.o.predict_f_samples(Xnew, num_samples)
end

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
    out = GPR(X, Y, kern, mean_function, name, nothing, nothing)
    compile!(out)
    out
end

function compile!(o::Union{GPR,Nothing})
    if o === nothing return nothing end
    kern_ = compile!(o.kern)
    mean_function_ = nothing # TODO: Update this once the mean_functions module is ready. compile!(o.mean_function)
    o.o = py_gpflow.models.GPR(o.X, o.Y, kern_, mean_function_, o.name)
    o.likelihood = likelihoods.Gaussian()
    o.likelihood.o = o.o.likelihood
    return o.o
end

end