__precompile__()
module likelihoods

using ..gpflow
import ..gpflow: compile!
export 
Gaussian,
compile!

mutable struct Gaussian<:Likelihood
    var::Real
    o::Union{PyObject,Nothing}
end

function Gaussian(;var::Real=1.0)
    Gaussian(var, nothing)
end

function compile!(o::Union{Gaussian,Nothing})
    if typeof(o.o)<:PyObject return o.o end
    if o === nothing return nothing end
    o.o = py_gpflow.likelihoods.Gaussian(var=o.var)
end

end