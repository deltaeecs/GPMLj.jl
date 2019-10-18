__precompile__()
module gpflow
using GPJ, PyCall
import ..GPJ: compile!
export  
py_gpflow, 
compile!,
kernels,
models,
likelihoods,
Model,
Kernel,
Likelihood,
MeanFunction,
ParameterPrior,
PyObject

abstract type Model end
abstract type Kernel end
abstract type Likelihood end
abstract type MeanFunction end
abstract type ParameterPrior end

py_gpflow=nothing;
function __init__()
global py_gpflow = pyimport("gpflow")
end


function compile!(o::Union{Model,Kernel,Likelihood,MeanFunction,ParameterPrior}) end

include("gpflow/models.jl")
using .models: compile!
include("gpflow/kernels.jl")
using .kernels: compile!
include("gpflow/likelihoods.jl")
using .likelihoods: compile!
include("gpflow/mean_functions.jl")
include("gpflow/parameter_priors.jl")

end