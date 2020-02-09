module gpflow

using GPnet, PyCall
import ..GPnet: instantiate!, minimize!, predict_f, predict_f_samples, GPFlowObject
export  
    py_gpflow, 
    instantiate!,
    minimize!,
    predict_f,
    predict_f_samples,
    kernels,
    models,
    likelihoods,
    Model,
    Kernel,
    AbstractLikelihood,
    AbstractMeanFunction,
    Prior,
    Optimizer,
    PyObject,
    GPFlowObject

abstract type Model <: GPFlowObject end
abstract type Kernel <: GPFlowObject end
abstract type AbstractLikelihood <: GPFlowObject end
abstract type AbstractMeanFunction <: GPFlowObject end
abstract type Prior <: GPFlowObject end
abstract type Optimizer <: GPFlowObject end

py_gpflow= nothing;
function __init__()
    global py_gpflow = pyimport("gpflow")
end


function instantiate!(o::Union{Model,Kernel,AbstractLikelihood,AbstractMeanFunction,Prior}) end
function minimize!(opt::Optimizer, m::Model) end

function predict_f(m::Model, Xnew) end
function predict_f_samples(m::Model, Xnew, num_samples) end


include("gpflow/models.jl")
using .models: predict_f, predict_f_samples

include("gpflow/kernels.jl")

include("gpflow/likelihoods.jl")

include("gpflow/train.jl")
using .train: minimize!

include("gpflow/mean_functions.jl")

include("gpflow/parameter_priors.jl")

end
