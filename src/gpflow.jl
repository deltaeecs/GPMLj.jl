__precompile__()
module gpflow
using GPJ, PyCall

export GPR

abstract type Model end
abstract type Kernel end
abstract type Likelihood end
abstract type MeanFunction end
abstract type ParameterPrior end


py_gpflow = pyimport("gpflow")


include("gpflow/models.jl")
include("gpflow/kernels.jl")
include("gpflow/likelihoods.jl")
include("gpflow/mean_functions.jl")
include("gpflow/parameter_priors.jl")


end