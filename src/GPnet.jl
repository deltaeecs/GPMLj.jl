module GPnet

using PyCall, Distances, Random
import Base.Broadcast: broadcast_shape
using LinearAlgebra

export
    gpflow,
    instantiate!,
    minimize!,
    predict_f,
    predict_f_samples,
    GPFlowObject


abstract type GPFlowObject end

function instantiate!(o) end
function instantiate!(o::Nothing) return nothing end
function minimize!(opt, m) end
function predict_f(m, Xnew) end
function predict_f_samples(m, Xnew, num_samples) end

function elementwise end

const pw = pairwise
const ew = elementwise

const AV{T} = AbstractVector{T}
const AM{T} = AbstractMatrix{T}
const AVM{T} = AbstractVecOrMat{T}

# Load necessary utilities
include(joinpath("utils", "zygote_rules.jl"))
include(joinpath("utils", "abstract_data_set.jl"))
include(joinpath("utils", "distances.jl"))


include("kernels.jl")

# Load the GPFlow Interface
include("gpflow.jl")


end # module
