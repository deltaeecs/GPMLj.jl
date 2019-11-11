module likelihoods

using ..gpflow
import ..gpflow: instantiate!
export 
    Bernoulli,
    Beta,
    Exponential,
    Gamma,
    Gaussian,
    Likelihood,
    MonteCarloLikelihood,
    MultiClass,
    Ordinal,
    Poisson,
    RobustMax,
    SoftMax,
    StudentT,
    SwitchedLikelihood,
    instantiate!

mutable struct Bernoulli<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    o::Union{PyObject,Nothing}
end

function Bernoulli()
    out = Bernoulli(nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Bernoulli,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Bernoulli()
end

mutable struct Beta{T}<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    scale::T
    o::Union{PyObject,Nothing}
end

function Beta(;scale::Real=1.0)
    out = Beta(scale, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Beta,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Beta(;scale=o.scale)
end

mutable struct Exponential<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    o::Union{PyObject,Nothing}
end

function Exponential()
    out = Exponential(nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Exponential,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Exponential()
end

mutable struct Gamma<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    o::Union{PyObject,Nothing}
end

function Gamma()
    out = Gamma(nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Gamma,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Gamma()
end

mutable struct Gaussian{T}<:AbstractLikelihood
    variance::T
    o::Union{PyObject,Nothing}
end

function Gaussian(;variance::T=1.0) where T <: Real
    out = Gaussian(variance, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Gaussian,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Gaussian(;variance=o.variance)
end

# TODO GaussianMC not implemented in GPFlow
function GaussianMC()
    @error "GaussianMC not implemented"
end

mutable struct Likelihood<:AbstractLikelihood
    o::Union{PyObject,Nothing}
end

function Likelihood()
    out = Likelihood(nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Likelihood,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Likelihood()
end

mutable struct MonteCarloLikelihood<:AbstractLikelihood
    o::Union{PyObject,Nothing}
end

function MonteCarloLikelihood()
    out = MonteCarloLikelihood(nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{MonteCarloLikelihood,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.MonteCarloLikelihood()
end

mutable struct MultiClass<:AbstractLikelihood
    num_classes::Integer
    # invlink # TODO: add capability to add custom invlink
    o::Union{PyObject,Nothing}
end

function MultiClass(num_classes::Integer)
    out = MultiClass(num_classes, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{MultiClass,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.MultiClass(o.num_classes)
end

mutable struct Ordinal{T}<:AbstractLikelihood
    bin_edges::T
    o::Union{PyObject,Nothing}
end

function Ordinal(bin_edges)
    out = Ordinal(bin_edges, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Ordinal,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Ordinal(o.bin_edges)
end

mutable struct Poisson{T}<:AbstractLikelihood
    # invlink # TODO: add capability to add custom invlink
    binsize::T
    o::Union{PyObject,Nothing}
end

function Poisson(;binsize::Real=1.0)
    out = Poisson(binsize, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Poisson,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.Poisson(;binsize=o.binsize)
end

mutable struct RobustMax{T}<:AbstractLikelihood where T <: Real
    num_classes::Integer
    epsilon::T
    o::Union{PyObject,Nothing}
end

function RobustMax(num_classes::Integer; epsilon::Real=1e-3)
    out = RobustMax(num_classes, epsilon, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{RobustMax,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.RobustMax(o.num_classes; epsilon=o.epsilon)
end

mutable struct SoftMax{T}<:AbstractLikelihood
    num_classes::Integer
    o::Union{PyObject,Nothing}
end

function SoftMax(num_classes::Integer)
    out = SoftMax(num_classes, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{SoftMax,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.SoftMax(o.num_classes)
end

mutable struct StudentT{T1,T2}<:AbstractLikelihood where {T1 <: Real, T2 <: Real}
    scale::T1
    df::T2
    o::Union{PyObject,Nothing}
end

function StudentT(;scale::Real=1.0, df::Real=3.0)
    out = StudentT(scale, df, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{StudentT,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.StudentT(;scale=o.scale, df=o.df)
end

mutable struct SwitchedLikelihood{T}<:AbstractLikelihood
    likelihood_list::T
    o::Union{PyObject,Nothing}
end

function SwitchedLikelihood(likelihood_list)
    out = SwitchedLikelihood(likelihood_list, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{SwitchedLikelihood,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.likelihoods.SwitchedLikelihood(likelihood_list)
end

end # module
