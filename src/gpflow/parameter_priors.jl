module priors

using ..gpflow
import ..gpflow: instantiate!

export
    Gaussian,
    LogNormal,
    Gamma,
    Laplace,
    Uniform,
    instantiate!,
    logp,
    sample

function logp(obj::T, x) where T <: Prior
    return  obj.o.logp(x)
end

function sample(obj::T; shape=(1,)) where T <: Prior
    return  obj.o.sample(;shape=shape)
end

mutable struct Gaussian{T1,T2} <: Prior
    mu::T1
    var::T2
    o::Union{PyObject, Nothing}
end

function Gaussian(mu, var)
    out = Gaussian(mu, var, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{Gaussian, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.priors.Gaussian(o.mu, o.var)
    return o.o
end

mutable struct LogNormal{T1,T2} <: Prior
    mu::T1
    var::T2
    o::Union{PyObject, Nothing}
end

function LogNormal(mu, var)
    out = LogNormal(mu, var, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{LogNormal, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.priors.LogNormal(o.mu, o.var)
    return o.o
end

mutable struct Gamma{T1,T2} <: Prior
    shape::T1
    scale::T2
    o::Union{PyObject, Nothing}
end

function Gamma(shape, scale)
    out = Gamma(shape, scale, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{Gamma, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.priors.Gamma(o.shape, o.scale)
    return o.o
end

mutable struct Laplace{T1,T2} <: Prior
    mu::T1
    sigma::T2
    o::Union{PyObject, Nothing}
end

function Laplace(mu, sigma)
    out = Laplace(mu, sigma, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{Laplace, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.priors.Laplace(o.mu, o.sigma)
    return o.o
end

mutable struct Uniform{T1,T2} <: Prior
    lower::T1
    upper::T2
    o::Union{PyObject, Nothing}
end

function Uniform(;lower=0.0, upper=1.0)
    out = Uniform(lower, upper, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{Uniform, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.priors.Uniform(;lower=o.lower, upper=o.upper)
    return o.o
end

end # module
