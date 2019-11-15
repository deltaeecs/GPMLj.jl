module mean_functions

using ..gpflow
import ..gpflow: instantiate!, predict_f, predict_f_samples
export
    Additive,
    Constant,
    Identity,
    Linear,
    MeanFunction,
    instantiate!

mutable struct Additive{T1,T2} <: AbstractMeanFunction
    first_part::T1
    second_part::T2
    o::Union{PyObject,Nothing}
end

function Additive(first_part, second_part)
    out = Additive(first_part, second_part, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{Additive,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.mean_functions.Additive(o.first_part, o.second_part)
    return o.o
end

function (o::Additive)(X)
    if  !(typeof(o.o)<:PyObject)
        instantiate!(o)
    end
    return o.o(X)
end

mutable struct Constant{T1} <: AbstractMeanFunction
    c::T1
    o::Union{PyObject,Nothing}
end

function Constant(;c=nothing)
    out = Constant(c, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{Constant,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.mean_functions.Constant(;c=o.c)
    return o.o
end

function (o::Constant)(X)
    if  !(typeof(o.o)<:PyObject)
        instantiate!(o)
    end
    return o.o(X)
end

abstract type AbstractLinear <: AbstractMeanFunction end

mutable struct Identity{T1} <: AbstractLinear
    input_dim::T1
    o::Union{PyObject,Nothing}
end

function Identity(;input_dim=nothing)
    out = Identity(input_dim, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{Identity,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.mean_functions.Identity(;input_dim=o.input_dim)
    return o.o
end

function (o::Identity)(X)
    if  !(typeof(o.o)<:PyObject)
        instantiate!(o)
    end
    return o.o(X)
end

mutable struct Linear{T1,T2} <: AbstractLinear
    A::T1
    b::T2
    o::Union{PyObject,Nothing}
end

function Linear(;A=nothing, b=nothing)
    out = Linear(A, b, nothing)
    instantiate!(out)
    out
end

function instantiate!(o::Union{Linear,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.mean_functions.Linear(;A=o.A, b=o.b)
    return o.o
end

function (o::Linear)(X)
    if  !(typeof(o.o)<:PyObject)
        instantiate!(o)
    end
    return o.o(X)
end

mutable struct MeanFunction{T1} <: AbstractMeanFunction
    name::T1
    o::Union{PyObject,Nothing}
end

function MeanFunction(;name=nothing)
    out = MeanFunction(name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{MeanFunction,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.mean_functions.MeanFunction(;name=o.name)
    return o.o
end

function (o::MeanFunction)(X)
    if  !(typeof(o.o)<:PyObject)
        instantiate!(o)
    end
    return o.o(X)
end

end # module
