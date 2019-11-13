module kernels

using ..gpflow
import ..gpflow: instantiate!
export 
    Matern52,
    ArcCosine,
    Periodic,
    Coregion,
    instantiate!

abstract type Stationary <: Kernel end

mutable struct Matern52 <: Stationary 
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String,Nothing}
    o::Union{PyObject,Nothing}
end

function Matern52(
    input_dim; 
    variance=1.0, 
    lengthscales=1.0, 
    active_dims=nothing, 
    ARD=nothing, 
    name=nothing
)
    out = Matern52(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Matern52,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Matern52(
        o.input_dim; 
        variance=o.variance, 
        lengthscales=o.lengthscales, 
        active_dims=o.active_dims, 
        ARD=o.ARD, 
        name=o.name
    )
    return o.o
end 

mutable struct ArcCosine{T1,T2,T3,T4,T5,T6,T7,T8} <: Kernel 
    input_dim::T1
    order::T2
    variance::T3
    weight_variances::T4
    bias_variance::T5
    active_dims::T6
    ARD::T7
    name::T8
    o::Union{PyObject,Nothing}
end

function ArcCosine(
    input_dim;
    order=0,
    variance=1.0, 
    weight_variances=1., 
    bias_variance=1.,
    active_dims=nothing, 
    ARD=nothing, 
    name=nothing
)
    out = ArcCosine(
        input_dim,
        order,
        variance,
        weight_variances,
        bias_variance,
        active_dims,
        ARD,
        name,
        nothing
    )
    instantiate!(out)
    return out
end

function instantiate!(o::Union{ArcCosine,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.ArcCosine(
        o.input_dim,
        order=o.order,
        variance=o.variance,
        weight_variances=o.weight_variances,
        bias_variance=o.bias_variance,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end 

mutable struct Periodic{T1,T2,T3,T4,T5,T6} <: Kernel
    input_dim::T1
    period::T2
    variance::T3
    lengthscales::T4
    active_dims::T5
    name::T6
    o::Union{PyObject,Nothing}
end

function Periodic(
    input_dim;
    period=1.0, 
    variance=1.0,
    lengthscales=1.0, 
    active_dims=nothing, 
    name=nothing
)
    out = Periodic(
        input_dim,
        period,
        variance,
        lengthscales,
        active_dims,
        name,
        nothing
    )
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Periodic,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Periodic(
        o.input_dim;
        period=o.period,
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        name=o.name
    )
    return o.o
end 

mutable struct Coregion{T1,T2,T3} <: Kernel
    output_dim::T1
    rank::T2
    active_dims::T3
    o::Union{PyObject,Nothing}
end

function Coregion(
    output_dim, 
    rank; 
    active_dims=nothing, 
)
    out = Coregion(
        output_dim,
        rank,
        active_dims,
        nothing
    )
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Coregion,Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Coregion(
        o.output_dim,
        o.rank;
        active_dims=o.active_dims
    )
    return o.o
end 

end # module
