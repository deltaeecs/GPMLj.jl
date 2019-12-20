module kernels

using ..gpflow
import ..gpflow: instantiate!
export
    Combination,
    Product
    Sum,
    Stationary,
    Convolutional,
    WeightedConvolutional,
    Cosine
    Exponential,
    RationalQuadratic,
    SquaredExponential,
    Linear
    Polynomial,
    Matern12
    Matern32,
    Matern52,
    Static,
    White,
    Bias,
    Constant,
    ArcCosine,
    Periodic
    Coregion,
    instantiate!

# Stationary Kernels

abstract type AbstractStationary <: Kernel end
abstract type AbstractLinear <: Kernel end
abstract type AbstractCombination <: Kernel end

mutable struct Combination <: AbstractCombination
    kernels
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Combination(
    kernels;
    name=nothing
)
    out = Combination(kernels, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Combination, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Combination(
        [kern.o for kern in o.kernels];
        name=o.name
    )
    return o.o
end

mutable struct Product <: AbstractCombination
    kernels
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Product(
    kernels;
    name=nothing
)
    out = Product(kernels, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Product, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Product(
        [kern.o for kern in o.kernels];
        name=o.name
    )
    return o.o
end

mutable struct Sum <: AbstractCombination
    kernels
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Sum(
    kernels;
    name=nothing
)
    out = Sum(kernels, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Sum, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Sum(
        [kern.o for kern in o.kernels];
        name=o.name
    )
    return o.o
end

mutable struct Stationary <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Stationary(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Stationary(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Stationary, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Stationary(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

mutable struct Convolutional <: AbstractStationary
    basekern
    img_size
    patch_size
    colour_channels
    o::Union{PyObject, Nothing}
end

function Convolutional(
    basekern,
    img_size,
    patch_size;
    colour_channels=1
)
    out = Convolutional(basekern, img_size, patch_size, colour_channels, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Convolutional, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Convolutional(
        o.basekern,
        o.img_size,
        o.patch_size;
        colour_channels=o.colour_channels
    )
    return o.o
end

mutable struct WeightedConvolutional <: AbstractStationary
    basekern
    img_size
    patch_size
    colour_channels
    o::Union{PyObject, Nothing}
end

function WeightedConvolutional(
    basekern,
    img_size,
    patch_size;
    colour_channels=1
)
    out = WeightedConvolutional(basekern, img_size, patch_size, colour_channels, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{WeightedConvolutional, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.WeightedConvolutional(
        o.basekern,
        o.img_size,
        o.patch_size;
        colour_channels=o.colour_channels
    )
    return o.o
end

mutable struct Cosine <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Cosine(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Cosine(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Cosine, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Cosine(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

mutable struct Exponential <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Exponential(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Exponential(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Exponential, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Exponential(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

mutable struct RationalQuadratic <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function RationalQuadratic(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = RationalQuadratic(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{RationalQuadratic, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.RationalQuadratic(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

mutable struct SquaredExponential <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function SquaredExponential(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = SquaredExponential(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{SquaredExponential, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.SquaredExponential(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

mutable struct Linear <: AbstractLinear
    input_dim
    variance
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Linear(
    input_dim;
    variance=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Linear(input_dim, variance, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Linear, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Linear(
        o.input_dim;
        variance=o.variance,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end



mutable struct Polynomial <: AbstractLinear
    input_dim
    degree
    variance
    offset
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Polynomial(
    input_dim;
    degree=3.0,
    variance=1.0,
    offset=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Polynomial(input_dim, degree, variance, offset, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Polynomial, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Polynomial(
        o.input_dim;
        degree=o.degree,
        variance=o.variance,
        offset=o.offset,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

mutable struct Matern12 <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Matern12(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Matern12(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Matern12, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Matern12(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

mutable struct Matern32 <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
end

function Matern32(
    input_dim;
    variance=1.0,
    lengthscales=1.0,
    active_dims=nothing,
    ARD=nothing,
    name=nothing
)
    out = Matern32(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Matern32, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Matern32(
        o.input_dim;
        variance=o.variance,
        lengthscales=o.lengthscales,
        active_dims=o.active_dims,
        ARD=o.ARD,
        name=o.name
    )
    return o.o
end

mutable struct Matern52 <: AbstractStationary
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String, Nothing}
    o::Union{PyObject, Nothing}
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

function instantiate!(o::Union{Matern52, Nothing})
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

# Static Kernels
mutable struct Static <: Kernel
    input_dim
    variance
    active_dims
    o::Union{PyObject, Nothing}
end

function Static(
    input_dim;
    variance=1.0,
    active_dims=nothing,
)
    out = Static(input_dim, variance, active_dims, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Static, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Static(
        o.input_dim;
        variance=o.variance,
        active_dims=o.active_dims,
    )
    return o.o
end

abstract type AbstractStatic <: Kernel end

mutable struct White <: AbstractStatic
    variance
    active_dims
    o::Union{PyObject, Nothing}
end

function White(;
    variance=1.0,
    active_dims=nothing,
)
    out = White(variance, active_dims, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{White, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.White(;
        variance=o.variance,
        active_dims=o.active_dims,
    )
    return o.o
end

mutable struct Bias <: AbstractStatic
    variance
    active_dims
    o::Union{PyObject, Nothing}
end

function Bias(;
    variance=1.0,
    active_dims=nothing,
)
    out = Bias(variance, active_dims, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Bias, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Bias(;
        variance=o.variance,
        active_dims=o.active_dims,
    )
    return o.o
end

mutable struct Constant <: AbstractStatic
    variance
    active_dims
    o::Union{PyObject, Nothing}
end

function Constant(;
    variance=1.0,
    active_dims=nothing,
)
    out = Constant(variance, active_dims, nothing)
    instantiate!(out)
    return out
end

function instantiate!(o::Union{Constant, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.kernels.Constant(;
        variance=o.variance,
        active_dims=o.active_dims,
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
    o::Union{PyObject, Nothing}
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

function instantiate!(o::Union{ArcCosine, Nothing})
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
    o::Union{PyObject, Nothing}
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

function instantiate!(o::Union{Periodic, Nothing})
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
    o::Union{PyObject, Nothing}
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

function instantiate!(o::Union{Coregion, Nothing})
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
