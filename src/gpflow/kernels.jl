module kernels

    using ..gpflow
    import ..gpflow: compile!
    export 
        Matern52,
        compile!

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
        compile!(out)
        return out
    end

    function compile!(o::Union{Matern52,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        @info string("Instantiating ", string(mf))
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

    mutable struct ArcCosine{T1,T2,T3,T4,T5,T6,T7} <: Kernel 
        order::T1
        variance::T2
        weight_variances::T3
        bias_variance::T4
        active_dims::T5
        ARD::T6
        name::T7
        o::Union{PyObject,Nothing}
    end

    function ArcCosine(;
        order=0,
        variance=1.0, 
        weight_variances=1., 
        bias_variance=1.,
        active_dims=nothing, 
        ARD=nothing, 
        name=nothing
    )
        out = ArcCosine(
            order,
            variance,
            weight_variances,
            bias_variance,
            active_dims,
            ARD,
            name,
            nothing
        )
        compile!(out)
        return out
    end

    function compile!(o::Union{ArcCosine,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        @info string("Instantiating ", string(mf))
        o.o = py_gpflow.kernels.ArcCosine(
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
        compile!(out)
        return out
    end

    function compile!(o::Union{Periodic,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        @info string("Instantiating ", string(mf))
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

    mutable struct Coregion{T1,T2,T3,T4,T5} <: Kernel
        input_dim::T1
        output_dim::T2
        rank::T3
        active_dims::T4
        name::T5
        o::Union{PyObject,Nothing}
    end

    function Coregion(
        input_dim, 
        output_dim, 
        rank; 
        active_dims=nothing, 
        name=nothing
    )
        out = Coregion(
            input_dim,
            output_dim,
            rank,
            active_dims,
            name,
            nothing
        )
        compile!(out)
        return out
    end

    function compile!(o::Union{Coregion,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        @info string("Instantiating ", string(mf))
        o.o = py_gpflow.kernels.Coregion(
            o.input_dim,
            o.output_dim,
            o.rank;
            active_dims=o.active_dims,
            name=o.name
        )
        return o.o
    end 

end # module
