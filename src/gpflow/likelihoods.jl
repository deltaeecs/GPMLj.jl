module likelihoods

    using ..gpflow
    import ..gpflow: instantiate!
    export 
        Gaussian,
        instantiate!

    mutable struct Bernoulli<:Likelihood
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

    mutable struct Beta{T}<:Likelihood
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

    mutable struct Exponential<:Likelihood
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

    mutable struct Gamma<:Likelihood
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

    mutable struct Gaussian{T}<:Likelihood
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

    # TODO Likelihood class not implemented


end # module
