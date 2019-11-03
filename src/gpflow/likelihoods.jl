module likelihoods

    using ..gpflow
    import ..gpflow: instantiate!
    export 
        Gaussian,
        instantiate!

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
        @info string("Instantiating ", string(mf))
        o.o = py_gpflow.likelihoods.Gaussian(;variance=o.variance)
    end

end # module
