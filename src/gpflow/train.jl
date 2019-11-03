module train
    using ..gpflow
    import ..gpflow: compile!, minimize!
    export 
        ScipyOptimizer,
        compile!,
        minimize!

    mutable struct ScipyOptimizer<:Optimizer
        o::Union{PyObject,Nothing}
    end

    function ScipyOptimizer()
        out = ScipyOptimizer(nothing)
        compile!(out)
        out
    end

    function minimize!(opt::Union{ScipyOptimizer,Nothing}, m::Model)
    opt.o.minimize(m.o)
    end

    function compile!(o::Union{ScipyOptimizer,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        @info string("Instantiating ", string(mf))
        o.o = py_gpflow.train.ScipyOptimizer()
    end

end # module
