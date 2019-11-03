module mean_functions

    using ..gpflow
    import ..gpflow: compile!, predict_f, predict_f_samples
    export
        compile!,
        Additive,
        Constant,
        Identity,
        Linear,
        MeanFunction

    # function (o::MeanFunction)(X)
    #     if  !(typeof(o.o)<:PyObject)
    #         compile!(o)
    #     end
    #     return o.o(X)
    # end

    mutable struct Additive{T1,T2} <: MeanFunction
        first_part::T1
        second_part::T2
        o::Union{PyObject,Nothing}
    end

    function Additive(first_part, second_part)
        out = Additive(first_part, second_part, nothing)
        compile!(out)
        out
    end

    function compile!(o::Union{Additive,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        @info string("Instantiating ", string(mf))
        o.o = py_gpflow.mean_functions.Additive(o.first_part, o.second_part)
        return o.o
    end

    mutable struct Constant{T1} <: MeanFunction
        c::T1
        o::Union{PyObject,Nothing}
    end

    function Constant(;c=nothing)
        out = Additive(c, nothing)
        compile!(out)
        out
    end

    function compile!(o::Union{Constant,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        @info string("Instantiating ", string(mf))
        o.o = py_gpflow.mean_functions.Constant(;c=o.c)
        return o.o
    end

    abstract type Linear_ <: MeanFunction end

    mutable struct Identity{T1} <: Linear_
        input_dim::T1
        o::Union{PyObject,Nothing}
    end

    function Identity(;input_dim=nothing)
        out = Identity(input_dim, nothing)
        compile!(out)
        out
    end

    function compile!(o::Union{Identity,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        @info string("Instantiating ", string(mf))
        o.o = py_gpflow.mean_functions.Identity(;input_dim=o.input_dim)
        return o.o
    end

    mutable struct Linear{T1,T2} <: MeanFunction
        A::T1
        B::T2
        o::Union{PyObject,Nothing}
    end

    function Linear(;A=nothing, B=nothing)
        out = Linear(A, B, nothing)
        compile!(out)
        out
    end

    function compile!(o::Union{Linear,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        @info string("Instantiating ", string(mf))
        o.o = py_gpflow.mean_functions.Linear(;A=o.A, B=o.B)
        return o.o
    end


    # mutable struct MeanFunction{T1} <: GPFlowObject
    #     name::T1
    #     o::Union{PyObject,Nothing}
    # end

    # function MeanFunction(;name=nothing)
    #     out = MeanFunction(name, nothing)
    #     compile!(out)
    #     return out
    # end

    # function compile!(o::Union{MeanFunction,Nothing})
        # if o === nothing return nothing end
        # if typeof(o.o)<:PyObject return o.o end
        # @info string("Instantiating ", string(mf))
    #     o.o = py_gpflow.mean_functions.MeanFunction(;name=o.name)
    #     return o.o
    # end

end # module
