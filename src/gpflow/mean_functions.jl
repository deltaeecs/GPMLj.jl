module mean_functions

    using ..gpflow
    import ..gpflow: instantiate!, predict_f, predict_f_samples
    export
        instantiate!,
        Additive,
        Constant,
        Identity,
        Linear,
        MeanFunction

    mutable struct Additive{T1,T2} <: MeanFunctionAbstract
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

    mutable struct Constant{T1} <: MeanFunctionAbstract
        c::T1
        o::Union{PyObject,Nothing}
    end

    function Constant(;c=nothing)
        out = Additive(c, nothing)
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

    abstract type LinearAbstract <: MeanFunctionAbstract end

    mutable struct Identity{T1} <: LinearAbstract
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
    
    mutable struct Linear{T1,T2} <: LinearAbstract
        A::T1
        B::T2
        o::Union{PyObject,Nothing}
    end

    function Linear(;A=nothing, B=nothing)
        out = Linear(A, B, nothing)
        instantiate!(out)
        out
    end

    function instantiate!(o::Union{Linear,Nothing})
        if o === nothing return nothing end
        if typeof(o.o)<:PyObject return o.o end
        o.o = py_gpflow.mean_functions.Linear(;A=o.A, B=o.B)
        return o.o
    end

    function (o::Linear)(X)
        if  !(typeof(o.o)<:PyObject)
            instantiate!(o)
        end
        return o.o(X)
    end

    mutable struct MeanFunction{T1} <: MeanFunctionAbstract
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
