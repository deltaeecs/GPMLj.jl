module gpflow
    using GPJ, PyCall
    import ..GPJ: compile!, minimize!, predict_f, predict_f_samples, GPFlowObject
    export  
        py_gpflow, 
        compile!,
        minimize!,
        predict_f,
        predict_f_samples,
        kernels,
        models,
        likelihoods,
        Model,
        Kernel,
        Likelihood,
        MeanFunction,
        ParameterPrior,
        Optimizer,
        PyObject,
        GPFlowObject

    abstract type Model <: GPFlowObject end
    abstract type Kernel <: GPFlowObject end
    abstract type Likelihood <: GPFlowObject end
    abstract type MeanFunction <: GPFlowObject end
    abstract type ParameterPrior <: GPFlowObject end
    abstract type Optimizer <: GPFlowObject end

    py_gpflow= nothing;
    function __init__()
        global py_gpflow = pyimport("gpflow")
    end


    function compile!(o::Union{Model,Kernel,Likelihood,MeanFunction,ParameterPrior}) end
    function minimize!(opt::Optimizer, m::Model) end

    function predict_f(m::Model, Xnew) end
    function predict_f_samples(m::Model, Xnew, num_samples) end


    include("gpflow/models.jl")
    using .models: predict_f, predict_f_samples

    include("gpflow/kernels.jl")

    include("gpflow/likelihoods.jl")

    include("gpflow/train.jl")
    using .train: minimize!

    include("gpflow/mean_functions.jl")

    include("gpflow/parameter_priors.jl")

end
