module GPMLj

using PyCall

export
    gpflow,
    instantiate!,
    minimize!,
    predict_f,
    predict_f_samples,
    GPFlowObject


abstract type GPFlowObject end

function instantiate!(o) end
function instantiate!(o::Nothing) return nothing end
function minimize!(opt, m) end
function predict_f(m, Xnew) end
function predict_f_samples(m, Xnew, num_samples) end

include("gpflow.jl")

end # module
