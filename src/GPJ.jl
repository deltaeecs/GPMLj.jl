module GPJ

using PyCall

export
gpflow,
compile!,
minimize!,
predict_f,
predict_f_samples

function compile!(o::Any) end
function minimize!(opt, m) end
function predict_f(m, Xnew) end
function predict_f_samples(m, Xnew, num_samples) end

include("gpflow.jl")

end
