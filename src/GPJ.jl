module GPJ

using PyCall

export
gpflow,
compile!

function compile!(o::Any) end

include("gpflow.jl")

end
