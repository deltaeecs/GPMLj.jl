module GPJ

using PyCall

export
gpflow,
compile!

# function compile!(o) end

include("gpflow.jl")

end
