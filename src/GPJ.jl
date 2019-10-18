module GPJ

using PyCall

export
gpflow,
compile!

# TODO: Export compile! 
# function compile!(o) end

include("gpflow.jl")

end
