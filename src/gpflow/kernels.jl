__precompile__()
module kernels

using ..gpflow
import ..gpflow: compile!
export 
Matern52,
compile!

abstract type Stationary<:Kernel end

mutable struct Matern52 <: Stationary 
    input_dim
    variance
    lengthscales
    active_dims
    ARD
    name::Union{String,Nothing}
    o::Union{PyObject,Nothing}
end

function Matern52(input_dim; variance=1.0, lengthscales=1.0, active_dims=nothing, ARD=nothing, name=nothing)

    Matern52(input_dim, variance, lengthscales, active_dims, ARD, name, nothing)
end

function compile!(o::Union{Matern52,Nothing})
    if o === nothing return nothing end
    o.o = py_gpflow.kernels.Matern52(
                                        o.input_dim; 
                                        variance=o.variance, 
                                        lengthscales=o.lengthscales, 
                                        active_dims=o.active_dims, 
                                        ARD=o.ARD, 
                                        name=o.name
                                    )
end 

end