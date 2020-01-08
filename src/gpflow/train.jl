module train
using ..gpflow
import ..gpflow: instantiate!, minimize!
export 
    ScipyOptimizer,
    instantiate!,
    minimize!

mutable struct ScipyOptimizer<:Optimizer
    o::Union{PyObject, Nothing}
end

function ScipyOptimizer()
    out = ScipyOptimizer(nothing)
    instantiate!(out)
    out
end

function minimize!(opt::Union{ScipyOptimizer, Nothing}, m::Model)
opt.o.minimize(m.o)
end

function instantiate!(o::Union{ScipyOptimizer, Nothing})
    if o === nothing return nothing end
    if typeof(o.o)<:PyObject return o.o end
    o.o = py_gpflow.train.ScipyOptimizer()
end

end # module
