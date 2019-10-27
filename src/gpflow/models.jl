__precompile__()
module models

using ..gpflow
import ..gpflow: compile!, predict_f, predict_f_samples
export 
GPR,
compile!,
predict_f,
predict_f_samples,
SGPR,
VGP,
SVGP,
GPMC,
SGPMC

abstract type GPModel <: Model end

function predict_f(m::GPModel, Xnew)
    return m.o.predict_f(Xnew)
end
function predict_f_samples(m::GPModel, Xnew, num_samples)
    return m.o.predict_f_samples(Xnew, num_samples)
end

mutable struct GPR{T1,T2} <: GPModel
    X::T1
    Y::T2
    kern::Union{Kernel,Nothing}
    mean_function::Union{MeanFunction,Nothing}
    name::Union{String,Nothing}
    likelihood::Union{Likelihood,Nothing}
    o::Union{PyObject,Nothing}
end

function GPR(X, Y, kern::Kernel; mean_function::Union{MeanFunction,Nothing}=nothing, name::Union{String,Nothing}=nothing)
    out = GPR(X, Y, kern, mean_function, name, nothing, nothing)
    compile!(out)
    out
end

function compile!(o::Union{GPR,Nothing})
    if typeof(o.o)<:PyObject return o.o end
    if o === nothing return nothing end
    kern_ = compile!(o.kern)
    mean_function_ = compile!(o.mean_function)
    o.o = py_gpflow.models.GPR(o.X, o.Y, kern_, mean_function_, o.name)
    #= TODO: Shoudld we manually initialize likelihood?
    o.likelihood = likelihoods.Gaussian()
    o.likelihood.o = o.o.likelihood =#
    return o.o
end


# TODO: SGPRUpperMixin required?
mutable struct SGPR{T1,T2,T3} <: GPModel
    X::T1
    Y::T2
    kern::Union{Kernel,Nothing}
    feat
    mean_function::Union{MeanFunction,Nothing}
    Z::T3
    name::Union{String,Nothing}
    o::Union{PyObject,Nothing}
end

function SGPR(  X, Y, kern::Union{Kernel,Nothing};
                feat=nothing,
                mean_function::Union{MeanFunction,Nothing}=nothing,
                Z=nothing,
                name::Union{String,Nothing}=nothing
                )
    out = SGPR(X, Y, kern, feat, mean_function, Z, name, nothing)
    compile!(out)
    out
end

function compile!(o::Union{SGPR,Nothing})
    if typeof(o.o)<:PyObject return o.o end
    if o === nothing return nothing end
    kern_ = compile!(o.kern)
    mean_function_ = compile!(o.mean_function)
    o.o = py_gpflow.models.SGPR(    o.X, o.Y, kern=kern_;
                                    mean_function=mean_function_,
                                    name=o.name
                                    )
    o.o
end

mutable struct VGP{T1,T2} <: GPModel
    X::T1
    Y::T2
    kern::Union{Kernel,Nothing}
    likelihood::Union{Likelihood,Nothing}
    mean_function::Union{MeanFunction,Nothing}
    num_latent::Union{Int,Nothing}
    o::Union{PyObject,Nothing}
end

function VGP(  X, Y, kern::Union{Kernel,Nothing}, likelihood::Union{Likelihood,Nothing};
                mean_function::Union{MeanFunction,Nothing}=nothing,
                num_latent::Union{Int,Nothing}=nothing
                )
    out = VGP(X, Y, kern, likelihood, mean_function, num_latent, nothing)
    compile!(out)
    out
end    

function compile!(o::VGP)
    if typeof(o.o)<:PyObject return o.o end
    if o === nothing return nothing end
    kern_ = compile!(o.kern)
    likelihood_ = compile!(o.likelihood)
    mean_function_ = compile!(o.mean_function)
    o.o = py_gpflow.models.VGP(o.X, o.Y, kern=kern_, likelihood=likelihood_;
                                mean_function=mean_function_,
                                num_latent=o.num_latent
                                )
    o.o
end

mutable struct SVGP{T1,T2,T3,T4,T5,T6} <: GPModel
    X::T1
    Y::T2
    kern::Union{Kernel,Nothing}
    likelihood::Union{Likelihood,Nothing}
    feat::T3
    mean_function::Union{MeanFunction,Nothing}
    num_latent::Union{Int,Nothing}
    q_diag::Bool
    whiten::Bool
    minibatch_size::Union{Int,Nothing}
    Z::T4
    num_data::Union{Int,Nothing}
    q_mu::T5
    q_sqrt::T6
    o::Union{PyObject,Nothing}
end

function SVGP(  X, Y, kern::Union{Kernel,Nothing}, likelihood::Union{Likelihood,Nothing};
                feat=nothing, 
                mean_function::Union{MeanFunction,Nothing}=nothing, 
                num_latent::Union{Int,Nothing}=nothing, 
                q_diag::Bool=false,
                whiten::Bool=true, 
                minibatch_size::Union{Int,Nothing}=nothing, 
                Z=nothing, 
                num_data=nothing, 
                q_mu=nothing, 
                q_sqrt=nothing)

    out = SVGP( X, Y, kern, likelihood,
                feat,
                mean_function,
                num_latent,
                q_diag,
                whiten,
                minibatch_size,
                Z,
                num_data,
                q_mu,
                q_sqrt,
                nothing
                )
    compile!(out)
    out
end

function compile!(o::SVGP)
    if typeof(o.o)<:PyObject return o.o end
    if o === nothing return nothing end
    kern_ = compile!(o.kern)
    likelihood_ = compile!(o.likelihood)
    mean_function_ = compile!(o.mean_function)
    o.o = py_gpflow.models.SGVP(o.X, o.Y, kern=kern_, likelihood=likelihood_;
                                feat=o.feat,
                                mean_function=mean_function_,
                                num_latent=o.num_latent,
                                q_diag=o.q_diag,
                                whiten=o.whiten,
                                minibatch_size=o.minibatch_size,
                                Z=o.Z,
                                num_data=o.num_data,
                                q_mu=o.q_mu,
                                q_sqrt=o.q_sqrt,
                                )
    o.o
end

mutable struct GPMC{T1,T2} <: GPModel
    X::T1
    Y::T2
    kern::Union{Kernel,Nothing}
    likelihood::Union{Likelihood,Nothing}
    mean_function::Union{MeanFunction,Nothing}
    num_latent::Union{Int,Nothing}
    o::Union{PyObject,Nothing}
end

function GPMC(  X, Y, kern::Union{Kernel,Nothing}, likelihood::Union{Likelihood,Nothing};
                mean_function::Union{MeanFunction,Nothing}=nothing, 
                num_latent::Union{Int,Nothing}=nothing,
                )
    out = GPMC( X,
                Y,
                kern,
                likelihood,
                mean_function,
                num_latent,
                nothing
                )
    compile!(out)
    out
end

function compile!(o::GPMC)
    if typeof(o.o)<:PyObject return o.o end
    if o === nothing return nothing end
    kern_ = compile!(o.kern)
    likelihood_ = compile!(o.likelihood)
    mean_function_ = compile!(o.mean_function)
    o.o = py_gpflow.models.GPMC(o.X, o.Y, kern=kern_, likelihood=likelihood_;
                                mean_function=mean_function_,
                                num_latent=o.num_latent                                
                                )
    o.o
end


mutable struct SGPMC{T1,T2,T3,T4} <: GPModel
    X::T1
    Y::T2
    kern::Union{Kernel,Nothing}
    likelihood::Union{Likelihood,Nothing}
    feat::T3
    mean_function::Union{MeanFunction,Nothing}
    num_latent::Union{Int,Nothing}
    Z::T4
    o::Union{PyObject,Nothing}
end

function SGPMC( X, Y, kern::Union{Kernel,Nothing}, likelihood::Union{Likelihood,Nothing};
                feat=nothing,
                mean_function::Union{MeanFunction,Nothing}=nothing, 
                num_latent::Union{Int,Nothing}=nothing,
                Z=nothing
                )
    out = SGPMC(X,
                Y,
                kern,
                likelihood,
                feat,
                mean_function,
                num_latent,
                Z,
                nothing
                )
    compile!(out)
    out
end

function compile!(o::SGPMC)
    if typeof(o.o)<:PyObject return o.o end
    if o === nothing return nothing end
    kern_ = compile!(o.kern)
    likelihood_ = compile!(o.likelihood)
    mean_function_ = compile!(o.mean_function)
    o.o = py_gpflow.models.SGPMC(o.X, o.Y, kern=kern_, likelihood=likelihood_;
                                fwat=o.feat,
                                mean_function=mean_function_,
                                num_latent=o.num_latent,
                                Z=o.Z
                                )
    o.o
end

end # models module end