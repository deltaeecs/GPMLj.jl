using GPJ, Test, Random, PyCall

Random.seed!(123)


@testset "Likelihoods" begin
    @testset "Bernoulli" begin
        likelihood = gpflow.likelihoods.Bernoulli()
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.Bernoulli
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "Beta" begin
        likelihood = gpflow.likelihoods.Beta()
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.Beta
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "Exponential" begin
        likelihood = gpflow.likelihoods.Exponential()
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.Exponential
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "Gamma" begin
        likelihood = gpflow.likelihoods.Gamma()
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.Gamma
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "Gaussian" begin
        likelihood = gpflow.likelihoods.Gaussian()
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.Gaussian
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "GaussianMC" begin
        @test_throws "GaussianMC not implemented" gpflow.likelihoods.GaussianMC()
    end

    @testset "Likelihood" begin
        likelihood = gpflow.likelihoods.Likelihood()
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.Likelihood
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "MonteCarloLikelihood" begin
        likelihood = gpflow.likelihoods.MonteCarloLikelihood()
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.MonteCarloLikelihood
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "MultiClass" begin
        likelihood = gpflow.likelihoods.MultiClass(3)
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.MultiClass
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "Ordinal" begin
        likelihood = gpflow.likelihoods.Ordinal([1,2])
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.Ordinal
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end
    
    @testset "Poisson" begin
        likelihood = gpflow.likelihoods.Poisson()
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.Poisson
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "RobustMax" begin
        likelihood = gpflow.likelihoods.RobustMax(3)
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.RobustMax
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "SoftMax" begin
        likelihood = gpflow.likelihoods.SoftMax(3)
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.SoftMax
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    @testset "StudentT" begin
        likelihood = gpflow.likelihoods.StudentT()
        @test typeof(likelihood)<:gpflow.AbstractLikelihood
        @test typeof(likelihood)<:gpflow.likelihoods.StudentT
        @test typeof(likelihood.o)<:PyObject
        temp = likelihood.o
        @test instantiate!(likelihood) == temp
    end

    # TODO
    # @testset "SwitchedLikelihood" begin
    #     likelihood = gpflow.likelihoods.SwitchedLikelihood()
    #     @test typeof(likelihood)<:gpflow.AbstractLikelihood
    #     @test typeof(likelihood)<:gpflow.likelihoods.SwitchedLikelihood
    #     @test typeof(likelihood.o)<:PyObject
    #     temp = likelihood.o
    #     @test instantiate!(likelihood) == temp
    # end

end #module
