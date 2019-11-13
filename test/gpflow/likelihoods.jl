using GPJ, Test, Random, PyCall

Random.seed!(123)


@testset "Likelihoods" begin
    @testset "Bernoulli" begin
        kern= gpflow.likelihoods.Bernoulli()
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.Bernoulli
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "Beta" begin
        kern= gpflow.likelihoods.Beta()
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.Beta
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "Exponential" begin
        kern= gpflow.likelihoods.Exponential()
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.Exponential
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "Gamma" begin
        kern= gpflow.likelihoods.Gamma()
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.Gamma
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "Gaussian" begin
        kern= gpflow.likelihoods.Gaussian()
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.Gaussian
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "GaussianMC" begin
        @test_throws "GaussianMC not implemented" gpflow.likelihoods.GaussianMC()
    end

    @testset "Likelihood" begin
        kern= gpflow.likelihoods.Likelihood()
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.Likelihood
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "MonteCarloLikelihood" begin
        kern= gpflow.likelihoods.MonteCarloLikelihood()
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.MonteCarloLikelihood
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "MultiClass" begin
        kern= gpflow.likelihoods.MultiClass(3)
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.MultiClass
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "Ordinal" begin
        kern= gpflow.likelihoods.Ordinal([1,2])
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.Ordinal
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end
    
    @testset "Poisson" begin
        kern= gpflow.likelihoods.Poisson()
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.Poisson
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "RobustMax" begin
        kern= gpflow.likelihoods.RobustMax(3)
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.RobustMax
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "SoftMax" begin
        kern= gpflow.likelihoods.SoftMax(3)
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.SoftMax
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    @testset "StudentT" begin
        kern= gpflow.likelihoods.StudentT()
        @test typeof(kern)<:gpflow.AbstractLikelihood
        @test typeof(kern)<:gpflow.likelihoods.StudentT
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == temp
    end

    # TODO
    # @testset "SwitchedLikelihood" begin
    #     kern= gpflow.likelihoods.SwitchedLikelihood()
    #     @test typeof(kern)<:gpflow.AbstractLikelihood
    #     @test typeof(kern)<:gpflow.likelihoods.SwitchedLikelihood
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == temp
    # end

end #module
