using GPMLj, Test, Random, PyCall

Random.seed!(123)

@testset "Priors" begin

    @testset "Gaussian" begin
        prior = gpflow.priors.Gaussian(0.0, 1.0)
        typeof(prior)<:gpflow.Prior
        @test typeof(prior)<:gpflow.priors.Gaussian
        @test typeof(prior.o)<:PyObject
        temp = prior.o
        @test instantiate!(prior) == temp
    end

    @testset "LogNormal" begin
        prior = gpflow.priors.LogNormal(0.0, 1.0)
        typeof(prior)<:gpflow.Prior
        @test typeof(prior)<:gpflow.priors.LogNormal
        @test typeof(prior.o)<:PyObject
        temp = prior.o
        @test instantiate!(prior) == temp
    end

    @testset "Gamma" begin
        prior = gpflow.priors.Gamma((1,), 1.0)
        typeof(prior)<:gpflow.Prior
        @test typeof(prior)<:gpflow.priors.Gamma
        @test typeof(prior.o)<:PyObject
        temp = prior.o
        @test instantiate!(prior) == temp
    end

    @testset "Laplace" begin
        prior = gpflow.priors.Laplace(0.0, 1.0)
        typeof(prior)<:gpflow.Prior
        @test typeof(prior)<:gpflow.priors.Laplace
        @test typeof(prior.o)<:PyObject
        temp = prior.o
        @test instantiate!(prior) == temp
    end

    @testset "Uniform" begin
        prior = gpflow.priors.Uniform()
        typeof(prior)<:gpflow.Prior
        @test typeof(prior)<:gpflow.priors.Uniform
        @test typeof(prior.o)<:PyObject
        temp = prior.o
        @test instantiate!(prior) == temp
    end


end
