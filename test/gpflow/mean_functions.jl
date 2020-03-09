using GPMLj, Test, Random, PyCall

Random.seed!(123)

@testset "MeanFunctions" begin

    @testset "Additive" begin
        identity(X) = X
        mf = gpflow.mean_functions.Additive(identity, identity)
        @test typeof(mf)<:gpflow.AbstractMeanFunction
        @test typeof(mf)<:gpflow.mean_functions.Additive
        @test typeof(mf.o)<:PyObject
        temp = mf.o
        @test instantiate!(mf) == temp
    end

    @testset "Constant" begin
        mf = gpflow.mean_functions.Constant()
        @test typeof(mf)<:gpflow.AbstractMeanFunction
        @test typeof(mf)<:gpflow.mean_functions.Constant
        @test typeof(mf.o)<:PyObject
        temp = mf.o
        @test instantiate!(mf) == temp
    end

    @testset "Identity" begin
        mf = gpflow.mean_functions.Identity()
        @test typeof(mf)<:gpflow.AbstractMeanFunction
        @test typeof(mf)<:gpflow.mean_functions.Identity
        @test typeof(mf.o)<:PyObject
        temp = mf.o
        @test instantiate!(mf) == temp
    end

    @testset "Linear" begin
        mf = gpflow.mean_functions.Linear()
        @test typeof(mf)<:gpflow.AbstractMeanFunction
        @test typeof(mf)<:gpflow.mean_functions.Linear
        @test typeof(mf.o)<:PyObject
        temp = mf.o
        @test instantiate!(mf) == temp
    end

    @testset "MeanFunction" begin
        mf = gpflow.mean_functions.MeanFunction()
        @test typeof(mf)<:gpflow.AbstractMeanFunction
        @test typeof(mf)<:gpflow.mean_functions.MeanFunction
        @test typeof(mf.o)<:PyObject
        temp = mf.o
        @test instantiate!(mf) == temp
    end

    @testset "SwitchedMeanFunction" begin
        mf1 = gpflow.mean_functions.Linear()
        mf2 = gpflow.mean_functions.Linear()
        mf3 = gpflow.mean_functions.Linear()
        mf = gpflow.mean_functions.SwitchedMeanFunction([mf1, mf2, mf3])
        @test typeof(mf)<:gpflow.AbstractMeanFunction
        @test typeof(mf)<:gpflow.mean_functions.SwitchedMeanFunction
        @test typeof(mf.o)<:PyObject
        temp = mf.o
        @test instantiate!(mf) == temp
    end

end #module
