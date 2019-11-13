using GPJ, Test, Random, PyCall

Random.seed!(123)


@testset "Kernels" begin
    @testset "Matern52" begin
        kern= gpflow.kernels.Matern52(2)
        @test typeof(kern)<:gpflow.Kernel
        @test typeof(kern)<:gpflow.kernels.Stationary
        @test typeof(kern)<:gpflow.kernels.Matern52
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == kern.o
    end

    @testset "ArcCosine" begin
        kern= gpflow.kernels.ArcCosine(2)
        @test typeof(kern)<:gpflow.Kernel
        @test typeof(kern)<:gpflow.kernels.ArcCosine
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == kern.o
    end

    @testset "Periodic" begin
        kern= gpflow.kernels.Periodic(2)
        @test typeof(kern)<:gpflow.Kernel
        @test typeof(kern)<:gpflow.kernels.Periodic
        @test typeof(kern.o)<:PyObject
        temp = kern.o
        @test instantiate!(kern) == kern.o
    end

    # TODO
    # @testset "Coregion" begin
    #     kern= gpflow.kernels.Coregion(2, 1)
    #     @test typeof(kern)<:gpflow.Kernel
    #     @test typeof(kern)<:gpflow.kernels.Coregion
    #     @test typeof(kern.o)<:PyObject
    #     temp = kern.o
    #     @test instantiate!(kern) == kern.o
    # end

    

end #module
