using GPJ, Test

function test()
    @testset "Test Sanity Check" begin
       @test 1 ≈ 1 
    end
end


test()

