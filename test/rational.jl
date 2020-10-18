
using MatrixAlgebra
using Test

@testset "arithmetic" begin
    a = RatApp(1.0, 0, 1, 0)
    @test a == one(RatApp{Float64})
end
