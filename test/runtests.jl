using MatrixAlgebra
using LinearAlgebra
using Test

@testset "MatrixAlgebra" begin
    @testset "tridiagonalize" begin include("tridiagonalize.jl") end
    @testset "zrst" begin include("zrst.jl") end
    @testset "determinant" begin include("determinant.jl") end
    @testset "rational approximation" begin include("rational.jl") end
end
