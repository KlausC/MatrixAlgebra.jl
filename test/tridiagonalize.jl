using MatrixAlgebra
using LinearAlgebra
using Test

let A = [1.0 2 4 7 11; 2 3 5 8 12; 4 5 6 9 13; 7 8 9 10 14; 11 12 13 14 15], n = size(A, 1)
    @testset "argument checks" begin
        @test_throws ArgumentError tridiag_real(A, n+1)
        @test_throws ArgumentError tridiag_real(A, 0)
        @test_throws DimensionMismatch tridiag_real(A[:,2:n], 1)
    end
    @testset "real $(2d-1)-diagonal" for d in 2:n
        M = [abs(i-j) < d ? A[i,j] : zero(eltype(A)) for i in 1:n, j=1:n]
        E = tridiag_real(M, d)
        @test eigvals(E) ≈ eigvals(M)
    end
end
let A = [1.0 2 4+im 7 11; 2 3 5 8+im 12; 4-im 5 6 9 13; 7 8-im 9 10 14; 11 12 13 14 15], n = size(A, 1)
    @testset "complex $(2d-1)-diagonal" for d in 2:n
        M = [abs(i-j) < d ? A[i,j] : zero(eltype(A)) for i in 1:n, j=1:n]
        E = tridiag_real(M, d)
        @test eigvals(E) ≈ eigvals(M)
    end
end

