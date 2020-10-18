
using BandedMatrices

function sband(n::Integer, a::Vector{T}) where T
    k = length(a)
    SH = T<:Complex ? Hermitian : Symmetric
    SH(BandedMatrix([-i => fill(a[i+1], n-i) for i = 0:min(k,n)-1]...), :L)
end

function example_1(n, T=Float64)
    tt = sband(n, T[4, 1])
    ss = sband(n, T[2, 1])
    tt, ss
end

det0(A, B, s) = det(Matrix(A - s * B))
function det1(A, B, s)
    h = sqrt(eps(max(abs(s), 1.0)))
    ( det0(A, B, s + h) - det0(A, B, s - h) ) / (2*h)
end
function det2(A, B, s)
    h = eps((max(abs(s), 1)))^(1/3)
    ( det0(A, B, s + h) + det0(A, B, s - h) -2*det0(A, B, s) ) / h^2
end


@testset "basics n=$n s=$s" for n = [1, 2, 3, 10], s = [1.0, 0.5]
    A, B = sband(n, [0.1]), sband(n, [1.0])
    κ, η, ζ = detderivates(A, B, s)
    @test κ == n 
    @test η ≈ -n / (0.1 - s)
    @test ζ ≈ n / (0.1 - s)^2
end

@testset "small real $n" for n = 1:4
    A, B = sband(n, [6.0, -4, 1]), sband(n, [2.0, 1])
    s = 0.001
    d0 = det0(A, B, s)
    κ, η, ζ = detderivates(A, B, s)
    @test κ == 0 
    @test isapprox(η, det1(A, B, s) / d0; rtol=1e-3)
    @test isapprox(η^2 - ζ, det2(A, B, s) / d0; atol=1e-5, rtol=1e-3)
end

@testset "small complex $n" for n = 1:4
    A, B = sband(n, [6.0, -4, 1+0im]), sband(n, [2.0, 1im])
    s = 0.001
    d0 = det0(A, B, s)
    κ, η, ζ = detderivates(A, B, s)
    @test η isa Real
    @test ζ isa Real
    @test κ == 0 
    @test isapprox(η, det1(A, B, s) / d0; rtol=1e-3)
    @test isapprox(η^2 - ζ, det2(A, B, s) / d0; atol=1e-5, rtol=1e-3)
end


kappa(x) = x < 0.0 ? 0 : 2
kappa1(x) = x < 1.0 ? 0 : 3
kappa2(x) = x < 0.2 ? 0 : x < 1 ? 1 : x < 2 ? 3 : typemax(Int)

MatrixAlgebra.detderivates!(f::Function, x) = f(x), 0.0, 0.0

@testset "bisection bounds 0-2" begin
    lb, ub = eigbounds(1, 2, kappa, 1.0)
    @test  all(isapprox.(lb, 0.0; atol=eps()))
    @test  all(isapprox.(ub, 0.0; atol=eps()))
end
@testset "bisection bounds 1-3" begin
    lb, ub = eigbounds(1, 3, kappa1, 1.0)
    @test  all(isapprox.(lb, 1.0; atol=eps()))
    @test  all(isapprox.(ub, 1.0; atol=eps()))
end
@testset "bisection bounds 2-4" begin
    lb, ub = eigbounds(1, 4, kappa2, 1.0)
    @test all(isapprox.(lb, [0, 1, 1, 1])) 
    @test all(isapprox.(ub, [0.5, 1, 1, 3])) 
end
@testset "bisection bounds 2-5" begin
    lb, ub = eigbounds(1, 5, kappa2, 1.0)
    @test all(isapprox.(lb, [0, 1, 1, 2, 2])) 
    @test all(isapprox.(ub, [0.5, 1, 1, 2, 2])) 
end
