using MatrixAlgebra
using BandedMatrices
using MiscUtilities
using DoubleFloats
using Plots

n = 10000

function sband(n::Integer, a::Vector{T}) where T
    k = length(a)
    SH = T<:Complex ? Hermitian : Symmetric
    SH(BandedMatrix([-i => fill(a[i+1], n-i) for i = 0:min(k,n)-1]...), :L) 
end

#A, B = sband(n, [6.0, -4, 1, 0.1]), sband(n, [2.0, 1, 0, 0])
#ws = MatrixAlgebra.make_workspace(A, B)
cws(T) = MatrixAlgebra.make_workspace(copy_elementtype(T, A), copy_elementtype(T, B))

eig(k) = MatrixAlgebra.eigval!(ws, k, lb[k], ub[k], 1)

function pl(a::T, b::T, v::AbstractVector = T[]) where T
    pf(x) = clamp(x - f(x), a, b)
    ph1(x) = clamp(x - h(x, 1), a, b)
    ph2(x) = clamp(x - h(x, 2), a, b)
    ph3(x) = clamp(x - h(x, 3), a, b)
    ph10(x) = clamp(x - h(x, 10), a, b)
    ph100(x) = clamp(x - h(x, 100), a, b)
    ph1000(x) = clamp(x - h(x, 1000), a, b)
    pg1(x) = clamp(x - g(x, v), a, b)
    p = plot(legend=nothing)
    plot!(p, [a; b], [a; b])
    plot!(p, range(a,stop=b,length=2001), [pf; ph1; pg1])
    #display(p)
end

function pl(a::Integer,b::Integer)
    pl(eve[a], eve[b])
end

function plfh(a, b)
    p = plot(legend=nothing)
    plot!(p, [a; b], [0; 0])
    plot!(p, range(a,stop=b,length=1001), [f; h])
    #display(p)
end

function h(x::AbstractFloat, r=1)
    κ, η, ζ = detderivates(A, B, x)
    n = size(A, 1)
    MatrixAlgebra.laguerre1(η, ζ, n, r)
end

function g(x::AbstractFloat, v::AbstractVector)
    n = size(A, 1)
    p = length(v)
    y = sum(inv.(x .- v))
    z = sum(inv.(x .- v) .^ 2)
    κ, η, ζ = detderivates(A, B, x)
    r1 = float(1)
    r2 = clamp(η^2 / ζ, 1, n)
    w = 0 # η^2 / ( ζ + η^2 )
    r = r1 + ( r2 - r1 ) * w^n 
    MatrixAlgebra.laguerre1(η - y, ζ - z, n - p, r)
end

function f(x)
    κ, η, ζ = detderivates(A, B, x)
    1 / η
end

function kappa(x, A=A, B=B)
    κ, η, ζ = detderivates(A, B, x)
    κ
end

function rest(x, A=A, B=B)
    κ, η, ζ = detderivates(A, B, x)
    η ^ 2 / ζ
end

nothing
