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

A, B = sband(n, [6.0, -4, 1]), sband(n, [2.0, 1, 0])
ws = MatrixAlgebra.make_workspace(A, B)

eig(k) = MatrixAlgebra.eigval!(ws, k, lb[k], ub[k], 1)

function pl(a, b)
    p = plot(legend=nothing)
    plot!(p, [a; b], [a; b])
    plot!(p, range(a,stop=b,length=10000), [pf; ph])
    #display(p)
end

function pl(a::Integer,b::Integer)
    p = plot(legend=nothing)
    scatter!(p, eve[a:b], eve[a:b])
    #scatter!(p, [eve[a]*0.98; eve[b]*1.02], [eve[a]*0.98; eve[b]*1.02])
    plot!(p, [eve[a]; eve[b]], [eve[a]; eve[b]])
    plot!(p, range(eve[a],stop=eve[b],length=10001), [pf; ph])
    #display(p)
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
    a = clamp(ζ / η^2, -Inf, 1)
    n / η / ( 1 + sqrt((n-r)/r*max(((n-1) - n * a), (n-1)*0.0)))
end

function h2(x::AbstractFloat, delta=2e-5, r=1)
    κ, η = detderivates(A, B, x)
    _, η2 = detderivates(A, B, x + delta)
    _, η1 = detderivates(A, B, x - delta)
    ζ = (η2 - η1) / 2delta + η^2
    #η = (2η1 + η2) / 3
    n = size(A, 1)
    a = clamp(ζ / η^2, -Inf, 1)
    n / η / ( 1 + sqrt((n-r)/r*max(((n-1) - n * a), (n-1)*0.0)))
end

function f(x)
    κ, η, ζ = detderivates(A, B, x)
    clamp(1 / η, -1e-3, 1e-3)
end

phi(f, x) = clamp(x - f(x), 0, 2x)
pf(x) = phi(f, x)
ph(x) = phi(h, x)


