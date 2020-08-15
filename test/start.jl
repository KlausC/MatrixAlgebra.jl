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
    pf(x) = clamp(x - f(x), a, b)
    ph(x) = clamp(x - h(x), a, b)
    ph2(x) = clamp(x - h2(x, (b-a)/100), a, b)
    p = plot(legend=nothing)
    plot!(p, [a; b], [a; b])
    plot!(p, range(a,stop=b,length=2001), [pf; ph; ph2])
    #display(p)
end

function pl(a::Integer,b::Integer)
    pl(eve[a], eve[b])
end

function plfh(a, b)
    p = plot(legend=nothing)
    plot!(p, [a; b], [0; 0])
    plot!(p, range(a,stop=b,length=1001), [f; h; h2])
    #display(p)
end

function h(x::AbstractFloat, r=1)
    κ, η, ζ = detderivates(A, B, x)
    n = size(A, 1)
    a = clamp(ζ / η^2, -Inf, 1)
    n / η / ( 1 + sqrt((n-r)/r*max(((n-1) - n * a), 0.0)))
end

function h2(x::AbstractFloat, delta=2e-5, r=1)
    κ, η = detderivates(A, B, x)
    if η > 0
    η2 = η
    _, η1 = detderivates(A, B, x - delta)
    else
        _, η2 = detderivates(A, B, x + delta)
        η1 = η
    end
     if abs(η * delta) < 100
        ζ = (1 - (inv(η2) - inv(η1)) / delta) * η^2
    else
        ζ = (η2 - η1) / delta + η^2
    end
    #η = (2η1 + η2) / 3
    n = size(A, 1)
    a = clamp(ζ / η^2, -Inf, 1)
    n / η / ( 1 + sqrt((n-r)/r*max(((n-1) - n * a), 0.0)))
end

function f(x)
    κ, η, ζ = detderivates(A, B, x)
    1 / η
end

nothing
