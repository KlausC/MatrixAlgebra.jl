
export mwiden, detderivates


using BandedMatrices
using DoubleFloats

const SymRealHerm{T} = Union{Symmetric{T,<:BandedMatrix},Hermitian{T,<:BandedMatrix}}

"""
    detderivates(A, B, s::Real)

For a symmetric real or Hermtian matriices `A` and `B` calculate determinant of `A-sB`
resp. derived values:

κ: number of zeros of characteristic polynomial `p(x) = det(A - x *  B)` below `s`
η: `p'(s) / p(s)`
ζ: `p''(s) / p(s) - η^2`
"""
function detderivates(A::M, B::M, s::Real) where {T,M<:SymRealHerm{T}}
    k = max(bandwidth(A), bandwidth(B))
    W = mwiden(T)
    Q = zeros(W, k+1, k+1)
    Qp = zeros(T, k+1, k+1, 2)
    detderivates!(Q, Qp, A, B, s)
end
function detderivates!(Q, Qp, A::M, B::M, s) where {T,M<:SymRealHerm{T}}
    R = real(T)
    n = size(A, 1)
    κ = 0
    η = zero(R)
    ζ = zero(R)
    Q, Qp = initQ!( Q, Qp, A, B, s)
    for i = 1:n
        ξ, ξp, ξpp = R(Q[1,1]), Qp[1,1,1], Qp[1,1,2]
        κ += ξ < 0
        dξ = ξp / ξ
        η += dξ
        ζ += ξpp / ξ - dξ^2
        updateQ!(Q, Qp)
        stepQ!(Q, Qp, i, A, B, s)
    end
    κ, η, ζ
end

function initQ!(Q, Qp, A, B, s)
    k = size(Q, 1)
    for j = 1:k
        for i = j:k
            Q[i,j] = A[i,j] - B[i,j] * s
            Qp[i,j,1] = -B[i,j]
        end
    end
    Q, Qp
end

function stepQ!(Q, Qp, k, A, B, s)
    q = size(Q, 1)
    n = size(A, 1)
    for j = 1:q-1
        for i = j:q-1
            Q[i,j] = Q[i+1,j+1]
            Qp[i,j,1] = Qp[i+1,j+1,1]
            Qp[i,j,2] = Qp[i+1,j+1,2]
        end
        if k + q <= n
            Q[q,j] = A[k+q,k+j] - B[k+q,k+j] * s
            Qp[q,j,1] = -B[k+q,k+j]
            Qp[q,j,2] = 0
        end
    end
    if k + q <= n
        Q[q,q] = A[k+q,k+q] - B[k+q,k+q] * s
        Qp[q,q,1] = -B[k+q,k+q]
        Qp[q,q,2] = 0
    end
    nothing
end

function updateQ!(Q, Qp)
    q = size(Q, 1)
    b = Q[1,1]
    bp = Qp[1,1,1]
    bpp = Qp[1,1,2]
    for j = 2:q
        aj = Q[j,1]'
        apj = Qp[j,1,1]'
        appj = Qp[j,1,2]'
        for i = j:q
            ai = Q[i,1]
            api = Qp[i,1,1]
            appi = Qp[i,1,2]
            ajib = aj * ai / b
            Q[i,j] -= ajib 
            apjib = ( apj * ai + aj * api - ajib * bp ) / b
            Qp[i,j,1] -= apjib
            appjib = (apj * api - apjib * bp) * 2 + appj * ai + aj * appi - ajib * bpp 
            Qp[i,j,2] -= appjib / b
        end
    end
    nothing
end

# widen Float64 to Double64
mwiden(x::Type) = widen(x)
mwiden(x::Type{Float64}) = Double64
mwiden(x::Type{Complex{T}}) where T = Complex{mwiden(T)}
mwiden(x::T) where T = mwiden(T)(x)
