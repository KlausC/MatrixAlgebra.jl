
export mwiden, detderivates


using BandedMatrices
using DoubleFloats

const SymRealHerm{T} = Union{Symmetric{T,<:BandedMatrix},Hermitian{T,<:BandedMatrix}}

"""
    detderivates(A::AbstractMatrix)

For a symmetric real or Hermtian matrix `A` calculate determinatnt and derived values.
"""
function detderivates(A::M, B::M, s) where {T,M<:SymRealHerm{T}}
    R = real(T)
    n = size(A, 1)
    q = max(bandwidth(A), bandwidth(B))
    W = mwiden(T)
    a, ap, app, Q, Qp, Qpp = initQ!(q, W, T, A, B, s)
    
    κ = 0
    η = zero(R)
    ζs = zero(R)
    for k = 1:n
        ξ, ξp, ξpp = Q[1,1], Qp[1,1], Qpp[1,1]
        println("$k: $Q")
        κ += ξ < 0
        dξ = ξp / ξ
        η += dξ
        ζs += ξpp / ξ - dξ^2
        k == n && break
        update!(a, ap, app, Q, Qp, Qpp)
        stepQ!(k, a, ap, app, Q, Qp, Qpp, A, B, s)
    end
    κ, η, ζs + η^2, Q, Qp, Qpp
end

function initQ!(q, W, T, A, B, s)
    Q = zeros(W, q, q)
    Qp = zeros(T, q, q)
    Qpp = zeros(T, q, q)
    a = zeros(T, q+1)
    ap = zeros(T, q+1)
    app = zeros(T, q+1)
    a[q+1] = 1
    for j = 1:q
        for i = j:q
            Q[i,j] = A[i,j] - B[i,j] * s
            Qp[i,j] = -B[i,j]
        end
    end
    a, ap, app, Q, Qp, Qpp
end

function stepQ!(k, a, ap, app, Q, Qp, Qpp, A, B, s)
    q = size(Q, 1)
    n = size(A, 1)
    a[q+1] = Q[1,1]
    ap[q+1] = Qp[1,1]
    app[q+1] = Qpp[1,1]
    q = min(q, n - k)
    for j = 1:q-1
        a[j] = Q[j+1,1]
        ap[j] = Qp[j+1,1]
        app[j] = Qpp[j+1,1]
    end
    a[q] = A[k+q,k] - B[k+q,k] * s
    ap[q] = -B[k+q,k]
    a[q] = 0
    for j = 1:q-1
        for i = j:q-1
            Q[i,j] = Q[i+1,j+1]
            Qp[i,j] = Qp[i+1,j+1]
            Qpp[i,j] = Qpp[i+1,j+1]
        end
        Q[q,j] = A[k+q,k+j] - B[k+q,k+j] * s
        Qp[q,j] = -B[k+q,k+j]
        Qpp[q,j] = 0
    end
    Q[q,q] = A[k+q,k+q] - B[k+q,k+q] * s
    Qp[q,q] = -B[k+q,k+q]
    Q[q,q] = 0
    nothing
end

function update!(a, ap, app, Q, Qp, Qpp)
    q = size(Q, 1)
    b = a[q+1]
    bp = ap[q+1]
    bpp = app[q+1]
    for j = 1:q
        aj = a[j]'
        apj = ap[j]'
        appj = app[j]'
        for i = j:q
            ai = a[i]
            api = ap[i]
            appi = app[i]
            ajib = aj * ai / b
            Q[i,j] -= ajib 
            qp = ( apj * ai + aj * api - ajib * bp ) / b
            Qp[i,j] -= qp
            qpp = appj * ai + 2apj * api + aj * appi - qp * bp - ajib * bpp + qp * bp / b
            Qpp[i,j] -= qpp / b
        end
    end
    nothing
end

# widen Float64 to Double64
mwiden(x::Type) = widen(x)
mwiden(x::Type{Float64}) = Double64
mwiden(x::Type{Complex{T}}) where T = Complex{mwiden(T)}
mwiden(x::T) where T = mwiden(T)(x)
