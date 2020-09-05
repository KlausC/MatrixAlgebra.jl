
export detderivates

using BandedMatrices
using StaticArrays

const SymRealHerm{T} = Union{Symmetric{T,<:BandedMatrix},Hermitian{T,<:BandedMatrix}}
struct Workspace{T,M<:SymRealHerm{T},TQ,TQp} 
    A::M
    B::M
    Q::TQ
    Qp::TQp
end

"""
    detderivates(A, B, s::Real)

For a symmetric real or Hermtian matriices `A` and `B` calculate determinant of
`A - s * B` respectively related values:

`κ`: number of zeros of characteristic polynomial `p(x) = det(A - x *  B)` below `s`

`η`: `p'(s) / p(s)`

`ζ`: `-(p'(s) / p(s))' = (p'(s) / p(s))^2 - p''(s) / p(s) `

!!! note
    For a zero of `p` with multplicity `r`,
    the Laguerre iteration formula is `s -= n / ( η + sqrt((n * ζ - η^2) * (n-r)/r) )`.

    This iteration has far better area- and order of convergence than
    for example Newton's iteration `s -= 1 / η`.
"""
function detderivates(A::M, B::M, s::Real) where {T,M<:SymRealHerm{T}}
    ws = make_workspace(A, B)
    detderivates!(ws, s)
end
function detderivates!(ws::Workspace{T,M}, s) where {T,M<:SymRealHerm{T}}
    A, B, Q, Qp = ws.A, ws.B, ws.Q, ws.Qp
    R = real(T)
    Z = zero(R)
    n = size(A, 1)
    ϵ = T(eps(real(T)))
    if isinf(s)
        return ifelse(s < 0, 0, n), Z, Z, Z, Z, Z, Z, Z, Z, Z, Z
    end
    κ = 0
    dξp = dξpp = η = zero(R)
    ξ, ξp, ξpp = zero(R), zero(R), zero(R)
    ζ = zero(R)
    Q, Qp = initQ!(Q, Qp, A, B, s)
    ld = zero(R)
    for i = 1:n
        ξ, ξp, ξpp = R(real(Q[1,1])), real(Qp[1,1,1]), real(Qp[1,1,2])
        ld += log(abs(ξ))
        if iszero(ξ)
            ξ = ϵ^1.5
        end
        dξp = ξp / ξ
        dξpp = ξpp / ξ
        #println("i=$i ξ=$ξ dξp=$dξp dξpp = $dξpp")
        κ += ξ < 0
        dξ2 = dξp^2
        if abs(η + dξp) < abs(η) * 1e-5
            # println("Δη small: i=$i $(abs(η + dξp) / abs(η))")
        end
        if abs(dξpp) > dξ2 * 1000
            #dξpp = zero(R)
        end
        dζ = dξ2 - dξpp
        ζ += dζ
        if ζ < -η^2 / i
            # println("discriminant negative: i=$i ξ=$ξ $ξp $ξpp  η=$η ζ=$ζ")
            ζ = R(NaN) # -η^2 / i
        end
        η += dξp
        if  i == n && ζ > 1.5e10 * η^2
            # println("discriminant too big: i=$i ξ=$ξ $ξp $ξpp  η=$η ζ=$ζ")
            # ζ = 1.5 * η^2
        end
        updateQ!(ξ, Q, Qp, i)
        stepQ!(Q, Qp, i, A, B, s)
    end
    λ = laguerre1(η, ζ, n, 1)
    κ, η, ζ, ld, λ, dξp, dξpp, ξ, ξp, ξpp
end

function initQ!(Q, Qp, A, B, s)
    k = min(size(Q, 1), size(A, 1))
    W = eltype(Q)
    for j = 1:k
        for i = j:k
            Q[i,j] = W(A[i,j]) - W(B[i,j]) * s
            Qp[i,j,1] = -B[i,j]
            Qp[i,j,2] = 0
        end
    end
    Q, Qp
end

function stepQ!(Q, Qp, k, A, B, s)
    q = size(Q, 1)
    n = size(A, 1)
    W = eltype(Q)
    for j = 1:q-1
        for i = j:q-1
            Q[i,j] = Q[i+1,j+1]
            Qp[i,j,1] = Qp[i+1,j+1,1]
            Qp[i,j,2] = Qp[i+1,j+1,2]
        end
        if k + q <= n
            Q[q,j] = W(A[k+q,k+j]) - W(B[k+q,k+j]) * s
            Qp[q,j,1] = -B[k+q,k+j]
            Qp[q,j,2] = 0
        end
    end
    if k + q <= n
        Q[q,q] = W(A[k+q,k+q]) - W(B[k+q,k+q]) * s
        Qp[q,q,1] = -B[k+q,k+q]
        Qp[q,q,2] = 0
    end
    nothing
end

function updateQ!(ξ, Q, Qp, k)
    T = eltype(Qp)
    R = real(T)
    q = size(Q, 1)
    aa = real(ξ)
    a = R(aa)
    ap = real(Qp[1,1,1])
    app = real(Qp[1,1,2])
    bb = inv(aa)
    b = R(bb)
    bp = -ap * b * b
    bpp = -(app * b + 2 * ap * bp) * b
    for j = 2:q
        aaj = Q[j,1]'
        aj = T(aaj)
        apj = Qp[j,1,1]'
        appj = Qp[j,1,2]'
        aajb = aaj * bb
        ajb = aj * b
        apjb = apj * b + aj * bp
        appjb = appj * b + 2 * apj * bp + aj * bpp
        for i = j:q
            aai = Q[i,1]
            ai = T(aai)
            api = Qp[i,1,1]
            appi = Qp[i,1,2]
            aajib = aajb * aai
            ajib = ajb * ai
            aij = Q[i,j]
            Q[i,j] -= aajib
            apjib = apjb * ai + ajb * api
            Qp[i,j,1] -= apjib
            appjib = appjb * ai + 2*apjb * api + ajb * appi
            appji = Qp[i,j,2]
            if abs(appji - appjib) < 1e-8*max(abs(appji), abs(appjib))
                # println("inaccurate at $k($i, $j): $(abs(appji) \ (appji - appjib))")
            end
            Qp[i,j,2] -= appjib
        end
    end
    nothing
end

function make_workspace(A::M, B::M) where {T,M<:SymRealHerm{T}}
    k = min(max(bandwidth(A), bandwidth(B)) + 1, size(A, 1))
    if isbitstype(T) 
        Q = MMatrix{k,k}(zeros(T, k, k))
        Qp = MArray{Tuple{k,k,2}}(zeros(T, k, k, 2))
    else
        Q = zeros(T, k, k)
        Qp = zeros(T, k, k, 2)
    end
    Workspace(A, B, Q, Qp)
end

