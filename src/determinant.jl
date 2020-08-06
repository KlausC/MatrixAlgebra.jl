
export mwiden, detderivates, eigbounds, make_kappa, eigval


using BandedMatrices
using DoubleFloats

const SymRealHerm{T} = Union{Symmetric{T,<:BandedMatrix},Hermitian{T,<:BandedMatrix}}

"""
    detderivates(A, B, s::Real)

For a symmetric real or Hermtian matriices `A` and `B` calculate determinant of
`A - s * B` respectively related values:

`κ`: number of zeros of characteristic polynomial `p(x) = det(A - x *  B)` below `s`

`η`: `p'(s) / p(s)`

`ζ`: `p''(s) / p(s)`

!!! note
    For a zero of `p` with multplicity `r`,
    the Laguerre iteration formula is `s -= n / ( η + sqrt(((n-1)*η² - n*ζ) * (n-r)/r) )`.

    This iteration has far better area- and order of convergence than
    for example Newton's iteration `s -= 1 / η`.
"""
function detderivates(A::M, B::M, s::Real) where {T,M<:SymRealHerm{T}}
    k = min(max(bandwidth(A), bandwidth(B)) + 1, size(A, 1))
    W = mwiden(T)
    Q = zeros(W, k, k)
    Qp = zeros(T, k, k, 2)
    detderivates!(Q, Qp, A, B, s)
end
function detderivates!(Q, Qp, A::M, B::M, s) where {T,M<:SymRealHerm{T}}
    R = real(T)
    Z = zero(R)
    n = size(A, 1)
    ϵ = T(eps(real(T)))
    if isinf(s)
        return ifelse(s < 0, 0, n), Z, Z, Z, Z, Z, Z, Z
    end
    κ = 0
    dξp = dξpp = η = zero(R)
    ξ, ξp, ξpp = zero(R), zero(R), zero(R)
    ζs = zero(R)
    Q, Qp = initQ!( Q, Qp, A, B, s)
    for i = 1:n
        ξ, ξp, ξpp = R(real(Q[1,1])), real(Qp[1,1,1]), real(Qp[1,1,2])
        if iszero(ξ)
            ξ = ϵ
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
            dξpp = zero(R)
        end
        dζs = dξ2 - dξpp
        ζs += dζs
        if ζs < -η^2 / i
            # println("discriminant negative: i=$i ξ=$ξ $ξp $ξpp  η=$η ζs=$ζs")
            ζs = -η^2 / i
        end
        η += dξp
        if  i == n && ζs > 1.5e10 * η^2
            # println("discriminant too big: i=$i ξ=$ξ $ξp $ξpp  η=$η ζs=$ζs")
            # ζs = 1.5 * η^2
        end
        updateQ!(ξ, Q, Qp, i)
        stepQ!(Q, Qp, i, A, B, s)
    end
    κ, η, η^2 - ζs, dξp, dξpp, ξ, ξp, ξpp
end

function initQ!(Q, Qp, A, B, s)
    k = min(size(Q, 1), size(A, 1))
    W = eltype(Q)
    for j = 1:k
        for i = j:k
            Q[i,j] = W(A[i,j]) - W(B[i,j]) * s
            Qp[i,j,1] = -B[i,j]
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

# widen Float64 to Double64
mwiden(x::Type) = widen(x)
mwiden(x::Type{Float64}) = Float64 # Double64
mwiden(x::Type{Complex{T}}) where T = Complex{mwiden(T)}
mwiden(x::T) where T = mwiden(T)(x)

function make_kappa(A, B)
    x -> begin k, = detderivates(A, B, x); k end
end

# determine lower and upper bounds for eigenvalues number k1:k2.
# Bisectional method for function bif, where bif(x) is the number of eigenvalues
# less than or equal to x.
# scale is an (under-) estimation for the maximal abolute value of eigenvectors. It is used
# to control interval size growth for smallest and largest ev.
function eigbounds(k1::Int, k2::Int, bif::Function, scale::T; rtol=T(Inf), atol=T(Inf), rtolg=eps(T), atolg=rtolg^2) where T<:AbstractFloat
    n = k2 - k1 + 1
    lb = Vector{T}(undef, n)
    ub = Vector{T}(undef, n)
    eigbounds!(lb, ub, k1, k2, bif, scale, rtol, atol, rtolg, atolg)
end
function eigbounds!(lb::V, ub::V, k1::Int, k2::Int, bif, scale::T, rtol, atol, rtolg, atolg) where {T<:AbstractFloat,V<:AbstractVector{T}}
    n = k2 - k1 + 1
    length(lb) == n == length(ub) || throw(ArgumentError("length mismatch"))
    bif(T(Inf)) >= k2 || throw(ArgumentError("total number of eigenvalues must be >= $k2"))
    fill!(lb, -T(Inf))
    fill!(ub, T(Inf))
    scalep = abs(scale)
    scalen = - scalep
    count = 0
    if isinf(atol) != isinf(rtol)
        atol = ifelse(isinf(atol), zero(atol), atol)
        rtol = ifelse(isinf(rtol), zero(rtol), rtol)
    end
    # evaluate bif(x) and improve bounds according to result.
    function setx!(lb, ub, x)
        k = bif(x) - k1 + 1 
        count += 1
        for j =  max(k-k1+1, 0):n-1
            if x > lb[j+1]
                lb[j+1] = x
            else
                break
            end
        end
        for j = min(n, k):-1:1
            if x < ub[j]
                ub[j] = x
            else
                break
            end
        end
    end

    # find (first) index j with ub[j] - lb[j+1] > tol
    function findgap(rtol, atol)
        for j = 1:n-1
            dx = ub[j] - lb[j+1]
            if dx >= max(max(abs(ub[j]), lb[j+1]) * rtol, atol)
                return j
            end
        end
        return 0
    end

    # find (first) index j with ub[j] - lb[j+1] > tol
    function findbad(rtol, atol)
        for j = 1:n
            dx = ub[j] - lb[j]
            if abs(dx) > max(max(abs(ub[j]), lb[j]) * rtol, atol)
                return j
            end
        end
        return 0
    end

    # midpoint for finite a, b, otherwise extend towards infinity
    function midpoint(a, b)
        x = a + (b - a) / 2
        isfinite(x) && return x
        if isinf(a)
            isinf(b) && return zero(x)
            x = b + scalen
            scalen *= 2
        else
            x = a + scalep
            scalep *= 2
        end
        x
    end

    while isinf(lb[1])
        x = midpoint(lb[1], ub[1])
        setx!(lb, ub, x)
    end
    while isinf(ub[n])
        x = midpoint(lb[n], ub[n])
        setx!(lb, ub, x)
    end
    while ( j = findgap(rtolg, atolg) ) != 0
        x = midpoint(lb[j+1], ub[j])
        setx!(lb, ub, x)
    end
    while ( j = findbad(rtol, atol) ) != 0
        x = midpoint(lb[j], ub[j])
        setx!(lb, ub, x)
    end
    lb, ub, count
end

function laguerre(η::T, ζ::T, n::Int, r::Int) where T<:AbstractFloat
    disc = max(η^2 * (n -1) - ζ * n, 0)
    sq = sqrt(disc * (n-1) / r)
    n / ( η + copysign(sq, η) )
end

function eigval(A, B, k::Int, a::T, b::T, tol::T=eps(T), r= 1) where T<:AbstractFloat
    n = size(A, 1)
    x = a + (b - a) / 2
    while b - a > tol
        κ, η, ζ = detderivates(A, B, x)
        if κ < k
            a = x
        else
            b = x
        end
        if κ == k - 1 && η <= 0 || κ == k && η >= 0
            dx = -laguerre(η, ζ, n, r)
            x += dx
            if abs(dx) < tol
                break
            end
        else
            x = a + (b - a) / 2
        end
    end
    x
end

