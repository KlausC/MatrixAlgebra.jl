
export detderivates, eigbounds, make_kappa, eigval


using BandedMatrices
using DoubleFloats
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

`ζ`: `p''(s) / p(s)`

!!! note
    For a zero of `p` with multplicity `r`,
    the Laguerre iteration formula is `s -= n / ( η + sqrt(((n-1)*η² - n*ζ) * (n-r)/r) )`.

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
        return ifelse(s < 0, 0, n), Z, Z, Z, Z, Z, Z, Z, Z
    end
    κ = 0
    dξp = dξpp = η = zero(R)
    ξ, ξp, ξpp = zero(R), zero(R), zero(R)
    ζs = zero(R)
    Q, Qp = initQ!(Q, Qp, A, B, s)
    for i = 1:n
        ξ, ξp, ξpp = R(real(Q[1,1])), real(Qp[1,1,1]), real(Qp[1,1,2])
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
        dζs = dξ2 - dξpp
        ζs += dζs
        if ζs < -η^2 / i
            # println("discriminant negative: i=$i ξ=$ξ $ξp $ξpp  η=$η ζs=$ζs")
            ζs = R(NaN) # -η^2 / i
        end
        η += dξp
        if  i == n && ζs > 1.5e10 * η^2
            # println("discriminant too big: i=$i ξ=$ξ $ξp $ξpp  η=$η ζs=$ζs")
            # ζs = 1.5 * η^2
        end
        updateQ!(ξ, Q, Qp, i)
        stepQ!(Q, Qp, i, A, B, s)
    end
    ζ = η^2 - ζs
    λ = laguerre(η, ζ, n, 1)
    κ, η, ζ, λ, dξp, dξpp, ξ, ξp, ξpp
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

# determine lower and upper bounds for eigenvalues number k1:k2.
# Bisectional method for function bif, where bif(x) is the number of eigenvalues
# less than or equal to x.
# scale is an (under-) estimation for the maximal abolute value of eigenvectors. It is used
# to control interval size growth for smallest and largest ev.
function eigbounds(k1::Int, k2::Int, ws::Workspace, scale::T; rtol=T(Inf), atol=T(Inf), rtolg=eps(T), atolg=T(0)) where T<:AbstractFloat
    n = k2 - k1 + 1
    lb = Vector{T}(undef, n)
    ub = Vector{T}(undef, n)
    eigbounds!(lb, ub, k1, k2, ws, scale, rtol, atol, rtolg, atolg)
end
function eigbounds!(lb::V, ub::V, k1::Int, k2::Int, ws, scale::T, rtol, atol, rtolg, atolg) where {T<:AbstractFloat,V<:AbstractVector{T}}
    n = k2 - k1 + 1
    length(lb) == n == length(ub) || throw(ArgumentError("length mismatch"))
    fill!(lb, -T(Inf))
    fill!(ub, T(Inf))
    scalep = abs(scale)
    scalen = -abs(scale)
    if isinf(atol) != isinf(rtol)
        atol = ifelse(isinf(atol), zero(atol), atol)
        rtol = ifelse(isinf(rtol), zero(rtol), rtol)
    end
    # find (first) index j with ub[j] - lb[j+1] > tol
    function findgap(rtol, atol)
        for j = 1:n-1
            dx = ub[j] - lb[j+1]
            if dx >= max(( ub[j+1] - lb[j] ) * rtol, atol)
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
        a + (b - a) / 2
    end

    setx!(lb, ub, 0.0, k1, ws)
    while isinf(lb[1])
        x = ub[1] + scalen
        scalen *= 2
        setx!(lb, ub, x, k1, ws)
    end
    while isinf(ub[n])
        x = lb[n] + scalep
        scalep *= 2
        setx!(lb, ub, x, k1, ws)
    end
    while ( j = findgap(rtolg, atolg) ) != 0
        x = midpoint(lb[j+1], ub[j])
        setx!(lb, ub, x, k1, ws)
    end
    while ( j = findbad(rtol, atol) ) != 0
        x = midpoint(lb[j], ub[j])
        setx!(lb, ub, x, k1, ws)
    end
    lb, ub
end

# evaluate bif(x) and improve bounds according to result.
function setx!(lb, ub, x, k1, ws)
    n = length(lb)
    k, = detderivates!(ws, x)
    k = k - k1 + 1 
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

function laguerre1(η::T, ζ::T, n::Int, r) where T<:AbstractFloat
    disc = max(η^2 * (n - 1) - ζ * n, 0)
    sq = sqrt(disc * (n - r) / r)
    n / ( η + copysign(sq, η) )
end

function laguerre2(η::T, ζ::T, n::Int, r) where T<:AbstractFloat
    1 / ( η * (1 - ζ / η^2) )
end

function eigval(A, B, k::Int, a::T, b::T, r=1) where T<:AbstractFloat
    ws = make_workspace(A, B)
    eigval!(ws, k, a, b, r)
end

function eigval!(ws::Workspace{T}, k::Int, a::S, b::S, r, check=true) where {S,T}
    n = size(ws.A, 1)
    a, b = promote(a, b, zero(real(T)))
    κa = k - 1; κb = k
    if check
        κa, = detderivates!(ws, a)
        κb, = detderivates!(ws, b)
        κa < k <= κb || throw(ArgumentError("eigenvalue($k) not in [$a,$b]"))
        r = max(κb - κa, 1)    
        println("start: κa=$κa κb=$κb r=$r")
    end
    println("start: κa=$κa κb=$κb r=$r")
    x = a + (b - a) / 2
    x1 = real(T)(NaN)
    step = 0
    η0 = NaN
    while step < 100
        step += 1
        κ, η, ζ0 = detderivates!(ws, x)
        if κ < k
            a = x
            κa = κ
            r = max(κb - κa, 1)
        else
            b = x
            κb = κ
            r = max(κb - κa, 1)
        end
        println("$step: κ=$κ η=$η ζ=$ζ0 r=$r")
        x0 = x1
        x1 = x
        if κ < k && η <= 0 || κ >= k && η >= 0 # Newton points to ev[k]
            if isnan(η0)
                ζ = zero(η)
            else
                Δ = (inv(η0) - inv(η)) / (x0 - x)
                ζ = (1 - Δ) * η^2
            end
            if k - 1 <= κ <= k # 
                η0 = η
                dx = laguerre(η, ζ0, n, r)
                if dx * η >= 0.5
                    x = min(b, max(a, x1 - dx))
                    op = "Laguerre r=$r"
                elseif isnan(dx) && (dx = inv(η)) |> isfinite && abs(dx) <  (b - a) / 2
                    x = min(b, max(a, x1 - dx))
                    op = "Newton"
                else
                    x = a + (b - a) / 2
                    op = "bisect1"
                end
            else
                x = κ > k ? a + (b - a) / (κ - k + 1) : b - (b - a) / (k - κ)
                op = "division $κ / $k"
            end
        else
            x = a + (b - a) / 2
            op = "bisect2"
        end
        dx = x - x1
        println("step $step: $x $dx $op")
        convergence2(x, x1, x0) && break
    end
    x
end

"""
    convergence(x2, x1, x0)

Determines convergence of a series, given 3 succcessive elements.

returns `true`, if `x1 == x2`
returns `true`, if not monotonous and `|x2 - x1| >= |x1 - x0|`
returns `true`, if `|x2 - x1|^2 <= (|x1 - x0| - |x2 - x1|) * ϵ
returns `false` in all other cases
"""
function convergence2(x2, x1, x0)
    ϵ = eps(x2)
    dx2 = abs(x2 - x1)
    x1 == x2 && return true
    dx1 = abs(x1 - x0)
    dx2 >= dx1 && (x2 > x1) != (x1 > x0) && return true
    dx2^2 <= (dx1 - dx2) * ϵ && return true
    return false
end

