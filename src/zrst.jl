export detf, quad, laguerre, estimate_mult, convergence, starting_points!
export iteration_one, merge12, merge2

"""
    detf(t, s, x)
    
Determinant quotient and derivatives.
"""
function detf(t::AbstractMatrix, s::AbstractMatrix, x::T) where T<:Real
    n = size(t, 2)
    S = real(promote_type(eltype(t), eltype(s), T))
    ϵ = eps(S)^2
    ξ = oneunit(S)
    η = zero(S)
    ζ = zero(S)
    κ = 0
    if n == 0
        return ξ, η, ζ, κ
    end
    ξ_1, η_1, ζ_1 = ξ, η, ζ
    tii, sii = real(t[1,1]), real(s[1,1])
    ξ = tii - sii * x
    if iszero(ξ)
        ξ = max(abs(tii), abs(sii * x), ξ_1) * ϵ
    end
    η = sii / ξ
    ζ = zero(S)
    κ += ξ < 0
    for i = 2:n
        η_0, ζ_0 = η_1, ζ_1
        ξ_1, η_1, ζ_1 = ξ, η, ζ
        tii, tim, sii, sim = real(t[1,i]), t[2,i-1], real(s[1,i]), s[2,i-1]
        f, fp, fpp = quad(tim, sim, x)
        tsi = tii - sii * x
        ξ = tsi - f / ξ_1
        if iszero(ξ)
            ξ = (abs(tim) + abs(sim) * abs(x))^2 * ϵ / ξ_1
        end
        η = (tsi * η_1 + sii - (f * η_0 - fp ) / ξ_1) / ξ
        ζ = (tsi * ζ_1 + 2 * sii * η_1 - (ζ_0 * f - 2 * η_0 * fp + fpp) / ξ_1) / ξ
        κ += (ξ < 0)
    end
    return ξ, η, ζ, κ
end

# real(a * b')
realprod(a::T, b::T) where T<:Real = a * b
realprod(a::T, b::T) where T<:Complex = real(a) * real(b) + imag(a) * imag(b)

"""
    quad(a, b, x::Real)

`f(x) = abs2(a - b * x)` and `f'(x), f''(x)`
"""
function quad(a::T, b::T, x::Real) where T<:Union{Real,Complex}
    f = abs2(a - b * x)
    bb = abs2(b) * 2
    fp = bb * x - realprod(a, b) * 2
    f, fp, bb
end


"""
    laguerre(t, s, x, r)

Calculate one iteration step of Laguerre's iteration method for polynomial roots.
The polynomial is given by `x -> det(t - s * x)`. 
"""
function laguerre(t::AbstractMatrix, s::AbstractMatrix, x::Real, r::Integer)
    n = size(t, 2)
    _, η, ζ, κ = detf(t, s, x)
    laguerre(η, ζ, n, r), κ
end
function laguerre(η::T, ζ::T, n::Integer, r::Integer) where T<:Real
    disc = (η^2 * (n-1) - ζ * n) * (n - r) / r
    @assert disc >= 0
    disc = copysign(sqrt(disc), η) 
    n / (η + disc)
end

"""
    estmlt

Estimate multiplicity of eigenvalue number j close to x
"""
function estimate_mult(x, sign::Integer, λ::Vector, j::Integer)
    n = size(λ, 1)
    1 <= j <= n || throw(ArgumentError("j = $j not in range 1:$n"))
    m = j
    lj = λ[j]
    tol = abs(x - lj) * 0.01
    mlt = 1
    while true
        m += sign
        1 <= m <= n || break
        abs(λ[m] - lj) >= tol && break 
        mlt += 1
    end
    mlt
end

function convergence(x2, x1, x0)
    ϵ = eps(x2)
    dx2 = abs(x2 - x1)
    dx2 <= ϵ && return true
    dx1 = abs(x1 - x0)
    dx2 >= dx1 && return true
    dx2^2 / (dx1 - dx2) <= ϵ && return true
    return false
end

κ(x, t, s) = detf(t, s, x)[4]

"""
    starting_points!(estev, t, s, a, b)

Select starting points for the eigenvalues of pencil(t, s) in interval `[a,b)`.
`estev` are the eigenvalues of the split system in the same interval, which have been
calculated accurately.
"""
function starting_points!(λ0::Vector{<:Real}, t, s, a::Real, b::Real)
    @assert issorted(λ0)
    @assert a <= λ0[1] <= λ0[end] < b

    m = length(λ0)  # number of ev in input data
    c, d = λ0[1], λ0[end]
    ka = κ(a, t, s)
    kb = κ(b, t, s)
    kc = κ(c, t, s)
    kd = κ(d, t, s)
    if ka < kc
        pushfirst!(λ0, a)
    end
    offseta = kc <= ka + 1
    if kb > kd
        push!(λ0, b)
    end
    offsetb = kd >= kb - 1
    @assert length(λ0) - offseta - offsetb == kb - ka
    λ0, ka, offseta, offsetb
end

"""
    interval(λ, j, i)

The interval for eigenvalue j in interval. Starting-point is the midpoint for inner
indices and one of the boundaries for first and last index.
"""
function interval(λ::Vector, j::Integer)
    n = size(λ, 1)
    λ[max(j-1, 1)], λ[min(j+1, n)]
end

"""
    iteration_one(a, x, b, j, n, r, t, s)

Iterate in interval `[a,b]` for the eigenvalue number `j` of `n`.
The eigenvalue number `j` with multiplicity `r` must be contained in [a, b].
Algorithm: First bisection loops until indication for Laguerre loops.
Test for convergence during Laguerre loop.
"""
function iteration_one(a::T, x::T, b::T, j::Integer, n::Integer, r::Integer, t, s) where T
    @assert a <= x <= b
    @assert 1 <= j <= n
    @assert 1 <= r
    mult = r

    η, ζ, κ = zero(x), zero(x), 0
    while a < b
        _, η, ζ, κ = detf(t, s, x)
        sig = κ < j 
        #println("iterone($a, $x, $b, $j) sig = $sig κ = $κ")
        if sig
            a = x
        else
            b = x
        end
        ( η > 0 ) == sig && ( κ == j || κ == j-1 ) && break
        x = (a + b) / 2
    end
    
    x1 = NaN
    while true
        κ1 = κ
        x0 = x1
        x1 = x
        dx = laguerre(η, ζ, n, mult)
        x = min(b, max(a, x1 + dx))
        convergence(x, x1, x0) && break
        while x != x1
            _, η, ζ, κ = detf(t, s, x)
            dκ = abs(κ - κ1) 
            ( mult <= 1 || dκ <= 1 ) && break
            mult = dκ
            x = (x + x1) / 2
        end
    end
    x
end

function pushres!(res, ev, a, b)
    if a <= ev < b
        push!(res, ev)
    end
    res
end

function merge12(t::AbstractMatrix{T}, s::AbstractMatrix{T}, a, b) where T
    n = size(t, 2)
    res = real(T)[]
    if n == 1
        ev = real(t[1,1]) / real(s[1,1])
        return pushres!(res, ev, a, b)
    elseif n == 2
        s11, s12, s22 = s[1,1], s[2,1], s[1,2]
        ss = s11 * s22 - abs2(s12)
        t11, t12, t22 = t[1,1], t[2,1], t[1,2]
        tt = t11 * t22 - abs2(t12)
        st = t11 * s22 + s11 * t22 - realprod(t12, s12) * 2
        iszero(ss) && return pushres!(res, tt / st, a, b)
        if iszero(tt)
            pushres!(res, T(0), a, b)
            return pushres!(res, st / ss, a, b)
        end
        u = st / ss / 2
        v = tt / ss
        d = u^2 - v
        d = sqrt(d)
        s = u + copysign(d, u)
        pushres!(res, s, a, b)
        pushres!(res, v / s, a, b)
    end
end

function merge2(t, s, a, b)
    n = size(t, 2)
    n <= 2 && return merge12(t, s, a, b)
    #println("merge($n)")
    T = real(eltype(t))
    n2 = (n + 1) ÷ 2
    res1 = merge2(view(t, :, 1:n2), s, a, b)
    res2 = merge2(view(t, :, n2+1:n), view(s, :, n2+1:n), a, b)
    μ = sort([res1; res2])

    μ, ka, offseta, offsetb = starting_points!(μ, t, s, a, b)
    p = length(μ) - offseta - offsetb
    λ = Vector{T}(undef, p)

    for i = 1 : p
        j = i + ka
        jj = i + offseta
        x = μ[jj]
        sig = κ(x, t, s) < j ? 1 : -1
        a, b = interval(μ, jj)
        mult = estimate_mult(x, sig, μ, jj)
        λ[i] = iteration_one(a, x, b, j, n, mult, t, s)    
        #println("merge($n): $(λ[i]) <- iterate ($a <= $x < $b) j = $j isig = $sig, mult = $mult" )
    end
    λ
end

