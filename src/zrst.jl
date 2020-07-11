export detf, quad, laguerre, estimate_mult, convergence, starting_points!
export iteration_one

"""
    detf(t, s, x)
    
Determinant quotient and derivatives.
"""
function detf(t::Matrix, s::Matrix, x::T) where T<:Real
    n = size(t, 2)
    S = real(promote_type(eltype(t), eltype(s), T))
    ϵ = eps(S)^2
    ξ = oneunit(S)
    η = zero(S)
    ζ = zero(S)
    κ = 0
    if n == 0
        det, detexp = frexp(ξ) 
        return ξ, η, ζ, κ, det, detexp
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
    det, detexp = frexp(ξ)
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
        det, dde = frexp(det * ξ)
        detexp += dde
    end
    return ξ, η, ζ, κ, det, detexp
end


"""
    quad(a, b, x::Real)

`f(x) = abs2(a - b * x)` and `f'(x), f''(x)`
"""
function quad(a::T, b::T, x::Real) where T<:Union{Real,Complex}
    f = abs2(a - b * x)
    bb = abs2(b) * 2
    fp = bb * x - 2 * (real(a) * real(b) + imag(a) * imag(b))
    f, fp, bb
end


"""
    laguerre(t, s, x, r)

Calculate one iteration step of Laguerre's iteration method for polynomial roots.
The polynomial is given by `x -> det(t - s * x)`. 
"""
function laguerre(t::Matrix, s::Matrix, x::Real, r::Integer)
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

"""
    starting_points!(estev, t, s, a, b)

Select starting points for the eigenvalues of pencil(t, s) in interval `[a,b)`.
`estev` are the eigenvalues of the split system in the same interval, which have been
calculated accurately.
"""
function starting_points!(λ0::Vector{<:Real}, t::Matrix, s::Matrix, a::Real, b::Real)
    @assert issorted(λ0)
    @assert a <= λ0[1] <= λ0[end] < b
    κ(x) = detf(t, s, x)[4]
    c, d = λ0[1], λ0[end]
    k1 = κ(a) 
    p = κ(b) - k1   # the number of ev in [a,b)
    k2 = κ(c)
    m = κ(d) - k2   # the number of estev in [a,b)
    @assert k1 <= k2 <= k1 + 1
    @assert abs(p - m) <= 2
    if k2 > k1
        pushfirst!(λ0, a)
    end
    while size(λ0, 1) < p
        push!(λ0, b)
    end
    if size(λ0, 1) > p
        resize!(λ0, p)
    end
    λ0
end

"""
    interval(λ, j)

The interval for eigenvalue j in interval. Starting-point is the midpoint for inner
indices and one of the boundaries for first and last index.
"""
function interval(λ::Vector, j::Integer)
    p = size(λ, 1)
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
            ( mult > 1 && dκ > 1 ) || break
            mult = dκ
            x = (x + x1) / 2
        end
    end
    x
end

