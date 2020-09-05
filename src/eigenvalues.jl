
export eigbounds, eigval

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
    disc = max(ζ * n - η^2, 0)
    sq = sqrt(disc * (n - r) / r)
    n / ( η + copysign(sq, η) )
end

function laguerre2(η::T, ζ::T, n::Int, r) where T<:AbstractFloat
    η / ζ
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
                dx = laguerre1(η, ζ0, n, r)
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

