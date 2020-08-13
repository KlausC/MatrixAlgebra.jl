
export RatApp, f0, f10, f20

import Base: +, -, *, /, \, inv, conj, sqrt
import Base: zero, one, oneunit, promote_rule

struct RatApp{T<:Number} <: Number
    a::T
    b::T
    c::T
    d::T
    RatApp(a::T, b::T, c::T, d::T) where T<:Number = new{T}(a, b, c, d)
    RatApp{T}(a::T, b::T, c::T, d::T) where T<:Number = new{T}(a, b, c, d)
end

function RatApp{T}(a::T...) where T
    RatApp(a...)
end
function RatApp{T}(a...) where T
    RatApp{T}(convert.(T, a)...)
end
function RatApp(a::Number, b::Number, c::Number, d::Number)
    RatApp(promote(a, b, c, d)...)
end
function RatApp(a::Number, b::Number)
    RatApp(promote(a, b, oftype(a, 1), oftype(a, 0))...)
end
function RatApp(a::Number)
    RatApp(promote(a, oftype(a, 0), oftype(a, 1), oftype(a, 0))...)
end
function convert(R::Type{RatApp{T}}, x::RatApp{S}) where {T,S}
    R(convert(T, x.a), convert(T, x.b), convert(T, x.c), convert(T, x.d))
end

function promote_rule(A::Type{RatApp{T}}, B::Type{RatApp{S}}) where {S,T}
    RatApp{promote_type(T, S)}
end
function promote_rule(A::Type{RatApp{T}}, B::Type{S}) where {S,T}
    RatApp{promote_type(T, S)}
end

zero(::Type{R}) where {T,R<:RatApp{T}} = R(zero(T), zero(T), oneunit(T), zero(T))
oneunit(::Type{R}) where {T,R<:RatApp{T}} = R(oneunit(T), zero(T), oneunit(T), zero(T))
one(::Type{R}) where R<:RatApp = oneunit(R)

-(x::T) where T<:RatApp = T(-x.a, -x.b, x.c, x.d)
-(x::T, y::T) where T<:RatApp = normalized(minus(x, y))
+(x::T, y::T) where T<:RatApp = normalized(minus(x, y))
inv(x::T) where T<:RatApp = T(x.c, x.d, x.a, x.b)
*(x::T, y::T) where T<:RatApp = normalized(times(x, y))
/(x::T, y::T) where T<:RatApp = normalized(divideby(x, y))
conj(x::RatApp{<:Real}) = x
conj(x::R) where R<:RatApp = R(conj(x.a), conj(x.b), conj(x.c), conj(x.d))

function f0(x::T) where T<:RatApp
    x.a / x.c
end
function f10(x::T) where T<:RatApp
    (x.b * x.c - x.a * x.d) / x.a * x.c
end
function f20(x::T) where T<:RatApp
    -2 * f10(x) * x.d / x.c
end

function plus(x::T, y::T) where T<:RatApp
    a, b, c, d = x.a, x.b, x.c, x.d
    A, B, C, D = y.a, y.b, y.c, y.d
    aa = a * C + c * A
    bb = a * D + b * C + c * B + d * A
    cc = c * C
    dd = c * D + d * C
    T(aa, bb, cc, dd)
end
function minus(x::T, y::T) where T<:RatApp
    a, b, c, d = x.a, x.b, x.c, x.d
    A, B, C, D = y.a, y.b, y.c, y.d
    aa = a * C - c * A
    bb = a * D + b * C - c * B - d * A
    cc = c * C
    dd = c * D + d * C
    T(aa, bb, cc, dd)
end

function times(x::T, y::T) where T<:RatApp
    a, b, c, d = x.a, x.b, x.c, x.d
    A, B, C, D = y.a, y.b, y.c, y.d
    aa = a * A
    bb = a * B + b * A
    cc = c * C
    dd = c * D + d * C
    T(aa, bb, cc, dd)
end
function divideby(x::T, y::T) where T<:RatApp
    a, b, c, d = x.a, x.b, x.c, x.d
    A, B, C, D = y.c, y.d, y.a, y.b
    aa = a * A
    bb = a * B + b * A
    cc = c * C
    dd = c * D + d * C
    T(aa, bb, cc, dd)
end

function sqrt(x::R) where R<:RatApp
    aa = sqrt(x.a)
    bb = x.b / 2aa
    cc = sqrt(x.c)
    dd = x.d / 2cc
    R(aa, bb, cc, dd)
end

@inline function normalized(x::T) where T<:RatApp
    a, b, c, d = x.a, x.b, x.c, x.d
    n1 = max(abs(a), abs(b))
    n2 = max(abs(c), abs(d))
    n = max(n1, n2)
    if iszero(n)
        x
    elseif iszero(n1)
        T(a, b, one(a), a)
    elseif iszero(n2)
        T(one(c), c, c, c)
    elseif a * d == b * c
        if iszero(a)
            if abs(d) >= abs(b)
                T(b / d, zero(a), one(a), zero(a))
            else
                T(one(a), zero(a), d / b, zero(a))
            end
        else
            if abs(c) >= abs(a)
                T(a / c, zero(a), one(a), zero(a))
            else
                T(one(a), zero(a), c / a, zero(a))
            end
        end
    elseif min(n1, n2) > 2^20
        n = oftype(n, 2^20)
        if real(c) < 0 || c == 0 && d < 0
            n = -n
        end
        T(a / n, b / n, c / n, d / n)
    else
        x
    end
end

