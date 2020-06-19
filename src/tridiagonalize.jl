
export tridiag_real
using LinearAlgebra

"""
    tridiag_real(A::AbstractMatrix, d::Integer)

The symmetric real or Hermitian input matrix `A` is transformed to a real
`SymTridiagonal` matrix.
For general matrices only `d` superdiagonals are processed.
Works efficiently for `BandedMatrices`.
"""
function tridiag_real(A::AbstractMatrix, d::Integer)
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square: dimensions are $((m, n))"))
    0 < d <= n || throw(ArgumentError("number of diagonals $d not in 1:$n"))

    B = copybands(A, d)
    td = make_tridiag!(B)
    dv = [real(B[i,1]) for i in 1:n]
    ev = d > 1 ? [abs(B[i,2]) for i = 2:n] : zeros(real(eltype(A)), n-1)
    SymTridiagonal(dv, ev)
end

# B[1:q,1:n] has the bands of a (2q-1)-banded hermitian nxn-matrix.
# superdiagonal i valid in [i+1:n].  
function make_tridiag!(B)
    n, d = size(B)
    0 < d <= n || throw(ArgumentError("number of diagonals $d not in 1:$n"))

    @inbounds for bm = d-1:-1:2
        for k = 1:n-bm
            kp = k 
            apiv = B[bm+kp,bm+1] 
            iszero(apiv) && continue
            for i = bm+k-1:bm:n-1
                b = B[i, i-kp+1]
                c, s, r, α = givens2(b, apiv)
                u, v = B[i,1], B[i+1,1]
                upx = (u + v) / 2
                B[i,1] = (u - v) / 2
                B[i,i-kp+1] = r
                for j = kp+1:i
                    u = B[i,i-j+1]
                    v = B[i+1,i-j+2] * α'
                    B[i,i-j+1], B[i+1,i-j+2] = u * c + v * s, -u * s + v * c
                end
                B[i+1,1] = -(B[i,1] * α)'
                ip = i + bm
                for j = i+1:min(ip, n)
                    u = B[j,j-i+1]
                    v = B[j,j-i] * α
                    B[j,j-i+1], B[j,j-i] = u * c + v * s, -u * s + v * c
                end
                w = real(B[i+1,1])
                B[i,1] = upx - w
                B[i+1,1] = upx + w
                if ip < n
                    v = B[ip+1,ip-i+1] * α
                    apiv, B[ip+1,ip-i+1] = v * s, v * c
                end
                kp = i
            end
        end
    end
    B
end

function givens2(a::T, b::T) where T <: AbstractFloat
    r = _hypot(a, b)
    s = b / r
    c = a / r
    c, s, r, one(T)
end
@inline function givens2(a::T, b::T) where T <: Complex
    aa = _abs(a)
    ba = _abs(b)
    ra = _hypot(aa, ba)
    s = ba / ra
    c = aa / ra
    ua = a / aa
    α = ua' * (b / ba)
    r = ua * ra
    c, s, r, α
end

# speedy versions of abs and hypot (in the common case)
function _abs(x::Complex)
    s = sqrt(abs2(real(x)) + abs2(imag(x)))
    _isclean(s) ? s : abs(x)
end
_abs(x) = abs(x)
function _hypot(x, y)
    s = sqrt(abs2(x) + abs2(y))
    _isclean(s) ? s : hypot(x, y)
end

function _isclean(a::T) where T<:AbstractFloat
    isfinite(a) &&
    a >= sqrt(2 * floatmin(T))
end

copybands(A, d) = copybands!(zeros(eltype(A), size(A,1), d), A)

function copybands!(B::AbstractMatrix, A::AbstractMatrix)
    n, d = size(B)
    for i = 1:d
        for j = i:n
            B[j,i] = A[j-i+1,j]
        end
    end
    B
end

