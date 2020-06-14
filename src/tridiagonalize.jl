
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
    B = copybands(A, d)
    td = make_tridiag!(B)
    dv = [real(B[1,i]) for i in 1:n]
    ev = [abs(B[2,i]) for i = 2:n]
    SymTridiagonal(dv, ev)
end

# B[1:q,1:n] has the bands of a (2q-1)-banded hermitian nxn-matrix.
# superdiagonal i valid in [i+1:n].  
function make_tridiag!(B)
    d, n = size(B)
    1 < d <= n || throw(ArgumentError("number of diagonals $d not in 2:$n"))

    for bm = d-1:-1:2
        for k = 1:n-bm
            kp = k 
            apiv = B[bm+1,bm+kp] 
            iszero(apiv) && continue
            for i = bm+k-1:bm:n-1
                b = B[i-kp+1, i]
                c, s, r, α = givens2(b, apiv)
                u, v = B[1,i], B[1,i+1]
                upx = (u + v) / 2
                B[1,i] = (u - v) / 2
                B[i-kp+1,i] = r
                for j = kp+1:i
                    u = B[i-j+1,i]
                    v = B[i-j+2,i+1] * α'
                    B[i-j+1,i], B[i-j+2,i+1] = u * c + v * s, -u * s + v * c
                end
                B[1,i+1] = -(B[1,i] * α)'
                ip = i + bm
                for j = i+1:min(ip, n)
                    u = B[j-i+1,j]
                    v = B[j-i,j] * α
                    B[j-i+1,j], B[j-i,j] = u * c + v * s, -u * s + v * c
                end
                w = real(B[1,i+1])
                B[1,i] = upx - w
                B[1,i+1] = upx + w
                if ip < n
                    v = B[ip-i+1,ip+1] * α
                    apiv, B[ip-i+1,ip+1] = v * s, v * c
                end
                kp = i
            end
        end
    end
    B
end

@inline function givens2(a::T, b::T) where T <: AbstractFloat
    r = hypot(a, b)
    s = b / r
    c = a / r
    c, s, r, one(T)
end
@inline function givens2(a::T, b::T) where T <: Complex
    aa = abs(a)
    ba = abs(b)
    ra = hypot(aa, ba)
    s = ba / ra
    c = aa / ra
    ua = a / aa
    α = ua' * (b / ba)
    r = ua * ra
    c, s, r, α
end

copybands(A, d) = copybands!(zeros(eltype(A), d+1, size(A,1)), A)

function copybands!(B::AbstractMatrix, A::AbstractMatrix)
    d, n = size(B)
    for i = 1:d
        for j = i:n
            B[i,j] = A[j-i+1,j]
        end
    end
    B
end

