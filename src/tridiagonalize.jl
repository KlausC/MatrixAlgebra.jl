
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
    n = size(A, 1)
    B = copybands(A, d)
    td = make_tridiag!(B, d)
    dv = [real(B[i,i]) for i in 1:n]
    ev = [abs(B[i-1,i]) for i = 2:n]
    SymTridiagonal(dv, ev)
end

function make_tridiag!(A, d)
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square: dimensions are $((m, n))"))
    0 < d < n || throw(ArgumentError("number of superdiagonals $d not in 1:$(n-1)"))

    for bm = d:-1:2
        for k = 1:n-bm
            kp = k 
            apiv = A[kp, bm+kp]
            iszero(apiv) && continue
            for i = bm+k-1:bm:n-1
                b = A[kp, i]
                c, s, r, α = givens2(b, apiv)
                u, v = A[i,i], A[i+1,i+1]
                upx = (u + v) / 2
                A[i,i] = (u - v) / 2
                A[kp,i] = r
                for j = kp+1:i
                    u = A[j,i]
                    v = A[j,i+1] * α'
                    A[j,i], A[j,i+1] = u * c + v * s, -u * s + v * c
                end
                A[i+1,i+1] = -(A[i,i] * α)'
                ip = i + bm
                for j = i+1:min(ip, n)
                    u = A[i,j]
                    v = A[i+1,j] * α
                    A[i,j], A[i+1,j] = u * c + v * s, -u * s + v * c
                end
                w = real(A[i+1,i+1])
                A[i,i] = upx - w
                A[i+1,i+1] = upx + w
                if ip < n
                    v = A[i+1,ip+1] * α
                    apiv, A[i+1,ip+1] = v * s, v * c
                end
                kp = i
            end
        end
    end
    A
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

copybands(A, d) = copybands!(zero(A), A, d)

function copybands!(B::AbstractMatrix, A::AbstractMatrix, d::Integer)
    m, n = size(A)
    for j = 1:n
        for i = max(j-d,1):min(j+d,m)
            B[i,j] = A[i,j]
        end
    end
    B
end
