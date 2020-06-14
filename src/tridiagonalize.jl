
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
    td = make_tridiag!(B, Val('L'), d)
    dv = [real(B[i,i]) for i in 1:n]
    ev = [abs(B[i,i-1]) for i = 2:n]
    SymTridiagonal(dv, ev)
end

function make_tridiag!(A, ::Val{uplo}, d) where uplo
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square: dimensions are $((m, n))"))
    0 < d < n || throw(ArgumentError("number of superdiagonals $d not in 1:$(n-1)"))
    T = eltype(A)

    for bm = d:-1:2
        for k = 1:n-bm
            kp = k
            apiv = uplo == 'L' ? A[bm+kp, kp] : A[kp, bm+kp]
            iszero(apiv) && continue
            for i = bm+k-1:bm:n-1
                b = uplo == 'L' ? A[i,kp] : A[kp, i]
                c, s, r, α = givens2(b, apiv)
                u, v = A[i,i], A[i+1,i+1]
                upx = (u + v) / 2
                A[i,i] = (u - v) / 2
                if uplo == 'L'
                    A[i,kp], A[i+1,kp] = r, 0
                else
                    A[kp,i], A[kp,i+1] = r, 0
                end
                for j = kp+1:i
                    if uplo == 'L'
                        u = A[i,j]
                        v = T <: Real ? A[i+1,j] : A[i+1,j] * α
                        A[i,j], A[i+1,j] = u * c' + v * s', -u * s + v * c
                    else
                        u = A[j,i]
                        v = T <: Real ? A[j,i+1] : A[j,i+1] * α
                        A[j,i], A[j,i+1] = u * c' + v * s', -u * s + v * c
                    end
                end
                A[i+1,i+1] = T <: Real ? -A[i,i]' : -A[i,i]' * α
                ip = i + bm
                for j = i+1:min(ip, n)
                    if uplo == 'L'
                        u = A[j,i]
                        v = T <: Real ? A[j,i+1] : A[j,i+1] * α'
                        A[j,i], A[j,i+1] = u * c + v * s, -u * s' + v * c'
                    else
                        u = A[i,j]
                        v = T <: Real ? A[i+1,j] : A[i+1,j] * α'
                        A[i,j], A[i+1,j] = u * c + v * s, -u * s' + v * c'
                    end
                end
                r = real(A[i+1,i+1])
                A[i,i] = upx - r
                A[i+1,i+1] = upx + r
                if ip < n
                    if uplo == 'L'
                        v = T <: Real ? A[ip+1,i+1] : A[ip+1,i+1] * α'
                        A[ip+1,i+1] = v * c'
                    else
                        v = T <: Real ? A[i+1,ip+1] : A[i+1,ip+1] * α'
                        A[i+1,ip+1] = v * c'
                    end
                    apiv = v * s
                end
                kp = i
            end
        end
    end
    A
end

function givens2(a::T, b::T) where T <: AbstractFloat
    r = hypot(a, b)
    s = b / r
    c = a / r
    c, s, r, nothing
end
function givens2(a::T, b::T) where T <: Complex
    aa = abs(a)
    ba = abs(b)
    ra = hypot(aa, ba)
    s = ba / ra
    c = aa / ra
    ua = a / aa
    α = ua * (b' / ba)
    r = ua * ra
    c, s, r, α
end

copybands(A, d) = copybands!(zero(A), A, d)

function copybands!(B::AbstractMatrix, A::AbstractMatrix, d::Integer)
    m, n = size(A)
    for j = 1:n
        for i = j:min(j+d,m)
            B[i,j] = A[i,j]
        end
    end
    B
end
