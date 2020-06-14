
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
    ev = [real(B[i,i-1]) for i = 2:n]
    SymTridiagonal(dv, ev)
end

function make_tridiag!(A, ::Val{uplo}, d) where uplo
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square: dimensions are $((m, n))"))
    0 < d < n || throw(ArgumentError("number of superdiagonals $d not in 1:$(n-1)"))

    for bm = d:-1:2
        for k = 1:n-bm
            kp = k
            apiv = uplo == 'L' ? A[bm+kp, kp] : A[kp, bm+kp]
            for i = bm+k-1:bm:n-1
                b = uplo == 'L' ? A[i,kp] : A[kp, i]
                #c2, s2, r2 = LinearAlgebra.givensAlgorithm(b, apiv)
                # f = r2' / abs(r2); (c, s, r) = (c2', s2', r2') .* f'
                r = hypot(b, apiv)
                c = b / r
                s = apiv / r
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
                        u, v = A[i,j], A[i+1,j]
                        A[i,j], A[i+1,j] = u * c' + v * s', -u * s + v * c
                    else
                        u, v = A[j,i], A[j,i+1]
                        A[j,i], A[j,i+1] = u * c' + v * s', -u * s + v * c
                    end
                end
                A[i+1,i+1] = -A[i,i]'
                ip = i + bm
                for j = i+1:min(ip, n)
                    if uplo == 'L'
                        u, v = A[j,i], A[j,i+1]
                        A[j,i], A[j,i+1] = u * c + v * s, -u * s' + v * c'
                    else
                        u, v = A[i,j], A[i+1,j]
                        A[i,j], A[i+1,j] = u * c + v * s, -u * s' + v * c'
                    end
                end
                r = real(A[i+1,i+1])
                A[i,i] = upx - r
                A[i+1,i+1] = upx + r
                if ip < n
                    if uplo == 'L'
                        v = A[ip+1,i+1]
                        A[ip+1,i+1] = v * c'
                    else
                        v = A[i+1,ip+1]
                        A[i+1,ip+1] = v * c'
                    end
                    apiv = v * s
                end
                kp = i
            end
        end
    end
    if uplo == 'L'
        A[n,n-1] = abs(A[n,n-1])
    else
        A[n-1,n] = abs(A[n-1,n])
    end
    A
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
