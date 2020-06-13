
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
    td = make_tridiag!(copy(A), d)
    SymTridiagonal(Symmetric(real(td), :L))
end

function make_tridiag!(A, d)
    n = size(A, 1)
    for bm = d:-1:2
        for k = 1:n-bm
            kp = k
            apiv = A[bm+kp, kp]
            #println("first piv A[$(bm+kp),$kp] = $apiv")
            for i = bm+k-1:bm:n-1
                b = A[i,kp]
                #c2, s2, r2 = LinearAlgebra.givensAlgorithm(b, apiv)
                # f = r2' / abs(r2); (c, s, r) = (c2', s2', r2') .* f'
                r = hypot(b, apiv)
                c = b / r
                s = apiv / r
                upx = (A[i,i] + A[i+1,i+1]) / 2
                umx = (A[i,i] - A[i+1,i+1]) / 2
                A[i,i] = umx
                A[i,kp] = r
                A[i+1,kp] = 0
                for j = kp+1:i
                    A[i,j], A[i+1,j] = A[i,j] * c' + A[i+1,j] * s', -A[i,j] * s + A[i+1,j] * c
                end
                A[i+1,i+1] = -A[i,i]'
                ip = i + bm
                for j = i+1:min(ip, n)
                    A[j,i], A[j,i+1] = A[j,i] * c + A[j,i+1] * s, -A[j,i] * s' + A[j,i+1] * c'
                end
                r = real(A[i+1,i+1])
                A[i,i] = upx - r
                A[i+1,i+1] = upx + r
                if ip < n
                    apiv, A[ip+1,i+1] = A[ip+1,i+1] * s, A[ip+1,i+1] * c'
                    #A[ip+1,i] = apiv
                    #println("piv A[$(ip+1),$(i+1)] = $apiv")
                end
                #println("bm = $bm i = $i k = $k ip = $ip")
                #display(A)
                kp = i
            end
        end
    end
    A[n,n-1] = abs(A[n,n-1])
    A
end
