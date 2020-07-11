
export Toeplitz, levinson, levinson!

struct Toeplitz{T<:AbstractFloat,S} <: AbstractMatrix{T}
    n::Int
    l::Int
    u::Int
    t::S # vector of length l + u + 1 < 2n
    f::Vector{T}
    b::Vector{T}
    function Toeplitz(n, l, u, t::AbstractVector)
        T = float(eltype(t))
        S = Vector{T}
        l <= u + 1 && -l + u + 1 <= 2n || throw(ArgumentError("l + u + 1 < 2n"))
        length(t) >= -l + u + 1 || throw(ArgumentError("vector insufficient size"))
        new{T,S}(n, l, u, t, Vector{T}(undef,0), Vector{T}(undef,0))
    end
end

function Base.getindex(A::Toeplitz{T}, i, j) where T
    A.l <= j - i <= A.u ? getindex(A.t, j-i-A.l+1) : zero(T)
end
Base.size(A::Toeplitz) = (A.n, A.n)


function levinson!(A::Toeplitz{T}) where T
    n = A.n
    if length(A.f) == 0
        resize!(A.f, n*(n+1)÷2)
        resize!(A.b, n*(n+1)÷2)
        fill!(A.f, zero(T))
        fill!(A.b, zero(T))

        toeplitz_forward(A, 1)[1] = toeplitz_backward(A, 1)[1] = inv(float(A[1,1]))
        for i = 2:n
            fi = toeplitz_forward(A, i)
            fi1 = toeplitz_forward(A, i-1)
            bi = toeplitz_backward(A, i)
            bi1 = toeplitz_backward(A, i-1)
            efi = sum(A[i,j] * fi1[j] for j = 1:i-1)
            ebi = sum(A[1,j+1] * bi1[j] for j = 1:i-1)
            ebf = inv(one(T) - efi*ebi)
            fi[1] = fi1[1] * ebf
            fi[2:i-1] = fi1[2:i-1] * ebf - bi1[1:i-2] * efi * ebf
            fi[i] = -bi1[i-1] * efi * ebf
            bi[1] = -fi1[1] * ebi * ebf
            bi[2:i-1] = bi1[1:i-2] * ebf - fi1[2:i-1] * ebi * ebf
            bi[i] = bi1[i-1] * ebf
        end
    end
    A
end

toeplitz_forward(A::Toeplitz, i) = view(A.f, i*(i-1)÷2+1:i*(i+1)÷2)
toeplitz_backward(A::Toeplitz, i) = view(A.b, i*(i-1)÷2+1:i*(i+1)÷2)


function levinson(A::Toeplitz{T}, B::AbstractArray{T,N}) where {T,N}
    N <= 2 || throw(ArgumentError("rhs must be vector or matrix"))
    n = size(B,1)
    n == A.n || throw(DimensionMismatch("rhs not correct size"))
    r = N == 2 ? size(B, 2) : 1
    x = N == 1 ? Vector{T}(undef, n) : Matrix{T}(undef, n, n)
    for k = 1:r
        x[1,k] = A[1,1] \ B[1,k]
        for i = 2:n
            bi = toeplitz_backward(A, i)
            exi = sum(A[i,j] * x[j,k] for j = 1:i-1)
            x[1:i,k] += bi * (B[i,k] - exi)
        end
    end
    x
end

