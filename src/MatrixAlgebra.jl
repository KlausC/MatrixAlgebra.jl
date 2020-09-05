module MatrixAlgebra


export MElement

using LinearAlgebra
using SparseArrays: iswrsparse

struct MElement{T,N,P,M} <: AbstractArray{T,N}
    data::P
    MElement(a::P) where {T,N,P<:AbstractArray{T,N}} = new{T,N,P,mode(P)}(a)
end
mode(P::Type{<:AbstractArray}) = iswrsparse(P) ? :sparse : :dense

struct MFactorization{T,P} <: Factorization{T}
    data::P
    MFactorization(p::P) where {T,P<:Factorization{T}} = new{T,P}(p)
end

import Base: ==, +, -, *
import Base: size, getindex
import LinearAlgebra: lu, ldiv!

size(a::MElement) = size(a.data)
getindex(a::MElement, i...) = getindex(a.data, i...)

-(a::MElement) = MElement(-data.a)
==(a::MElement, b::MElement) = MElement(a.data == b.data)
+(a::MElement, b::MElement) = MElement(a.data + b.data)
-(a::MElement, b::MElement) = MElement(a.data - b.data)
*(a::MElement, b::MElement) = MElement(a.data * b.data)



lu(a::MElement{T,2}) where T = MFactorization(lu(a.data))
ldiv!(a::MFactorization, b) = MElement(ldiv!(a.data, b))


include("synchron.jl")
include("tridiagonalize.jl")
include("toeplitz.jl")
include("givens.jl")
include("zrst.jl")
include("determinant.jl")
include("eigenvalues.jl")
include("rational.jl")

end # module
