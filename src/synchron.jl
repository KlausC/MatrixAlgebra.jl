
import Base: iterate
import Base.Order.Forward

struct SIter{UA<:UnitRange,GA<:AbstractArray,UB<:UnitRange,GB<:AbstractArray}
    ra::UA
    ga::GA
    rb::UB
    gb::GB
end

Base.IteratorSize(::Type{<:SIter}) = Base.SizeUnknown()

function iterate(si::SIter)
    iterate(si, (first(si.ra), first(si.rb)))
end
function iterate(si::SIter, (ia, ib))
    ra, ga, rb, gb = si.ra, si.ga, si.rb, si.gb
    la = last(ra)
    lb = last(rb)
    while ia <= la && ib <= lb
        rga = ga[ia]
        rgb = gb[ib]
        rga == rgb && return (ia, ib), (ia+1, ib+1)
        if rga > rgb
            ib += 1
        else
            ia += 1
        end
    end
    nothing
end

function iterate_slower(si::SIter, (ia, ib))
    ra, ga, rb, gb = si.ra, si.ga, si.rb, si.gb
    la = last(ra)
    lb = last(rb)
    ia <= la && ib <= lb || return nothing
    rga = ga[ia]
    rgb = gb[ib]
    rga == rgb && return (ia, ib), (ia+1, ib+1)
    if rga > rgb
        ib = searchsortedfirst(gb, rga, ib+1, lb, Forward)
        ib <= lb || return nothing
        rgb = gb[ib]
        rgb == rga && return (ia, ib), (ia+1, ib+1)
    end
    while true
        ia = searchsortedfirst(ga, rgb, ia+1, la, Forward)
        ia <= la || return nothing
        rga = ga[ia]
        rgb == rga && return (ia, ib), (ia+1, ib+1)

        ib = searchsortedfirst(gb, rga, ib+1, lb, Forward)
        ib <= lb || return nothing
        rgb = gb[ib]
        rgb == rga && return (ia, ib), (ia+1, ib+1)
    end
    nothing
end

using SparseArrays
import SparseArrays: nzrange, rowvals, nonzeros, estimate_mulsize

SparseArrays.nzrange(A::SparseVector, j::Integer) = 1:(j == 1 ? length(A.nzind) : 0)
SparseArrays.rowvals(A::SparseVector) = A.nzind
SparseArrays.nonzeros(A::SparseVector) = A.nzval

function syncsum(f, A, i, B, j)
    si = MatrixAlgebra.SIter(nzrange(A, i), rowvals(A), nzrange(B, j), rowvals(B))
    nza = nonzeros(A)
    nzb = nonzeros(B)
    s = 0.0
    for (ia, ib) in si
        s += f(nza[ia]) * nzb[ib]
    end
    s
end

function spmulatb(f, A, B)
    mA, nA = size(A)
    mA == size(B, 1) || throw(DimensionMismatch())
    nB = size(B, 2)
    nnzC = min(max(estimate_mulsize(mA, nnz(A), nA, nnz(B), nB) * 11 ÷ 10, mA), nA*nB)
    Ti = promote_type(eltype(rowvals(A)), eltype(rowvals(B)))
    Tv = promote_type(eltype(nonzeros(A)), eltype(nonzeros(B)))
    colptrC = Vector{Ti}(undef, nB+1)
    rowvalC = Vector{Ti}(undef, nnzC)
    nzvalC = Vector{Tv}(undef, nnzC)
    nza = nonzeros(A)
    nzb = nonzeros(B)

    Z = f(zero(eltype(A)))*zero(eltype(B))
    ip = 1
    @inbounds begin
        colptrC[1] = ip
        for i = 1:nB
            if ip + nA - 1 > nnzC 
                nnzC += max(nA, nnzC>>2)
                resize!(rowvalC, nnzC)
                resize!(nzvalC, nnzC)
            end
            for j = 1:nA
                si = SIter(nzrange(A, j), rowvals(A), nzrange(B, i), rowvals(B))
                isempty(si) && continue
                s = Z
                for (ia, ib) in si
                    s += f(nza[ia]) * nzb[ib]
                end
                iszero(s) && continue
                rowvalC[ip] = j
                nzvalC[ip] = s
                ip += 1
            end
            colptrC[i+1] = ip
        end
        if ip < nnzC
            resize!(rowvalC, ip)
            resize!(nzvalC, ip)
        end
    end
    SparseMatrixCSC(nA, nB, colptrC, rowvalC, nzvalC)
end

