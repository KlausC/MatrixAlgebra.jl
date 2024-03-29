
function blockeig(A::AbstractMatrix{T}, s::Integer; rtol=1e-14, maxiter=1000) where {T}
    n = LinearAlgebra.checksquare(A)
    na = norm(A)
    V = randn(T, n, s)
    S = T[]
    err = Inf
    k = 0
    H = real(T)[]
    while err > rtol * na && k < maxiter
        k += 1
        B = A * V
        Q, R = qr(B)
        V = Q[:, 1:s]
        S = R[1:s, 1:s]
        ef = eigen(V' * A * V, sortby = x -> -abs(x))
        V = V * ef.vectors
        S = Diagonal(ef.values)
        err = norm(A * V - V * S)
        push!(H, err / na)
        println("k = $k err/|A| = $(err / na)")
    end
    Diagonal(diag(S)), V, err / na, H
end

function blocksvd(A::AbstractMatrix{T}, s::Integer; rtol=1e-14, maxiter=1000) where {T}
    m, n = size(A)
    na = norm(A)
    V = rand(T, n, s)
    U = T[]
    S = T[]
    err = Inf
    k = 0
    while err > rtol * na && k < maxiter
        k += 1
        B = A * V
        Q, R = qr(B)
        U = Q[:, 1:s]
        C = A' * U
        Q, R = qr(C)
        V = Q[:, 1:s]
        S = R[1:s, 1:s]
        err = norm(A * V - U * S)
        println("k = $k err/|A| = $(err / na)")
    end
    s = diag(S)
    LinearAlgebra.rmul!(V, Diagonal([x < 0 ? -1 : 1 for x in s]))
    Diagonal(abs.(s)), U, V, err
end
