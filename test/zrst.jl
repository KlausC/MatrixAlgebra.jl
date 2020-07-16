
using BandedMatrices

function example(n)
    t = [ones(n)'*4; ones(n)']
    s = [2ones(n)'; ones(n)']
    tt = Symmetric(BandedMatrix(0=>t[1,:], -1=>t[2,1:end-1]), :L)
    ss = Symmetric(BandedMatrix(0=>s[1,:], -1=>s[2,1:end-1]), :L)
    t, s, tt, ss
end

@testset "bounds($n)" for n = [1,2,3,4,5,7,10,199]
    t, s, = example(n)
    lb00, ub00 = bounds(t, s)
    #lb01, ub01 = bounds(t, -s)
    lb10, ub10 = bounds(-t, s)
    #lb11, ub11 = bounds(-t, -s)
    @test lb00 <= ub00
    #@test lb00 == lb11 && ub00 == ub11
    #@test lb01 == lb10 && ub01 == ub10
    @test lb10 == -ub00 && ub10 == -lb00
end

