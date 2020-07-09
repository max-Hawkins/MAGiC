using Statistics
using BenchmarkTools
using CUDA

function kurtosis(a)
    u = mean(a)
    s = stdm(a, u)
    k = map(x->(x .- u) .^ 4 /  s .^ 4, a)
    return k
end

N = 516608 # number of samples in Voyager 1 guppi raw block
array = rand(N)
array = [12,13,54,56,25]

print("N: ", N, "\n")
print("Mean")
@btime u = mean(array)
u = mean(array)

print("STD")
@btime s = std(array)
s = std(array)

print("STDM")
@btime sm = stdm(array, u)
sm = stdm(array, u)
print(s,sm)

print("kurtosis")
@btime kurtosis(array)
print(kurtosis(array))

@btime kurtosis(CuArray(array))
