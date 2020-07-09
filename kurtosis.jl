module kurtosis_module
    using Statistics
    using BenchmarkTools
    using CUDA

    function kurtosis(a)
        u = mean(a)
        s = stdm(a, u)
        k = map(x->(x .- u) .^ 4 /  s .^ 4, a)
        return k
    end

    # TODO: Put mean and std^4 in shared memory, figure out exponent problem
    function kurtosis_gpu(power_block, kurt_block, mean, std4)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        m_diff = power_block[i] - mean
        kurt = m_diff * m_diff * m_diff * m_diff / std4
        kurt_block[i] = kurt
        return
    end

    function bench_kurtosis_gpu()
        d_pow, d_kurt, u, s4 = create_arrays()
        CUDA.@sync begin
            @cuda threads=(1024) blocks=(32288) kurtosis_gpu(d_pow, d_kurt, u, s4)
        end
        h_kurt = Array(d_kurt)
    end

    function create_arrays()
        sampPerChan = 516608 # number of samples in Voyager 1 guppi raw block
        nChan = 64
        N = sampPerChan * nChan
        h_power = rand(-128 : 127, N)
        h_power = convert(Array{Int8}, h_power)

        u = mean(h_power)
        s = stdm(h_power, u)
        s4 = s ^ 4

        

        # if time
        #     print("Mean")
        #     @btime u = mean(h_power)

        #     print("STD")
        #     @btime s = std(h_power)

        #     print("STDM")
        #     @btime sm = stdm(h_power, u)
        # end

        d_power = CuArray(h_power)
        d_kurt = CUDA.zeros(N)

        return d_power, d_kurt, u, s4
    end

    function main()
        
        print("N: ", sampPerChan * nChan, "\n")
        print("Mean")
        

        print("Map function")
        @btime k = map(x->(x .- u) .^ 4 /  s .^ 4, array)

        print("kurtosis")
        @btime kurtosis(array)
        # print(kurtosis(array))
    end

end