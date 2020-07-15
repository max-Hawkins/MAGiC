"""
Guppi data processing algorithms for energy detection.
"""
module search
    using Statistics
    using BenchmarkTools
    using CUDA
    using Plots
    using Random
    using Main.Blio.GuppiRaw
    using Main.Blio.Filterbank

    "CUDA-defined limit on number of threads per block." #TODO: See if 2080Ti is different
    const global MAX_THREADS_PER_BLOCK = 1024

    "Load a Guppi RAW file and return the file and header variables."
    function load_guppi(fn)
        raw = open(fn)
        rh = GuppiRaw.Header()
        return raw, rh
    end

    "Read in the next Guppi block header and return the corresponding block data."
    function read_block_gr(raw, rh)
        read!(raw, rh)
        data = Array(rh)
        read!(raw, data)
        return data
    end

    "Compute the power of complex voltage data and return the spectrogram.
        If sum_pols is true and the complex data is polarized, the polarized
        powers are summed."
    function power_spec(complex_block, sum_pols=true)
        power_block = abs2.(Array{Complex{Int16}}(complex_block))

        if sum_pols && size(complex_block, 1) == 2
            power_block = sum(power_block[:,:,:], dims=1)
        end
        return power_block
    end

    "Calculate the kurtosis values for each coarse channel in a power spectrum.
        If by_chan is false, the statistics used when computing the kurtosis
        are calculated from the entire block of data."
    function kurtosis(power_block, by_chan=true)
        kurtosis_block = Array{Float32}(power_block)
        if !by_chan
            println("Calculating kurtosis - using CPU and block statistics")
            u = mean(power_block)
            s4 = stdm(power_block, u) ^ 4
            kurtosis_block = map(x->(x .- u) .^ 4 /  s4, power_block)
        else
            # TODO: had issues with mapping optimization, reverted to for loops. Check back later
            println("Calculating kurtosis by channel - using CPU")
            for pol = 1:size(power_block, 1)

                u = mean(power_block[pol,:,:], dims = 1)
                s4 = std(power_block[pol,:,:], dims = 1) .^ 4

                for i = 1:size(power_block, 3)
                    kurtosis_block[pol, :, i] = map(x->(x .- u[i]) .^ 4 /  s4[i], power_block[pol, :, i])
                end
            end
        end
        return kurtosis_block
    end

    # TODO: Put mean and std^4 in shared memory, figure out exponent problem
    function kurtosis_kernel(power_block, kurt_block, mean, std4, num_elements)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i > num_elements
            return
        end
        m_diff = power_block[i] - mean
        kurt = m_diff * m_diff * m_diff * m_diff / std4
        kurt_block[i] = kurt
        return
    end


    function power_spec_kernel(complex_block, power_block, nint)
        start_i = ((blockIdx().x - 1) * blockDim().x + threadIdx().x) * nint

        @cuprint("$(i)")
    end

    function power_spec_gpu(h_complex_block, nint)
	    npol, ntime, nchan  = size(h_complex_block)
        if npol == 1
            println("Complex data is not polarized. npols = $(npol)")
            return
        end
        if nint > ntime
            println("Cannot integrate over longer length than data block.")
            return
        end
        if ntime % nint != 0
            println("integration length $(nint) doesn't evenly divide ntime $(ntime). Not integrating")
            return
        end

        println("integration len: $(nint)  npol: $(npol)  ntime: $(ntime)  nchan: $(nchan)")
        println("max threads/block: $(MAX_THREADS_PER_BLOCK)")
        
        nthreads_per_chan = ntime / nint

        block_dim = (ceil(nthreads_per_chan / MAX_THREADS_PER_BLOCK), nchan)
        grid_dim  = (MAX_THREADS_PER_BLOCK)
        println("block_dim: $(block_dim)  grid_dim: $(grid_dim)")

        d_complex_block = CuArray{Int8}(h_complex_block)

        h_power_block = zeros{Int16}(1, nthreads_per_chan, nchan)
        d_power_block = CuArray(h_power_block)
            

        CUDA.@sync begin
            @cuda threads=block_dim blocks=grid_dim power_spec_kernel(d_complex_block, d_power_block, nint)
        end
        
        h_power_block = Array{Int16}(d_power_block)
        return h_power_block
    end

    # TODO: Include mean and std calculations into kernel, create indexing without flattening
    "Calculate the kurtosis of a power spectrum block using the GPU."
    function kurtosis_gpu(h_power_block)
        println("Calculating kurtosis - using GPU")
        h_power_size = size(h_power_block)
        nblocks = Int64(ceil(length(h_power_block) / MAX_THREADS_PER_BLOCK))
        h_power_block = reshape(h_power_block, (:))

        d_power_block = CuArray(h_power_block)
        h_kurtosis_block = similar(h_power_block, Float32)
        d_kurtosis_block = CuArray(h_kurtosis_block)

        u = mean(d_power_block)
        s4 = stdm(d_power_block, u) ^ 4
        
        CUDA.@sync begin
            @cuda threads=(MAX_THREADS_PER_BLOCK, 1, 1) blocks=(nblocks, 1, 1) kurtosis_kernel(d_power_block, d_kurtosis_block, u, s4, length(h_power_block))
        end
        h_kurtosis_block = Array(d_kurtosis_block)
        h_kurtosis_block = reshape(h_kurtosis_block, h_power_size)
        return h_kurtosis_block
    end

    "Create arrays with different distributions and plot the kurtosis values."
    function kurtosis_demo()
        N = 10000
        x = 1:N
        rand_a = rand(N,N)
        randn_a = randn(N,N)
        rande_a = randexp(N,N)
        

        rand_k = [mean(kurtosis(rand_a[i,:])) for i=1:N]
        randn_k = [mean(kurtosis(randn_a[i,:])) for i=1:N]
        rande_k = [mean(kurtosis(rande_a[i,:])) for i=1:N]

        p_rand_k = plot(x, rand_k, title="Kurtosis Values (N = $(N))", labels="Uniform ($(mean(rand_k)))")
        plot!(x, randn_k, title="Kurtosis Values", labels="Normal ($(mean(randn_k)))")
        plot!(x, rande_k, title="Kurtosis Values", labels="Exponential ($(mean(rande_k)))")

        println("Uniform kurtosis std: $(std(rand_k))")
        println("Normal kurtosis std:  $(std(randn_k))")
        println("Expo kurtosis std:    $(std(rande_k))")

        p_rand_h = histogram(vec(rand_a), title="Uniform Array Histogram")
        p_randn_h = histogram(vec(randn_a), title="Normal Array Histogram")
        p_rande_h = histogram(vec(rande_a), title="Expo Array Histogram")
        p = plot(p_randn_h, p_rand_h, p_rande_h, layout=(3,1), legend=false)
        display(p)
        display(p_rand_k)
        savefig(p, "array_histograms.png")
        savefig(p_rand_k, "kurtosis_plots.png")
        
    end

end
