"""
Guppi data processing algorithms for energy detection.
"""
module Search
    using Statistics
    using CUDA
    using Plots
    using Random
    using Printf
    using Main.Blio.GuppiRaw
    using Main.Blio.Filterbank

    "CUDA-defined limit on number of threads per block." #TODO: See if 2080Ti is different
    const global MAX_THREADS_PER_BLOCK = 1024

    "Load a Guppi RAW file and return the file and header variables."
    function load_guppi(fn::String)
        raw = open(fn)
        rh = GuppiRaw.Header()
        return raw, rh
    end

    "Read in the next Guppi block header and return the corresponding block data."
    function read_block_gr(raw, rh::GuppiRaw.Header)
        read!(raw, rh)
        data = Array(rh)
        read!(raw, data)
        return data
    end

    "Development function - do not use"
    function get_test_data(fn::String)
        raw, rh = Search.load_guppi(fn)
        complex = Search.read_block_gr(raw, rh)
        return complex
    end

    "Average over nint number of data points along the time axis for GUPPI data.
        Assumes the time dimension is the second dimension."
    function average(data, nint)
        ntime = size(data, 2)
        if ntime % nint != 0
            println("Number of time samples is not divisible by nint. Returning original data.")
            return data
        end

        data = reshape(data, (size(data,1), nint, Int(ntime / nint), size(data,3)))
        data = mean(data, dims=2)[:,1,:,:]
        return data
    end

    "Compute the power of complex voltage data and return the spectrogram.
        If sum_pols is true and the complex data is polarized, the polarized
        powers are summed."
    function power_spec(complex_block, avg_dim=-1, use_gpu=true, return_gpu_data=false)
        println("Calculating power spectrum")

        if use_gpu # TODO: Check for GPU availability
            println("Using GPU")
            complex_block = CuArray{Complex{Int32}}(complex_block)
            power_block   = CuArray{Int16}(undef, size(complex_block))            
        else
            complex_block = Array{Complex{Int32}}(complex_block)
        end

        power_block = abs2.(complex_block)

        if avg_dim > 0 && avg_dim <= ndims(complex_block) && size(complex_block, avg_dim) > 1
            println("Averaging along dim $avg_dim")
            power_block = mean(power_block, dims=avg_dim)
        end

        if use_gpu && !return_gpu_data
            power_block = Array(power_block)
        elseif use_gpu && return_gpu_data
            println("Returning GPU data")
        end

        return power_block
    end

    "Calculate population kurtosis taking advantage of Julia broadcasting.
        Consumes more memory than kurtosis_safe() that uses for loops."
    function kurtosis(a::Array; dims::Int=2)
        m = mean(a, dims=dims)
        aa = a .- m
        k = mean(aa .^ 4 ) ./ (mean(aa .^ 2, dims=dims) .^ 2)
        return k
    end

    "Calculate the population kurtosis values for each coarse channel in a power spectrum.
        If by_chan is false, the statistics used when computing the kurtosis
        are calculated from the entire block of data."
    function kurtosis_safe(power_block, by_chan=true)
        kurtosis_block = Array{Float32}(power_block)
        if !by_chan
            println("Calculating kurtosis - using CPU and block statistics")
            u = mean(power_block)
            s4 = stdm(power_block, u) ^ 4
            for pol = 1:size(power_block, 1)
                println("Pol $pol")
                for chan = 1:size(power_block, 3)
                    for samp = 1:size(power_block, 2)
                        kurtosis_block[pol, samp, chan] = (power_block[pol, samp, chan] - u) ^ 4 /  s4

                    end
                end
            end
        else
            # TODO: had issues with mapping optimization, reverted to for loops. Check back later
            println("Calculating kurtosis by channel - using CPU")

            u = mean(power_block, dims = 2)
            s4 = std(power_block, dims = 2) .^ 4

            for pol = 1:size(power_block, 1)
                println("Pol $pol")
                for chan = 1:size(power_block, 3)
                    for samp = 1:size(power_block, 2)
                        kurtosis_block[pol, samp, chan] = (power_block[pol, samp, chan] - u[pol,1,chan]) ^ 4 /  s4[pol,1,chan]

                    end
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

    # TODO: Include mean and std calculations into kernel, create indexing without flattening
    "Calculate the kurtosis of a power spectrum block using block statistics and the GPU."
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

    "Create the ticks and labels for frequency axes from GUPPI headers"
    function get_fticks(rh::GuppiRaw.Header, data_length, num_ticks=10)
        spec_flip = rh.obsbw < 0
        obsbw = spec_flip ? rh.obsbw : -1 * rh.obsbw

        start_f = rh.obsfreq - 0.5 * obsbw
        end_f   = rh.obsfreq + 0.5 * obsbw
        tick_width = obsbw / num_ticks
        labels = collect(start_f:tick_width:end_f)
        labels = [ @sprintf("%5.0f",x) for x in labels ]
        ticks = 0:data_length / num_ticks:data_length
        return ticks, labels, spec_flip
    end

    "Create the ticks and labels for time axes from GUPPI headers"
    function get_tticks(rh::GuppiRaw.Header, data_length, num_blocks=1, num_ticks=5)
        ntime = rh.blocsize / rh.npol / rh.nbin
        timespan = num_blocks * rh.tbin * ntime
        units = "s"

        if timespan < 1
            timespan /= 1000.0
            units = "ms"
        end
        println("ntime: $ntime  timespan: $timespan  Units: $units")

        tick_width = timespan / num_ticks
        labels = collect(0:tick_width:timespan)
        labels = [ @sprintf("%5.3f",x) for x in labels ]
        ticks = 0:data_length / num_ticks:data_length

        label = "Time ($units)"

        return ticks, labels, label 
    end

    # Plot and display/save/return t
    function plot_1d(block_data, rh::GuppiRaw.Header, type::String, disp=true, save=false)
        gr()
        show_legend = false
        num_pols = size(block_data, 1)
        if num_pols > 1
            show_legend = true
        end

        # Bandpass plotting
        if type=="b" || type=="bandpass"
            title_prefix = "Bandpass"
            save_prefix = "bandpass"
            ylab = "Power"
        # Kurtosis plotting with channel statistics
        elseif type=="k" || type=="kurtosis"
            title_prefix = "Kurtosis (chan stats)"
            save_prefix = "kurtosis_bychan"
            ylab = "Kurtosis"
        # Kurtosis plotting with block statistics
        elseif type=="kb" || type=="kurtosis_block"
            title_prefix = "Kurtosis (block stats)"
            save_prefix = "kurtosis_byblock"
            ylab = "Kurtosis"
        else
            println("Please enter a valid plot type string (b, k, or kb)")
            return nothing
        end
        ticks, labels, xinvert = get_fticks(rh, size(block_data,3))
        println("xflip: $xinvert")
        
        p = plot()
        for pol = 1:num_pols
            plot!(p, block_data[pol,1,:],
                    title="$title_prefix of $(rh.src_name) - Single Block",
                    legend=show_legend,
                    label = "Pol $pol",
                    xlabel="Frequency (MHz)",
                    ylabel=ylab,
                    xticks=(ticks, labels),
                    xflip=xinvert)
        end

        if disp
            println("Displaying plot")
            display(p)
        end
        if save
            save_fn = "$(save_prefix)_$(rh.src_name)_block$(lpad(rh.curblock, 3, '0')).png"
            println("Saveing plot to '$(save_fn)'")
            savefig(p,save_fn)
        end
        return p
    end

    "Plot 2D histogram data."
    function plot_2d(data, rh::GuppiRaw.Header, type::String)
        plotly()
        npol = size(data, 1)

        # Power plotting
        if type=="p" || type=="power"
            title_prefix = "Power"
            ylab = "Power"
            max_value = 3000
        # Kurtosis plotting with channel statistics
        elseif type=="k" || type=="kurtosis"
            title_prefix = "Kurtosis (chan stats)"
            ylab = "Kurtosis"
            max_value = 15
        # Kurtosis plotting with block statistics
        elseif type=="kb" || type=="kurtosis_block"
            title_prefix = "Kurtosis (block stats)"
            ylab = "Kurtosis"
            max_value = 15
        else
            println("Please enter a valid plot type string (p, k, or kb)")
            return nothing
        end

        yticks, ylabels, yinvert = get_fticks(rh, size(data,3))
        xticks, xlabels, xlabel = get_tticks(rh, size(data, 2))
        data = permutedims(data,[1,3,2])

        if npol == 1 
            heatmap(data[1,:,:], 
                title="$title_prefix of $(rh.src_name)",
                xlabel=xlabel,
                xticks=(xticks, xlabels),
                ylabel="Frequency (MHz)",
                yflip=yinvert,
                yticks=(yticks, ylabels),
                clim=(0,max_value))
        else
            # TODO: Plot polarized heatmaps vertically stacked with pol labels
        end
    end




    #----------------#
    # In Development #
    #----------------#


    function power_spec_kernel(complex_block, power_block, nint, ntime)
        pol = blockIdx().x
        samp = (blockIdx().y - 1) * blockDim().x + threadIdx().x
        chan = blockIdx().z
        
        if samp > ntime - 1
            return nothing
        end
        @inbounds real = complex_block[1, 1, 1]
        #@inbounds imag = complex_block[pol, samp, chan]
        @inbounds power_block[1, 1, 1] = real ^ 2 
           
        # if pol == 1 && chan==50 && (samp >= 1032700 || samp <= 1)
        #     @cuprint("Start: (pol=$pol, samp=$samp, chan=$chan) = \t
        #              Complex: $(real) Power: $(power_block[pol, samp, chan])\t
        #              Block = ($(blockIdx().x), $(blockIdx().y), $(blockIdx().z))\t
        #              Grid = ($(threadIdx().x), $(threadIdx().y), $(threadIdx().z))\n")
        # end
        
        return nothing
    end

    function power_spec_gpu(h_complex_block, nint=1)
        h_complex_block = reinterpret(Int8, h_complex_block)
        npol, ntime, nchan  = size(h_complex_block)

        if nint > ntime
            println("Cannot integrate over longer length than data block.")
            return
        end
        if ntime % nint != 0
            println("integration length $(nint) doesn't evenly divide ntime $(ntime). Not integrating")
            # TODO: Calculate next closest factor and use that averaging length
            return
        end

        println("integration len: $(nint)  npol: $(npol)  ntime: $(ntime)  nchan: $(nchan)")
        println("max threads/block: $(MAX_THREADS_PER_BLOCK)")
        
        nthreads_per_chan = ntime / nint

        block_dim = (Int(npol / 2), Int(ceil(nthreads_per_chan / MAX_THREADS_PER_BLOCK)), nchan)
        grid_dim  = (MAX_THREADS_PER_BLOCK, 1, 1)
        println("block_dim: $(block_dim)  grid_dim: $(grid_dim)")


        d_complex_block = CuArray{Int8}(h_complex_block)
        h_power_block = Array{UInt16}(undef, Int(npol / 2), Int(nthreads_per_chan), nchan)
        d_power_block = CuArray(h_power_block)

        println("Complex block size: $(size(d_complex_block))  Power block size: $(size(d_power_block))")

        CUDA.@sync begin
            @cuda threads=grid_dim blocks=block_dim power_spec_kernel(d_complex_block, d_power_block, nint, ntime)
        end
        
        h_power_block = Array{Int16}(d_power_block)
        return h_power_block
    end

    function test_kernel(data)

        @inbounds @cuprint("Data: $(data[threadIdx().x,blockIdx().y,blockIdx().x] ^ 2)\t
                     Block = ($(blockIdx().x), $(blockIdx().y), $(blockIdx().z))\t
                     Grid = ($(threadIdx().x), $(threadIdx().y), $(threadIdx().z))\n")
        return nothing
    end

    function gpu_test(data::AbstractArray)
        data = reinterpret(Int8, data)
        d_data = CuArray(data)

        @cuda threads=(4,1,1) blocks=(2,3,1) test_kernel(d_data)

    end
    

    "Calculate the kurtosis by channel averaged over an entire GUPPI file."
    function kurtosis(fn::String)
        println("Calculating entire Guppi Raw kurtosis")
        raw, rh = load_guppi(fn)
        complex_data = read_block_gr(raw, rh)
        i = 0
        kurtosis_blocks = zeros((size(complex_data, 3),128))
        println("Kurtosis block size: $(size(kurtosis_blocks))")

        while complex_data != -1
            i += 1
            println("Reading block $i")
            if i != 1
                try
                    complex_data = read_block_gr(raw, rh)
                catch
                    println("Error reading next block - skipping.")
                    break
                end
            end
            power_data = power_spec(complex_data)
            kurtosis_block = kurtosis(power_data)
            kurtosis_block = mean(kurtosis_block, dims = 2)
            kurtosis_blocks[:, i] = kurtosis_block
        end
        return kurtosis_blocks

    end

    

    "Calculate the kurtosis of complex samples.
        Probably not very useful."
    function kurtosis_complex(complex_block, bychan=false)
        kurtosis_block = Array{Complex{Float32}}(undef, (size(complex_block, 1),1,size(complex_block, 3)))

        u = mean(complex_block, dims=2)
        s4_real = std(real(complex_block), dims=2) .^ 4
        s4_imag = std(imag(complex_block), dims=2) .^ 4

        for pol = 1:size(complex_block, 1)
            println("Pol $pol")
            for i = 1:size(complex_block, 3)
                kurtosis_pol = Array{Complex{Float32}}(complex_block[pol,:,i])

                for samp = 1:size(complex_block, 2)
                    
                    kurtosis_real = (real(kurtosis_pol[samp]) - u[pol,1,i]) ^ 4 / s4_real[pol,1,i]
                    kurtosis_imag = (imag(kurtosis_pol[samp]) - u[pol,1,i]) ^ 4 / s4_imag[pol,1,i]

                    kurtosis_pol[samp] = kurtosis_real + (kurtosis_imag)im
                    
                    # real(pol_complex_chan) =  map(x->(x .- real(u[pol,1,i])) .^ 4 /  s4_real[pol,1,i], real(pol_complex_chan))
                    # imag(pol_complex_chan) =  map(x->(x .- imag(u[pol,1,i])) .^ 4 /  s4_imag[pol,1,i], imag(pol_complex_chan))
            
                    
                end
                kurtosis_block[pol,1,i] = mean(real(kurtosis_pol)) + (mean(imag(kurtosis_pol)))im
                
            end
        end

        return kurtosis_block
    end
end
