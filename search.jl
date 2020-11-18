"""
Data processing algorithms for energy detection.
NOTE: Most functions found here are in development and may not be usable.
""" 
module Search
    using Statistics
    using BenchmarkTools
    using CUDA
    using Plots
    using Random
    using Printf
    # For spectral kurtosis calculations
    using Roots
    using SpecialFunctions
    using SimpleRoots
    using NLsolve

    using Blio.GuppiRaw
    using Blio.Filterbank

    using Main.Hashpipe

    # Enum type for the current MeerKAT data format to help with indexing
    @enum MK_DIMS::Int8 D_POL=1 D_TIME D_CHAN D_ANT

    "CUDA-defined limit on number of threads per block." #TODO: See if 2080Ti is different
    const global MAX_THREADS_PER_BLOCK = 1024

    function pin_databuf_mem(db, bytes=-1)
        if(bytes==-1) # If bytes not specified, use databuf block size (may be incorrect)
            bytes = db.block_size
        end

        hp_databuf = unsafe_wrap(Array{Main.Hashpipe.hashpipe_databuf_t}, db.p_hpguppi_db, (1))[1];
        println("Pinning $bytes of Memory:")
        for i in 1:hp_databuf.n_block
            println("Block: $i")
            # Get correct buffer size from databuf!
            CUDA.Mem.register(CUDA.Mem.HostBuffer,db.blocks[i].p_data , bytes)
        end
    end


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

    """
    Compute the power of complex voltage data and return the spectrogram.
    If sum_pols is true and the complex data is polarized, the polarized powers are summed.
    """
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

    """
    Calculate the population kurtosis values for each coarse channel in a power spectrum.
    If by_chan is false, the statistics used when computing the kurtosis
    are calculated from the entire block of data.
    """
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

    

    """
    Calculate the kurtosis of complex samples.
    Probably not very useful. Distribution does not change much for voltages.
    """
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

    
    #-------------------#
    # Spectral Kurtosis #
    # Ref: Gelu Nita    #
    #-------------------#
    """
    Returns the spectral kurtosis values of an array of data.
    This spectral kurtosis algorithm is based off Gelu Nita's
    2010 paper: #TODO: Link
    """
    function spectral_kurtosis(power_array, p2_array, nints::Int, dims=2) #TODO: implement variations of dim - probably wouldn't be used
        p_size = size(power_array)
        if p_size[dims] % nints != 0
            println("Selected dimension $dims is not divisible by nints $nints.")
            nints -= p_size[dims] % nints
            if(nints < 1 || nints > p_size[dims])
                println("Could not calculate usable nints. Setting nints to 1.")
                nints = 1
            end 
            println("New nints: $nints")
        end
        new_dim_size = Int(p_size[dims] / nints)

        sum_p  = dropdims(sum(reshape(power_array,      (p_size[1], nints, new_dim_size, p_size[3], p_size[4])); dims=2), dims=2)
        sum_p2 = dropdims(sum(reshape(p2_array, (p_size[1], nints, new_dim_size, p_size[3], p_size[4])); dims=2), dims=2)
        sk = @. ((nints + 1)/(nints - 1)) * ((nints * sum_p2)/(sum_p ^ 2) - 1)
        return sk
    end
    
    """
    Calculate the upper root for finding the upper SK threshold given an integration length and p.
    """
    function upperRoot(x, m2, m3, p)
        if (-(m3-2*m2^2)/m3+x)/(m3/2/m2) < 0
            return 0
        end
        println((-(m3-2*m2^2)/m3+x)/(m3/2/m2))
        return abs((1 - gamma_inc( (4 * m2^3)/m3^2, (-(m3-2*m2^2)/m3 + x)/(m3/2/m2), 1)[1]) -p)
    end

    """
    Calculate the lower root for finding the lower SK threshold given an integration length and p.
    """
    function lowerRoot(x, m2, m3, p)
        println("lower x: $x")
        if (-(m3-2*m2^2)/m3+x)/(m3/2/m2) < 0
            println("lower return 0")
            return 0
        end
        println((-(m3-2*m2^2)/m3+x)/(m3/2/m2))
        lower = abs(gamma_inc( (4 * m2^3)/m3^2, (-(m3-2*m2^2)/m3 + x)/(m3/2/m2), 0)[1] -p)
        println("Lower: $lower")
        return lower
    end

    """
    calc_sk_thresholds

    Calculates the asymmetric spectral kurtosis thresholds given an integration length.
    Adapted from Gelu Nita's IDL code (https://github.com/Gelu-Nita/GSK/blob/master/gsk.pro) 
    and Nick Joslyn's Python code (https://github.com/NickJoslyn/helpful-BL/edit/master/helpful_BL_programs.py)
    """
    function calc_sk_thresholds(M, N=1, d=1, p=0.0013499)
        Nd = N * d
        if((p==0.0013499) && ((expo = log(2,M)) % 1 == 0))
            # Create look-up for sk_thresholds when p=0.0013499
            # Index: 2^i integration length (Remember: 1-indexed in Julia)
            sk_thresholds_p0013499 = [-0.9491207671598745 4.451577708347232;
                0 0;
                0 0;
                0 0;
                0 0;
                0.6007695589928923 2.090134929397901;
                0.660792564715807 1.7174329413624772;
                0.7249537831632763 1.4733902568788164;
                0.7859397345336379 1.3156809453381224;
                0.8383281718532006 1.2131125067280124;
                0.880396432714329 1.1454837933406543;
                0.9127540558562938 1.1002263183494991;
                0.9369638151287253 1.069536473556811;
                0.9547530883154925 1.0484995859651514;
                0.9676684770523931 1.0339584931405572;
                0.9769695436055237 1.0238440998251523;]
            return Tuple(sk_thresholds_p0013499[Int(expo), :])
        end
        # Moment Calculations
        m_1 = 1
        m2 = (2*(M^2) * Nd * (1 + Nd) ) / ( (M - 1) * (6 + 5*M*Nd + (M^2)*(Nd^2)) )
        m3 = (8*(M^3)*Nd * (1 + Nd) * (-2 + Nd * (-5 + M * (4+Nd)))) / (((M-1)^2) * (2+M*Nd) *(3+M*Nd)*(4+M*Nd)*(5+M*Nd))
        m4 = (12*(M^4)*Nd*(1+Nd)*(24+Nd*(48+84*Nd+M*(-32+Nd*(-245-93*Nd+M*(125+Nd*(68+M+(3+M)*Nd))))))) / (((M-1)^3)*(2+M*Nd)*(3+M*Nd)*(4+M*Nd)*(5+M*Nd)*(6+M*Nd)*(7+M*Nd) )

        upperThreshold = secant_method(x->upperRoot(x,m2,m3,p), 1, atol=1e-8, rtol=1e-8, maxevals=1000)
        lowerThreshold = secant_method(x->lowerRoot(x,m2,m3,p), 1, atol=1e-4, rtol=1e-12, maxevals=1000)
        #lowerThreshold = SimpleRoots.findzero(x->lowerRoot(x,m2,m3,p), [0,1])
        #lowerThreshold = Roots.fzeros(x->lowerRoot(x,m2,m3,p), [0,100], rtol=1e-8, maxevals=1000)
        #lowerThreshold = Roots.find_zero(x->lowerRoot(x, m2, m3, p), 1, rtol=1e-8, maxevals=1000)
        return lowerThreshold, upperThreshold
    end

    # Data for each spectral kurtosis array
    mutable struct sk_array_t
        sk_data_gpu::CuArray
        nint::Int
        sk_low_lim::Float32
        sk_up_lim::Float32
    end
    # Data necessary for calculating spectral kurtosis on GPU
    mutable struct sk_plan_t
        complex_data_gpu::CuArray
        power_gpu::CuArray
        power2_gpu::CuArray
        sk_arrays::Array{sk_array_t}
        sk_thresh::Float32
        sk_pizazz_gpu::CuArray
    end

    # Create high-level structs containing information needed during real-time SK processing
    # Allocates GPU memory as well
    function create_sk_plan(raw_eltype, raw_size, nint_array, sk_thresh; dims=2, t_min=0.001)
        complex_eltype = raw_eltype
        complex_size = raw_size # TODO: Calculate using raw header
        power_eltype  = Int32 # Allows for long summation
        power2_eltype = Int32 # Allows for long summation

        # TODO: Calculate nint_min based off t_min
        nint_min = 256

        # Allocate space for complex data on GPU and wrap with CuArray
        p_buf_complex_gpu = CUDA.Mem.alloc(CUDA.Mem.DeviceBuffer, prod(complex_size)*sizeof(complex_eltype))
        # Need to convert because by default CUDA.Mem.alloc creates CuPtr{nothing}
        p_complex_gpu = convert(CuPtr{complex_eltype}, p_buf_complex_gpu)
        # wrap data for later easier use
        complex_gpu = unsafe_wrap(CuArray, p_complex_gpu, complex_size)
        
        # Allocate space for intermediate power sk_arrays
        # Power array
        p_buf_power_gpu = CUDA.Mem.alloc(CUDA.Mem.DeviceBuffer, prod(complex_size)*sizeof(power_eltype))
        p_power_gpu = convert(CuPtr{power_eltype}, p_buf_power_gpu)
        power_gpu = unsafe_wrap(CuArray, p_power_gpu, complex_size)
        # Power squared array
        p_buf_power2_gpu = CUDA.Mem.alloc(CUDA.Mem.DeviceBuffer, prod(complex_size)*sizeof(power2_eltype))
        p_power2_gpu = convert(CuPtr{power2_eltype}, p_buf_power2_gpu)
        power2_gpu = unsafe_wrap(CuArray, p_power2_gpu, complex_size)

        #Pizazz Array allocation
		sk_pizazz_size = (1, Int(floor(complex_size[2] / nint_min)), complex_size[3], 1) # TODO: change to not use int and floor
		println(sk_pizazz_size)
		sk_pizazz_type = Float16
		
		# Allocate memory for SK pizazz score
		p_buf_sk_pizazz_gpu = CUDA.Mem.alloc(CUDA.Mem.DeviceBuffer, prod(sk_pizazz_size)*sizeof(sk_pizazz_type))
		p_sk_pizazz_gpu = convert(CuPtr{sk_pizazz_type}, p_buf_sk_pizazz_gpu)
        # wrap data for later easier use
        sk_pizazz_gpu = unsafe_wrap(CuArray, p_sk_pizazz_gpu, sk_pizazz_size)

        # Create sk_array_t for each
        sk_arrays = []
        for nint in nint_array
            sk_eltype = Float16
            sk_size = (complex_size[1],
                        Int(floor(complex_size[dims] / nint)),
                        complex_size[3],
                        complex_size[4])  # TODO, WARNING: NOT dims variable

            p_buf_sk_array_gpu = CUDA.Mem.alloc(CUDA.Mem.DeviceBuffer, prod(sk_size)*sizeof(sk_eltype))
            p_sk_array_gpu = convert(CuPtr{sk_eltype}, p_buf_sk_array_gpu)
            sk_data_gpu = unsafe_wrap(CuArray, p_sk_array_gpu, sk_size)

            low_lim, up_lim = calc_sk_thresholds(nint)
            sk_array = sk_array_t(sk_data_gpu,
                        nint,
                        low_lim,
                        up_lim)
            push!(sk_arrays, sk_array)
        end
        # Create sk_plan struct
        sk_plan = sk_plan_t(complex_gpu,
                        power_gpu,
                        power2_gpu,
                        sk_arrays,
                        sk_thresh,
                        sk_pizazz_gpu)
        return sk_plan
    end

    function exec_plan(plan, data_ptr)	
		# Transfer raw data to GPU
        unsafe_copyto!(plan.complex_data_gpu.ptr,
                        Ptr{eltype(plan.complex_data_gpu)}(data_ptr),
                        length(plan.complex_data_gpu))
		
		tic = time_ns() # To calculate execution time without transfer
				
		# Populate power and power-squared arrays
		@. plan.power_gpu = abs2(Complex{Int16}.(plan.complex_data_gpu))
		@. plan.power2_gpu = plan.power_gpu ^ 2
		
		for sk_array in plan.sk_arrays
			#println("=============")
			#display(sk_array)		
			sk_array.sk_data_gpu = Search.spectral_kurtosis(plan.power_gpu, plan.power2_gpu, sk_array.nint) # Unoptimized!!! TODO: Sum power/power2 as nint increases
		end
        
        toc = time_ns()
		#println("Time withouth transfer: $((toc-tic) / 1E9)")
    end

    function create_hit_info(plan)
        hits = findall(>(plan.sk_thresh), plan.sk_pizazz_gpu)
        return hits
    end
    
    #~3ms per integration length on GUPPI Block
    # Return pizazz matrix
    function hit_mask(plan)
		complex_size = size(plan.complex_data_gpu)

		low_pizazz_coef = 25.0
		up_pizazz_coef  = 0.5
		
		max_pizazz = (low_pizazz_coef + up_pizazz_coef) * complex_size[1] * complex_size[4] * length(plan.sk_arrays)
		
		for sk_array in plan.sk_arrays
			#println("i: $(size(sk_array.sk_data_gpu))")
			
			pizazz_up = sum(sum(sk_array.sk_data_gpu .> sk_array.sk_up_lim, dims=1), dims = 4) .* up_pizazz_coef;
			pizazz_low = sum(sum(sk_array.sk_data_gpu .< sk_array.sk_low_lim, dims=1), dims = 4) .* low_pizazz_coef;
			
			if sk_array == plan.sk_arrays[1]
				plan.sk_pizazz_gpu .= pizazz_up .+ pizazz_low
				
			else 
				total_pizazz = pizazz_up .+ pizazz_low
				
				temp_n_time = size(sk_array.sk_data_gpu,2)
				stride = Int(size(plan.sk_pizazz_gpu, 2) / temp_n_time)
				#println("Stride: $stride")
				
				# TODO: Figure out how to broadcast with addition to larger dimensions
				# for i in 1:temp_n_time
				# 	println("I: $i")
				# 	start = (i - 1) * stride + 1
				# 	stop  = i * stride
				# 	println("$start , $stop")
				# 	println(size(total_pizazz))
				# 	println(size(plan.sk_pizazz_gpu[1, start:stop, : , 1]))
				# 	#plan.sk_pizazz_gpu[1, start:stop, : , 1] += total_pizazz[1, i, :, 1]
				# end
				
				for i in 1:size(plan.sk_pizazz_gpu, 2)
					index = Int(ceil(i / temp_n_time))
					plan.sk_pizazz_gpu[1, i, :, 1] .+= total_pizazz[1, index, :, 1]
				end
			end
			
		end
        plan.sk_pizazz_gpu ./= max_pizazz # Scales plan.sk_pizazz_gpu to 0-1
        
        # Creates hit meta-data records for sending to the CPU

        #TODO: place this function in here after timing
        hits_metadata = create_hit_info(plan);

		return hits_metadata
	end

end

function gen_data(_size=(2,32768,16,64), sig_chans=[])::AbstractArray
    normal_data = rand(Normal(0,1), _size) .+ rand(Normal(0,1), _size)im
    data = Array{Complex{Int8}}(undef, _size)
    @. data =  Int8(floor(real(normal_data * 16))) + Int8(floor(imag(normal_data * 16)))im

    # Insert signal into channels
    for chan in sig_chans
        data[:,:,:, chan] = rand(Complex{Int8}, _size[1:3])
    end
    return data
end
