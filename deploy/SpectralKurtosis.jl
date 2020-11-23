export SpectralKurtosis
"""
    SpectralKurtosis

Module containing functions and parameters about Spectral Kurtosis.
"""
module SpectralKurtosis

export gpu_memory

using ..Search: SearchAlgoPlan, hit_info_t
using CUDA
using Roots: secant_method

"""
    sk_array_t

Data for each spectral kurtosis array.
"""
struct sk_array_t
    sk_data_gpu::CuArray
    nint::Int
    sk_low_lim::Float32
    sk_up_lim::Float32
end

# TODO: Split dynamic thresholding variables from static CuArray?
#   This could allow for a set of mutable items to have their own space
#
# struct sk_thresh_t
#   sk_thresh::Float32
#   sk_up_weight::Float32
#   sk_low_weight::Float32
#   hit_meta_cpu::Array{hit_info_t}
#   hit_meta_gpu::CuArray{hit_info_t}
# end
"""
    sk_plan_t

Data necessary for calculating spectral kurtosis on GPU.
"""
struct sk_plan_t <: SearchAlgoPlan
    complex_data_gpu::CuArray
    power_gpu::CuArray
    power2_gpu::CuArray
    sk_arrays::Array{sk_array_t}
    sk_thresh::Float32
    sk_pizazz_gpu::CuArray
end

"""
    spectral_kurtosis(power_array, p2_array, nints::Int, dims=2)

Returns the spectral kurtosis values of an array of data given power and power squared.

Based off Gelu Nita's 2010 paper: https://arxiv.org/abs/1005.4371
"""
function spectral_kurtosis(power, power2, nints::Int, dims=2) #TODO: implement variations of dim - probably wouldn't be used
    p_size = size(power)
    # Check for correct dimensions and integration length
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

    # Sum over nints number of samples for both power and power-squared arrays
    sum_p  = dropdims(sum(reshape(power,  (p_size[1], nints, new_dim_size, p_size[3], p_size[4])); dims=2), dims=2)
    sum_p2 = dropdims(sum(reshape(power2, (p_size[1], nints, new_dim_size, p_size[3], p_size[4])); dims=2), dims=2)

    # Actual spectral kurtosis calculation. Additional scaling accounts for sample statistics
    sk = @. ((nints + 1)/(nints - 1)) * ((nints * sum_p2)/(sum_p ^ 2) - 1)
    return sk
end

# TODO: Multi-scale spectral kurtosis
function spectral_kurtosis_multi(power, power2, min_nint::Int, dims=2)

end

"""
    upperRoot(x, m2, m3, p)

Calculate the upper root for finding the upper SK threshold given an integration length and p.
"""
function upperRoot(x, m2, m3, p)
    if (-(m3-2*m2^2)/m3+x)/(m3/2/m2) < 0
        return 0
    end
    return abs((1 - gamma_inc( (4 * m2^3)/m3^2, (-(m3-2*m2^2)/m3 + x)/(m3/2/m2), 1)[1]) -p)
end

"""
    lowerRoot(x, m2, m3, p)

Calculate the lower root for finding the lower SK threshold given an integration length and p.
"""
function lowerRoot(x, m2, m3, p)
    if (-(m3-2*m2^2)/m3+x)/(m3/2/m2) < 0
        return 0
    end
    lower = abs(gamma_inc( (4 * m2^3)/m3^2, (-(m3-2*m2^2)/m3 + x)/(m3/2/m2), 0)[1] -p)
    return lower
end

"""
    calc_sk_thresholds(M, N=1, d=1, p=0.0013499)

Calculates the asymmetric spectral kurtosis thresholds given an integration length.

Adapted from Gelu Nita's IDL code (https://github.com/Gelu-Nita/GSK/blob/master/gsk.pro) 
and Nick Joslyn's Python code (https://github.com/NickJoslyn/helpful-BL/edit/master/helpful_BL_programs.py)
Currently uses a lookup table for the thresholds with the default p value because the root finding
algorithms in Julia were not working over a large enough range.
"""
function calc_sk_thresholds(M, N=1, d=1, p=0.0013499)
    Nd = N * d
    if((p==0.0013499) && ((expo = log(2,M)) % 1 == 0))
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

    upperThreshold = Roots.secant_method(x->upperRoot(x,m2,m3,p), 1, atol=1e-8, rtol=1e-8, maxevals=1000)
    lowerThreshold = Roots.secant_method(x->lowerRoot(x,m2,m3,p), 1, atol=1e-4, rtol=1e-12, maxevals=1000)
    return lowerThreshold, upperThreshold
end

"""
    create_sk_plan(raw_eltype, raw_size, nint_array, sk_thresh; dims=2, t_min=0.001)

Predetermine and allocate all necessary GPU arrays for calculating spectral kurtosis.
Also create high-level structs containing information needed during real-time SK processing.

"""
function create_sk_plan(raw_eltype::Type, raw_size, nint_array::Array{Int}, sk_thresh::AbstractFloat; dims::Int=2)
    complex_eltype = raw_eltype
    complex_size = raw_size # TODO: Calculate using raw header
    power_eltype  = Int32 # Allows for long summation
    power2_eltype = Int32 # Allows for long summation

    # TODO: Calculate nint_min based off t_min
    # Currently base it off of the lowest integration length found in nint_array
    # Assumes sorted in ascending order and powers of two
    nint_min = nint_array[1]

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

# TODO: Make function to create plan from GuppiRaw Header
"""
    exec_plan(plan::sk_plan_t, data_ptr::Ptr{Any})

Execute the Spectral Kurtosis plan on the data starting at data_ptr on the CPU. 
"""
function exec_plan(plan::sk_plan_t, data_ptr::Ptr{Any})	
    # Transfer raw data to GPU
    unsafe_copyto!(plan.complex_data_gpu.ptr,
                    Ptr{eltype(plan.complex_data_gpu)}(data_ptr),
                    length(plan.complex_data_gpu))
    
    # Populate power and power-squared arrays
    @. plan.power_gpu = abs2(Complex{Int16}.(plan.complex_data_gpu))
    @. plan.power2_gpu = plan.power_gpu ^ 2
    
    for sk_array in plan.sk_arrays		
        sk_array.sk_data_gpu = spectral_kurtosis(plan.power_gpu, plan.power2_gpu, sk_array.nint) # Unoptimized!!! TODO: Sum power/power2 as nint increases
    end
end

"""
    hit_mask(plan::sk_plan_t)

Return the metadata of the hit data to later save to disk.

"""
function hit_mask(plan::sk_plan_t)
    # hit_meta_array::CuArray{hit_info_t} = []
    complex_size = size(plan.complex_data_gpu)

    low_pizazz_coef = 25.0 # TODO: Make part of plan fields
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
            
            for i in 1:size(plan.sk_pizazz_gpu, 2)
                index = Int(ceil(i / temp_n_time))
                plan.sk_pizazz_gpu[1, i, :, 1] .+= total_pizazz[1, index, :, 1]
            end
        end
        
    end
    plan.sk_pizazz_gpu ./= max_pizazz # Scales plan.sk_pizazz_gpu to 0-1
    
    # Creates hit meta-data records for sending to the CPU

    #TODO: Create structured hit_info_t
    hits_metadata = findall(>(plan.sk_thresh), plan.sk_pizazz_gpu)

    #TODO:
    # return Array(hit_meta_array)
    return hits_metadata
end

function gpu_memory(plan::sk_plan_t)::Int
    bytes::Int = 0

    bytes += sizeof(plan.power_gpu) + sizeof(plan.power2_gpu)
            + sizeof(plan.sk_pizazz_gpu)

    for sk_array in plan.sk_arrays
        bytes += sizeof(sk_array.sk_data_gpu)
    end

    return bytes
end

end # Module SpectralKurtosis