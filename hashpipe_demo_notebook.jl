### A Pluto.jl notebook ###
# v0.12.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 36cb96dc-f48b-11ea-3e7c-4d91d2ceac72
push!(LOAD_PATH,"/home/mhawkins/jl-blio/src/")

# ╔═╡ 7aeea79e-f48d-11ea-314c-3d88b920173c
push!(LOAD_PATH,"/home/mhawkins/MAGiC/")

# ╔═╡ 782c7b3a-f591-11ea-2226-03ff2eac5908
begin
	include("/home/mhawkins/jl-blio/src/GuppiRaw.jl")
	using Pkg, Statistics, Plots, Blio, CUDA, BenchmarkTools
	include("hashpipe.jl")
	
end

# ╔═╡ 3b4ad73c-1587-11eb-33b8-9b41cddd8eec
using Main.workspace3.Hashpipe

# ╔═╡ 9ecbd766-2572-11eb-162e-dd6b5e05685a
include("/home/mhawkins/MAGiC/search.jl")

# ╔═╡ 87a1962e-f602-11ea-37a1-f96f23b1f41a


# ╔═╡ c494912a-f603-11ea-341f-d5d7be4bfb32
@bind disp_ant html"<input type='range' min=1 max=64>"

# ╔═╡ 808797e2-f9dd-11ea-3593-570577bd00a2
@bind sk_pol html"<input type='radio'>" #TODO: Implement radio polarization selection

# ╔═╡ 38ca937a-f605-11ea-2a0a-25897a39c96b
@bind expo html"<input type='range' max=15>"

# ╔═╡ 961f870e-f9dd-11ea-2ab4-1d01a22e0312
sk_lower, sk_upper = Search.calc_sk_thresholds(2^expo)

# ╔═╡ 09dea1d6-f606-11ea-0ff0-a72a809b569a
@bind sk_disp_ant html"<input type='range' min=1 max=64>"

# ╔═╡ add4cd66-1231-11eb-2104-a1e7bac0dbd9
#@time Search.spectral_kurtosis(power, 32768);

# ╔═╡ 7d145c2c-1232-11eb-3839-734a669e0925
#CUDA.@time Search.spectral_kurtosis(power_gpu, 32768)

# ╔═╡ 381e32fa-1232-11eb-15cb-d5940afe8444
#power_gpu = CuArray(power);

# ╔═╡ 9c66de88-1237-11eb-32f0-59796da93462
begin
	function display(a::Search.sk_array_t)
		println("Nint: $(a.nint)")
		println("SK Upper Limit: $(a.sk_up_lim)")	
		println("SK Lower Limit: $(a.sk_low_lim)")	
	end
end

# ╔═╡ c358cff6-13cc-11eb-1aac-33f55a0a3c38


# ╔═╡ d5d39dfc-17ba-11eb-2c9c-9b4520e934c2
#sk_test = Search.spectral_kurtosis(sk_plan.power_gpu, 2048)

# ╔═╡ 82ad8f16-1a69-11eb-0d65-81c7cbf3f634
@bind sk_array_index html"<input type='range' min=1 max=4>"

# ╔═╡ 2503c04c-1596-11eb-14fd-a9d429c9e742
#CUDA.@time begin sk_plan.power_gpu = abs2.(sk_plan.complex_data_gpu); end

# ╔═╡ 8858ba20-1592-11eb-2eb1-4381c593d38c


# ╔═╡ 949f36d8-1592-11eb-289c-39279d70a873


# ╔═╡ 18e46bfc-158b-11eb-19d3-9bc69368a2d8


# ╔═╡ 97bc6692-1240-11eb-0a03-933307758e29
begin
	# Hit info struct for later saving back into the databuf header 
	#	to later save raw data segments to disk
	mutable struct hit_info_t
		freq_chan_i
		freq_chan_f
		t_i
		t_f
		pizazz #Interestingness value (0-1)
	end
end

# ╔═╡ b9066556-123f-11eb-15ac-bfb24274fff1
begin
	# Implement "interestingness" function on generated data for later saving
	# Generates array with dimensions of raw_data except for time which is
	# 	minimum time of interest per t_step
	# Later has an adjustable threshold that, once exceeded, is masked for
	# 	saving on the GPU. The time and frequency (maybe antenna and polarization too?)
	#	is saved in a struct hit_info_t along with the original interestingness value
	# 	to help out with later data triaging
	# An array of hit_info_t is passed back to the CPU
	#	for saving/writing to disk or buffer space
	# 
	function hit_mask(plan::Search.sk_plan_t)
		complex_size = size(plan.complex_data_gpu)
		sk_pizazz = CuArray{Float64}(undef, 
					(1,
					size(plan.sk_arrays[1].sk_data_gpu,2),
					complex_size[3],
					1))

		low_pizazz_coef = 25.0
		up_pizazz_coef  = 0.5
		
		max_pizazz = (low_pizazz_coef + up_pizazz_coef) * complex_size[1] * complex_size[4] * length(plan.sk_arrays)
		
		for sk_array in plan.sk_arrays
			println("i: $(size(sk_array.sk_data_gpu))")
			
			pizazz_up = sum(sum(sk_array.sk_data_gpu .> sk_array.sk_up_lim, dims=1), dims = 4) .* up_pizazz_coef;
			pizazz_low = sum(sum(sk_array.sk_data_gpu .< sk_array.sk_low_lim, dims=1), dims = 4) .* low_pizazz_coef;
			
			if sk_array == plan.sk_arrays[1]
				sk_pizazz .= pizazz_up .+ pizazz_low
				
			else 
				total_pizazz = pizazz_up .+ pizazz_low
				
				temp_n_time = size(sk_array.sk_data_gpu,2)
				stride = Int(size(sk_pizazz, 2) / temp_n_time)
				println("Stride: $stride")
				
				# TODO: Figure out how to broadcast with addition to larger dimensions
				# for i in 1:temp_n_time
				# 	println("I: $i")
				# 	start = (i - 1) * stride + 1
				# 	stop  = i * stride
				# 	println("$start , $stop")
				# 	println(size(total_pizazz))
				# 	println(size(sk_pizazz[1, start:stop, : , 1]))
				# 	#sk_pizazz[1, start:stop, : , 1] += total_pizazz[1, i, :, 1]
				# end
				
				for i in 1:size(sk_pizazz, 2)
					index = Int(ceil(i / temp_n_time))
					sk_pizazz[1, i, :, 1] .+= total_pizazz[1, index, :, 1]
				end
			end
			#println(typeof(pizazz_up))
			# pizazz_up  .*= up_pizazz_coef;
			# pizazz_low .*= low_pizazz_coef;
			# println(typeof(pizazz_up))
			
		end
		sk_pizazz ./= max_pizazz # Scales sk_pizazz to 0-1
		return sk_pizazz
	end
end

# ╔═╡ 549a09f4-253d-11eb-2629-b59545c82c1e
@bind input_db_block html"<input type='range' min=1 max=24>"

# ╔═╡ d212d4c4-f487-11ea-0412-c70aaa045350
begin 
	hp_status_num = 0
	hp_databuf_num = 2
	p_input_db = Hashpipe.hashpipe_databuf_attach(0,2);
	input_db = Hashpipe.databuf_init(p_input_db);
	grh, raw_data = Hashpipe.get_data(input_db.blocks[input_db_block]);
	size(raw_data)
end

# ╔═╡ 3340e7b2-2573-11eb-0343-0b111c236d10
input_db.blocks[1].p_data

# ╔═╡ 5d974e64-2573-11eb-28ef-134d036cfc17
size(raw_data)

# ╔═╡ 2a8ff99a-f494-11ea-36c6-690a33c35eb4
power = abs2.(Array{Complex{Int32}}(raw_data));

# ╔═╡ 62a0bbda-f494-11ea-2230-c7507f82bb96
avg_power = mean(power, dims=1);

# ╔═╡ a1cd2cc6-f494-11ea-1b44-a9fc419fc0ad
disp_avg_power = reshape(avg_power, (32768, :));

# ╔═╡ be4ce3aa-f494-11ea-220f-8350fda08d4b
heatmap(disp_avg_power[1:1024,1:256])

# ╔═╡ b907faca-f600-11ea-0330-a54f043afa9d
bandpass = mean(avg_power, dims=2)[1,1,:,:];

# ╔═╡ 68d2989c-f602-11ea-21cc-735df353494c
heatmap(permutedims(bandpass,[2,1]), title="Bandpasses of All Antennas", xlabel="Coarse Channel", ylabel="Antenna")


# ╔═╡ 123d2d80-f5fd-11ea-079e-5ddf323bee5d
plot(1:16,bandpass, legend=false, title="Bandpass of all Antennas", xlabel="Coarse Channel", ylabel="Intensity")

# ╔═╡ 13082a38-f604-11ea-066e-f9345e644d98
plot(1:16, bandpass[:,disp_ant], title="Bandpass of Antenna $disp_ant", xlabel="Channel", ylabel="Intensity")

# ╔═╡ a7203906-f495-11ea-0a37-051bf1620a5c
incoherent_sum = sum(avg_power, dims=3);

# ╔═╡ e810355c-f499-11ea-29bd-7169859b8ef4
size(incoherent_sum)

# ╔═╡ a235300e-f495-11ea-24cd-81d25398aed6
heatmap(incoherent_sum[1,:,1,1:16], title="Incoherent Sum")

# ╔═╡ 45789cee-1591-11eb-1b90-092db5a7ddb8
bp = mean(incoherent_sum[:,:,:,1:16], dims=2)[1,1,1,:]

# ╔═╡ 93795b72-1591-11eb-02dd-d9252d405fbe
plot(bp)

# ╔═╡ d5e5db12-f592-11ea-3fed-cf4251f4c50d
begin 
	sk = Search.spectral_kurtosis(power,2^expo);
	sk_array = sk[2,:,:,sk_disp_ant]
	
	#SK plot
	sk_plot = heatmap(sk_array, 
		title="Spectral Kurtosis - Antenna: $sk_disp_ant Nints: $(2^expo)",
		xlabel="Channel",
		ylabel="Time")
	
	function sk_mask(sk_value)::Int8
		if(sk_value > sk_upper)
			return 1
		elseif(sk_value < sk_lower)
			return -1
		end
		return 0
	end
	
	#SK mask plot
	sk_mask_array = map(sk_mask, sk_array)
	sk_mask_plot = heatmap(sk_mask_array, 
		title="Spectral Kurtosis Mask - Antenna: $sk_disp_ant Nints: $(2^expo)",
		xlabel="Channel",
		ylabel="Time",
		clim=(-1,1))
	plot(sk_plot, sk_mask_plot, layout=(2,1))
end

# ╔═╡ d88c2eca-2571-11eb-1724-0b50006fb6bc
sk_plan = Search.create_sk_plan(Complex{Int8}, size(raw_data), [256]);

# ╔═╡ 5b9285d4-1597-11eb-12f4-6f3e93584da9
begin
	sk_a = sk_plan.sk_arrays[1]
	for t in 1:size(sk_a.sk_data_gpu, 2)
					println("size: $t")
					t_i = 1 + (t - 1) * sk_a.nint
					t_f = t * sk_a.nint
					println("$(t_i) : $(t_f)")
					s = sum(sk_a.sk_data_gpu[1,t, :, 1] .> sk_a.sk_up_lim)
					println(s)
	end
end

# ╔═╡ d2bfcd0e-17ba-11eb-094e-1d6a9864ec01
size(sk_plan.sk_arrays[1].sk_data_gpu)

# ╔═╡ d07e7e96-17ba-11eb-03bf-27fa8ea470b4
begin
	pizazz_high = sum(sum(sk_plan.sk_arrays[sk_array_index].sk_data_gpu .> sk_plan.sk_arrays[sk_array_index].sk_up_lim, dims=1), dims = 4);
	pizazz_low = sum(sum(sk_plan.sk_arrays[sk_array_index].sk_data_gpu .< sk_plan.sk_arrays[sk_array_index].sk_low_lim, dims=1), dims = 4);
	println(size(pizazz_low))
	#heatmap((pizazz_high.+pizazz_low)[1,:,:,1])
	sk_up_plot = heatmap(pizazz_high[1,:,:,], 
		title="Spectral Kurtosis Upper Cutoff",
		xlabel="Channel",
		ylabel="Time")
	
	sk_low_plot = heatmap(pizazz_low[1,:,:,], 
		title="Spectral Kurtosis Lower Cutoff",
		xlabel="Channel",
		ylabel="Time")
	plot(sk_up_plot, sk_low_plot, layout=(2,1))
end

# ╔═╡ acb463e8-22b3-11eb-0ce6-bda9fcfb21a7
CUDA.@elapsed hit_mask(sk_plan)

# ╔═╡ 2fc55022-2577-11eb-2630-89d7de8ad500
size(raw_data)

# ╔═╡ 0ba6fc04-2577-11eb-358b-5b83ad4349ca
Search.create_sk_plan(Complex{Int8}, size(raw_data), [256])

# ╔═╡ 60282c7c-1237-11eb-0ac7-e7031445c568
begin
	# Takes ~1ms per SK array of certain integration length
	function exec_plan(plan::Search.sk_plan_t)
		nint_min = 256 #slightly longer than 1ms
		
		_size = size(plan.complex_data_gpu)
		sk_pizazz_size = (1, Int(floor(_size[2] / nint_min)), _size[3], 1) # TODO: change to not use int and floor
		println(sk_pizazz_size)
		sk_pizazz_type = Float16
		
		# Allocate memory for SK pizazz score. TODO: Make per plan or global?
		p_buf_sk_pizazz_gpu = CUDA.Mem.alloc(CUDA.Mem.DeviceBuffer, prod(sk_pizazz_size)*sizeof(sk_pizazz_type))
		p_sk_pizazz_gpu = convert(CuPtr{sk_pizazz_type}, p_buf_sk_pizazz_gpu)
        # wrap data for later easier use
        sk_pizazz_gpu = unsafe_wrap(CuArray, p_sk_pizazz_gpu, sk_pizazz_size)
		
		# Calculate the 
		
		# Transfer raw data to GPU
		# TODO: Later, do this outside of any plan executions
		unsafe_copyto!(plan.complex_data_gpu.ptr, pointer(raw_data), length(raw_data))
		
		tic = time_ns() # To calculate execution time without transfer
		
		n_sk_array_pols = length(plan.sk_arrays) * size(_size, 1)
		pizazz_over_lim::Float16  = 0.5 / n_sk_array_pols
		pizazz_under_lim::Float16 = 1 / n_sk_array_pols
		
		# Populate power and power-squared arrays
		@. plan.power_gpu = abs2(Complex{Int16}.(plan.complex_data_gpu))
		@. plan.power2_gpu = plan.power_gpu ^ 2
		
		for sk_array in plan.sk_arrays
			println("=============")
			display(sk_array)
			
			sk_array.sk_data_gpu = Search.spectral_kurtosis(plan.power_gpu, sk_array.nint) # Unoptimized!!! TODO: Sum power/power2 as nint increases
			
			sk_pizazz_lower = 1
			sk_pizazz_upper = 0.5
			
			# Add to pizazz array
# 			for t in 1:size(sk_array.sk_data_gpu, 2)
# 				println("size: $t")
# 				t_i = 1 + (t - 1) * sk_array.nint
# 				t_f = t * sk_array.nint
# 				println("$(t_i) : $(t_f)")
# 				sk_pizazz_gpu[1, t_i:t_f, :, 1] += sum(sk_array.sk_data_gpu[1,t, :, 1] .> sk_array.sk_up_lim) * sk_pizazz_upper + sum(sk_array.sk_data_gpu[1,t, :, 1] .< sk_array.sk_low_lim) * sk_pizazz_lower
				
# 			end
			
		end
		
		
		toc = time_ns()
		println("Time withouth transfer: $((toc-tic) / 1E9)")
		#return sk_pizazz_gpu
	end
end

# ╔═╡ 0c1a7f96-1238-11eb-37ca-1187f8c6c64e
CUDA.@elapsed exec_plan(sk_plan)

# ╔═╡ 618d60c6-2573-11eb-188a-6928c65a0755
_data = unsafe_wrap(Array{eltype(raw_data)}, Ptr{eltype(raw_data)}(input_db.blocks[input_db_block].p_data), size(raw_data))

# ╔═╡ 92dbe4e0-2573-11eb-3075-1da10605b1e4
length(_data)

# ╔═╡ bac8e8e0-22b4-11eb-3036-09495ecc2c1d
begin
	xtick = 
	hit_data = hit_mask(sk_plan)
	heatmap(transpose(hit_data[1,:,:,1]),
	xaxis=(font(10), "Time (ms)", 0:grh.tbin*128*1000*32:1000*grh.tbin*32768),
	yaxis=(font(10), "Frequency (Coarse Channel)"),
	title="SK Interestingness of 1 GUPPI Block of VELA/MeerKAT Data")
end

# ╔═╡ aeb37570-256e-11eb-21f9-33f954ad2a63
hit_data

# ╔═╡ Cell order:
# ╠═36cb96dc-f48b-11ea-3e7c-4d91d2ceac72
# ╠═7aeea79e-f48d-11ea-314c-3d88b920173c
# ╠═782c7b3a-f591-11ea-2226-03ff2eac5908
# ╠═9ecbd766-2572-11eb-162e-dd6b5e05685a
# ╠═3b4ad73c-1587-11eb-33b8-9b41cddd8eec
# ╠═d212d4c4-f487-11ea-0412-c70aaa045350
# ╠═3340e7b2-2573-11eb-0343-0b111c236d10
# ╠═5d974e64-2573-11eb-28ef-134d036cfc17
# ╠═618d60c6-2573-11eb-188a-6928c65a0755
# ╠═92dbe4e0-2573-11eb-3075-1da10605b1e4
# ╠═2a8ff99a-f494-11ea-36c6-690a33c35eb4
# ╟─87a1962e-f602-11ea-37a1-f96f23b1f41a
# ╠═62a0bbda-f494-11ea-2230-c7507f82bb96
# ╠═a1cd2cc6-f494-11ea-1b44-a9fc419fc0ad
# ╠═b907faca-f600-11ea-0330-a54f043afa9d
# ╟─68d2989c-f602-11ea-21cc-735df353494c
# ╠═123d2d80-f5fd-11ea-079e-5ddf323bee5d
# ╠═c494912a-f603-11ea-341f-d5d7be4bfb32
# ╠═13082a38-f604-11ea-066e-f9345e644d98
# ╠═a7203906-f495-11ea-0a37-051bf1620a5c
# ╠═e810355c-f499-11ea-29bd-7169859b8ef4
# ╠═a235300e-f495-11ea-24cd-81d25398aed6
# ╠═45789cee-1591-11eb-1b90-092db5a7ddb8
# ╠═93795b72-1591-11eb-02dd-d9252d405fbe
# ╠═be4ce3aa-f494-11ea-220f-8350fda08d4b
# ╠═808797e2-f9dd-11ea-3593-570577bd00a2
# ╠═38ca937a-f605-11ea-2a0a-25897a39c96b
# ╠═961f870e-f9dd-11ea-2ab4-1d01a22e0312
# ╠═09dea1d6-f606-11ea-0ff0-a72a809b569a
# ╠═d5e5db12-f592-11ea-3fed-cf4251f4c50d
# ╠═add4cd66-1231-11eb-2104-a1e7bac0dbd9
# ╠═7d145c2c-1232-11eb-3839-734a669e0925
# ╠═381e32fa-1232-11eb-15cb-d5940afe8444
# ╠═d88c2eca-2571-11eb-1724-0b50006fb6bc
# ╠═2fc55022-2577-11eb-2630-89d7de8ad500
# ╠═0ba6fc04-2577-11eb-358b-5b83ad4349ca
# ╠═9c66de88-1237-11eb-32f0-59796da93462
# ╠═c358cff6-13cc-11eb-1aac-33f55a0a3c38
# ╠═60282c7c-1237-11eb-0ac7-e7031445c568
# ╠═5b9285d4-1597-11eb-12f4-6f3e93584da9
# ╠═d5d39dfc-17ba-11eb-2c9c-9b4520e934c2
# ╠═d2bfcd0e-17ba-11eb-094e-1d6a9864ec01
# ╠═82ad8f16-1a69-11eb-0d65-81c7cbf3f634
# ╠═d07e7e96-17ba-11eb-03bf-27fa8ea470b4
# ╠═2503c04c-1596-11eb-14fd-a9d429c9e742
# ╟─8858ba20-1592-11eb-2eb1-4381c593d38c
# ╟─949f36d8-1592-11eb-289c-39279d70a873
# ╟─18e46bfc-158b-11eb-19d3-9bc69368a2d8
# ╠═97bc6692-1240-11eb-0a03-933307758e29
# ╠═b9066556-123f-11eb-15ac-bfb24274fff1
# ╠═0c1a7f96-1238-11eb-37ca-1187f8c6c64e
# ╠═acb463e8-22b3-11eb-0ce6-bda9fcfb21a7
# ╠═549a09f4-253d-11eb-2629-b59545c82c1e
# ╠═bac8e8e0-22b4-11eb-3036-09495ecc2c1d
# ╠═aeb37570-256e-11eb-21f9-33f954ad2a63
