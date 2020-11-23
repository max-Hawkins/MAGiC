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

# ╔═╡ 675110f4-2af3-11eb-021f-ad1e7f91a1fc
using Revise

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

# ╔═╡ 3e947bbe-2b6f-11eb-33c5-07ff645609fb
# TODO: show Spectral kurtosis
# Compute SK for lowest integration length at certain start time
# Increase integration length by small amount from same start time
# Recompute SK with new threshold/nint
# Increase nint by 1 (or smallest nint length) until calc SK across whole block
# Plot SK vs integration length per channel

# ╔═╡ a9d9279e-2b71-11eb-1124-b9e96bae09a3
32768/64

# ╔═╡ 488ad006-2b6e-11eb-0c78-d96c3c479f7f


# ╔═╡ add4cd66-1231-11eb-2104-a1e7bac0dbd9
#@time Search.spectral_kurtosis(power, 32768);

# ╔═╡ 7d145c2c-1232-11eb-3839-734a669e0925
#CUDA.@time Search.spectral_kurtosis(power_gpu, 32768)

# ╔═╡ 381e32fa-1232-11eb-15cb-d5940afe8444
#power_gpu = CuArray(power);

# ╔═╡ d88c2eca-2571-11eb-1724-0b50006fb6bc
begin
	sk_nints = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
	sk_plan = Search.create_sk_plan(Complex{Int8}, (2,32768,16,64), sk_nints, 0.06);
end

# ╔═╡ 9c66de88-1237-11eb-32f0-59796da93462
begin
	function display(a::Search.sk_array_t)
		println("Nint: $(a.nint)")
		println("SK Upper Limit: $(a.sk_up_lim)")	
		println("SK Lower Limit: $(a.sk_low_lim)")	
	end
end

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

# ╔═╡ 82ad8f16-1a69-11eb-0d65-81c7cbf3f634
@bind sk_array_index html"<input type='range' min=1 max=8>"

# ╔═╡ d07e7e96-17ba-11eb-03bf-27fa8ea470b4
begin
	pizazz_high = sum(sum(sk_plan.sk_arrays[sk_array_index].sk_data_gpu .> sk_plan.sk_arrays[sk_array_index].sk_up_lim, dims=1), dims = 4);
	pizazz_low = sum(sum(sk_plan.sk_arrays[sk_array_index].sk_data_gpu .< sk_plan.sk_arrays[sk_array_index].sk_low_lim, dims=1), dims = 4);
	
	sk_up_plot = heatmap(transpose(pizazz_high[1,:,:,]), 
		title="Spectral Kurtosis Upper Cutoff",
		ylabel="Channel",
		xlabel="Time")
	
	sk_low_plot = heatmap(transpose(pizazz_low[1,:,:,]), 
		title="Spectral Kurtosis Lower Cutoff",
		ylabel="Channel",
		xlabel="Time")
	plot(sk_up_plot, sk_low_plot, layout=(2,1))
end

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


# ╔═╡ 0c1a7f96-1238-11eb-37ca-1187f8c6c64e
#CUDA.@elapsed Search.exec_plan(sk_plan, input_db.blocks[input_db_block].p_data)

# ╔═╡ acb463e8-22b3-11eb-0ce6-bda9fcfb21a7
#CUDA.@elapsed Search.hit_mask(sk_plan)

# ╔═╡ 8e44821a-2afb-11eb-3a5b-cb2bcc756d95
maximum(sk_plan.sk_pizazz_gpu)

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
	sk = Search.spectral_kurtosis(power, power.^2, 2^expo);
	sk_array = sk[1,:,:,sk_disp_ant]
	
	#SK plot
	sk_plot = heatmap(transpose(sk_array), 
		title="Spectral Kurtosis - Antenna: $sk_disp_ant Nints: $(2^expo)",
		ylabel="Channel",
		xlabel="Time")
	
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
	num_flagged = length(sk_mask_array[sk_mask_array .!= 0]) / length(sk_mask_array) * 100
	sk_mask_plot = heatmap(transpose(sk_mask_array), 
		title="Spectral Kurtosis Mask - Antenna: $sk_disp_ant Nints: $(2^expo) % flagged: $num_flagged",
		ylabel="Channel",
		xlabel="Time",
		clim=(-1,1))
	plot(sk_plot, sk_mask_plot, layout=(2,1))
	
end

# ╔═╡ 499af45c-2b71-11eb-2d1d-957e11ff8bf3
begin
	pol = 1
	ant = 1
	sk_multi = zeros(512,16)
	sk
	n_64 = Search.spectral_kurtosis(power, power.^2, 64)
	for i in 1:512
		sk_testing = sum(n_64[pol, 1:i, :, ant], dims=2)[1,:,:,1]
	end

end

# ╔═╡ b474abe6-2b72-11eb-2b26-7f7ce5e9f32e
sum(n_64[1, 1:4, :, 1], dims=3)

# ╔═╡ f290f25e-2b6d-11eb-1284-7bf14cb3a7fc
sk_chan_flag_percent = sum(sk_mask_array .!=0, dims=1) ./ size(sk_mask_array, 1) .* 100

# ╔═╡ 747c565c-2b6e-11eb-0e4d-6bdb53fba522
plot(sk_chan_flag_percent[1,:])

# ╔═╡ 5d3eec82-2afb-11eb-00a2-97a584e958ce
size(raw_data)

# ╔═╡ 618d60c6-2573-11eb-188a-6928c65a0755
_data = unsafe_wrap(Array{eltype(raw_data)}, Ptr{eltype(raw_data)}(input_db.blocks[input_db_block].p_data), size(raw_data))

# ╔═╡ 93a7fa10-2afc-11eb-29f3-8575f67fb16c
heatmap(transpose(sum(sum(dropdims(sum(reshape(power,      (size(power)[1], sk_nints[1], :, size(power)[3], size(power)[4])); dims=2), dims=2), dims=1), dims=4)[1,:,:,1]))

# ╔═╡ bac8e8e0-22b4-11eb-3036-09495ecc2c1d
begin
	Search.exec_plan(sk_plan, input_db.blocks[input_db_block].p_data)
	hit_data = Search.hit_mask(sk_plan)
	
	heatmap(transpose(sk_plan.sk_pizazz_gpu[1,:,:,1]),
		xaxis=(font(10), "Time (ms)", 0:grh.tbin*128*1000*32:1000*grh.tbin*32768),
		yaxis=(font(10), "Frequency (Coarse Channel)"),
		title="SK Interestingness of 1 GUPPI Block of VELA/MeerKAT Data")
end

# ╔═╡ aeb37570-256e-11eb-21f9-33f954ad2a63
hit_data

# ╔═╡ d25635de-2b69-11eb-1478-47b92898c4f1
minimum(sk_plan.

# ╔═╡ 71ae0630-2af8-11eb-0929-81bfedc64a26
size(sk_plan.sk_pizazz_gpu)

# ╔═╡ Cell order:
# ╠═36cb96dc-f48b-11ea-3e7c-4d91d2ceac72
# ╠═7aeea79e-f48d-11ea-314c-3d88b920173c
# ╠═675110f4-2af3-11eb-021f-ad1e7f91a1fc
# ╠═782c7b3a-f591-11ea-2226-03ff2eac5908
# ╠═3b4ad73c-1587-11eb-33b8-9b41cddd8eec
# ╠═9ecbd766-2572-11eb-162e-dd6b5e05685a
# ╠═d212d4c4-f487-11ea-0412-c70aaa045350
# ╠═5d974e64-2573-11eb-28ef-134d036cfc17
# ╠═618d60c6-2573-11eb-188a-6928c65a0755
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
# ╠═747c565c-2b6e-11eb-0e4d-6bdb53fba522
# ╠═3e947bbe-2b6f-11eb-33c5-07ff645609fb
# ╠═a9d9279e-2b71-11eb-1124-b9e96bae09a3
# ╠═499af45c-2b71-11eb-2d1d-957e11ff8bf3
# ╠═b474abe6-2b72-11eb-2b26-7f7ce5e9f32e
# ╠═d5e5db12-f592-11ea-3fed-cf4251f4c50d
# ╠═f290f25e-2b6d-11eb-1284-7bf14cb3a7fc
# ╠═488ad006-2b6e-11eb-0c78-d96c3c479f7f
# ╠═add4cd66-1231-11eb-2104-a1e7bac0dbd9
# ╠═7d145c2c-1232-11eb-3839-734a669e0925
# ╠═381e32fa-1232-11eb-15cb-d5940afe8444
# ╠═5d3eec82-2afb-11eb-00a2-97a584e958ce
# ╠═d88c2eca-2571-11eb-1724-0b50006fb6bc
# ╠═9c66de88-1237-11eb-32f0-59796da93462
# ╠═5b9285d4-1597-11eb-12f4-6f3e93584da9
# ╠═82ad8f16-1a69-11eb-0d65-81c7cbf3f634
# ╠═d07e7e96-17ba-11eb-03bf-27fa8ea470b4
# ╠═2503c04c-1596-11eb-14fd-a9d429c9e742
# ╟─8858ba20-1592-11eb-2eb1-4381c593d38c
# ╟─949f36d8-1592-11eb-289c-39279d70a873
# ╟─18e46bfc-158b-11eb-19d3-9bc69368a2d8
# ╠═97bc6692-1240-11eb-0a03-933307758e29
# ╟─b9066556-123f-11eb-15ac-bfb24274fff1
# ╠═0c1a7f96-1238-11eb-37ca-1187f8c6c64e
# ╠═acb463e8-22b3-11eb-0ce6-bda9fcfb21a7
# ╠═8e44821a-2afb-11eb-3a5b-cb2bcc756d95
# ╠═549a09f4-253d-11eb-2629-b59545c82c1e
# ╠═93a7fa10-2afc-11eb-29f3-8575f67fb16c
# ╠═bac8e8e0-22b4-11eb-3036-09495ecc2c1d
# ╠═aeb37570-256e-11eb-21f9-33f954ad2a63
# ╠═d25635de-2b69-11eb-1478-47b92898c4f1
# ╠═71ae0630-2af8-11eb-0929-81bfedc64a26
