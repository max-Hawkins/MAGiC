### A Pluto.jl notebook ###
# v0.11.12

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
	using Pkg, Statistics, Plots, Blio
	include("hashpipe.jl")
end

# ╔═╡ a5acc0cc-f48f-11ea-297c-d34901987cec
using Main.workspace3.Hashpipe

# ╔═╡ f4250d22-f58e-11ea-3633-43fa2864c296
include("/home/mhawkins/MAGiC/search.jl")

# ╔═╡ ff828e60-f58e-11ea-21b7-05feba7a7f40
hp_status_num = 0

# ╔═╡ 351112e6-f5fc-11ea-1453-11c042b2bef7
hp_databuf_num = 2

# ╔═╡ d212d4c4-f487-11ea-0412-c70aaa045350
begin 
	p_input_db = Hashpipe.hashpipe_databuf_attach(0,2);
	input_db = Hashpipe.databuf_init(p_input_db);
	grh, raw_data = Hashpipe.get_data(input_db.blocks[3]);
	size(raw_data)
end

# ╔═╡ 2a8ff99a-f494-11ea-36c6-690a33c35eb4
power = abs2.(Array{Complex{Int32}}(raw_data));

# ╔═╡ 87a1962e-f602-11ea-37a1-f96f23b1f41a
grh.tbin * grh.


# ╔═╡ 7fd891f8-f494-11ea-22bb-d734af134c18
maximum(power)

# ╔═╡ 318cae2a-f494-11ea-2d3d-35e8a4265497
mean(power)

# ╔═╡ 62a0bbda-f494-11ea-2230-c7507f82bb96
avg_power = mean(power, dims=1);

# ╔═╡ 9bc91074-f494-11ea-1577-b59d7275a1e9
size(avg_power)

# ╔═╡ a1cd2cc6-f494-11ea-1b44-a9fc419fc0ad
disp_avg_power = reshape(avg_power, (32768, :));

# ╔═╡ b907faca-f600-11ea-0330-a54f043afa9d
bandpass = mean(avg_power, dims=2)[1,1,:,:];

# ╔═╡ d780aac8-f601-11ea-26d5-b7ed093c5404
size(bandpass)

# ╔═╡ 87ddedbc-f603-11ea-14a1-bb0531d52d2b


# ╔═╡ 68d2989c-f602-11ea-21cc-735df353494c
heatmap(permutedims(bandpass,[2,1]), title="Bandpasses of All Antennas", xlabel="Coarse Channel", ylabel="Antenna")


# ╔═╡ 123d2d80-f5fd-11ea-079e-5ddf323bee5d
plot(1:16,bandpass, legend=false, title="Bandpass of all Antennas", xlabel="Coarse Channel", ylabel="Intensity")

# ╔═╡ c494912a-f603-11ea-341f-d5d7be4bfb32
@bind disp_ant html"<input type='range' min=1 max=64>"

# ╔═╡ 13082a38-f604-11ea-066e-f9345e644d98
plot(1:16, bandpass[:,disp_ant], title="Bandpass of Antenna $disp_ant", xlabel="Channel", ylabel="Intensity")

# ╔═╡ a7203906-f495-11ea-0a37-051bf1620a5c
incoherent_sum = sum(avg_power, dims=3);

# ╔═╡ e810355c-f499-11ea-29bd-7169859b8ef4
size(incoherent_sum)

# ╔═╡ a235300e-f495-11ea-24cd-81d25398aed6
heatmap(incoherent_sum[1,:,1,1:16], title="Incoherent Sum")

# ╔═╡ be4ce3aa-f494-11ea-220f-8350fda08d4b
heatmap(disp_avg_power[1:1024,1:256])

# ╔═╡ 9fb6e33c-f499-11ea-0e36-91e1ad3def19
size(power)

# ╔═╡ 808797e2-f9dd-11ea-3593-570577bd00a2
@bind sk_pol html"<input type='radio'>"

# ╔═╡ 38ca937a-f605-11ea-2a0a-25897a39c96b
@bind expo html"<input type='range' max=15>"

# ╔═╡ 961f870e-f9dd-11ea-2ab4-1d01a22e0312
sk_lower, sk_upper = (0.9676684770523931,1.0339584931405572) #Search.calc_sk_thresholds(2^expo) Need to use preset value until lookup table is created

# ╔═╡ 09dea1d6-f606-11ea-0ff0-a72a809b569a
@bind sk_disp_ant html"<input type='range' min=1 max=64>"

# ╔═╡ d5e5db12-f592-11ea-3fed-cf4251f4c50d
begin 
	sk = Main.workspace3.Search.spectral_kurtosis(power,2^expo);
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

# ╔═╡ Cell order:
# ╠═36cb96dc-f48b-11ea-3e7c-4d91d2ceac72
# ╠═7aeea79e-f48d-11ea-314c-3d88b920173c
# ╠═782c7b3a-f591-11ea-2226-03ff2eac5908
# ╠═a5acc0cc-f48f-11ea-297c-d34901987cec
# ╠═f4250d22-f58e-11ea-3633-43fa2864c296
# ╠═ff828e60-f58e-11ea-21b7-05feba7a7f40
# ╠═351112e6-f5fc-11ea-1453-11c042b2bef7
# ╠═d212d4c4-f487-11ea-0412-c70aaa045350
# ╠═2a8ff99a-f494-11ea-36c6-690a33c35eb4
# ╠═87a1962e-f602-11ea-37a1-f96f23b1f41a
# ╠═7fd891f8-f494-11ea-22bb-d734af134c18
# ╠═318cae2a-f494-11ea-2d3d-35e8a4265497
# ╠═62a0bbda-f494-11ea-2230-c7507f82bb96
# ╠═9bc91074-f494-11ea-1577-b59d7275a1e9
# ╠═a1cd2cc6-f494-11ea-1b44-a9fc419fc0ad
# ╠═b907faca-f600-11ea-0330-a54f043afa9d
# ╠═d780aac8-f601-11ea-26d5-b7ed093c5404
# ╠═87ddedbc-f603-11ea-14a1-bb0531d52d2b
# ╟─68d2989c-f602-11ea-21cc-735df353494c
# ╠═123d2d80-f5fd-11ea-079e-5ddf323bee5d
# ╠═c494912a-f603-11ea-341f-d5d7be4bfb32
# ╠═13082a38-f604-11ea-066e-f9345e644d98
# ╠═a7203906-f495-11ea-0a37-051bf1620a5c
# ╠═e810355c-f499-11ea-29bd-7169859b8ef4
# ╠═a235300e-f495-11ea-24cd-81d25398aed6
# ╠═be4ce3aa-f494-11ea-220f-8350fda08d4b
# ╠═9fb6e33c-f499-11ea-0e36-91e1ad3def19
# ╠═808797e2-f9dd-11ea-3593-570577bd00a2
# ╠═38ca937a-f605-11ea-2a0a-25897a39c96b
# ╠═961f870e-f9dd-11ea-2ab4-1d01a22e0312
# ╠═09dea1d6-f606-11ea-0ff0-a72a809b569a
# ╠═d5e5db12-f592-11ea-3fed-cf4251f4c50d
