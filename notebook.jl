### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ 36cb96dc-f48b-11ea-3e7c-4d91d2ceac72
push!(LOAD_PATH,"/home/mhawkins/jl-blio/src/")

# ╔═╡ 7aeea79e-f48d-11ea-314c-3d88b920173c
push!(LOAD_PATH,"/home/mhawkins/MAGiC/")

# ╔═╡ a5acc0cc-f48f-11ea-297c-d34901987cec
using Pkg, Statistics, Plots, Blio

# ╔═╡ 34a6e1f6-f592-11ea-0a88-cdfb0c83fdab
using Search

# ╔═╡ aa336062-f48d-11ea-0900-ed0ebc66aadb
begin
	using Main.workspace3.Hashpipe
	#using Main.workspace3.Search
end

# ╔═╡ 782c7b3a-f591-11ea-2226-03ff2eac5908
b = include("/home/mhawkins/jl-blio/src/GuppiRaw.jl")

# ╔═╡ a2d69d6c-f487-11ea-2dbf-a34c7dd4b238
begin
	include("hashpipe.jl")
	#include("search.jl")
end

# ╔═╡ f4250d22-f58e-11ea-3633-43fa2864c296
include("/home/mhawkins/MAGiC/search.jl")

# ╔═╡ ff828e60-f58e-11ea-21b7-05feba7a7f40


# ╔═╡ d212d4c4-f487-11ea-0412-c70aaa045350
p_input_db = Hashpipe.hashpipe_databuf_attach(0,2)

# ╔═╡ cef3c082-f487-11ea-36f4-677b961afe6f
input_db = Hashpipe.databuf_init(p_input_db)

# ╔═╡ 1c869ca2-f488-11ea-3741-355a37309322
grh, data = Hashpipe.get_data(input_db.blocks[4]);

# ╔═╡ 281f2172-f494-11ea-35e4-7725c2383794
size(data)

# ╔═╡ b7ddf570-f496-11ea-188e-094fddd6656f
corr_out = data[1,:,1,13] .* conj.(data[1,:,2,13])

# ╔═╡ df813d2e-f496-11ea-1fcb-79c315dc9b91
phase = angle.(corr_out)

# ╔═╡ 63f3fd96-f497-11ea-310e-ef1e1a39485c


# ╔═╡ 3fa2a726-f497-11ea-252b-913ea0751649
size(phase)

# ╔═╡ 5f24a98a-f497-11ea-1e6e-290cdefb5bb2


# ╔═╡ 53cb2a34-f497-11ea-378d-bb0bec903000
typeof(phase)

# ╔═╡ 5a99214a-f497-11ea-2bab-ef160e5d24c9


# ╔═╡ 506b24a2-f497-11ea-2eb2-f746fee5374b


# ╔═╡ 2a249f08-f497-11ea-31d8-695c286f3e32
plot(1:32768,phase)


# ╔═╡ 78150310-f497-11ea-0cdf-37de8c1733c9


# ╔═╡ 3400348a-f497-11ea-2679-472724daaf39
Pkg.add(StatsBase)

# ╔═╡ 2e3507f0-f499-11ea-0e31-8f8f7bd7fbd6


# ╔═╡ 2b5c3e9e-f499-11ea-0394-472f36be84f2


# ╔═╡ 21e7d574-f499-11ea-0e65-b5d981e67ec4


# ╔═╡ d393f238-f496-11ea-123d-f3904b906f15


# ╔═╡ 2a8ff99a-f494-11ea-36c6-690a33c35eb4
power = abs2.(Array{Complex{Int32}}(data));

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

# ╔═╡ 98f583a0-f494-11ea-1ca5-955e6d9710b9
size(disp_avg_power)

# ╔═╡ 144932e0-f495-11ea-30ee-3b1ae1fc2599
disp_avg_power

# ╔═╡ 316571d6-f495-11ea-144e-897deb42bbdc
bandpass = mean(disp_avg_power, dims=1)[1,:]

# ╔═╡ a7203906-f495-11ea-0a37-051bf1620a5c
incoherent_sum = sum(avg_power, dims=4)[1,:,:,1]

# ╔═╡ e810355c-f499-11ea-29bd-7169859b8ef4


# ╔═╡ a235300e-f495-11ea-24cd-81d25398aed6
heatmap(incoherent_sum[:,1:16])

# ╔═╡ 899afcae-f495-11ea-3ace-6122c1e3e27e
x_vals = collect(1:1024)

# ╔═╡ 45003910-f495-11ea-29e8-936efecb9c07
plot(x_vals[1:256],bandpass[1:256])

# ╔═╡ be4ce3aa-f494-11ea-220f-8350fda08d4b
heatmap(disp_avg_power[1:1024,1:256])

# ╔═╡ 9fb6e33c-f499-11ea-0e36-91e1ad3def19
data

# ╔═╡ 8fc42f0e-f499-11ea-0a94-c1b1ebaaf4e2
p = abs2.(Array{Complex{Int32}}(data))

# ╔═╡ 965c969c-f593-11ea-2197-afcf6a298421


# ╔═╡ edeed3da-f592-11ea-2009-452f56930a9f
size(p)

# ╔═╡ f03fd382-f592-11ea-3f31-491920707978


# ╔═╡ d5e5db12-f592-11ea-3fed-cf4251f4c50d
sk = Main.workspace4.Search.spectral_kurtosis(p,2048)

# ╔═╡ 51a1e92a-f594-11ea-0cee-d9bcb81df4d8


# ╔═╡ 4d897dc8-f594-11ea-11d5-0f97e021bedf


# ╔═╡ 47a79df2-f594-11ea-3da6-556c32036252


# ╔═╡ 2c82f2ce-f594-11ea-3783-03e5a4cc6fd6


# ╔═╡ ac6d32de-f593-11ea-3b1a-05c95b79d588
heatmap(sk[1,:,14,:])

# ╔═╡ 06887410-f594-11ea-10e2-e100d41a5eef
maximum(sk)


# ╔═╡ 1e83e516-f594-11ea-2d89-25b37ea643f6
mean(sk)

# ╔═╡ f5af6bba-f593-11ea-3b07-89ad8519cf75


# ╔═╡ f2f96916-f593-11ea-333e-cdb011981bf2


# ╔═╡ ef5cd554-f593-11ea-38c4-b10dea199e36


# ╔═╡ ec25f816-f593-11ea-3ee8-1dd8783f2630


# ╔═╡ dc723f76-f593-11ea-11a4-ff75159e3627


# ╔═╡ b62ec3f8-f593-11ea-329a-6d13db49f7a2
size(sk)

# ╔═╡ c17a92b6-f593-11ea-1ec6-4110476c3488


# ╔═╡ fde850e0-f592-11ea-13a6-5fa09cb8b1bb
abs2((-1-4im))

# ╔═╡ 6aeea23e-f593-11ea-2661-57961f4ed2aa


# ╔═╡ ca62cf68-f487-11ea-3265-0f08c21e42e0


# ╔═╡ Cell order:
# ╠═36cb96dc-f48b-11ea-3e7c-4d91d2ceac72
# ╠═7aeea79e-f48d-11ea-314c-3d88b920173c
# ╠═782c7b3a-f591-11ea-2226-03ff2eac5908
# ╠═a5acc0cc-f48f-11ea-297c-d34901987cec
# ╠═34a6e1f6-f592-11ea-0a88-cdfb0c83fdab
# ╠═a2d69d6c-f487-11ea-2dbf-a34c7dd4b238
# ╠═aa336062-f48d-11ea-0900-ed0ebc66aadb
# ╠═f4250d22-f58e-11ea-3633-43fa2864c296
# ╠═ff828e60-f58e-11ea-21b7-05feba7a7f40
# ╠═d212d4c4-f487-11ea-0412-c70aaa045350
# ╠═cef3c082-f487-11ea-36f4-677b961afe6f
# ╠═1c869ca2-f488-11ea-3741-355a37309322
# ╠═281f2172-f494-11ea-35e4-7725c2383794
# ╠═b7ddf570-f496-11ea-188e-094fddd6656f
# ╠═df813d2e-f496-11ea-1fcb-79c315dc9b91
# ╠═63f3fd96-f497-11ea-310e-ef1e1a39485c
# ╠═3fa2a726-f497-11ea-252b-913ea0751649
# ╠═5f24a98a-f497-11ea-1e6e-290cdefb5bb2
# ╠═53cb2a34-f497-11ea-378d-bb0bec903000
# ╠═5a99214a-f497-11ea-2bab-ef160e5d24c9
# ╠═506b24a2-f497-11ea-2eb2-f746fee5374b
# ╠═2a249f08-f497-11ea-31d8-695c286f3e32
# ╠═78150310-f497-11ea-0cdf-37de8c1733c9
# ╠═3400348a-f497-11ea-2679-472724daaf39
# ╠═2e3507f0-f499-11ea-0e31-8f8f7bd7fbd6
# ╠═2b5c3e9e-f499-11ea-0394-472f36be84f2
# ╠═21e7d574-f499-11ea-0e65-b5d981e67ec4
# ╠═d393f238-f496-11ea-123d-f3904b906f15
# ╠═2a8ff99a-f494-11ea-36c6-690a33c35eb4
# ╠═7fd891f8-f494-11ea-22bb-d734af134c18
# ╠═318cae2a-f494-11ea-2d3d-35e8a4265497
# ╠═62a0bbda-f494-11ea-2230-c7507f82bb96
# ╠═9bc91074-f494-11ea-1577-b59d7275a1e9
# ╠═a1cd2cc6-f494-11ea-1b44-a9fc419fc0ad
# ╠═98f583a0-f494-11ea-1ca5-955e6d9710b9
# ╠═144932e0-f495-11ea-30ee-3b1ae1fc2599
# ╠═316571d6-f495-11ea-144e-897deb42bbdc
# ╠═a7203906-f495-11ea-0a37-051bf1620a5c
# ╠═e810355c-f499-11ea-29bd-7169859b8ef4
# ╠═a235300e-f495-11ea-24cd-81d25398aed6
# ╠═899afcae-f495-11ea-3ace-6122c1e3e27e
# ╠═45003910-f495-11ea-29e8-936efecb9c07
# ╠═be4ce3aa-f494-11ea-220f-8350fda08d4b
# ╠═9fb6e33c-f499-11ea-0e36-91e1ad3def19
# ╠═8fc42f0e-f499-11ea-0a94-c1b1ebaaf4e2
# ╠═965c969c-f593-11ea-2197-afcf6a298421
# ╠═edeed3da-f592-11ea-2009-452f56930a9f
# ╠═f03fd382-f592-11ea-3f31-491920707978
# ╠═d5e5db12-f592-11ea-3fed-cf4251f4c50d
# ╠═51a1e92a-f594-11ea-0cee-d9bcb81df4d8
# ╠═4d897dc8-f594-11ea-11d5-0f97e021bedf
# ╠═47a79df2-f594-11ea-3da6-556c32036252
# ╠═2c82f2ce-f594-11ea-3783-03e5a4cc6fd6
# ╠═ac6d32de-f593-11ea-3b1a-05c95b79d588
# ╠═06887410-f594-11ea-10e2-e100d41a5eef
# ╠═1e83e516-f594-11ea-2d89-25b37ea643f6
# ╠═f5af6bba-f593-11ea-3b07-89ad8519cf75
# ╠═f2f96916-f593-11ea-333e-cdb011981bf2
# ╠═ef5cd554-f593-11ea-38c4-b10dea199e36
# ╠═ec25f816-f593-11ea-3ee8-1dd8783f2630
# ╠═dc723f76-f593-11ea-11a4-ff75159e3627
# ╠═b62ec3f8-f593-11ea-329a-6d13db49f7a2
# ╠═c17a92b6-f593-11ea-1ec6-4110476c3488
# ╠═fde850e0-f592-11ea-13a6-5fa09cb8b1bb
# ╠═6aeea23e-f593-11ea-2661-57961f4ed2aa
# ╠═ca62cf68-f487-11ea-3265-0f08c21e42e0
