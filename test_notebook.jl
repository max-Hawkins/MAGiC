### A Pluto.jl notebook ###
# v0.11.12

using Markdown
using InteractiveUtils

# ╔═╡ 80f04f3e-f2e0-11ea-0822-db2baa17090c
begin
	using ArgParse
	 # HPGUPPI_DATABUF.h constants
	 const ALIGNMENT_SIZE = 4096
	 const N_INPUT_BLOCKS = 24
	 const BLOCK_HDR_SIZE = 5*80*512
	 const BLOCK_DATA_SIZE = 128*1024*1024
	 const PADDING_SIZE = ALIGNMENT_SIZE - (sizeof(hashpipe_databuf_t)%ALIGNMENT_SIZE)
	 const BLOCK_SIZE = BLOCK_HDR_SIZE + BLOCK_DATA_SIZE
end

# ╔═╡ 8239a990-f2df-11ea-2ea7-d7408fc11be9
include("../jl-blio/src/GuppiRaw.jl")


# ╔═╡ 488744c8-f2df-11ea-221d-99013965272c
pwd()

# ╔═╡ 56e9814a-f2e2-11ea-27ae-537542703a3a


# ╔═╡ b60c00cc-f2e1-11ea-0d58-4f660584662f


# ╔═╡ afa45912-f2e1-11ea-3995-55f5c58d5248


# ╔═╡ 58dae80c-f2df-11ea-2c89-db86a5698a7e
readdir("../jl-blio/src/")

# ╔═╡ e8dc3262-f2e2-11ea-07a2-4764e2c2803d
GuppiRaw


# ╔═╡ 54c97766-f2e3-11ea-2f4d-3f79dd50ba7c


# ╔═╡ 5100a1d6-f2e3-11ea-04b9-15f99a4f76d8


# ╔═╡ f1a25f04-f2e2-11ea-139e-79871bed5301


# ╔═╡ e64d1e64-f2e2-11ea-0761-b1e57374cc8f


# ╔═╡ e353840a-f2e2-11ea-24c4-03c2ef197359


# ╔═╡ d8a85738-f2e2-11ea-36cf-f3bf9d6bbccf


# ╔═╡ 3450d338-f2e2-11ea-2550-3531d95a0e1d


# ╔═╡ 317a5c00-f2e1-11ea-1db8-39da0ceb489a
begin
	struct hpguppi_input_block_t
	     p_hdr::Ptr{UInt8}
	     p_data::Ptr{Any}
	 end
	  mutable struct hpguppi_input_databuf_t
	     p_hpguppi_db::Ptr{hashpipe_databuf_t}
	     blocks::Array{hpguppi_input_block_t}
	 end
end

# ╔═╡ 57ae3400-f2e1-11ea-2dc2-11f84c83b7e0
function databuf_init(p_input_db::Ptr{hashpipe_databuf_t})
    blocks_array::Array{hpguppi_input_block_t} = []
    p_blocks = p_input_db + sizeof(hashpipe_databuf_t) + PADDING_SIZE
    for i = 0:N_INPUT_BLOCKS - 1
        p_header = p_blocks + i * BLOCK_SIZE
        p_data = p_header + BLOCK_HDR_SIZE
        push!(blocks_array, hpguppi_input_block_t(p_header, p_data))
    end
    return hpguppi_input_databuf_t(p_input_db, blocks_array)
end

# ╔═╡ e2cf2012-f2e1-11ea-3043-47f35b65d589
function get_data(input_block::hpguppi_input_block_t)
      grh = GuppiRaw.Header()
      # TODO: Fix Int8 conversion
      buf = reshape(unsafe_wrap(Array, input_block.p_hdr, BLOCK_HDR_SIZE), (GuppiRaw.HEADER_REC_SIZE, :))
      endidx = findfirst(c->buf[1:4,c] == GuppiRaw.END, 1:size(buf,2))
 
      for i in 1:endidx-1
          rec = String(buf[:,i])
          k, v = split(rec, '=', limit=2)
          k = Symbol(lowercase(strip(k)))
          v = strip(v)
          if v[1] == '\''
              v = strip(v, [' ', '\''])
          elseif !isnothing(match(r"^[+-]?[0-9]+$", v))
              v = parse(Int, v)
          elseif !isnothing(tryparse(Float64, v))
              v = parse(Float64, v)
          end
	end
     model_array = Array(grh)
     dims = size(model_array)
     data = unsafe_wrap(Array{eltype(model_array)}, Ptr{eltype(model_array)}(input_block.p_data), dims)
     return grh, data
end

# ╔═╡ 001b30fc-f2e2-11ea-3e22-a97e6fb9704c


# ╔═╡ 2e0d66c0-f2e1-11ea-1f79-bf70df5e422a
p_input_db = hashpipe_databuf_attach(0,2)

# ╔═╡ 82be4090-f2e1-11ea-3e98-37157274f459
input_databuf = databuf_init(p_input_db)

# ╔═╡ c1646aa4-f2e1-11ea-2a44-ef56f2a529eb
grh, raw_block = get_data(input_databuf.blocks[1])

# ╔═╡ 1a7371e4-f2e2-11ea-149f-49944509c532


# ╔═╡ 0d007f0c-f2e2-11ea-2e08-c519e6af591d


# ╔═╡ 29260464-f2e1-11ea-17d7-5b3536a544d3


# ╔═╡ 1e9d8f80-f2e1-11ea-2422-1f5dab5e805e


# ╔═╡ 560a530a-f2e0-11ea-029f-1b2a31bf4537


# ╔═╡ 54249884-f2e0-11ea-03f0-19b3d936db2f


# ╔═╡ 50068afa-f2e0-11ea-130c-9dbb9b45bcd1


# ╔═╡ 2f32f110-f2e0-11ea-1019-49844e4c5d51


# ╔═╡ 06edd0e4-f2e0-11ea-1c3d-dbcf82c1b6ac


# ╔═╡ 9256d96a-f2df-11ea-0736-1151890f949e


# ╔═╡ 7ae134ce-f2df-11ea-3e6f-cf8d46f965ca


# ╔═╡ 77bfc508-f2df-11ea-1cbd-8f696f8410c7


# ╔═╡ 7402fa36-f2df-11ea-2084-65507b5c961a


# ╔═╡ 62319842-f2df-11ea-239e-07885043a53a


# ╔═╡ 571c07b2-f2df-11ea-1770-a5b23b34dd08


# ╔═╡ Cell order:
# ╠═488744c8-f2df-11ea-221d-99013965272c
# ╠═56e9814a-f2e2-11ea-27ae-537542703a3a
# ╠═b60c00cc-f2e1-11ea-0d58-4f660584662f
# ╠═afa45912-f2e1-11ea-3995-55f5c58d5248
# ╠═58dae80c-f2df-11ea-2c89-db86a5698a7e
# ╠═8239a990-f2df-11ea-2ea7-d7408fc11be9
# ╠═e8dc3262-f2e2-11ea-07a2-4764e2c2803d
# ╠═54c97766-f2e3-11ea-2f4d-3f79dd50ba7c
# ╠═5100a1d6-f2e3-11ea-04b9-15f99a4f76d8
# ╠═f1a25f04-f2e2-11ea-139e-79871bed5301
# ╠═e64d1e64-f2e2-11ea-0761-b1e57374cc8f
# ╠═e353840a-f2e2-11ea-24c4-03c2ef197359
# ╠═d8a85738-f2e2-11ea-36cf-f3bf9d6bbccf
# ╠═3450d338-f2e2-11ea-2550-3531d95a0e1d
# ╠═80f04f3e-f2e0-11ea-0822-db2baa17090c
# ╠═317a5c00-f2e1-11ea-1db8-39da0ceb489a
# ╠═57ae3400-f2e1-11ea-2dc2-11f84c83b7e0
# ╠═e2cf2012-f2e1-11ea-3043-47f35b65d589
# ╠═001b30fc-f2e2-11ea-3e22-a97e6fb9704c
# ╠═2e0d66c0-f2e1-11ea-1f79-bf70df5e422a
# ╠═82be4090-f2e1-11ea-3e98-37157274f459
# ╠═c1646aa4-f2e1-11ea-2a44-ef56f2a529eb
# ╠═1a7371e4-f2e2-11ea-149f-49944509c532
# ╠═0d007f0c-f2e2-11ea-2e08-c519e6af591d
# ╠═29260464-f2e1-11ea-17d7-5b3536a544d3
# ╠═1e9d8f80-f2e1-11ea-2422-1f5dab5e805e
# ╠═560a530a-f2e0-11ea-029f-1b2a31bf4537
# ╠═54249884-f2e0-11ea-03f0-19b3d936db2f
# ╠═50068afa-f2e0-11ea-130c-9dbb9b45bcd1
# ╠═2f32f110-f2e0-11ea-1019-49844e4c5d51
# ╠═06edd0e4-f2e0-11ea-1c3d-dbcf82c1b6ac
# ╠═9256d96a-f2df-11ea-0736-1151890f949e
# ╠═7ae134ce-f2df-11ea-3e6f-cf8d46f965ca
# ╠═77bfc508-f2df-11ea-1cbd-8f696f8410c7
# ╠═7402fa36-f2df-11ea-2084-65507b5c961a
# ╠═62319842-f2df-11ea-239e-07885043a53a
# ╠═571c07b2-f2df-11ea-1770-a5b23b34dd08
