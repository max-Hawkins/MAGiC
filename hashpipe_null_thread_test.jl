include("./hashpipe.jl")
include("../jl-blio/src/GuppiRaw.jl")

using ArgParse
# HPGUPPI_DATABUF.h constants
const ALIGNMENT_SIZE = 4096
const N_INPUT_BLOCKS = 24
const BLOCK_HDR_SIZE = 5*80*512
const BLOCK_DATA_SIZE = 128*1024*1024
const PADDING_SIZE = ALIGNMENT_SIZE - (sizeof(hashpipe_databuf_t)%ALIGNMENT_SIZE)
const BLOCK_SIZE = BLOCK_HDR_SIZE + BLOCK_DATA_SIZE

# typedef struct hpguppi_input_block {
#     char hdr[BLOCK_HDR_SIZE];
#     char data[BLOCK_DATA_SIZE];
#   } hpguppi_input_block_t;
  
#   // Used to pad after hashpipe_databuf_t to maintain data alignment
#   typedef uint8_t hashpipe_databuf_alignment[
#     ALIGNMENT_SIZE - (sizeof(hashpipe_databuf_t)%ALIGNMENT_SIZE)
#   ];
  
#   typedef struct hpguppi_input_databuf {
#     hashpipe_databuf_t header;
#     hashpipe_databuf_alignment padding; // Maintain data alignment
#     hpguppi_input_block_t block[N_INPUT_BLOCKS];
#   } hpguppi_input_databuf_t;
  
# looked into using StaticArrays, but stated large arrays are best kept as Array
struct hpguppi_input_block_t
    p_hdr::Ptr{UInt8}
    p_data::Ptr{Any}
end

mutable struct hpguppi_input_databuf_t
    p_hpguppi_db::Ptr{hashpipe_databuf_t}
    blocks::Array{hpguppi_input_block_t}
end

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
        grh[k] = v
    end
    # TODO: Make custom function in GuppiRaw.jl to do this parsing from a pointer. Figure out ideal array resizing for CUDA
    model_array = Array(grh)
    dims = size(model_array)
    data = unsafe_wrap(Array{eltype(model_array)}, Ptr{eltype(model_array)}(input_block.p_data), dims)
    return grh, data
end


function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table s begin
		"--inst_id"
			help = "hashpipe instance id to attach to"
			arg_type = Int
			default = 0
		"--input_db"
			help = "input hashpipe databuf to get raw data from"
			arg_type = Int
			default = 2
		"--verbose"
			help = "verbose output"
			action = :store_true
	end
		
	args = parse_args(s)
	println(args)
	if args["verbose"]
		println("Parsed args:")
		for (arg,val) in args
			println(" $arg => $val")
		end
	end
	return args
end

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

function main()
	args = parse_commandline()
	
	instance_id = args["inst_id"]
	input_db_id = args["input_db"]
	verbose     = args["verbose"]
	
	instance_id = 0
	input_db_id = 2
	cur_block_in = 0
	cur_block_out = 0
	status_key = "GPUSTAT"  
	
	status = hashpipe_status_t(0,0,0,0)
	r_status = Ref(status)

	hashpipe_status_attach(instance_id, r_status)
	display(status)

	input_db = hashpipe_databuf_attach(instance_id, input_db_id)
    
    # get_block_arrays(input_db)

	println(status.p_buf)
	time_per_block = 1

	while true

		hashpipe_status_buf_lock_unlock(r_status) do 
				hputi4(status.p_buf, Cstring(pointer("GPUBLKIN")), Cint(cur_block_in));
				hputs(status.p_buf, status_key, "waiting");
				hputi4(status.p_buf, Cstring(pointer("GPUBKOUT")), Cint(cur_block_out));
		end

		while (rv=hashpipe_databuf_wait_filled(input_db, cur_block_in)) != HASHPIPE_OK
			if rv==HASHPIPE_TIMEOUT
				# println("GPU Timeout")
			else
				println("Error waiting for filled databuf")
			end 
			# TODO: Finish checking
		end
		
        tick = time_ns()

		hashpipe_status_buf_lock_unlock(r_status) do
			hputs(status.p_buf, status_key, "processing gpu");
			hputs(status.p_buf, "GPUBLKMS", string(time_per_block));
		end

		hashpipe_databuf_set_free(input_db, cur_block_in)
		cur_block_in = (cur_block_in + 1) % N_INPUT_BLOCKS
		tock = time_ns()
		time_per_block = Int(tock - tick) / 1e6
		print("Elapsed (ms): ",time_per_block) 
	end
end

main()

