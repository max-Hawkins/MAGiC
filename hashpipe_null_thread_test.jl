include("./hashpipe.jl")
using ArgParse
# HPGUPPI_DATABUF.h constants
const ALIGNMENT_SIZE = 4096
const N_INPUT_BLOCKS = 24
const BLOCK_HDR_SIZE = 5*80*512
const BLOCK_DATA_SIZE = 128*1024*1024
const padding_size = ALIGNMENT_SIZE - (sizeof(hashpipe_databuf_t)%ALIGNMENT_SIZE)
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
mutable struct hpguppi_input_block_t
    hdr::Ptr{Int8}
    data::Ptr{Int8}
end

mutable struct hpguppi_input_databuf_t
    p_header::hashpipe_databuf_t
    padding::NTuple{padding_size, Int8}
    p_blocks::Ptr{Any}
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

function get_block_arrays(p_input_db)
    blocks_array = Array{N_INPUT_BLOCKS, hpguppi_input_block_t}[]
    p_blocks = sizeof(hashpipe_databuf_t) + padding_size 

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
    
    get_block_arrays(input_db)

	println(status.p_buf)
	time_per_block = 1

	while true
		tick = time_ns()

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

		hashpipe_status_buf_lock_unlock(r_status) do
			hputs(status.p_buf, status_key, "processing gpu");
			#hputs(status.p_buf, "T/BLKMS", time_per_block);
		end

		hashpipe_databuf_set_free(input_db, cur_block_in)
		cur_block_in = (cur_block_in + 1) % N_INPUT_BLOCKS
		tock = time_ns()
		time_per_block = Int(tock - tick) / 1e6
		print("Elapsed (ms): ",time_per_block) 
	end
end

main()

