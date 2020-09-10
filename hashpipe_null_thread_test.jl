include("./hashpipe.jl")
include("../jl-blio/src/GuppiRaw.jl")

using ArgParse, Statistics

"""
Parse commandline arguments for hashpipe calculation thread.
"""
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

"""
process_block(GuppiRaw.Header, hpguppi_input_block)

Function where the actual block calculations go.
"""
function process_block(guppi_db::hpguppi_input_databuf_t, cur_block_in::Int)
	# grh: The GuppiRaw header dictionary of the current input block
	# raw_block: The array of complex voltage data of the current input block
	grh, raw_block = get_data(guppi_db.blocks[cur_block_in + 1])

	# GuppiRaw block processing here
	avg = mean(abs2.(raw_block)) # ~200ms
	println("Block $cur_block_in avg: $avg")
end

"""
main

The primary calculation thread for hashpipe pipeline.
"""
function main()
	args = parse_commandline()
	
	instance_id = args["inst_id"]
	input_db_id = args["input_db"]
	verbose     = args["verbose"]
	
	instance_id = 0
	hp_input_db_id = 2
	cur_block_in = 0
	cur_block_out = 0
	
	status = hashpipe_status_t(0,0,0,0)
	r_status = Ref(status)

	hashpipe_status_attach(instance_id, r_status)
	display(status)

	hp_input_db = hashpipe_databuf_attach(instance_id, hp_input_db_id)
    
    guppi_db = databuf_init(hp_input_db)

	println(status.p_buf)
	time_per_block = 1

	# Main hashpipe calculation loop that waits for filled blocks and processes them
	while true

		hashpipe_status_buf_lock_unlock(r_status) do 
				hputi4(status.p_buf, Cstring(pointer("GPUBLKIN")), Cint(cur_block_in));
				hputi4(status.p_buf, Cstring(pointer("GPUBKOUT")), Cint(cur_block_out));
				hputs(status.p_buf,  "GPUSTAT", "Waiting");
				
		end
		# Busy loop to wait for filled block
		while (rv=hashpipe_databuf_wait_filled(hp_input_db, cur_block_in)) != HASHPIPE_OK
			if rv==HASHPIPE_TIMEOUT
				println("GPU thread timeout waiting for filled block")
			else
				println("Error waiting for filled databuf")
			end 
			# TODO: Finish checking
		end
		
        tick = time_ns()

		hashpipe_status_buf_lock_unlock(r_status) do
			hputs(status.p_buf, "GPUSTAT", "Processing");
			hputs(status.p_buf, "GPUBLKMS", string(time_per_block));
		end

		# Calculation on block data
		process_block(guppi_db, cur_block_in)

		hashpipe_databuf_set_free(hp_input_db, cur_block_in)
		cur_block_in = (cur_block_in + 1) % N_INPUT_BLOCKS

		# Calculate time elapsed and print
		tock = time_ns()
		time_per_block = Int(tock - tick) / 1e6
		println("Elapsed/Block (ms): ", time_per_block) 
	end
end

main()

