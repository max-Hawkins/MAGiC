include("../../jl-blio/src/GuppiRaw.jl")
using .GuppiRaw
include("./hashpipe.jl")

using .Hashpipe: update_status, hashpipe_status_t, HASHPIPE_OK, HASHPIPE_TIMEOUT
using .HPGuppi: hpguppi_input_databuf_t, N_INPUT_BLOCKS

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
		"--start_block"
			help = "hashpipe block to start processing on"
			arg_type = Int
			default = 0
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
	process_block(guppi_db::hpguppi_input_databuf_t, cur_block_in::Int)

Function where the actual block calculations go.
"""
function process_block(db::hpguppi_input_databuf_t, cur_block_in::Int)
	# grh: The GuppiRaw header dictionary of the current input block
	# raw_block: The array of complex voltage data of the current input block
	grh, raw_block = HPGuppi.get_data(db.blocks[cur_block_in + 1])

	# GuppiRaw block processing here
	#avg = mean(abs2.(raw_block)) # ~200ms
	#println("Block $cur_block_in avg: $avg")
end

"""
	main()

The primary calculation thread for hashpipe pipeline.
"""
function main()
	args = parse_commandline() # Parse command-line args
	
	instance_id    = args["inst_id"] # Hashpipe instance to attach to
	hp_input_db_id = args["input_db"] # Hashpipe databuf to read data from
	cur_block_in::Int   = args["start_block"] # Starting input block to process
	verbose        = args["verbose"]
	
	cur_block_out::Int = cur_block_in
	
	# Create hashpipe status and populate it with selected instance status values
	status = hashpipe_status_t(0,0,0,0)
	Hashpipe.hashpipe_status_attach(instance_id, Ref(status))
	# Hashpipe.display(status) # Uncomment to display the hashpipe instance status before starting work

	# Attach to hpguppi databuf
	hp_input_db = Hashpipe.hashpipe_databuf_attach(instance_id, hp_input_db_id)
    # Populate hpguppi_db with values of selected databuf
    hpguppi_db = HPGuppi.hpguppi_input_databuf_t(hp_input_db)

	time_per_block = 0

	# Main hashpipe calculation loop that waits for filled blocks and processes them
	while true
		# Lock status buffer before updating key/value pairs
		Hashpipe.hashpipe_status_buf_lock_unlock(Ref(status)) do 
				Hashpipe.update_status(status, "GPUBLKIN", cur_block_in);
				Hashpipe.update_status(status, "GPUBKOUT", cur_block_out);
				Hashpipe.update_status(status,  "GPUSTAT", "Waiting");
		end

		# Busy loop to wait for filled block
		while (rv=Hashpipe.hashpipe_databuf_wait_filled(hp_input_db, cur_block_in)) != Hashpipe.HASHPIPE_OK
			if rv==HASHPIPE_TIMEOUT
				println("GPU thread timeout waiting for filled block")
			else
				println("Error waiting for filled databuf - Error: $rv")
			end 
			# TODO: Finish checking
		end
		
        tick = time_ns()

		Hashpipe.hashpipe_status_buf_lock_unlock(Ref(status)) do
			Hashpipe.update_status(status, "GPUSTAT", "Processing");
			Hashpipe.update_status(status, "GPUBLKMS", string(time_per_block));
		end

		# Calculation on block data
		process_block(hpguppi_db, cur_block_in)

		# Free block after all calculations
		Hashpipe.hashpipe_databuf_set_free(hp_input_db, cur_block_in)

		# Increment through input blocks cycle
		cur_block_in  = (cur_block_in  + 1) % N_INPUT_BLOCKS
		cur_block_out = (cur_block_out + 1) % N_INPUT_BLOCKS

		# Calculate time elapsed
		tock = time_ns()
		time_per_block = Int(tock - tick) / 1e6
		#println("Elapsed/Block (ms): ", time_per_block) 
	end
end

main()

