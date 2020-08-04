include("./hashpipe.jl")

mutable struct demo1_input_block_header_t
    mcnt::UInt64
  end

mutable struct demo1_input_block_t
    mcnt::Int64
    padding::Int64
    number1::Int64
    number2::Int64
  end

function load_data()

end
instance_id = 0
NUM_BLOCKS = 3
cur_block_in  = 0 # Zero-indexed!
cur_block_out = 0 # Zero-indexed
status_key = "GPUSTAT"

status = hashpipe_status_t(0,0,0,0)
r_status = Ref(status)

hashpipe_status_attach(instance_id, r_status)
display(status)

input_db  = hashpipe_databuf_attach(instance_id, 1)
output_db = hashpipe_databuf_attach(instance_id, 2)

while true

    hashpipe_status_lock(r_status);
    hputi4(r_stat.p_buf, "GPUBLKIN", cur_block_in);
    hputs(r_stat.p_buf, status_key, "waiting");
    hputi4(r_stat.p_buf, "GPUBKOUT", cur_block_out);
    # hputi8(r_stat.p_buf,"GPUMCNT",mcnt);
    hashpipe_status_unlock(r_status);
	# sleep(1);
    #     // Wait for new input block to be filled
    #     while ((rv=demo1_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
    #         if (rv==HASHPIPE_TIMEOUT) {
    #             hashpipe_status_lock(r_status);
    #             hputs(r_stap_buf, status_key, "blocked");
    #             hashpipe_status_unlock(r_status);
    #             continue;
    #         } else {
    #             hashpipe_error(__FUNCTION__, "error waiting for filled databuf");
    #             pthread_exit(NULL);
    #             break;
    #         }
    #     }

    while (rv=hashpipe_databuf_wait_filled(input_db, cur_block_in)) != HASHPIPE_OK
        if rv==HASHPIPE_TIMEOUT
            # println("GPU Timeout")
        else
            println("Error waiting for filled databuf")
        end 
        # TODO: Finish checking
    end

    hashpipe_status_lock(r_status);
    hputs(r_stat.p_buf, status_key, "processing gpu");
    hashpipe_status_unlock(r_status);

    println("\nInput DB Block $cur_block_in filled")

    in_data = unsafe_wrap(Array, Ptr{Int64}(hashpipe_databuf_data(input_db, cur_block_in)), 4)
    println("In data: $in_data")
    out_sum = in_data[3] + in_data
    println("Out sum: $out_sum")
    
    hashpipe_databuf_set_free(input_db, cur_block_in)
    global cur_block_in = (cur_block_in + 1) % NUM_BLOCKS

    out_data = unsafe_wrap(Array, Ptr{Int64}(hashpipe_databuf_data(output_db, cur_block_out)), 3)
    println("Out_data before: $out_data")
    out_data[3] = out_sum
    println("Out after: $out_data")

    hashpipe_databuf_set_filled(output_db, cur_block_out)
    global cur_block_out = (cur_block_out + 1) % NUM_BLOCKS

    hashpipe_status_lock(r_stat);
	hputi4(r_stat.p_buf,"GPUSUM",out_sum);
	hashpipe_status_unlock(r_stat);
    
end