include("./hashpipe.jl")
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
    hdr::NTuple{BLOCK_HDR_SIZE, Cchar}
    data::NTuple{BLOCK_DATA_SIZE, Int8}
end

mutable struct hpguppi_input_databuf_t
    p_header::hashpipe_databuf_t
    padding::NTuple{padding_size, Int8}
    p_blocks::Ptr{Any}
end

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


while true

    hashpipe_status_lock(r_status);
    hputi4(status.p_buf, Cstring(pointer("GPUBLKIN")), Cint(cur_block_in));
    hputs(status.p_buf, status_key, "waiting");
    hputi4(status.p_buf, Cstring(pointer("GPUBKOUT")), Cint(cur_block_out));
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
    hputs(status.p_buf, status_key, "processing gpu");
    hashpipe_status_unlock(r_status);

    println("\nInput DB Block $cur_block_in filled")

    
    
    hashpipe_databuf_set_free(input_db, cur_block_in)
    global cur_block_in = (cur_block_in + 1) % N_INPUT_BLOCKS
end
