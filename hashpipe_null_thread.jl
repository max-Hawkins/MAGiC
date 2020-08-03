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
cur_block::Int8  = 0 # Zero-indexed!

status = hashpipe_status_t(0,0,0,0)
r_status = Ref(status)

hashpipe_status_attach(instance_id, r_status)
display(status)

input_db  = hashpipe_databuf_attach(instance_id, 1)
output_db = hashpipe_databuf_attach(instance_id, 2)

while true
    while hashpipe_databuf_wait_filled(input_db, cur_block) != HASHPIPE_OK
        # TODO: Check for timeouts etc
    end

    println("\nInput DB Block $cur_block filled")

    in_data = unsafe_wrap(Array, Ptr{Int64}(hashpipe_databuf_data(input_db, cur_block)), 4)
    println("In data: $in_data")
    out_sum = in_data[3] + in_data[4]
    hashpipe_databuf_set_free(input_db, cur_block)

    println("Out sum: $out_sum")
    out_data = unsafe_wrap(Array, Ptr{Int64}(hashpipe_databuf_data(output_db, cur_block)), 3)
    println("Out_data before: $out_data")
    out_data[3] = out_sum
    println("Out after: $out_data")

    hashpipe_databuf_set_filled(output_db, cur_block)
    global cur_block = (cur_block + 1) % NUM_BLOCKS
end