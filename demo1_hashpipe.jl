const global CACHE_ALIGNMENT = 8
const global N_INPUT_BLOCKS  = 3
const global N_OUTPUT_BLOCKS = 3

struct demo1_input_block_header {
  mcnt::UInt64
} demo1_input_block_header_t;

# TODO: cache alignment

struct demo1_input_block {
  header::demo1_input_block_header_t
  padding::demo1_input_header_cached_alignment
  number1::UInt64
  number2::UInt64
} demo1_input_block;

struct demo1_input_databuf {
  header::hashpipe_databuf_t
  padding::demo1_input_header_cache_alignment
  block::Array{demo1_input_block_t, N_INPUT_BLOCKS} 
} demo1_input_databuf_t;


# Output


