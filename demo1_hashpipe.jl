const global CACHE_ALIGNMENT = 8
const global N_INPUT_BLOCKS  = 3
const global N_OUTPUT_BLOCKS = 3

mutable struct demo1_input_block_header_t
  mcnt::UInt64
end

const global demo1_input_header_cache_alignment = CACHE_ALIGNMENT - (sizeof(demo1_input_block_header_t)%CACHE_ALIGNMENT)

mutable struct demo1_input_block_t
  header::demo1_input_block_header_t
  padding::Int = demo1_input_header_cached_alignment
  number1::UInt64
  number2::UInt64
end

mutable struct demo1_input_databuf_t 
  header::hashpipe_databuf_t
  padding::demo1_input_header_cache_alignment
  block::Array{demo1_input_block_t, N_INPUT_BLOCKS} 
end

# Output

mutable struct demo1_output_block_header_t
  mcnt::UInt64
end

const global demo1_output_header_cache_alignment = CACHE_ALIGNMENT - (sizeof(demo1_output_block_header_t)%CACHE_ALIGNMENT)

mutable struct demo1_output_block_t
  header::demo1_output_block_header_t
  padding::demo1_output_header_cached_alignment
  sum::UInt64
end

mutable struct demo1_output_databuf_t
  header::hashpipe_databuf_t
  padding::Int = demo1_output_header_cache_alignment
  block::Array{demo1_output_block_t, N_OUTPUT_BLOCKS}
end


