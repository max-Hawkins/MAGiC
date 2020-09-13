"""
hashpipe.jl

C Hashpipe functions ported for Julia usability.
Written by Max Hawkins
Hashpipe C code written by Dave MacMahon: https://github.com/david-macmahon/hashpipe
"""
module Hashpipe

include("/home/mhawkins/jl-blio/src/GuppiRaw.jl")
using .GuppiRaw
# Hashpipe error Codes
const global HASHPIPE_OK         =  0
const global HASHPIPE_TIMEOUT    =  1 # Call timed out 
const global HASHPIPE_ERR_GEN    = -1 # Super non-informative
const global HASHPIPE_ERR_SYS    = -2 # Failed system call
const global HASHPIPE_ERR_PARAM  = -3 # Parameter out of range
const global HASHPIPE_ERR_KEY    = -4 # Requested key doesn't exist
const global HASHPIPE_ERR_PACKET = -5 # Unexpected packet size

"""Hashpipe databuf struct"""
struct hashpipe_databuf_t
    data_type::NTuple{64, UInt8}
    header_size::Int # May need to change to Csize_t
    block_size::Int # May need to change to Csize_t
    n_block::Cint
    shmid::Cint
    semid::Cint
end

# Status constants
const global HASHPIPE_STATUS_TOTAL_SIZE = 184320 # 2880 * 64
const global HASHPIPE_STATUS_RECORD_SIZE = 80
# HPGUPPI_DATABUF.h constants
const ALIGNMENT_SIZE = 4096
const N_INPUT_BLOCKS = 24
const BLOCK_HDR_SIZE = 5*80*512
const BLOCK_DATA_SIZE = 128*1024*1024
const PADDING_SIZE = ALIGNMENT_SIZE - (sizeof(hashpipe_databuf_t)%ALIGNMENT_SIZE)
const BLOCK_SIZE = BLOCK_HDR_SIZE + BLOCK_DATA_SIZE


"""
Hashpipe Status struct

May need to create empty status struct before trying to attaching
to existing status buffer.
Example:
    instance_id = 0
    status = hashpipe_status_t(0,0,0,0)
    r_status = Ref(status)
    hashpipe_status_attach(instance_id, r_status)
"""
mutable struct hashpipe_status_t
    instance_id::Cint
    shmid::Cint
    p_lock::Ptr{UInt8} 
    p_buf::Ptr{UInt8}
end

"""
hpguppi input block

Used to hold the pointers to individual GuppiRaw
header and data pointers.
"""
struct hpguppi_input_block_t
    p_hdr::Ptr{UInt8}
    p_data::Ptr{Any}
end

"""
hpguppi input databuf


"""
mutable struct hpguppi_input_databuf_t
    p_hpguppi_db::Ptr{hashpipe_databuf_t}
    blocks::Array{hpguppi_input_block_t}
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

#----------#
# Displays #
#----------#

"Display hashpipe status"
function display(s::hashpipe_status_t)
    BUFFER_MAX_RECORDS = Int(HASHPIPE_STATUS_TOTAL_SIZE / 80)
    println("Instance ID: $(s.instance_id)")
    println("shmid: $(s.shmid)")
    lock = unsafe_wrap(Array, s.p_lock, (1))[1]
    println("Lock: $lock")

    println("Buffer:")    
    string_array = unsafe_wrap(Array, s.p_buf, (HASHPIPE_STATUS_RECORD_SIZE, BUFFER_MAX_RECORDS))
    for record in 1:size(string_array, 2)
        record_string = String(string_array[:, record])
        println("\t", record_string)
        if record_string[1:3] == "END"
            return nothing
        end
    end
    return nothing
end

"Display hashpipe status from reference"
function display(r::Ref{hashpipe_status_t})
    display(r[])
    return nothing
end

"Display hashpipe buffer"
function display(d::hashpipe_databuf_t)
    # Convert Ntuple to array and strip 0s before converting to string
    data_type_string = String(filter(x->x!=0x00, collect(d.data_type)))
    println("Data Type: $(data_type_string)")
    println("Header Size: $(d.header_size)")
    println("Num Blocks: $(d.n_block)")
    println("Block Size: $(d.block_size)")
    println("shmid: $(d.shmid)")
    println("semid: $(d.semid)")
    return nothing
end

"Display hashpipe databuf from pointer"
function display(p::Ptr{hashpipe_databuf_t})
    databuf = unsafe_wrap(Array, p, 1)[]
    display(databuf)
    return nothing
end

#---------------------------#
# Hashpipe Status Functions #
#---------------------------#

# TODO: wrap with error checking based on function
# Returns 0 with error
function hashpipe_status_exists(instance_id::Int)
    exists::Int8 = ccall((:hashpipe_status_exists, 
                "libhashpipestatus.so"), 
                Int8, (Int8,), instance_id)
    return exists
end

function hashpipe_status_attach(instance_id::Int, p_hashpipe_status::Ref{hashpipe_status_t})
    error::Int8 = ccall((:hashpipe_status_attach, "libhashpipestatus.so"),
                    Int, (Int8, Ref{hashpipe_status_t}), instance_id, p_hashpipe_status)
    return error
end

function hashpipe_status_lock(p_hashpipe_status::Ref{hashpipe_status_t})
    error::Int8 = ccall((:hashpipe_status_lock, "libhashpipestatus.so"),
                    Int, (Ref{hashpipe_status_t},), p_hashpipe_status)
    return error
end

function hashpipe_status_unlock(p_hashpipe_status::Ref{hashpipe_status_t})
    error::Int8 = ccall((:hashpipe_status_unlock, "libhashpipestatus.so"),
                    Int, (Ref{hashpipe_status_t},), p_hashpipe_status)
    return error
end

function hashpipe_status_buf_lock_unlock(f::Function, st::Ref{hashpipe_status_t})
        try
            hashpipe_status_lock(st)
            f() # or f(st) TODO: test which of these is better
        catch
            println("Error locking hashpipe status buffer.")
        finally
            hashpipe_status_unlock(st)
        end
end

function hashpipe_status_clear(p_hashpipe_status::Ref{hashpipe_status_t})
    ccall((:hashpipe_status_clear, "libhashpipestatus.so"),
            Int, (Ref{hashpipe_status_t},), p_hashpipe_status)
    return nothing
end

#----------------------------#
# Hashpipe Databuf Functions #
#----------------------------#

function hashpipe_databuf_data(p_databuf::Ptr{hashpipe_databuf_t}, block_id::Int)
    p_data::Ptr{UInt8} = ccall((:hashpipe_databuf_data, "libhashpipe.so"),
                            Ptr{UInt8}, (Ptr{hashpipe_status_t}, Int8), p_databuf, block_id)
    return p_data
end

function hashpipe_databuf_create(instance_id::Int, db_id::Int,
            header_size::Int, block_size::Int, n_block::Int)
    p_databuf::Ptr{hashpipe_databuf_t} = 
            ccall((:hashpipe_databuf_create, "libhashpipe.so"),
                Ptr{hashpipe_databuf_t},
                (Int8, Int8, Int, Int, Int),
                instance_id, db_id, header_size, block_size, n_block)
    return p_databuf
end

function hashpipe_databuf_clear(p_databuf::Ptr{hashpipe_databuf_t})
    ccall((:hashpipe_databuf_clear, "libhashpipe.so"),
            Cvoid, (Ptr{hashpipe_status_t},), p_databuf)
    return nothing
end
function hashpipe_databuf_attach(instance_id::Int, db_id::Int)
    p_databuf::Ptr{hashpipe_databuf_t} = ccall((:hashpipe_databuf_attach, "libhashpipe.so"),
                    Ptr{hashpipe_databuf_t}, (Int8, Int8), instance_id, db_id)
    return p_databuf
end

function hashpipe_databuf_detach(p_databuf::Ptr{hashpipe_databuf_t})
    error::Int = ccall((:hashpipe_databuf_attach, "libhashpipe.so"),
                    Int, (Ptr{hashpipe_databuf_t},), p_databuf)
    return error
end

# Check hashpipe databuf status
function hashpipe_check_databuf(instance_id::Int = 0, db_id::Int = 1)
    p_databuf = hashpipe_databuf_attach(instance_id, db_id)
    if p_databuf == C_NULL
        println("Error attaching to databuf $db_id (may not exist).")
        return nothing
    end
    println("--- Databuf $db_id Stats ---")
    display(p_databuf)
    return nothing
end

function hashpipe_databuf_block_status(p_databuf::Ptr{hashpipe_databuf_t}, block_id::Int)
    block_status::Int = ccall((:hashpipe_databuf_block_status, "libhashpipe.so"),
                    Int, (Ptr{hashpipe_databuf_t}, Int), p_databuf, block_id)
    return block_status
end

# Return total lock status for databuf
function hashpipe_databuf_total_status(p_databuf::Ptr{hashpipe_databuf_t})
    total_status::Int = ccall((:hashpipe_databuf_total_status, "libhashpipe.so"),
                    Int, (Ptr{hashpipe_databuf_t},), p_databuf)
    return total_status
end

function hashpipe_databuf_total_mask(p_databuf::Ptr{hashpipe_databuf_t})
    total_mask::UInt64 = ccall((:hashpipe_databuf_total_mask, "libhashpipe.so"),
                    UInt64, (Ptr{hashpipe_databuf_t},), p_databuf)
    return total_mask
end

# Databuf locking functions.  Each block in the buffer
# can be marked as free or filled.  The "wait" functions
# block (i.e. sleep) until the specified state happens.
# The "busywait" functions busy-wait (i.e. do NOT sleep)
# until the specified state happens.  The "set" functions
# put the buffer in the specified state, returning error if
# it is already in that state.
 
function hashpipe_databuf_wait_filled(p_databuf::Ptr{hashpipe_databuf_t}, block_id::Int)
    error::Int = ccall((:hashpipe_databuf_wait_filled, "libhashpipe.so"),
                    Int, (Ptr{hashpipe_databuf_t}, Int), p_databuf, block_id)
    return error
end

function hashpipe_databuf_wait_free(p_databuf::Ptr{hashpipe_databuf_t}, block_id::Int)
    error::Int = ccall((:hashpipe_databuf_wait_free, "libhashpipe.so"),
                    Int, (Ptr{hashpipe_databuf_t}, Int), p_databuf, block_id)
    return error
end

function hashpipe_databuf_set_filled(p_databuf::Ptr{hashpipe_databuf_t}, block_id::Int)
    error::Int = ccall((:hashpipe_databuf_set_filled, "libhashpipe.so"),
                    Int, (Ptr{hashpipe_databuf_t}, Int), p_databuf, block_id)
    return error
end

function hashpipe_databuf_set_free(p_databuf::Ptr{hashpipe_databuf_t}, block_id::Int)
    error::Int = ccall((:hashpipe_databuf_set_free, "libhashpipe.so"),
                    Int, (Ptr{hashpipe_databuf_t}, Int), p_databuf, block_id)
    return error
end

#----------------#
# Hput Functions #
#----------------#

function hputs(p_hstring::Ptr{UInt8}, p_keyword::Cstring, p_cval::Cstring )
    error::Int = ccall((:hputs, "libhashpipestatus.so"),
                    Int, (Ptr{UInt8}, Cstring, Cstring),
                    p_hstring, p_keyword, p_cval)
    return error
end
# Auto-convert Julia string to Cstring
function hputs(p_hstring::Ptr{UInt8}, p_keyword::String, p_cval::String)
    error::Int = ccall((:hputs, "libhashpipestatus.so"),
                    Int, (Ptr{UInt8}, Cstring, Cstring),
                    p_hstring, Cstring(pointer(p_keyword)), Cstring(pointer(p_cval)))
    return error
end

function hputi4(p_hstring::Ptr{UInt8}, p_keyword::Cstring, p_ival::Cint)
    error::Int = ccall((:hputi4, "libhashpipestatus.so"),
                    Int, (Ptr{UInt8}, Cstring, Cint),
                    p_hstring, p_keyword, p_ival)
    return error
end
# Auto-convert julia string/int to Cstring/Cint
function hputi4(p_hstring::Ptr{UInt8}, p_keyword::String, p_ival::Int)
    error::Int = ccall((:hputi4, "libhashpipestatus.so"),
                    Int, (Ptr{UInt8}, Cstring, Cint),
                    p_hstring, Cstring(pointer(p_keyword)), Cint(p_ival))
    return error
end

end # Module Hashpipe