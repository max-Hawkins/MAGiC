export Hashpipe

"""
    Hashpipe Module

C Hashpipe functions ported for Julia usability.
Written by Max Hawkins
Hashpipe C code written by Dave MacMahon: https://github.com/david-macmahon/hashpipe
"""
module Hashpipe

export hashpipe_databuf_t, hashpipe_status_t

# Hashpipe error Codes
const global HASHPIPE_OK         =  0
const global HASHPIPE_TIMEOUT    =  1 # Call timed out 
const global HASHPIPE_ERR_GEN    = -1 # Super non-informative
const global HASHPIPE_ERR_SYS    = -2 # Failed system call
const global HASHPIPE_ERR_PARAM  = -3 # Parameter out of range
const global HASHPIPE_ERR_KEY    = -4 # Requested key doesn't exist
const global HASHPIPE_ERR_PACKET = -5 # Unexpected packet size

"""
    hashpipe_databuf_t
    
"""
struct hashpipe_databuf_t
    data_type::NTuple{64, UInt8}
    header_size::Int # May need to change to Csize_t
    block_size::Int # May need to change to Csize_t
    n_block::Cint
    shmid::Cint
    semid::Cint
end

# Status constants
const global STATUS_TOTAL_SIZE = 184320 # 2880 * 64
const global STATUS_RECORD_SIZE = 80

"""
Hashpipe Status struct

May need to create empty status struct before trying to attaching
to existing status buffer.
Example:
'''
    instance_id = 0
    status = hashpipe_status_t(0,0,0,0)
    hashpipe_status_attach(instance_id, Ref(r_status))
'''
"""
mutable struct hashpipe_status_t
    instance_id::Cint
    shmid::Cint
    p_lock::Ptr{UInt8} 
    p_buf::Ptr{UInt8}
end



#----------#
# Displays #
#----------#

"Display hashpipe status"
function Base.display(s::hashpipe_status_t)
    BUFFER_MAX_RECORDS = Int(STATUS_TOTAL_SIZE / STATUS_RECORD_SIZE)
    println("Instance ID: $(s.instance_id)")
    println("shmid: $(s.shmid)")

    # Check to see if valid pointer
    if s.p_lock != C_NULL
        lock = unsafe_wrap(Array, s.p_lock, (1))[1]
        println("Lock: $lock")
    else
        println("Lock: NULL")
    end

    # Check to see if valid pointer
    if s.p_buf != C_NULL
        println("Buffer:")    
        string_array = unsafe_wrap(Array, s.p_buf, (STATUS_RECORD_SIZE, BUFFER_MAX_RECORDS))
        for record in 1:size(string_array, 2)
            record_string = String(string_array[:, record])
            println("\t", record_string)
            if record_string[1:3] == "END"
                return nothing
            end
        end
    else
        println("Buffer: NULL")
    end
    return nothing
end

"Display hashpipe status from reference"
function Base.display(r::Ref{hashpipe_status_t})
    display(r[])
    return nothing
end

"Display hashpipe buffer"
function Base.display(d::Hashpipe.hashpipe_databuf_t)
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
function Base.display(p::Ptr{hashpipe_databuf_t})
    databuf::hashpipe_databuf_t = unsafe_wrap(Array, p, 1)[]
    display(databuf)
    return nothing
end

#---------------------------#
# Hashpipe Status Functions #
#---------------------------#


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

function hashpipe_status_buf_lock_unlock(f::Function, r_status::Ref{hashpipe_status_t})
        try
            hashpipe_status_lock(r_status)
            f() # or f(st) TODO: test which of these is better
        catch e
            println("Error locking hashpipe status buffer - Error: $e")
            perror(e)
        finally
            hashpipe_status_unlock(r_status)
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

"""
    hashpipe_databuf_create(instance_id::Int, db_id::Int,
                            header_size::Int, block_size::Int, n_block::Int)

"""
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
    total_status::UInt64 = ccall((:hashpipe_databuf_total_status, "libhashpipe.so"),
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


function update_status(status::hashpipe_status_t, key::String, value::String)::Int8
    key = Cstring(pointer(key)) # Need to convert for hput functions
    error::Int8 = ccall((:hputs, "libhashpipestatus.so"),
                    Int, (Ptr{UInt8}, Cstring, Cstring),
                    status.p_buf, Cstring(pointer(key)), Cstring(pointer(value)))
    return error
end

function update_status(status::hashpipe_status_t, key::String, value::Int)::Int8
    key = Cstring(pointer(key)) # Need to convert for hput functions
    error::Int8 = ccall((:hputi4, "libhashpipestatus.so"),
                    Int, (Ptr{UInt8}, Cstring, Cint),
                    status.p_buf, Cstring(pointer(key)), Cint(value))
    return error
end

# hput functions...remove?

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


function pin_databuf_mem(db, bytes=-1)
    if(bytes==-1) # If bytes not specified, use databuf block size (may be incorrect)
        bytes = db.block_size
    end

    hp_databuf = unsafe_wrap(Array{Main.Hashpipe.hashpipe_databuf_t}, db.p_hpguppi_db, (1))[1];
    println("Pinning $bytes of Memory:")
    for i in 1:hp_databuf.n_block
        println("Block: $i")
        # Get correct buffer size from databuf!
        CUDA.Mem.register(CUDA.Mem.HostBuffer,db.blocks[i].p_data , bytes)
    end
end

export HPGuppi

module HPGuppi

using ..Hashpipe: hashpipe_databuf_t
using ..GuppiRaw

export hpguppi_input_block_t

# HPGUPPI_DATABUF.h constants
const ALIGNMENT_SIZE = 4096
const N_INPUT_BLOCKS = 24
const BLOCK_HDR_SIZE = 5*80*512
const BLOCK_DATA_SIZE = 128*1024*1024
const PADDING_SIZE = ALIGNMENT_SIZE - (sizeof(hashpipe_databuf_t) % ALIGNMENT_SIZE)
const BLOCK_SIZE = BLOCK_HDR_SIZE + BLOCK_DATA_SIZE

"""
    hpguppi_input_block_t

Used to hold the pointers to individual GuppiRaw
header and data pointers.
"""
struct hpguppi_input_block_t
    p_hdr::Ptr{UInt8}
    p_data::Ptr{Any}
end

"""
    hpguppi_input_databuf_t
"""
struct hpguppi_input_databuf_t
    p_hpguppi_db::Ptr{hashpipe_databuf_t}
    blocks::Array{hpguppi_input_block_t}

    function hpguppi_input_databuf_t(p_hp_db::Ptr{hashpipe_databuf_t})
        blocks_array = Array{hpguppi_input_block_t}(undef, N_INPUT_BLOCKS)
        p_blocks = p_hp_db + sizeof(hashpipe_databuf_t) + PADDING_SIZE
        for i = 0:N_INPUT_BLOCKS - 1
            p_header = p_blocks + i * BLOCK_SIZE
            p_data = p_header + BLOCK_HDR_SIZE
            blocks_array[i+1] = hpguppi_input_block_t(p_header, p_data)
        end
        new(p_hp_db, blocks_array)
    end
end

"""
    find_first_full_block(db::hpguppi_input_databuf_t)::Int

Return the index of the first full block in hpguppi databuf. Return 0 if none ready.

NOTE: 1-indexed and in development
"""
function find_first_full_block(db::hpguppi_input_databuf_t)::Int
    lock_mask = Hashpipe.hashpipe_databuf_total_mask(db.p_hpguppi_db)
    # If no blocks ready, start at 1 (1-indexed blocks array)
    if lock_mask == 0
        return 1

    # TODO: Create function to find first filled block to start processing at
    # bitstring(lock_mask) 
    return index
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

end # End Module HPGuppi