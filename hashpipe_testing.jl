# /* Structure describes status memory area */
# typedef struct {
#     int instance_id; /* Instance ID of this status buffer (DO NOT SET/CHANGE!) */
#     int shmid;   /* Shared memory segment id */
#     sem_t *lock; /* POSIX semaphore descriptor for locking */
#     char *buf;   /* Pointer to data area */
# } hashpipe_status_t;

# // Define hashpipe_databuf structure
# typedef struct {
#     char data_type[64]; /* Type of data in buffer */
#     size_t header_size; /* Size of each block header (bytes) */
#     size_t block_size;  /* Size of each data block (bytes) */
#     int n_block;        /* Number of data blocks in buffer */
#     int shmid;          /* ID of this shared mem segment */
#     int semid;          /* ID of locking semaphore set */
# } hashpipe_databuf_t;

# TODO create sem_t
# Example Julia creation for argument passing: status = Ref{hashpipe_status_t}(0,0,0,0)
mutable struct hashpipe_status_t
    instance_id::Cint
    shmid::Cint
    p_lock::Ptr{UInt8} 
    p_buf::Ptr{UInt8}
end

struct hashpipe_databuf_t
    data_type::Cstring
    header_size::Csize_t
    block_size::Csize_t
    n_block::Cint
    shmid::Cint
    semid::Cint
end

# Display hashpipe status
function display(s::hashpipe_status_t)
    BUFFER_MAX_RECORDS = 20
    println("Instance ID: $(s.instance_id)")
    println("shmid: $(s.shmid)")
    lock = unsafe_wrap(Array, s.p_lock, (1))[1]
    println("Lock: $lock")

    println("Buffer:")    
    string_array = unsafe_wrap(Array, s.p_buf, (80, BUFFER_MAX_RECORDS))
    mapslices(x->all(x.==0x00) ? nothing : println("\t", String(x)), string_array, dims=1)
    return nothing
end

# Display hashpipe status from reference
function display(r::Ref{hashpipe_status_t})
    display(r[])
end

# TODO: wrap with error checking based on function
# Returns 0 with error
function hashpipe_status_exists(instance_id::Int)
    ccall((:hashpipe_status_exists, 
            "libhashpipestatus.so"), 
            Int8, (Int8,), instance_id)
end

function hashpipe_status_attach(instance_id::Int, p_hashpipe_status::Ref{hashpipe_status_t})
    ccall((:hashpipe_status_attach, "libhashpipestatus.so"),
            Int, (Int8, Ref{hashpipe_status_t}), instance_id, p_hashpipe_status)
end

function hashpipe_status_lock(p_hashpipe_status::Ref{hashpipe_status_t})
    ccall((:hashpipe_status_lock, "libhashpipestatus.so"),
            Int, (Ref{hashpipe_status_t},), p_hashpipe_status)
end

function hashpipe_status_unlock(p_hashpipe_status::Ref{hashpipe_status_t})
    ccall((:hashpipe_status_unlock, "libhashpipestatus.so"),
            Int, (Ref{hashpipe_status_t},), p_hashpipe_status)
end

function hashpipe_status_clear(p_hashpipe_status::Ref{hashpipe_status_t})
    ccall((:hashpipe_status_clear, "libhashpipestatus.so"),
            Int, (Ref{hashpipe_status_t},), p_hashpipe_status)
end

#-------------#
# Development #
#-------------#

function hashpipe_databuf_attach(instance_id::Int, db_id::Int)
    ccall((:hashpipe_databuf_attach, "libhashpipestatus.so"),
            Ptr{hashpipe_databuf_t}, (Int8, Int8), instance_id, db_id)
end