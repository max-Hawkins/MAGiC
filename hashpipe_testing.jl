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
#define HASHPIPE_STATUS_TOTAL_SIZE (2880*64) // FITS-style buffer
#define HASHPIPE_STATUS_RECORD_SIZE 80 // Size of each record (e.g. FITS "card")
const global HASHPIPE_STATUS_TOTAL_SIZE = 184320 # 2880 * 64
const global HASHPIPE_STATUS_RECORD_SIZE = 80

# TODO create sem_t
# Example Julia creation for argument passing: status = Ref{hashpipe_status_t}(0,0,0,0)
mutable struct hashpipe_status_t
    instance_id::Cint
    shmid::Cint
    p_lock::Ptr{UInt8} 
    p_buf::Ptr{UInt8}
end

struct hashpipe_databuf_t
    data_type::NTuple{64, UInt8}
    header_size::Int # May need to change to Csize_t
    block_size::Int # May need to change to Csize_t
    n_block::Cint
    shmid::Cint
    semid::Cint
end

# Display hashpipe status
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

# Display hashpipe status from reference
function display(r::Ref{hashpipe_status_t})
    display(r[])
end

# Display hashpipe buffer
function display(d::hashpipe_databuf_t)
    # Convert Ntuple to array and strip 0s before converting to string
    data_type_string = String(filter(x->x!=0x00, collect(d.data_type)))
    println("Data Type: $(data_type_string)")
    println("Header Size: $(d.header_size)")
    println("Block Size: $(d.block_size)")
    println("shmid: $(d.shmid)")
    println("semid: $(d.semid)")
end

# Display hashpipe databuf from pointer
function display(p::Ptr{hashpipe_databuf_t})
    databuf = unsafe_wrap(Array, p, 1)[]
    display(databuf)
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
    ccall((:hashpipe_databuf_attach, "libhashpipe.so"),
            Ptr{hashpipe_databuf_t}, (Int8, Int8), instance_id, db_id)
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
end