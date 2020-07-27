# /* Structure describes status memory area */
# typedef struct {
#     int instance_id; /* Instance ID of this status buffer (DO NOT SET/CHANGE!) */
#     int shmid;   /* Shared memory segment id */
#     sem_t *lock; /* POSIX semaphore descriptor for locking */
#     char *buf;   /* Pointer to data area */
# } hashpipe_status_t;


function hashpipe_status_exists(instance_id::Int)
    ccall((:hashpipe_status_exists, 
            "/home/max/btl_workspace/hashpipe/src/.libs/libhashpipestatus.so"), 
            Int8, (Int8,), instance_id)
    
end