# Test file to experiment with creating hashpipe pipeline using Julia
include("../../jl-blio/src/Blio.jl")
using Main.Blio.GuppiRaw
include("./hashpipe.jl")
using Main.Hashpipe

instance_id = 0
status = Hashpipe.hashpipe_status_t(0,0,0,0) # Create dummy status to populate later
r_status = Ref(status) # Need reference to status
Hashpipe.hashpipe_status_attach(instance_id, r_status) # Populate hashpipe status with values
Hashpipe.display(r_status)

# Insert into status header
Hashpipe.hputi4(status.p_buf, "Test Num", 7)
Hashpipe.hputs(status.p_buf, "Hello", "Test")



# TODO: put GUPPIRAW header into status and keep track of data rate and change DESTIP to the filename
# Make databuf block size 128MB max size and only partially fill




# "Load a Guppi RAW file and return the file and header variables."
# function load_guppi(fn::String)
#     raw = open(fn)
#     rh = GuppiRaw.Header()
#     return raw, rh
# end

# "Read in the next Guppi block header and return the corresponding block data."
# function read_block_gr(raw, rh::GuppiRaw.Header)
#     read!(raw, rh)
#     data = Array(rh)
#     read!(raw, data)
#     return data
# end
