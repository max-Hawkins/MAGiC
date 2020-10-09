using CUDA, BenchmarkTools
# Couldn't make variable memory size with CUDA.reclaim()
println(CUDA.device())
CUDA.reclaim()

data = Array{Int8}(zeros(134217728))
time = @belapsed CuArray($data)
data_rate = Base.format_bytes(sizeof(data) / time) * "/s"
println("Naive CuArray: ", data_rate)

gpu_data = CuArray(data)
time = @belapsed copyto!($gpu_data, $data)
data_rate = Base.format_bytes(sizeof(data) / time) * "/s"
println("Without allocation: ", data_rate)

CUDA.reclaim()

gpu = Mem.alloc(Mem.Device, sizeof(data))
gpu_ptr = convert(CuPtr{Int8}, gpu)
time = @belapsed unsafe_copyto!($gpu_ptr, $(pointer(data)), 134217728)
data_rate = Base.format_bytes(sizeof(data) / time) * "/s"
println("Low-level API: ", data_rate)

CUDA.reclaim()

cpu = Mem.alloc(Mem.Host, 134217728)
cpu_ptr = convert(Ptr{Int8}, cpu)
time = @belapsed unsafe_copyto!($gpu_ptr, $cpu_ptr, 134217728)
data_rate = Base.format_bytes(sizeof(data) / time) * "/s"
println("Pinned memory: ", data_rate)

CUDA.reclaim();
