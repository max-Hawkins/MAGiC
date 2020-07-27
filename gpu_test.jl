include("../jl-blio/src/Blio.jl")
include("./search.jl")

fn = "../guppi_58626_J0332+5434_0018.0000.raw"
raw, rh = Search.load_guppi(fn)
complex_data = Search.read_block_gr(raw, rh)
power_data = Search.power_spec_gpu(complex_data)
println(power_data)
