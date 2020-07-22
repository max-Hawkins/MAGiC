using PyCall


function frb_pipeline(fn::String)
    raw, rh = load_guppi(fn)
    complex_block = read_block_gr(raw, rh)
    power_block = power_spec(complex_block, -1, true, true)
    power_block = average(power_block, 512)

    power_avg_py = np.float32(Array(power_block))

end