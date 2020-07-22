using PyCall
cv2 = pyimport("cv2")
np = pyimport("numpy")

"Implement contrast limiting adaptive histogram equalization
    on a 256x256 power spectrogram."
function clahe(power_spec)
    cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
end

function pulse_search()

end

function frb_pipeline(fn::String)
    

    raw, rh = load_guppi(fn)
    complex_block = read_block_gr(raw, rh)
    power_block = power_spec(complex_block, -1, true, true)
    power_block = average(power_block, 512)

    power_avg_py = np.float32(Array(power_avg))

end