println("Loading modules...")
include("../jl-blio/src/Blio.jl")
include("./search.jl")

println("Loading python packages...")
using PyCall
using Plots
# include current directory in python search path
pushfirst!(PyVector(pyimport("sys")."path"), "")
cv2 = pyimport("cv2")
np = pyimport("numpy")
model = pyimport("model") # import custom python TF model and inference code

fn = ARGS[1]
model_filename = ARGS[2]

println("Loading file: $fn")
println("Model file: $model_filename")

# Setup CLAHE function
# Example of calling imported python function straight from Julia
clahe_f = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Read in GUPPI data
raw, rh = Search.load_guppi(fn)
complex_block = Search.read_block_gr(raw, rh)
# Calculate power spectrum, average pols, and average along time axis on GPU
power_block = Search.power_spec(complex_block, 1, true, true)
power_block = Search.average(power_block, 512)
println("Power block type: $(eltype(power_block))")
# Pass numpy array to python functions
# Row/column ordering matters greatly, opencv and tensorflow are picky
# Plenty of room for optimization here! TODO
# Not the way to do things but is meant to show different ways of calling
# python funtions
power_npy = np.uint16(Array(power_block)[1,:,:])
plotly()
p = heatmap(power_npy)
display(p)
println("power_npy size: $(np.shape(power_npy))")
power_npy = clahe_f.apply(power_npy)

println("Running inference on block")
# Run custom python function from local file
model_pulse_confidence_block = model.inference(model_filename, power_npy, true)

