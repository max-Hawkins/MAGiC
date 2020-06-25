import numpy as np
import matplotlib.pyplot as plt
from blimpy import GuppiRaw
from IPython import get_ipython
nchan = 64


filename = 'blc3_guppi_57386_VOYAGER1_0004.0000_block000_pol_power.dat'
f = open(filename, "r")
cuda_spec = np.fromfile(f, dtype=np.int32)
cuda_spec = np.reshape(cuda_spec, (64, -1, 2))
cuda_spec.shape

pol_diff = np.abs(cuda_spec[:,:,0] - cuda_spec[:,:,1])
print(np.max(pol_diff))
print(np.mean(pol_diff))
plt.imshow(pol_diff[:,:], interpolation='None', aspect='auto', cmap='hot')

plt.show()
print("Cuda Spectrogram")
