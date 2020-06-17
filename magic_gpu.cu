extern "C" {
#include <stdio.h>
#include "magic.h"
}
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime.h>

__global__ void power_spectrum(int8_t *complex_block){
    printf("Alive!");
}

extern "C" void process_cuda_block(int8_t *data, raw_file_t *raw_file){
    int8_t *d_data;
    int8_t *spectrum;
    printf("CudaMalloc size: %0.3f GB\n", (float) raw_file->blocsize / BYTES_PER_GB);

    cudaMalloc((void **) &d_data, raw_file->blocsize);
    cudaHostAlloc((void **) &data, raw_file->blocsize, cudaHostAllocDefault);
    cudaMemcpy(d_data, data, raw_file->blocsize, cudaMemcpyHostToDevice);

    const dim3 grid_dim = {1024, 1, 1};
    const dim3 block_dim = {1009, 32, 1};

    power_spectrum<<<2, 4>>> (d_data);

    cudaFree(d_data);
    cudaFreeHost(data);

}

