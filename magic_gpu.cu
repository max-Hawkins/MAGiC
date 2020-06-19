extern "C" {
#include <stdio.h>
#include "magic.h"
}
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/cuda_device_runtime_api.h>

#define MAX_THREADS_PER_BLOCK (1024) // For my personal desktop (2070 Super) - TODO: Change to MeerKAT size

__global__ void power_spectrum(int8_t *complex_block, int8_t *power_block, unsigned long blocsize){
    unsigned long i = (blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x)  * 4;
    if(i > blocsize){
        printf("Exceeded array bounds in kernel");
        return;
    }
    unsigned long power = complex_block[i] * complex_block[i]
                            + complex_block[i+1] * complex_block[i+1]
                            + complex_block[i+2] * complex_block[i+2]
                            + complex_block[i+3] * complex_block[i+3];
    // if(i==10000){
    //     printf(" Block:: %s", complex_block);
    //     printf("Index: %ld  (%d, %d), (%d, %d)  Pow: %ld \n\n", 
    //             i, complex_block[i], complex_block[i+1], complex_block[i+2], complex_block[i+3], power);
    // } 
    power_block[i] = power;
}

extern "C" void process_cuda_block(int8_t *data, raw_file_t *raw_file){
    int8_t *d_data;
    int8_t *d_spectrum;
    int8_t *h_spectrum;

    printf("CudaMalloc size: %0.3f GB\n", (float) raw_file->blocsize / BYTES_PER_GB);

    for(int i = 0; i< raw_file->blocsize / 4; ++i){
        if(data[i]){
            printf("data %d: %d\n", i, data[i]);
        }
    }

    cudaMalloc(&d_data, raw_file->blocsize);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&d_spectrum, raw_file->blocsize / 4);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    cudaHostAlloc(&data, raw_file->blocsize, cudaHostAllocDefault);
        printf("CudaHostAlloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaHostAlloc(&h_spectrum, raw_file->blocsize / 4, cudaHostAllocDefault);
        printf("CudaHostAlloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(d_data, data, raw_file->blocsize, cudaMemcpyHostToDevice);
        printf("CudaMemcpy:\t%s\n", cudaGetErrorString(cudaGetLastError()));


    unsigned long grid_dim_x = raw_file->blocsize / (MAX_THREADS_PER_BLOCK);
    dim3 griddim(5, 1, 1);
    dim3 blockdim(MAX_THREADS_PER_BLOCK / raw_file->obsnchan, raw_file->obsnchan);

    power_spectrum<<<griddim, blockdim>>>(d_data, d_spectrum, raw_file->blocsize);
        printf("Kernel launch:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    cudaDeviceSynchronize();

    cudaMemcpy(h_spectrum, d_spectrum, raw_file->blocsize / 4, cudaMemcpyDeviceToHost);
        printf("CudaMemcpy:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    
    for(int i = 0; i< raw_file->blocsize / 4; ++i){
        if(h_spectrum[i]){
            printf("H_Spectrum %d: %d\n", i, h_spectrum[i]);
        }
    }
    

    cudaFree(d_data);
    cudaFreeHost(h_spectrum);

}

