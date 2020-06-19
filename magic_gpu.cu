extern "C" {
#include <stdio.h>
#include "magic.h"
}
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/cuda_device_runtime_api.h>

#define MAX_THREADS_PER_BLOCK (1024) // For my personal desktop (2070 Super) - TODO: Change to MeerKAT size

__global__ void power_spectrum(int8_t *complex_block, int *power_block, unsigned long blocsize){
    unsigned long i = (blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x)  * 4;
    if(i > blocsize){
        printf("Exceeded array bounds in kernel");
        return;
    }
    // TODO: use dp4a 8 bit math acceleration
    unsigned long power = complex_block[i] * complex_block[i]
                            + complex_block[i+1] * complex_block[i+1]
                            + complex_block[i+2] * complex_block[i+2]
                            + complex_block[i+3] * complex_block[i+3];
    if(i==TEST_INDEX){
        printf("In Kernel!\tIndex: %ld  (%d, %d), (%d, %d)  Pow: %ld \n\n", 
                i, complex_block[i], complex_block[i+1], complex_block[i+2], complex_block[i+3], power);
    } 
    power_block[i / 4] = power;
}

extern "C" void process_cuda_block(int8_t *h_complex_block, raw_file_t *raw_file){
    int8_t *d_complex_block;
    int *d_spectrum;
    int *h_spectrum;

    printf("CudaMalloc size: %0.3f GB\n", (float) raw_file->blocsize / BYTES_PER_GB);

    // for(int i = 0; i< 100 / 4; ++i){
    //     if(h_complex_block[i]){
    //         printf("data %d: %d\n", i, h_complex_block[i]);
    //     }
    // }

    cudaMalloc(&d_complex_block, raw_file->blocsize);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&d_spectrum, sizeof(int) * raw_file->blocsize / 4);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    cudaHostAlloc(&h_complex_block, raw_file->blocsize, cudaHostAllocMapped);
       printf("CudaHostAlloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaHostAlloc(&h_spectrum, sizeof(int) * raw_file->blocsize / 4, cudaHostAllocDefault);
        printf("CudaHostAlloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(d_complex_block, h_complex_block, raw_file->blocsize, cudaMemcpyHostToDevice);
        printf("CudaMemcpy:\t%s\n", cudaGetErrorString(cudaGetLastError()));


    unsigned long grid_dim_x = raw_file->blocsize / (MAX_THREADS_PER_BLOCK);
    dim3 griddim(5, 1, 1);
    dim3 blockdim(MAX_THREADS_PER_BLOCK / raw_file->obsnchan, raw_file->obsnchan);

    power_spectrum<<<griddim, blockdim>>>(d_complex_block, d_spectrum, raw_file->blocsize);
        printf("Kernel launch:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    cudaDeviceSynchronize();

    cudaMemcpy(h_spectrum, d_spectrum, sizeof(int) * raw_file->blocsize / 4, cudaMemcpyDeviceToHost);
        printf("CudaMemcpy:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    
    // for(int i = 0; i< raw_file->blocsize / 4; ++i){
    //     if(h_spectrum[i]){
    //         printf("H_Spectrum %d: %d\n", i, h_spectrum[i]);
    //     }
    // }
    printf("After Kernel!\tH_Complex (%d, %d), (%d, %d)\n", 
                    h_complex_block[TEST_INDEX], h_complex_block[TEST_INDEX+1], h_complex_block[TEST_INDEX+2], h_complex_block[TEST_INDEX+3]);
    printf("After Kernel!\tH_Spectrum %d: %d\n", TEST_INDEX / 4, h_spectrum[TEST_INDEX / 4]);
    

    cudaFree(d_complex_block);
    cudaFree(d_spectrum);
    cudaFreeHost(h_spectrum);
    cudaFreeHost(h_complex_block);
        printf("CudaFree:\t%s\n", cudaGetErrorString(cudaGetLastError()));
}

extern "C" void get_device_info(){
    int devCount;
    int current_device;
    cudaGetDeviceCount(&devCount);
    cudaGetDevice(&current_device);

    printf("-------------- CUDA Device Query ---------------\n");
    printf("CUDA devices: %d\n", devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        
        printf("\nCUDA Device #%d\n", i);
        if(i == current_device){
            printf("--- Device being used ---\n");
        }
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Device %d: %s\n", i, devProp.name);
    }
    cudaFree(0);
    printf("------------------------------------------------\n\n");
}
