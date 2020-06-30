extern "C" {
#include <stdio.h>
#include "magic.h"
}
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/cuda_device_runtime_api.h>

#define MAX_THREADS_PER_BLOCK (1024) // For my personal desktop (2070 Super) - TODO: Change to MeerKAT size

// CUDA kernel that takes a single GUPPI block and creates a power spectrum from it
__global__ void power_spectrum(int8_t *complex_block, unsigned int *power_block, unsigned long blocsize){
    unsigned long i = (blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x)  * 4;
    if(i > blocsize){ // changes to blocsize
        return;
    }
    // TODO: use dp4a 8 bit math acceleration
    unsigned int power = complex_block[i] * complex_block[i]
                            + complex_block[i+1] * complex_block[i+1]
                            + complex_block[i+2] * complex_block[i+2]
                            + complex_block[i+3] * complex_block[i+3];

    // printf("In Kernel!\tIndex: %ld  (%d, %d), (%d, %d)  Pow: %ld \n\n", 
    //             i, complex_block[i], complex_block[i+1], complex_block[i+2], complex_block[i+3], power);
    if(i == TEST_INDEX){
        printf("In Kernel!\tIndex: %ld  (%d, %d), (%d, %d)  Pow: %d\n\n", 
                i, complex_block[i], complex_block[i+1], complex_block[i+2], complex_block[i+3], power);
    } 
    power_block[i / 4] = power;
}

// CUDA kernel that takes a single GUPPI block and creates a linearly polarized power spectrum
// TODO: Create circularly polarized power kernel
__global__ void polarized_power(int8_t *complex_block, unsigned int *pol_power_block, unsigned long blocsize){
    
    unsigned long i = (blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x)  * 4;
    if(i > blocsize){
        return;
    }
    // TODO: use dp4a 8 bit math acceleration
    unsigned int power_x = (unsigned int) (complex_block[i] * complex_block[i]
                            + complex_block[i+1] * complex_block[i+1]);

    unsigned int power_y = (unsigned int) (complex_block[i+2] * complex_block[i+2]
                            + complex_block[i+3] * complex_block[i+3]);

    if(i == TEST_INDEX){
        printf("In Kernel!\tIndex: %ld  (%d, %d), (%d, %d)  Pow: (%d, %d) \n\n", 
                i, complex_block[i], complex_block[i+1], complex_block[i+2], complex_block[i+3], power_x, power_y);
    } 

    pol_power_block[i / 2] = power_x;
    pol_power_block[i / 2 + 1] = power_y;

}

void create_power_spectrum(int fd, raw_file_t *raw_file, int num_streams){

    off_t pos;
    ssize_t bytes_read;

    unsigned long grid_dim_x = raw_file->blocsize / (MAX_THREADS_PER_BLOCK);
    dim3 griddim(grid_dim_x, 1, 1);
    dim3 blockdim(MAX_THREADS_PER_BLOCK / raw_file->obsnchan, raw_file->obsnchan);

    // int8_t complex_block[raw_file->blocsize];
    int8_t *h_complex_block;
    int8_t *d_complex_block;
    unsigned int *h_power_block;
    unsigned int *d_power_block;

    size_t complex_block_size = raw_file->blocsize * num_streams;
    size_t power_block_size   = raw_file->blocsize * num_streams / 4 * sizeof(unsigned int);

    cudaHostAlloc(&h_complex_block, complex_block_size, cudaHostAllocDefault);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaHostAlloc(&h_power_block, power_block_size, cudaHostAllocDefault);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&d_complex_block, complex_block_size);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&d_power_block, power_block_size);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    pos = lseek(fd, 0, SEEK_SET);

    for(int block = 0; block < raw_file->nblocks; block++){
        printf("--------- Block %d ----------\n", block);
        pos = lseek(fd, raw_file->hdr_size, SEEK_CUR);
        // printf("Now at pos: %ld\n", pos);
        // printf("H complex address: %p\n", (void *) h_complex_block);

        bytes_read = read(fd, h_complex_block, raw_file->blocsize);
        if(bytes_read == -1){
            perror("Read block error\n");
            return;
        } 
        else if(bytes_read < raw_file->blocsize){
            printf("----- Didn't read in full block - skipping.\n");
            return;
        }

        // cudaHostRegister(h_complex_block, raw_file->blocsize, cudaHostRegisterDefault);
        // printf("Bytes read: %ld", bytes_read);

        printf("Test Data: %d\n", h_complex_block[TEST_INDEX]);

        cudaMemcpy(d_complex_block, h_complex_block, raw_file->blocsize, cudaMemcpyHostToDevice);
            printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
        polarized_power<<<griddim, blockdim>>>(d_complex_block, d_power_block, raw_file->blocsize);
            printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(h_power_block, d_power_block, raw_file->blocsize / 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));

        // Write block to file
        char *save_block_append = (char *) malloc(50);
        if(sprintf(save_block_append, "_block%03d_power.dat", block) < 0){
            printf("Error creating save_filename. Couldn't save file.");
        }
        else {
            char *save_filename = (char *) malloc(70);
            strcpy(save_filename, raw_file->trimmed_filename);
            strcat(save_filename, save_block_append);

            FILE *f = fopen(save_filename, "wb");
            int status = fwrite(h_power_block, sizeof(unsigned int), raw_file->blocsize / 4, f);
            if(!status){
                perror("Error writing array to file!");
            }
            printf("Num Elements written: %d", status);
            fclose(f);
            free(save_filename);
        }
        free(save_block_append);
    }


    cudaFree(d_complex_block);
    cudaFree(d_power_block);
    cudaFreeHost(h_complex_block);
    cudaFreeHost(h_power_block);
    
    // for (int i = 0; i < num_streams; ++i)
    // {
    //     cudaStreamSynchronize(streams[i]);
    //     cudaStreamDestroy(streams[i]);
    // }

    
        // printf("CudaFree:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    
}

// Creates linearly polarized power spectrum for an entire GUPPI raw file
void create_polarized_power(int fd, raw_file_t *raw_file){
    off_t pos;
    ssize_t bytes_read;

    unsigned long grid_dim_x = raw_file->blocsize / (MAX_THREADS_PER_BLOCK);
    dim3 griddim(grid_dim_x, 1, 1);
    dim3 blockdim(MAX_THREADS_PER_BLOCK / raw_file->obsnchan, raw_file->obsnchan);

    // int8_t complex_block[raw_file->blocsize];
    int8_t *h_complex_block;
    int8_t *d_complex_block;
    unsigned int *h_polarized_block;
    unsigned int *d_polarized_block;
    
    cudaHostAlloc(&h_complex_block, raw_file->blocsize, cudaHostAllocDefault);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaHostAlloc(&h_polarized_block, raw_file->blocsize / 2 * sizeof(unsigned int), cudaHostAllocDefault);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&d_complex_block, raw_file->blocsize);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&d_polarized_block, raw_file->blocsize / 2 * sizeof(unsigned int));
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    pos = lseek(fd, 0, SEEK_SET);

    for(int block = 0; block < raw_file->nblocks; block++){
        printf("--------- Block %d ----------\n", block);
        pos = lseek(fd, raw_file->hdr_size, SEEK_CUR);
        // printf("Now at pos: %ld\n", pos);
        // printf("H complex address: %p\n", (void *) h_complex_block);

        bytes_read = read(fd, h_complex_block, raw_file->blocsize);
        if(bytes_read == -1){
            perror("Read block error\n");
            return;
        } 
        else if(bytes_read < raw_file->blocsize){
            printf("----- Didn't read in full block - skipping.\n");
            return;
        }

        // cudaHostRegister(h_complex_block, raw_file->blocsize, cudaHostRegisterDefault);
        // printf("Bytes read: %ld", bytes_read);

        printf("Test Data: %d\n", h_complex_block[TEST_INDEX]);

        cudaMemcpy(d_complex_block, h_complex_block, raw_file->blocsize, cudaMemcpyHostToDevice);
            printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
        polarized_power<<<griddim, blockdim>>>(d_complex_block, d_polarized_block, raw_file->blocsize);
            printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(h_polarized_block, d_polarized_block, raw_file->blocsize / 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));

        // Write block to file
        char *save_block_append = (char *) malloc(50);
        if(sprintf(save_block_append, "_block%03d_pol_power.dat", block) < 0){
            printf("Error creating save_filename. Couldn't save file.");
        }
        else {
            char *save_filename = (char *) malloc(70);
            strcpy(save_filename, raw_file->trimmed_filename);
            strcat(save_filename, save_block_append);

            FILE *f = fopen(save_filename, "wb");
            int status = fwrite(h_polarized_block, sizeof(unsigned int), raw_file->blocsize / 2, f);
            if(!status){
                perror("Error writing array to file!");
            }
            printf("Num Elements written: %d", status);
            fclose(f);
            free(save_filename);
        }
        free(save_block_append);
    }


    cudaFree(d_complex_block);
    cudaFree(d_polarized_block);
    cudaFreeHost(h_complex_block);
    cudaFreeHost(h_polarized_block);
}

void ddc_coarse_chan(int fd, raw_file_t *raw_file, int chan, double i_freq){
    

}


void get_device_info(){
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
                if(i == current_device){
            printf("--- Device being used ---\n");
        }
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Device %d: %s\n", i, devProp.name);
    }
    cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaFree(0);
    printf("------------------------------------------------\n\n");
}
