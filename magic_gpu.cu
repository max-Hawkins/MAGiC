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
    if(i > blocsize){ // changes to blocsize
        return;
    }
    // TODO: use dp4a 8 bit math acceleration
    unsigned long power = complex_block[i] * complex_block[i]
                            + complex_block[i+1] * complex_block[i+1]
                            + complex_block[i+2] * complex_block[i+2]
                            + complex_block[i+3] * complex_block[i+3];

    // printf("In Kernel!\tIndex: %ld  (%d, %d), (%d, %d)  Pow: %ld \n\n", 
    //             i, complex_block[i], complex_block[i+1], complex_block[i+2], complex_block[i+3], power);
    if(i == TEST_INDEX){
        printf("In Kernel!\tIndex: %ld  (%d, %d), (%d, %d)  Pow: %ld \n\n", 
                i, complex_block[i], complex_block[i+1], complex_block[i+2], complex_block[i+3], power);
    } 
    power_block[i / 4] = power;
}

extern "C" void create_power_spectrum(int8_t *file_mmap, raw_file_t *raw_file, int num_streams){

    unsigned long grid_dim_x = raw_file->blocsize / (MAX_THREADS_PER_BLOCK);
    dim3 griddim(grid_dim_x, 1, 1);
    dim3 blockdim(MAX_THREADS_PER_BLOCK / raw_file->obsnchan, raw_file->obsnchan);
    
    cudaStream_t streams[num_streams];
    int8_t *h_complex_blocks[num_streams];
    int8_t *d_complex_blocks[num_streams];
    int *d_spectra[num_streams];
    int *h_spectra[num_streams];

    // Create streams and malloc data for initial streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
            printf("Stream creation:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    }
        
    
    for(int block = 0; block < raw_file->nblocks; block+= num_streams){
        // Malloc and allocate memory for all streams
        for (int i = 0; i < num_streams; ++i) {
            if(block + i >= raw_file->nblocks){
                printf("Block number exceeded (%d). Skipping.\n", block + i);
                break;
            }
            unsigned long block_index = raw_file->hdr_size + (block + i) * (raw_file->hdr_size + raw_file->blocsize);
            h_complex_blocks[i] = &file_mmap[block_index];
            // int8_t *temp_ptr = &file_mmap[block_index];

            printf("h complex: %p\t val: %d\n", h_complex_blocks[i], h_complex_blocks[i][1000]);

            cudaMalloc(&d_complex_blocks[i], raw_file->blocsize);
                printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
            cudaMalloc(&d_spectra[i], sizeof(int) * raw_file->blocsize / 4);
                printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));

            cudaHostAlloc(&h_complex_blocks[i], raw_file->blocsize, cudaHostAllocPortable);
                printf("CudaHostAlloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
            cudaHostAlloc(&h_spectra[i], sizeof(int) * raw_file->blocsize / 4, cudaHostAllocMapped);
                printf("CudaHostAlloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));

            // cudaHostGetDevicePointer((void **) &d_complex_blocks[i], (void *) h_complex_blocks[i], 0);
                printf("CudaHostAlloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
            printf("h complex: %p\t val: %d\n", h_complex_blocks[i], h_complex_blocks[i][1000]);
            //printf("Temp ptr: %p, %d\n", &d_complex_blocks[i], d_complex_blocks[i][1000]);

        }

        cudaDeviceSynchronize();
        // Launch streams with 1 kernel launch per stream
        for(int cur_stream = 0; cur_stream < num_streams; cur_stream++){
            int cur_block = block + cur_stream;
            if(cur_block >= raw_file->nblocks){
                printf("Block number exceeded (%d). Skipping.", cur_block);
                break;
            }

            printf("\n\n--------- Block %d  Stream %d ----------\n", cur_block, cur_stream);

            printf("h_complex: Location: %p\tPointer to val: %p\tVal: %d", &h_complex_blocks[cur_stream], h_complex_blocks[cur_stream], h_complex_blocks[cur_stream][10]);

            for(int i = 500; i< 508; i += 4){
                printf("I: %i  Address: %p\n", i, &h_complex_blocks[cur_stream][i]);
                printf("(%d, %d), (%d, %d)\n\n", h_complex_blocks[cur_stream][i], h_complex_blocks[cur_stream][i+1], h_complex_blocks[cur_stream][i+2], h_complex_blocks[cur_stream][i+3]);
            }
            cudaMemcpyAsync(d_complex_blocks[cur_stream], h_complex_blocks[cur_stream], raw_file->blocsize, cudaMemcpyHostToDevice, streams[cur_stream]);
                printf("CudaMemcpy:\t%s\n", cudaGetErrorString(cudaGetLastError()));

            power_spectrum<<<griddim, blockdim, 0, streams[cur_stream]>>>(d_complex_blocks[cur_stream], d_spectra[cur_stream], raw_file->blocsize);
                printf("Kernel launch:\t%s\n", cudaGetErrorString(cudaGetLastError()));
            cudaMemcpyAsync(h_spectra[cur_stream], d_spectra[cur_stream], sizeof(int) * raw_file->blocsize / 4, cudaMemcpyDeviceToHost, streams[cur_stream]);
                printf("CudaMemcpy:\t%s\n", cudaGetErrorString(cudaGetLastError()));
            // printf("Block: %d  Index: %ld  Contents: %d\n", block, block_index, block_address);
            // printf("Block: %d  Index: %d  Contents: %d\n", block, TEST_INDEX, file_mmap[block_index + TEST_INDEX]);
        }
        cudaDeviceSynchronize();

        // Write data - TODO: implement with callbacks
        for (int i = 0; i < num_streams; ++i) {
            int cur_block = block + i;
            if(cur_block >= raw_file->nblocks){
                printf("Block number exceeded (%d). Skipping.", cur_block);
                break;
            }
            char *save_block_append = (char *) malloc(50);
            if(sprintf(save_block_append, "_block%03d_power.dat", cur_block) < 0){
                printf("Error creating save_filename. Couldn't save file.");
            }
            else {
                char *save_filename = (char *) malloc(70);
                strcpy(save_filename, raw_file->trimmed_filename);
                strcat(save_filename, save_block_append);

                FILE *f = fopen(save_filename, "wb");
                int status = fwrite(h_spectra[i], sizeof(int), raw_file->blocsize / 4, f);
                if(!status){
                    perror("Error writing array to file!");
                }
                fclose(f);
                free(save_filename);
            }
            free(save_block_append);

            cudaFree(d_complex_blocks[i]);
            cudaFree(d_spectra[i]);
            cudaFreeHost(h_spectra[i]);
            cudaFreeHost(h_complex_blocks[i]);
        }
    

    }

    // for(int i = 0; i< 100 / 4; ++i){
    //     if(h_complex_block[i]){
    //         printf("data %d: %d\n", i, h_complex_block[i]);
    //     }
    // }

    
       // printf("CudaHostAlloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    
        // printf("CudaMemcpy:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    
    // for(int i = 0; i< raw_file->blocsize / 4; ++i){
    //     if(h_spectrum[i]){
    //         printf("H_Spectrum %d: %d\n", i, h_spectrum[i]);
    //     }
    // }
    // printf("After Kernel!\tH_Complex (%d, %d), (%d, %d)\n", 
    //                 h_complex_block[TEST_INDEX], h_complex_block[TEST_INDEX+1], h_complex_block[TEST_INDEX+2], h_complex_block[TEST_INDEX+3]);
    // printf("After Kernel!\tH_Spectrum %d: %d\n", TEST_INDEX / 4, h_spectrum[TEST_INDEX / 4]);

    // Save individual block arrays to file
    
    
    for (int i = 0; i < num_streams; ++i)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    
        // printf("CudaFree:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    
}
// TODO: Create circularly polarized power kernel
__global__ void polarized_power(int8_t *complex_block, unsigned int *pol_power_block, unsigned long blocsize){
    unsigned long i = (blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x)  * 4;
    if(i > blocsize){
        return;
    }
    // TODO: use dp4a 8 bit math acceleration
    pol_power_block[i / 2] = (unsigned int) complex_block[i] * complex_block[i]
                            + complex_block[i+1] * complex_block[i+1];

    pol_power_block[i / 2 + 1] = (unsigned int) complex_block[i+2] * complex_block[i+2]
                            + complex_block[i+3] * complex_block[i+3];

    // if(i == TEST_INDEX){
    //     printf("In Kernel!\tIndex: %ld  (%d, %d), (%d, %d)  Pow: (%d, %d) \n\n", 
    //             i, complex_block[i], complex_block[i+1], complex_block[i+2], complex_block[i+3], power_x, power_y);
    // } 

}

void create_polarized_power(int fd, raw_file_t *raw_file){
    off_t pos;
    ssize_t bytes_read;

    unsigned long grid_dim_x = raw_file->blocsize / (MAX_THREADS_PER_BLOCK) / 4;
    dim3 griddim(grid_dim_x, 1, 1);
    dim3 blockdim(MAX_THREADS_PER_BLOCK / raw_file->obsnchan, raw_file->obsnchan);

    // int8_t complex_block[raw_file->blocsize];
    int8_t *h_complex_block;
    int8_t *d_complex_block;
    unsigned int *h_polarized_block;
    unsigned int *d_polarized_block;
    
    cudaHostAlloc(&h_complex_block, raw_file->blocsize, cudaHostAllocDefault);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaHostAlloc(&h_polarized_block, raw_file->blocsize  / 2, cudaHostAllocDefault);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&d_complex_block, raw_file->blocsize * sizeof(unsigned int));
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

        printf("Data: %d", h_complex_block[1000]);

        cudaMemcpy(d_complex_block, h_complex_block, raw_file->blocsize, cudaMemcpyHostToDevice);
            printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
        polarized_power<<<griddim, blockdim>>>(d_complex_block, d_polarized_block, raw_file->blocsize);
            printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
        // cudaDeviceSynchronize();
            printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaMemcpy(h_polarized_block, d_polarized_block, raw_file->blocsize / 2, cudaMemcpyDeviceToHost);
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
            int status = fwrite(h_polarized_block, sizeof(int), raw_file->blocsize / 2, f);
            if(!status){
                perror("Error writing array to file!");
            }
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
