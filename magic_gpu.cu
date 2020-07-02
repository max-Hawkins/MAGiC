extern "C" {
#include <stdio.h>
#include <math.h>
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


void create_power_spectrum(int fd, rawspec_raw_hdr_t *raw_file, int num_streams){

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
void create_polarized_power(int fd, rawspec_raw_hdr_t *raw_file){
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

__global__ void ddc_channel(int8_t *raw_chan, double *ddc_chan, unsigned int raw_chan_size, int block, double t_per_samp, double lo_freq){
    unsigned long i = (blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x)  * 4;
    double time = t_per_samp * (i + block * raw_chan_size) / 4;

    double cosine = cospi(2 * time * lo_freq * 1000000);
    double sine   = sinpi(2 * time * lo_freq * 1000000);

    double ddc_x_real = cosine * raw_chan[i]   - sine * raw_chan[i+1];  
    double ddc_x_imag = cosine * raw_chan[i+1] + sine * raw_chan[i];
    double ddc_y_real = cosine * raw_chan[i+2] - sine * raw_chan[i+3];
    double ddc_y_imag = cosine * raw_chan[i+3] + sine * raw_chan[i+2];


    if(i == TEST_INDEX){
        printf("In Kernel!\tBlock: %d\tIndex: %ld\tTime: %f\tLO Freq: %f MHz\n", 
                block, i, time, lo_freq);
        printf("Cosine: %f\tSine: %f\n", cosine, sine);
        printf("\t  (x_real,  \tx_imag),  (y_real,  \ty_imag)\n");
        printf("Raw Data: (%9d, %9d), (%8d, %9d)\n", raw_chan[i], raw_chan[i+1], raw_chan[i+2], raw_chan[i+3]);
        printf("DDC Data: (%f, %f), (%f, %f)\n\n", ddc_x_real, ddc_x_imag, ddc_y_real, ddc_y_imag);
    }

    
    ddc_chan[i]   = ddc_x_real;
    ddc_chan[i+1] = ddc_x_imag;
    ddc_chan[i+2] = ddc_y_real;
    ddc_chan[i+3] = ddc_y_imag;    
}

void ddc_coarse_chan(int fd, rawspec_raw_hdr_t *raw_file, int chan, double lo_freq){
    off_t pos = lseek(fd, 0, SEEK_SET);
    size_t bytes_to_chan;
    size_t bytes_to_next_bloc_chan;
    size_t bytes_read;
    size_t raw_chan_size   = raw_file->blocsize / raw_file->obsnchan;
    size_t ddc_chan_size   = raw_file->blocsize / raw_file->obsnchan * sizeof(double);

    int status;
    char *save_block_append = (char *) malloc(50);
    char *save_filename = (char *) malloc(70);
    if(sprintf(save_block_append, "_chan%05d_ddc.dat", chan) < 0){
        printf("Error creating save_filename. Couldn't save file.");
        return;
    }
    else {
        strcpy(save_filename, raw_file->trimmed_filename);
        strcat(save_filename, save_block_append);
    }
    FILE *out_file = fopen(save_filename,"wb");
    
    unsigned long grid_dim_x = (int) ceil(raw_chan_size / MAX_THREADS_PER_BLOCK / 4);
    dim3 griddim(grid_dim_x, 1, 1);
    dim3 blockdim(MAX_THREADS_PER_BLOCK, 1, 1);

    int8_t *h_raw_chan;
    double *h_ddc_chan;
    int8_t *d_raw_chan;
    double *d_ddc_chan;

    bytes_to_chan = raw_file->hdr_size + chan * raw_chan_size;
    bytes_to_next_bloc_chan = raw_file->hdr_size + ((raw_file->obsnchan - chan - 1) * raw_chan_size) + chan * raw_chan_size;
    printf("Raw chan size: %ld\n", raw_chan_size);
    printf("bytes to chan: %ld\n", bytes_to_chan);
    printf("bytes to next bloc chan: %ld\n", bytes_to_next_bloc_chan);

    cudaHostAlloc(&h_raw_chan, raw_chan_size, cudaHostAllocDefault);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaHostAlloc(&h_ddc_chan, ddc_chan_size, cudaHostAllocDefault);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&d_raw_chan, raw_chan_size);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&d_ddc_chan, ddc_chan_size);
        printf("CudaMalloc:\t%s\n", cudaGetErrorString(cudaGetLastError()));

    pos = lseek(fd, bytes_to_chan, SEEK_CUR);

    for(int block = 0; block < raw_file->nblocks; ++block){
        
        bytes_read = read(fd, h_raw_chan, raw_chan_size);
        if(bytes_read != raw_chan_size){
            printf("Error reading GUPPI file. Bytes read: %ld", bytes_read);
            return;
        }
        print_complex_data(h_raw_chan, 1000*4);

        cudaMemcpy(d_raw_chan, h_raw_chan, raw_chan_size, cudaMemcpyHostToDevice);
            printf("CudaMemcpy:\t%s\n", cudaGetErrorString(cudaGetLastError()));

        ddc_channel<<<griddim, blockdim>>> (d_raw_chan, d_ddc_chan, raw_chan_size, block, raw_file->tbin, lo_freq);
            printf("Kernel launch:\t%s\n", cudaGetErrorString(cudaGetLastError()));

        cudaDeviceSynchronize();
            printf("cuda device sync:\t%s\n", cudaGetErrorString(cudaGetLastError()));

        cudaMemcpy(h_ddc_chan, d_ddc_chan, ddc_chan_size, cudaMemcpyDeviceToHost);
            printf("cudamemcpy:\t%s\n", cudaGetErrorString(cudaGetLastError()));


        
        status = fwrite(h_ddc_chan, sizeof(double), raw_chan_size, out_file);
        if(!status){
            perror("Error writing array to file!");
        }
        printf("Num Elements written: %d\n", status);
        

        pos = lseek(fd, bytes_to_next_bloc_chan, SEEK_CUR);
    }

    fclose(out_file);
    cudaFree(d_raw_chan);
    cudaFree(d_ddc_chan);
    cudaFreeHost(h_raw_chan);
    cudaFreeHost(h_ddc_chan);
    free(save_filename);
    free(save_block_append);
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

void print_complex_data(int8_t *data, unsigned long index){
    printf("Index: %ld  (%d, %d), (%d, %d)\n", 
                index, data[index], data[index+1], data[index+2], data[index+3]);
}