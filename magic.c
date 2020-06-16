#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime.h>

// TODO: Check to see if this large of size is needed
#define MAX_RAW_HDR_SIZE (25600)
#define MAX_CHUNKSIZE (5368709120) // 5GB - set so that too much memory isn't pinned
#define BYTES_PER_GB (1000000000)

typedef struct {
  char * filename;
  size_t filesize;
  unsigned int nblocks;
  long int chunksize;
  unsigned int nchunks;
  int directio; // whether or not DIRECTIO flag was size
  size_t blocsize;
  unsigned int npol;
  unsigned int obsnchan;
  unsigned int overlap;
  double obsfreq;
  double obsbw;
  double tbin;
  double mjd;
  //char src_name[81];
  //char telescop[81];
  off_t hdr_pos; // Offset of start of header
  size_t hdr_size; // Size of header in bytes including DIRECTIO padding if applicable
} raw_file_t;


int parse_raw_header(char * hdr, size_t len, raw_file_t * raw_hdr);
void process_cuda(int8_t *data, raw_file_t *raw_file);
void parse_use_open(char * fname);
void parse_use_fopen(char *fname);
void calc_chunksize(raw_file_t *raw_file);


int main(int argc, char *argv[]){

    //parse_use_fopen(argv[1]);

    parse_use_open(argv[1]);
    
    return 0;
};

void process_cuda(int8_t *data, raw_file_t *raw_file){
    int8_t *d_data;
    printf("CudaMalloc size: %0.3f GBs\n", (float) raw_file->chunksize / BYTES_PER_GB);

    cudaMalloc((void *) &d_data, raw_file->chunksize);
    //cudaHostAlloc((void *) data, raw_file->chunksize, cudaHostAllocDefault);
    cudaMemcpy(d_data, data, raw_file->chunksize, cudaMemcpyHostToDevice);
}



void parse_use_open(char * fname){
    int fd;
    raw_file_t raw_file;
    char buffer[MAX_RAW_HDR_SIZE];
    off_t pos;
    // Get page size for reading later
    long PAGESIZE = sysconf(_SC_PAGESIZE);
    printf("Pagesize: %ld\n", PAGESIZE);

    raw_file.filename = fname;
    fd = open(fname, O_RDONLY);
    if(fd == -1) {
        printf("Couldn't open file %s", fname);
        return;
    }    
    // Read in header data and parse it
    raw_file.hdr_size = read(fd, buffer, MAX_RAW_HDR_SIZE);
    raw_file.hdr_size = parse_raw_header(buffer, sizeof(buffer), &raw_file);

    pos = lseek(fd, raw_file.hdr_size, SEEK_SET);
    printf("Now at pos: %ld\n", pos);

    // Need padding to create an mmap offset size of a multiple system's page size
    //int mmap_pad = raw_file.hdr_size % PAGESIZE;
    //printf("mmap pad: %i\n", mmap_pad);

    raw_file.filesize = lseek(fd, 0, SEEK_END);
    printf("file size: %ld", raw_file.filesize);

    calc_chunksize(&raw_file);

    if(!raw_file.chunksize){
      printf("Error calculating chunksize for given raw_file.");
      return;
    }
    printf("Chunksize: %ld", raw_file.chunksize);

    int8_t *bloc_data = (int8_t *) mmap(NULL, raw_file.chunksize, PROT_READ, MAP_SHARED, fd, 0);

    // Print out complex polarization values
    for(int i =raw_file.hdr_size - 4; i< raw_file.hdr_size + 16; i += 4){
      printf("I: %i\n", i);
      printf("(%d, %d), (%d, %d)\n\n", bloc_data[i], bloc_data[i+1], bloc_data[i+2], bloc_data[i+3]);
      
    }
    printf("Size of bloc_data: %ld\n", raw_file.chunksize );

    process_cuda(bloc_data, &raw_file);
    

    close(fd);

    //fwrite(&bloc_data, 1, raw_file.hdr_size + mmap_pad, stdout);

};

// Returns the last byte location of the header
// Mainly copied from rawspec_rawutils.c in rawspec
// NOTE: Using hget like in rawspec seems like it might be inefficient.
//       This does too. Need to benchmark this against it. 
int parse_raw_header(char * hdr, size_t len, raw_file_t * raw_hdr)
{
  size_t i;
  char * endptr;

  // Loop over the 80-byte records
  // Compare the first characters for record headers
  // Then save the information after the = sign into the header struct
  for(i=0; i<len; i += 80) {
    // First check for DIRECTIO
    if (!strncmp(hdr+i, "DIRECTIO", 8)){
        raw_hdr->directio = strtoul(hdr+i+9, &endptr, 10);
        printf("DirectIO: %i\n", raw_hdr->directio);
        //printf("Endptr: %.50s|\n", endptr);
        //printf("Found DirectIO at %d\n", i);
    }
    else if (!strncmp(hdr+i, "BLOCSIZE", 8)){
        raw_hdr->blocsize = strtoul(hdr+i+9, &endptr, 10);
        printf("BLOCSIZE: %ld\n", raw_hdr->blocsize);
    }
    // If we found the "END " record
    else if(!strncmp(hdr+i, "END ", 4)) {
      // Move to just after END record
      i += 80;
      // Account for DirectIO
      if(raw_hdr->directio) {
        i += (MAX_RAW_HDR_SIZE - i) % 512;
      }
      printf("hdr_size: found END at record %ld\n", i);
      return i;
    }
    
  }
  return 0;
}

// Calculates the number of data chunks to pass to the GPU
// Dependent on the MAX_CHUNKSIZE
void calc_chunksize(raw_file_t *raw_file){
    //TODO: Create good chunksize algorithm
    size_t chunksize = raw_file->filesize;

    for(int chunks = 1; chunks < 128; chunks *= 2){
      chunksize = raw_file->filesize / chunks;
      printf("Chunksize in function: %ld\n", chunksize);

      if(chunksize < MAX_CHUNKSIZE){
          raw_file->chunksize = chunksize;
          raw_file->nchunks = chunks;
          return;
      }
    }
}

void parse_use_fopen(char *fname){
    FILE* fp;
    char buffer[MAX_RAW_HDR_SIZE];
    raw_file_t raw_file;
    int i=0;
    

    if(!fname){
      printf("Please enter a GUPPI RAW file to parse.\n");
      return;
    }

    printf("Opening File: %s\n", fname);    
    fp = fopen(fname,"rb");         

    if(fp == NULL){
        printf("Error opening file");
    }

    fread(&buffer,sizeof(buffer),1,fp);

    int hdr_size = parse_raw_header(buffer, MAX_RAW_HDR_SIZE, &raw_file);

    // if(!hdr_size){
    //     printf("Error parsing header. Couldn't find END record.");
    //     return 0;
    // }

    //fwrite(&buffer, 1, hdr_size, stdout);

    printf("\n");
    fclose(fp);
}