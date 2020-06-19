#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include "magic.h"

int main(int argc, char *argv[]){

    int fd;
    raw_file_t raw_file;
    char buffer[MAX_RAW_HDR_SIZE];
    off_t pos;
    char *fname = argv[1]; 
    // Get page size for reading later
    long PAGESIZE = sysconf(_SC_PAGESIZE);
    printf("Pagesize: %ld\n", PAGESIZE);

    raw_file.filename = fname;
    fd = open(fname, O_RDONLY);
    if(fd == -1) {
        printf("Couldn't open file %s", fname);
        return -1;
    }    
    // Initializes cuda context and show GPU names 
    get_device_info();

    // Read in header data and parse it
    raw_file.hdr_size = read(fd, buffer, MAX_RAW_HDR_SIZE);
    raw_file.hdr_size = parse_raw_header(buffer, sizeof(buffer), &raw_file);

    raw_file.filesize = lseek(fd, 0, SEEK_END);
    printf("file size: %ld", raw_file.filesize);

    raw_file.nblocks = raw_file.filesize / (raw_file.hdr_size + raw_file.blocsize);

    pos = lseek(fd, raw_file.hdr_size, SEEK_SET);
    printf("Now at pos: %ld\n", pos);

    // mmaps the entire GUPPI file before breaking into individual blocks - need to change for concurrency
    int8_t *file_mmap = (int8_t *) mmap(NULL, raw_file.filesize, PROT_READ, MAP_SHARED, fd, 0);
    for(int block = 0; block < raw_file.nblocks; block++){
      printf("\n\n--------- Block %d ----------\n", block);
      unsigned long block_index = raw_file.hdr_size + block * (raw_file.hdr_size + raw_file.blocsize);
      int8_t block_address = file_mmap[block_index];

      for(unsigned long int i = block_index - 4; i< block_index + 8; i += 4){
        printf("I: %li  Address: %p\n", i, &file_mmap[i]);
        printf("(%d, %d), (%d, %d)\n\n", file_mmap[i], file_mmap[i+1], file_mmap[i+2], file_mmap[i+3]);
      }
      process_cuda_block(&file_mmap[block_index], &raw_file);

      // printf("Block: %d  Index: %ld  Contents: %d\n", block, block_index, block_address);
      printf("Block: %d  Index: %d  Contents: %d\n", block, TEST_INDEX, file_mmap[block_index + TEST_INDEX]);

    }

    close(fd);
    return 0;
};

// Returns the last byte location of the header
// Mainly copied from rawspec_rawutils.c in rawspec
// NOTE: Using hget like in rawspec seems like it might be inefficient.
//       This does too. Need to benchmark this against it. 
// Doesn't have to be efficient though since only ran once per large GB file
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
    else if (!strncmp(hdr+i, "OBSNCHAN", 8)){
        raw_hdr->obsnchan = strtoul(hdr+i+9, &endptr, 10);
        printf("OBSNCHAN: %dd\n", raw_hdr->obsnchan);
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
// void calc_chunksize(raw_file_t *raw_file){
//     // First calculate nblocks for raw file
//     
//     //TODO: Create good chunksize algorithm
//     size_t chunksize = raw_file->filesize;

//     for(int chunks = 1; chunks <= 128; chunks *= 2){
//       chunksize = raw_file->filesize / chunks;

//       if(chunksize < MAX_CHUNKSIZE){
//           raw_file->chunksize = chunksize;
//           raw_file->nchunks = chunks;
//           raw_file->blocks_per_chunk = raw_file->nblocks / chunks;
//           printf("Blocks per chunk: %d", raw_file->blocks_per_chunk);
//           return;
//       }
//     }
// }
