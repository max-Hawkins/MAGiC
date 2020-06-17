
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

    //parse_use_fopen(argv[1]);

    parse_use_open(argv[1]);
    
    return 0;
};

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

    raw_file.filesize = lseek(fd, 0, SEEK_END);
    printf("file size: %ld", raw_file.filesize);

    raw_file.nblocks = raw_file.filesize / (raw_file.hdr_size + raw_file.blocsize);

    pos = lseek(fd, raw_file.hdr_size, SEEK_SET);
    printf("Now at pos: %ld\n", pos);

    
    

    int8_t *file_mmap = (int8_t *) mmap(NULL, raw_file.filesize, PROT_READ, MAP_SHARED, fd, 0);
    for(int block = 0; block < raw_file.nblocks; block++){
      unsigned long block_index = raw_file.hdr_size + block * (raw_file.hdr_size + raw_file.blocsize);
      int8_t block_address = file_mmap[block_index];

      for(unsigned long int i = block_index - 4; i< block_index + 8; i += 4){
        printf("I: %li\n", i);
        printf("Address: %p\n", &file_mmap[i]);
        printf("(%d, %d), (%d, %d)\n\n", file_mmap[i], file_mmap[i+1], file_mmap[i+2], file_mmap[i+3]);
      }
      process_cuda_block(&file_mmap[block_index], &raw_file);

      printf("Block: %d  Index: %ld  Contents: %d\n", block, block_index, block_address);

    }

    close(fd);
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