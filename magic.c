#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <math.h>

#include <fcntl.h>
#include "magic.h"

int main(int argc, char *argv[]){

    int fd;
    raw_file_t raw_file;
    char buffer[MAX_RAW_HDR_SIZE] __attribute__ ((aligned (512)));
    off_t pos; 
    int num_cuda_streams = 4;
    // Get page size for reading later
    long PAGESIZE = sysconf(_SC_PAGESIZE);
    printf("Pagesize: %ld\n", PAGESIZE);

    if(!argv[1]){
      printf("Please input a GUPPI file to process.\n");
      return -1;
    }

    raw_file.filename = argv[1];
    raw_file.trimmed_filename = trim_filename(raw_file.filename);

    fd = open(raw_file.filename, O_RDONLY);
    if(fd == -1) {
        printf("Couldn't open file %s", raw_file.filename);
        return -1;
    }    
    printf("Opened file: %s\n", raw_file.filename);
    // Initializes cuda context and show GPU names 
    get_device_info();

    // Read in header data and parse it
    raw_file.hdr_size = read(fd, buffer, MAX_RAW_HDR_SIZE);
    raw_file.hdr_size = parse_raw_header(buffer, sizeof(buffer), &raw_file);

    raw_file.filesize = lseek(fd, 0, SEEK_END);
    printf("file size: %ld", raw_file.filesize);

    raw_file.nblocks = raw_file.filesize / (raw_file.hdr_size + raw_file.blocsize);
    printf("Nblocks: %d", raw_file.nblocks);
    

    // mmaps the entire GUPPI file before breaking into individual blocks - need to change for concurrency
    // int8_t *file_mmap = (int8_t *) mmap(NULL, raw_file.filesize, PROT_READ, MAP_SHARED, fd, 0);
    // create_power_spectrum(file_mmap, &raw_file, num_cuda_streams);
    
    create_polarized_power(fd, &raw_file);

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

// strips the user-supplied filename 
char *trim_filename(char *str)
{
    char *trimmed_filename = strdup(str);
    size_t len = 0;
    char *endp = NULL;

    // Remove any relative pathing to get GUPPI base filename
    trimmed_filename = strrchr(trimmed_filename, '/') + 1;

    len = strlen(trimmed_filename);
    endp = trimmed_filename + len;
    // Remove '.raw' at end of filename
    endp[-4] = '\0';
   
    return trimmed_filename;
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
