#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

// TODO: Check to see if this large of size is needed
#define MAX_RAW_HDR_SIZE (25600)

typedef __int32_t int32_t;


typedef struct {
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
  size_t hdr_size; // Size of header in bytes including DIRECTIO padding
} raw_hdr_t;

int parse_raw_header(char * hdr, size_t len, raw_hdr_t * raw_hdr);
void parse_use_open(char * fname);


int main(int argc, char *argv[]){

    FILE* fp;
    char buffer[MAX_RAW_HDR_SIZE];
    raw_hdr_t raw_hdr;
    int i=0;
    

    if(!argv[1]){
      printf("Please enter a GUPPI RAW file to parse./n");
      return 0;
    }

    printf("Opening File: %s\n", argv[1]);

    
    
    fp = fopen(argv[1],"rb");         

    if(fp == NULL){
        printf("Error opening file");
    }

    fread(&buffer,sizeof(buffer),1,fp);

    // int hdr_size = parse_raw_header(buffer, MAX_RAW_HDR_SIZE, &raw_hdr);

    // if(!hdr_size){
    //     printf("Error parsing header. Couldn't find END record.");
    //     return 0;
    // }

    //fwrite(&buffer, 1, hdr_size, stdout);

    printf("\n");
    fclose(fp);

    parse_use_open(argv[1]);
    

};

void parse_use_open(char * fname){
    int fd;
    raw_hdr_t raw_hdr;
    char buffer[MAX_RAW_HDR_SIZE];
    off_t pos;
    char * bloc_data;
    long PAGESIZE = sysconf(_SC_PAGESIZE);

    printf("Pagesize: %ld\n", PAGESIZE);

    fd = open(fname, O_RDONLY);
    if(fd == -1) {
        printf("Couldn't open file %s", fname);
        return;
    }

    

    raw_hdr.hdr_size = read(fd, buffer, MAX_RAW_HDR_SIZE);

    raw_hdr.hdr_size = parse_raw_header(buffer, sizeof(buffer), &raw_hdr);

    pos = lseek(fd, raw_hdr.hdr_size, SEEK_SET);
    printf("Now at pos: %ld\n", pos);

    // Need padding to create an mmap offset size of a multiple system's page size
    int mmap_pad = PAGESIZE - raw_hdr.hdr_size % PAGESIZE;
    printf("mmap pad: %i\n", mmap_pad);

    bloc_data = mmap(NULL, raw_hdr.blocsize + mmap_pad, PROT_READ, MAP_SHARED, fd, raw_hdr.hdr_size + mmap_pad);
    
    printf("Bloc data: %s\n", bloc_data);

    close(fd);

};

// Returns the last byte location of the header
// Mainly copied from rawspec_rawutils.c in rawspec
// NOTE: Using hget like in rawspec seems like it might be inefficient.
//       This does too. Need to benchmark this against it. 
int parse_raw_header(char * hdr, size_t len, raw_hdr_t * raw_hdr)
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
    // If we found the "END " record
    else if(!strncmp(hdr+i, "END ", 4)) {
      // Move to just after END record
      i += 80;
      // TODO: Have this explained and where the value for MAX_RAW_HDR_SIZE came from
      // Account for DirectIO
      if(raw_hdr->directio) {
        i += (MAX_RAW_HDR_SIZE - i) % 512;
      }
      printf("hdr_size: found END at record %ld\n", i);
      return i;
    }
    
    else if (!strncmp(hdr+i, "BLOCSIZE", 8)){
        raw_hdr->blocsize = strtoul(hdr+i+9, &endptr, 10);
        printf("BLOCSIZE: %ld\n", raw_hdr->blocsize);
    }
    
  }
  return 0;
}


