#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>

// TODO: Check to see if this large of size is needed
#define MAX_RAW_HDR_SIZE (25600)

typedef __int32_t int32_t;

typedef struct {
  int directio;
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
  size_t hdr_size; // Size of header in bytes (not including DIRECTIO padding)
} raw_hdr_t;

int parse_raw_header(char * hdr, size_t len, raw_hdr_t * raw_hdr);


int main(int argc, char *argv[]){

    FILE* fp;
    char buffer[6402];
    raw_hdr_t raw_hdr;
    int i=0;
    printf("Opening File: %s\n", argv[1]);
    fp = fopen(argv[1],"rb");         

    if(fp == NULL){
        printf("Error opening file");
    }

    fread(&buffer,sizeof(buffer),1,fp);

    int hdr_size = parse_raw_header(buffer, 6402, &raw_hdr);

    if(!hdr_size){
        printf("Error parsing header. Couldn't find END record.");
        return 0;
    }

    //fwrite(&buffer, 1, 6402, stdout);

    printf("\n");
    fclose(fp);

    //parse_use_open(argv[1]);
    

};

// void parse_use_open(char * fname){
//     int fdin;
//     fdin = open(fname, O_RDONLY);
//     if(fdin == -1) {
//         printf("Couldn't open file %s", fname);
//         return;
//     }

//     raw_hdr_t raw_hdr;

//     off_t offset;

//     offset = rawspec_raw_read_header(fdin, &raw_hdr);

//     printf("BLOCSIZE: %d", raw_hdr.blocsize);


//     close(fdin);

// };

// Returns the last byte location of the header
// Mainly copied from rawspec_rawutils.c in rawspec
// NOTE: Using hget like in rawspec seems like it might be inefficient.
//       This does too. Need to benchmark this against it. 
int parse_raw_header(char * hdr, size_t len, raw_hdr_t * raw_hdr)
{
  int i;
  char * endptr;

  // Loop over the 80-byte records
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
      printf("hdr_size: found END at record %d\n", i);
      return i;
    }
    
    else if (!strncmp(hdr+i, "BLOCSIZE", 8)){
        raw_hdr->blocsize = strtoul(hdr+i+9, &endptr, 10);
        printf("BLOCSIZE: %ld\n", raw_hdr->blocsize);
    }
    
  }
  return 0;
}

// int parse_raw_header_loop(char * hdr, size_t len, int directio)
// {
//     char *p;

//     p = strstr(hdr, "END ");
//     if(!p){
//         printf("Error - Couldn't find end of header.");
//         return 0;
//     }
//     else {
//         return p;
//     }
//     printf("Found: %.80s\n", p);

//     p = strstr(hdr, "BLOCSIZE");
//     printf("Found: %.80s\n", p);

//     p = strstr(hdr, "DIRECTIO");
//     printf("Found: %.80s\n", p);


//     return 0;
// }

