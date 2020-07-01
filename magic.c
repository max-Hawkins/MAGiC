#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <math.h>
#include <fcntl.h>
#include "rawspec_rawutils.h"
#include "magic.h"


int main(int argc, char *argv[]){
    // Command-line arg variables
    extern int optind;
    int c;
    int power_flag = 0;
    int linear_pol_flag = 0;
    int ddc_flag = 0;
    int ddc_chan = -1;
    double ddc_lo_freq = 0;

    int fd;
    size_t pos;
    rawspec_raw_hdr_t rawspec_hdr;
    int num_cuda_streams = 1;
    long PAGESIZE = sysconf(_SC_PAGESIZE); // Get page size for reading later

    if(!argv[1]){
      printf("Please input a GUPPI file to process.\n");
      usage();
      return -1;
    }
    // Account for flagless filename argument
    if(optind < argc){
      optind += 1;
    }
    // Process command line arguments - TODO: long opts
    while ((c = getopt (argc, argv, "hpld:f:")) != -1){
      switch (c)
        {
        case 'h':
          usage();
          return 0;
          break;
        case 'p':
          power_flag = 1;
          break;
        case 'l':
          linear_pol_flag = 1;
          break;
        case 'd':
          ddc_flag = 1;
          ddc_chan = atoi(optarg);
          break;
        case 'f':
          ddc_lo_freq = strtod(optarg, NULL);
          break;
        case '?':
        default:
          printf("Error parsing user input.\n\n");
          usage();
          return 1;
          break;
        }
    }
    // Check for correct flagging when DDC-ing
    if(ddc_flag && (ddc_chan < 0 || ddc_lo_freq <= 0)){
      printf("Error: Need to give channel and LO to DDC.\n");
      usage();
      return -1;
    }

    rawspec_hdr.filename = argv[1];
    rawspec_hdr.trimmed_filename = trim_filename(rawspec_hdr.filename);

    fd = open(rawspec_hdr.filename, O_RDONLY);
    if(fd == -1) {
        printf("Couldn't open file %s\n", rawspec_hdr.filename);
        return -1;
    }    
    printf("Opened file: %s\n", rawspec_hdr.filename);

    // Parse raw header using rawspec functions - Credit: Dave McMahon
    rawspec_raw_read_header(fd, &rawspec_hdr);
    printf("Nchan: %d\n", rawspec_hdr.obsnchan);
    printf("Hdr size: %ld\n", rawspec_hdr.hdr_size);
    printf("Blocsize: %ld\n", rawspec_hdr.blocsize);

    // Validate DDC channel exists
    if(ddc_flag && ddc_chan > rawspec_hdr.obsnchan - 1){
      printf("Selected channel doesn't exist. File only has %d coarse channels.", rawspec_hdr.obsnchan);
      return -1;
    }

    // Initializes cuda context and show GPU names 
    get_device_info();

    pos = lseek(fd, 0, SEEK_SET);
    // // Read in header data and parse it
    // raw_file.hdr_size = read(fd, buffer, MAX_RAW_HDR_SIZE);
    // raw_file.hdr_size = parse_raw_header(buffer, sizeof(buffer), &raw_file);

    // raw_file.filesize = lseek(fd, 0, SEEK_END);
    // printf("file size: %ld\n", raw_file.filesize);
    // raw_file.nblocks = raw_file.filesize / (raw_file.hdr_size + raw_file.blocsize);
    // printf("Nblocks: %d\n", raw_file.nblocks);
    
    if(linear_pol_flag){
      printf("\n---Creating linearly polarized power spectrum.\n");
      create_polarized_power(fd, &rawspec_hdr);
    }    

    if(power_flag){
      printf("\n---Creating power spectrum.\n");
      create_power_spectrum(fd, &rawspec_hdr, 1);
    }

    if(ddc_flag){
      printf("\n---Down-converting channel %d\t LO: %f MHz\n", ddc_chan, ddc_lo_freq);
      ddc_coarse_chan(fd, &rawspec_hdr, ddc_chan, ddc_lo_freq);
    }
    

    close(fd);
    return 0;
};

// Returns the last byte location of the header
// Mainly copied from rawspec_rawutils.c in rawspec
// NOTE: Using hget like in rawspec seems like it might be inefficient.
//       This does too. Need to benchmark this against it. 
// Doesn't have to be efficient though since only ran once per large GB file
// int parse_raw_header(char * hdr, size_t len, raw_file_t * raw_hdr)
// {
//   size_t i;
//   char * endptr;

//   // Loop over the 80-byte records
//   // Compare the first characters for record headers
//   // Then save the information after the = sign into the header struct
//   for(i=0; i<len; i += 80) {
//     // First check for DIRECTIO
//     if (!strncmp(hdr+i, "DIRECTIO", 8)){
//         raw_hdr->directio = strtoul(hdr+i+9, &endptr, 10);
//         printf("DirectIO: %i\n", raw_hdr->directio);
//         //printf("Endptr: %.50s|\n", endptr);
//         //printf("Found DirectIO at %d\n", i);
//     }
//     else if (!strncmp(hdr+i, "BLOCSIZE", 8)){
//         raw_hdr->blocsize = strtoul(hdr+i+9, &endptr, 10);
//         printf("BLOCSIZE: %ld\n", raw_hdr->blocsize);
//     }
//     else if (!strncmp(hdr+i, "OBSNCHAN", 8)){
//         raw_hdr->obsnchan = strtoul(hdr+i+9, &endptr, 10);
//         printf("OBSNCHAN: %dd\n", raw_hdr->obsnchan);
//     }
//     // If we found the "END " record
//     else if(!strncmp(hdr+i, "END ", 4)) {
//       // Move to just after END record
//       i += 80;
//       // Account for DirectIO
//       if(raw_hdr->directio) {
//         i += (MAX_RAW_HDR_SIZE - i) % 512;
//       }
//       printf("hdr_size: found END at record %ld\n", i);
//       return i;
//     }
    
//   }
//   return 0;
// }

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
void usage() {
    fprintf(stderr,
    "\nUsage: ./magic [GUPPI_file] [options]\n"
    "\n"
    "Options:\n"
    "  -p,                Calculates and saves the power spectrum\n"
    "  -l,                Calculates and saves the linearly polarized power\n"
    "  -d [coarse_chan],  Digitally down-convert coarse channel (see -f flag)"
    "  -f [i_frequency],  Mix selected channel with i_frequency in MHz"
    "\n"
    "  -h,                Show this message\n"
  );
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
