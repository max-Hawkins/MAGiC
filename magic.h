#include <unistd.h>
#include <string.h>

#define MAX_RAW_HDR_SIZE (25600) // From rawspec - TODO: Check to see if this large of size is needed
#define MAX_CHUNKSIZE (5368709120 / 2.5) // 5GB - set so that too much memory isn't pinned
#define BYTES_PER_GB (1073741824)
#define TEST_INDEX (10000) // Index to compare computed results

typedef struct {
  char * filename;
  char * trimmed_filename;
  size_t filesize;
  size_t blocsize;
  size_t hdr_size; // Size of header in bytes including DIRECTIO padding if applicable
  int directio; // whether or not DIRECTIO flag was size
  unsigned int nblocks;
  unsigned int npol;
  unsigned int obsnchan;
  unsigned int overlap;
  double obsfreq;
  double obsbw;
  double tbin;
  double mjd;
  // long int chunksize;
  // unsigned int blocks_per_chunk;
  // unsigned int nchunks;
  // char src_name[81];
  // char telescop[81];
} raw_file_t;

int parse_raw_header(char * hdr, size_t len, raw_file_t * raw_hdr);
void create_power_spectrum(int8_t *file_mmap, raw_file_t *raw_file, int num_cuda_streams);
void get_device_info();
char *trim_filename(char *str);
// void calc_chunksize(raw_file_t *raw_file);


