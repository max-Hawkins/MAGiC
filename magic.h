#include <unistd.h>

// TODO: Check to see if this large of size is needed
#define MAX_RAW_HDR_SIZE (25600)
#define MAX_CHUNKSIZE (5368709120 / 2.5) // 5GB - set so that too much memory isn't pinned
#define BYTES_PER_GB (1000000000)

typedef struct {
  char * filename;
  size_t filesize;
  unsigned int nblocks;
  unsigned int blocks_per_chunk;
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
void parse_use_open(char * fname);
void process_cuda_block(int8_t *data, raw_file_t *raw_file);
// void calc_chunksize(raw_file_t *raw_file);


