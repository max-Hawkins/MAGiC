#include <stdint.h>
#include <stdio.h>
#include "hashpipe.h"
#include "hashpipe_databuf.h"

#define CACHE_ALIGNMENT         4096
#define N_INPUT_BLOCKS          3 
#define N_OUTPUT_BLOCKS         3

/* INPUT BUFFER STRUCTURES
  */
typedef struct array_demo_input_block_header {
   uint64_t mcnt;                    // mcount of first packet
} array_demo_input_block_header_t;

typedef uint8_t array_demo_input_header_cache_alignment[
   CACHE_ALIGNMENT - (sizeof(array_demo_input_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct array_demo_input_block {
   array_demo_input_block_header_t header;
   array_demo_input_header_cache_alignment padding; // Maintain cache alignment
   uint64_t number1;
   uint64_t number2;
} array_demo_input_block_t;

typedef struct array_demo_input_databuf {
   hashpipe_databuf_t header;
   array_demo_input_header_cache_alignment padding;
   array_demo_input_block_t block[N_INPUT_BLOCKS];
} array_demo_input_databuf_t;


/*
  * OUTPUT BUFFER STRUCTURES
  */
typedef struct array_demo_output_block_header {
   uint64_t mcnt;
} array_demo_output_block_header_t;

typedef uint8_t array_demo_output_header_cache_alignment[
   CACHE_ALIGNMENT - (sizeof(array_demo_output_block_header_t)%CACHE_ALIGNMENT)
];

typedef struct array_demo_output_block {
   array_demo_output_block_header_t header;
   array_demo_output_header_cache_alignment padding; // Maintain cache alignment
   uint64_t sum;
} array_demo_output_block_t;

typedef struct array_demo_output_databuf {
   hashpipe_databuf_t header;
   array_demo_output_header_cache_alignment padding;
   //hashpipe_databuf_cache_alignment padding; // Maintain cache alignment
   array_demo_output_block_t block[N_OUTPUT_BLOCKS];
} array_demo_output_databuf_t;

/*
 * INPUT BUFFER FUNCTIONS
 */
hashpipe_databuf_t *array_demo_input_databuf_create(int instance_id, int databuf_id);

static inline array_demo_input_databuf_t *array_demo_input_databuf_attach(int instance_id, int databuf_id)
{
    return (array_demo_input_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int array_demo_input_databuf_detach(array_demo_input_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline void array_demo_input_databuf_clear(array_demo_input_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline int array_demo_input_databuf_block_status(array_demo_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_input_databuf_total_status(array_demo_input_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int array_demo_input_databuf_wait_free(array_demo_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_input_databuf_busywait_free(array_demo_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_input_databuf_wait_filled(array_demo_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_input_databuf_busywait_filled(array_demo_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_input_databuf_set_free(array_demo_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_input_databuf_set_filled(array_demo_input_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

/*
 * OUTPUT BUFFER FUNCTIONS
 */

hashpipe_databuf_t *array_demo_output_databuf_create(int instance_id, int databuf_id);

static inline void array_demo_output_databuf_clear(array_demo_output_databuf_t *d)
{
    hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

static inline array_demo_output_databuf_t *array_demo_output_databuf_attach(int instance_id, int databuf_id)
{
    return (array_demo_output_databuf_t *)hashpipe_databuf_attach(instance_id, databuf_id);
}

static inline int array_demo_output_databuf_detach(array_demo_output_databuf_t *d)
{
    return hashpipe_databuf_detach((hashpipe_databuf_t *)d);
}

static inline int array_demo_output_databuf_block_status(array_demo_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_block_status((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_output_databuf_total_status(array_demo_output_databuf_t *d)
{
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

static inline int array_demo_output_databuf_wait_free(array_demo_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_output_databuf_busywait_free(array_demo_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_free((hashpipe_databuf_t *)d, block_id);
}
static inline int array_demo_output_databuf_wait_filled(array_demo_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_output_databuf_busywait_filled(array_demo_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_busywait_filled((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_output_databuf_set_free(array_demo_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

static inline int array_demo_output_databuf_set_filled(array_demo_output_databuf_t *d, int block_id)
{
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}


