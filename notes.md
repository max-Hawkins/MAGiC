# MAGiC Notes

Summer 2020 - Breakthrough Listen Internship

  

## Goals

### Accomplished:

 - Created basic Makefile
 - Basic IO working with fopen and open
 - Test transfer speeds to GPU
 - Test cudaMalloc vs cudaHostAlloc speeds - pinned vs non-pinned
 - Create CUDA-enabled Makefile
 - Test if cudaRegisterHost is faster with mmaped data than cudaHostAlloc


  

### TODO:

 - Time different read methods
 - Create simple GPU accelerated energy detection algorithm
 - Test synchronous GPU memory exchange and processing
 - Time entire process with detection write out to NVME module
 - Confirm largest header size
 - Test if increasing the system pagesize decreases time cost of pinning large chunks of memory
 

  
## Assumptions

 - 8 bit data
 - Header and bloc sizes don't change throughout a GUPPI file
 - All blocks are complete - will discard partial blocks 

## Potential Problems

 1. I/O speeds



## Notes

- Why pinned memory is so fast: <https://stackoverflow.com/questions/5736968/why-is-cuda-pinned-memory-so-fast>
    - My understanding: Pinned memory is stored in RAM and can be directly transferred to GPU using DMA. Non-pinned memory could also be in swap, so the CPU has to transfer the data to a pinned buffer before transferring to the GPU using DMA. Seems like pinning larges amounts of memory at once is better than making the CPU do it when you try to transfer non-pinned memory to the GPU. Reportedly faster speeds if the pagesize is increased. Need to test.
    - From my knowledge, pinning memory is good for this application because even though we won't read from this data multiple times, we will need to use asynchronous copy to the GPU. This is only possible with pinned memory. It also lets us use the full speed of the PCIE lines. However, we have to be careful with RAM usage. I'm unsure how much RAM the MeerKAT machines have.

### On personal machine (2070 super)
- mmap -> cudaMalloc then cudaMemcpy of 4.232 GBs of data took ~ 0.7 seconds (6.05 GB/s)
- mmap -> cudaAllocHost -> cudaMemcpy = Transfer rate of 12.21 GB/s
- mmap -> cudaMemcpy = Transfer rate of ~1 GB/s

### On blpc1 Titan Xp
    - CUDA context initialization can take 20-40 seconds
    - Pinned memory transfer (cudamemcpy): 8.02 GB/s
    - cudaHostAlloc by block (0.133 GB per block, 17GB total): 2.643 GB/s - NEED TO DO BETTER

### Thoughts on CUDA optimizations
 - Need to optimize for contiguous memory blocks for easy global memory acces for warps. This lends itself to threads accessing time segments of a single channel in a block. Warps hopefully covering contiguous threads. TODO: calculate if a single warp can capture entire memory of a coarse channel in a block.
 - End goal? mmap an entire GUPPI FILE, cudaAllocHost it in chunks, then launch kernels giving them each a single block. Implement streams with asynchronous copying.
 - Can you pin memory directly from a binary file??? - Dave 
 - SegFault with pinning memory directly with mmap?
 - Is mmap or read faster for cudaAllocHost?


- Using cudaHostRegister is 2x slower than using cudaAllocHost
- Helpful program for getting CUDA device properties: http://www.cs.fsu.edu/~xyuan/cda5125/examples/lect24/devicequery.cu (TODO: run this on MeerKAT hardware)

### Power Spectrum Creation
 - One block on python: ~8 minutes
 - One block on CUDA/TitanXp (only kernel exec time with naive multiplication): 1.8164 ms 
