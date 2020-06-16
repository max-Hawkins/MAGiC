# MAGiC Notes

Summer 2020 - Breakthrough Listen Internship

  

## Goals

Accomplished:

 - Created basic Makefile
 - Basic IO working with fopen and open
 - Test transfer speeds to GPU

  

TODO:

 - Time different read methods
 - Test cudaMalloc vs cudaHostAlloc speeds - pinned vs non-pinned
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

- mmap -> cudaMalloc then cudaMemcpy of 4.232 GBs of data took ~ 0.7 seconds (6.05 GB/s)