# Overview

# Summary

Breakthrough Listen works with many radio telescope facilities around the world. At most of these locations, the data from the telescopes is reduced at the facility and analyzed by astronomers after the observation is finished (offline). However, this limits the amount of information we are able to analyze in the observed radio emissions. Any reduction in data (integrating, averaging, squaring, or decimating in time or frequency) potentially removes interesting signals. These reductions are necessary for offline processing though because of the imbalance between signal acquisition rates and available storage. We simply can’t store all the raw (unreduced) data that radio telescopes can produce. At telescope arrays, this problem is further compounded by the number of antennas in the array.

Because we can’t store all the raw data, real-time or nearly real-time processing is required. The data is streamed in from the array, searched by Breakthrough Listen’s hardware for interesting signals, and then the raw data surrounding and including these signals is saved to our limited disk space. This project involves developing a portion of this pipeline.

# Diagrams

![Overview%205b38cfef2eeb4037a23d639d8028e16d/meerkat_diagram.jpg](Overview%205b38cfef2eeb4037a23d639d8028e16d/meerkat_diagram.jpg)

High-level diagram of MeerKAT data flow. Breakthrough Listen User-Supplied-Equipment = BLUSE

![Overview%205b38cfef2eeb4037a23d639d8028e16d/bluse_diagram.jpg](Overview%205b38cfef2eeb4037a23d639d8028e16d/bluse_diagram.jpg)

Diagram of BLUSE dataflow

# Project Aspects

[People Involved](Overview%205b38cfef2eeb4037a23d639d8028e16d/People%20Involved%20f0016c5e4a454c97a838c5e602524f52.csv)

## Data Rates

- For one MeerKAT antenna using L-band:

    856 MHz bandwidth * 2 (nyquist samp) * 2 polarizations * 8 bits per sample 

    = 856,000,000 * 2 * 2 * 8 = 27.392 Gb/s = **3.424 GB/s**

- All 64 MeerKAT antennas at L-band:

    27.392 Gb/s/antenna * 64 antennas = 1,753 Gb/s = **219 GB/s**

- For one SKA -Mid antenna using band 1:

    810 MHz bandwidth * 2 (nyquist samp) * 2 pols (assumed) * 8 bits/sample 

    = 810,000,000 * 2 * 2 * 8 = 25.92 Gb/s = **3.24 GB/s**

- All 197 SKA1-Mid antennas (133 additional from original MeerKAT 64):

    25.92 Gb/s/antenna * 197 antennas = 5,106 Gb/s = **638 GB/s**

## High Performance

- For reasons listed above, the data processing rates must nearly match the data acquisition rates and only a very small subset of the raw data should be saved. Thus, most of the data running through this pipeline will be ‘dumped’ at the end. We’re filtering the data through a strainer, not saving it in a giant pool for astronomers to drown in.

## Commensal

- Breakthrough Listen (BL) is not an observer (maybe rarely later on) at MeerKAT. We simply collect data from a region of the sky that the primary observer decided to point at. This is called a commensal observation. We’re passengers in somebody else’s car - we don’t choose where we go, but we get to see the same things as the driver.
- MeerKAT allows for this type of observation mode because they have multicast abilities. Essentially, BL and other commensal observers tell MeerKAT when they want the observation data, and MeerKAT distributes the same data packets to each requester (subscriber).
- This may seem limiting, and it is, but because MeerKAT is a telescope array, commensal observers can form smaller beams anywhere within the primary beam. This is called beamforming and allows for artificial steering of the telescope using only data. More information on this can be found below.
- You may see the term USE or BLUSE used. This is related to the commensal backend equipment. USE = User Supplied Equipment. BLUSE = Breakthrough Listen User Supplied Equipment. It’s the equipment that’s ‘added on’ to MeerKAT by users.

## SETI

- We’re looking for signs of technology from extraterrestrial intelligence in the radio spectrum. Unlike most astronomers, we don’t know what the signals we’re looking for actually look like. We have some general ideas based on our own technology and biases, but this pipeline shouldn’t be too restrictive in the signals it searches for.

## Modularity

- From the point above, we want to look for different families of signals. Thus this won’t be a pipeline running the same search algorithms for its entire lifespan. We want astronomers to be able to choose and develop their own algorithms as the SETI field and target selection grows and varies. However, with the immense data rates, any developed algorithm must fit within the compute and memory constraints of the MeerKAT system.
- The primary signal BL will be searching for is narrow-band signals that doppler drift in frequency.

## Generalizable

- This pipeline should work for almost any configuration of telescopes, feeds, targets, and frequency/time resolution and bandwidth. We’ve assumed 8 bit complex data so far in the development. Maybe this changes?

## Widely Deployable

- The long term goal is to have similar pipelines to this at other radio telescope arrays - both established and future.

# Pipelines

## MeerKAT Data Pipeline

1. Incoming radio waves hit each dish’s receiver
2. The analog data is digitized
3. The data is channelized by the F-Engines (polyphase filterbank on FPGAs)
4. The data leaves each dish and is transported to the MeerKAT datacenter (KAPB)
5. Intermediate data shuttling
6. Data is sent to BLUSE where each compute node gets a subset of frequencies but from every antenna

## BLUSE Pipeline

1. The data at this point comes in the form of STREAM packets
2. MeerKAT’s current hashpipe setup has two sequential input threads that re-assemble the packets into a GUPPI block of data. This block consists of a header with 80 byte records and a data array (freq vs time) that contains the subset of frequencies from each antenna in use
3. The second input thread puts this data into a block in a ring buffer and sets the corresponding flag to ‘filled’
4. A Julia-created GPU (or calc) thread is constantly waiting for a filled input block and notices the newly filled block. It takes the data and processes it according to the selected search algorithm.
5. The calc thread marks the input data block free and if applicable, marks the output data block filled
6. Raw data determined to contain anomalous data is written to disk as FITS files and the data block is marked as free
    1. The implementation of this is still up in the air
    2. This must be a very small percentage of the raw data

The general idea of this project is to have hashpipe manage the data flow and have a custom Julia calculation thread framework work with this data flow at various steps.

## Julia Calc Thread

### One-time operations:

1. Allocate GPU memory

### Per-block operations:

1. Get pointer to CPU complex memory
2. Transfer data to GPU
3. Perform GPU calculations
4. Return hit metadata back to CPU
5. Use hit metadata to save complex data of interest to disc