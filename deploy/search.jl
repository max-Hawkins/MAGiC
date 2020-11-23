"""
    Search

Data processing algorithms for energy detection.
""" 
module Search

export SearchAlgoPlan, hit_info_t

# Enum type for the current MeerKAT data format to help with indexing
@enum MK_DIMS::Int8 D_POL=1 D_TIME D_CHAN D_ANT

"""
    SearchAlgo

Abstract type for all search algorithms to inherit from.
"""
abstract type SearchAlgoPlan end

"""
    hit_info_t

Search algorithm hit information metadata to store with raw voltage data.

Fields:
    freq_chan_i::Int - The coarse channel start index
    freq_chan_f::Int - The coarse channel end index
    t_i::Int         - Raw resolution time sample start index
    t_f::Int         - Raw resolution time sample end index
    pizazz::Float32  - "Interestingness" value between 0-1

"""
struct hit_info_t
    freq_chan_i::Int
    freq_chan_f::Int
    t_i::Int
    t_f::Int
    pizazz::Float32
end

"""
    exec_plan(plan::SearchAlgoPlan, data_ptr::Ptr{Any})

Execute the Search Algorithm Plan to calculate intermediate and final pizazz arrays.
"""
function exec_plan(plan::SearchAlgoPlan, data_ptr::Ptr{Any}) end

"""
    hit_mask(plan::SearchAlgoPlan)

Return the hit meta data to the CPU for saving raw data to disk.
"""
function hit_mask(plan::SearchAlgoPlan) end

"""
    gpu_memory(plan::SearchAlgoPlan)

Calculate the GPU memory usage of a single search algorithm plan instance.
"""
function gpu_memory(plan::SearchAlgoPlan) end

end # Module Search

