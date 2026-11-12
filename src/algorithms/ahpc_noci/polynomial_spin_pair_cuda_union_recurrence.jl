#==============================================================================#
#      ___           ___           ___                                         #
#     /\__\         /\  \         /\  \          ___                           #
#    /::|  |       /::\  \       /::\  \        /\  \                          #
#   /:|:|  |      /:/\:\  \     /:/\:\  \       \:\  \         ___        ___  #
#  /:/|:|  |__   /:/  \:\  \   /:/  \:\  \      /::\__\       /\  \      /\__\ #
# /:/ |:| /\__\ /:/__/ \:\__\ /:/__/ \:\__\  __/:/\/__/       \:\  \    /:/  / #
# \/__|:|/:/  / \:\  \ /:/  / \:\  \  \/__/ /\/:/  /      ___ /::\__\  /:/  /  #
#     |:/:/  /   \:\  /:/  /   \:\  \       \::/__/      /\  /:/\/__/ /:/__/   #
#     |::/  /     \:\/:/  /     \:\  \       \:\__\      \:\/:/  /    \:\  \   #
#     /:/  /       \::/  /       \:\__\       \/__/       \::/  /      \:\  \  #
#     \/__/         \/__/         \/__/               ██   \/__/        \/__/  #
#                                                                              #
#==============================================================================#

#=
 NOCI.jl — Nonorthogonal Configuration Interaction in Julia

 File:            src/algorithms/ahpc_noci/polynomial_spin_pair_cuda_union_recurrence.jl
 Purpose:         Union-contiguous CUDA recurrence storage and one-thread-per-overlap-union polynomial cofactor evaluation.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _launch_polynomial_spin_pair_hessian_pair_per_union_gpu!(buffers, rank, width)

Evaluate the determinant and first/second complementary cofactors with one CUDA thread per occupied-overlap union.
The first five recurrence arrays use union-contiguous physical storage. The first five recurrence arrays use union-contiguous physical storage; the quantum-chemical polynomial expressions are unchanged.
"""
function _launch_polynomial_spin_pair_hessian_pair_per_union_gpu!(buffers::Tuple, rank::Int32, width::Int32)
    recurrence_buffers = ntuple(index -> _union_contiguous_polynomial_recurrence_buffer(buffers[index]), Val(5))
    coalesced_buffers = (recurrence_buffers..., buffers[6], buffers[7], buffers[8], buffers[9], buffers[10], buffers[11])
    return CUDA.@cuda threads=AHPC_POLYNOMIAL_GPU_THREADS blocks=cld(Int(width), AHPC_POLYNOMIAL_GPU_THREADS) _polynomial_spin_pair_hessian_pair_gpu_kernel!(coalesced_buffers..., rank, width)
end

"""
    _union_contiguous_polynomial_recurrence_buffer(buffer)

Borrow a CUDA recurrence allocation with adjacent union IDs contiguous in physical storage and return a transposed indexing wrapper with the original logical dimensions.
The input is a preallocated response-by-batch device matrix; this host-only device-pointer reshape/permutation changes no values, creates no reference-counted CUDA aliases or device storage, retains dimensionless recurrence units, requires all subsequent recurrence reads/writes to use the same wrapper, and must never be applied to uploaded overlaps, index tables, or externally transferred outputs.
"""
function _union_contiguous_polynomial_recurrence_buffer(buffer::CUDA.CuMatrix)
    return PermutedDimsArray(reshape(CUDA.cudaconvert(buffer), (size(buffer, 2), size(buffer, 1))), (2, 1))
end
