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

 File:            src/algorithms/ahpc_noci/polynomial_spin_pair_cuda_support.jl
 Purpose:         Shared constants, safety reserves, and helpers for the polynomial spin-pair CUDA kernels.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

const AHPC_SPIN_PAIR_GPU_SAFETY_RESERVE_BYTES = 64 * 1024^2
const AHPC_POLYNOMIAL_GPU_THREADS = 32

"""
    _polynomial_gpu_fill_newton!(newton_cofactor_matrix, elementary_symmetric_coefficients, matrix_powers, n_local_active_responses, union_id, order)

Fill one Newton/Faddeev transformation matrix used by the rank-uniform polynomial
complementary-cofactor CUDA circuit. All device arrays are caller-owned, one
union column is mutated, and the arithmetic order is identical to the validated
polynomial implementation.
"""
@inline function _polynomial_gpu_fill_newton!(
    newton_cofactor_matrix,
    elementary_symmetric_coefficients,
    matrix_powers,
    n_local_active_responses::Int,
    union_id::Int,
    order::Int,
)
    @inbounds for response in 1:n_local_active_responses
        newton_cofactor_matrix[response, union_id] = zero(eltype(newton_cofactor_matrix))
    end
    @inbounds for power in 0:order
        coefficient = elementary_symmetric_coefficients[order - power + 1, union_id]
        isodd(power) && (coefficient = -coefficient)
        power_offset = power * n_local_active_responses
        for response in 1:n_local_active_responses
            newton_cofactor_matrix[response, union_id] += coefficient * matrix_powers[power_offset + response, union_id]
        end
    end
    return nothing
end
