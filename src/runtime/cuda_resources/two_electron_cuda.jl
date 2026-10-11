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

 File:            src/algorithms/dp_noci/two_electron_cuda.jl
 Purpose:         CUDA direct two-electron contraction with uploaded ERIs and reusable device workspaces.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
 Algorithm origin: The determinant-pair method originates from the GronOR program (T. P. Straatsma, R. Broer,
                   A. Sanchez-Mansilla, C. Sousa, and C. de Graaf, J. Chem. Theory Comput. 18, 3549-3565 (2022),
                   doi:10.1021/acs.jctc.2c00266; gitlab.com/gronor/gronor, Apache-2.0). NOCI.jl rewrites the method in
                   Julia with extensive single-node performance optimizations, keeping most functionality and settings
                   consistent with GronOR.
=#

const DP_TWO_ELECTRON_GPU_THREADS = 256

"""
    dp_two_electron_gpu_workspace_memory(T, integrals)

Estimate device-resident storage for the reusable conventional DP-NOCI CUDA one-/two-body workspace, excluding the separately estimated CPU SVD preparation/workspace.
The integral object is read-only, returned units are bytes with zero host bytes, no CUDA context is initialized, and checked dimension overflow raises before allocation.
"""
function dp_two_electron_gpu_workspace_memory(
    ::Type{T},
    integrals::PackedERIValues,
) where {T<:Union{Float32,Float64}}
    n_basis = integrals.n_orbitals
    n_pairs = canonical_pair_count(n_basis)
    integral_bytes = estimate_array_bytes(T, length(integrals.values))
    label_bytes = estimate_array_bytes(Int32, 2, n_pairs)
    matrix_bytes = estimate_array_bytes(T, 8, n_basis, n_basis)
    null_bytes = estimate_array_bytes(T, 5, n_basis)
    partial_bytes = estimate_array_bytes(T, n_pairs)
    device_bytes = Base.Checked.checked_add(
        Base.Checked.checked_add(integral_bytes, label_bytes),
        Base.Checked.checked_add(matrix_bytes, Base.Checked.checked_add(null_bytes, partial_bytes)),
    )
    return MemoryEstimate(0, device_bytes)
end

"""
    _dp_gpu_pair_labels(n_basis)

Return one-based canonical orbital-pair labels as `Int32` CPU vectors for device upload.
The basis count is dimensionless, outputs own their storage, no CUDA operation occurs, and nonpositive counts or `Int32` overflow raise before allocation.
"""
function _dp_gpu_pair_labels(n_basis::Integer)
    n = Int(n_basis)
    n >= 1 || throw(ArgumentError("n_basis must be positive"))
    n <= typemax(Int32) || throw(OverflowError("DP CUDA orbital labels exceed Int32 capacity"))
    lower, upper = _canonical_orbital_pair_labels(n)
    return Int32.(lower), Int32.(upper)
end

"""
    _dp_regular_two_electron_term_gpu(alpha_factor, beta_factor, pair_sum, p, q, r, s)

Return the full-rank dimensionless direct-minus-exchange DP cofactor term from device-resident AO factors.
Inputs are read-only, no allocation or synchronization occurs, and orbital bounds are guaranteed by the calling CUDA kernel.
"""
@inline function _dp_regular_two_electron_term_gpu(alpha_factor, beta_factor, pair_sum, p::Int, q::Int, r::Int, s::Int)
    return @inbounds pair_sum[p, q] * pair_sum[r, s] -
                     alpha_factor[p, s] * alpha_factor[r, q] - alpha_factor[s, p] * alpha_factor[q, r] -
                     beta_factor[p, s] * beta_factor[r, q] - beta_factor[s, p] * beta_factor[q, r] -
                     alpha_factor[p, r] * alpha_factor[s, q] - alpha_factor[r, p] * alpha_factor[q, s] -
                     beta_factor[p, r] * beta_factor[s, q] - beta_factor[r, p] * beta_factor[q, s]
end

"""
    _dp_singular_two_electron_term_gpu(alpha_factor, beta_factor, pair_sum, left_alpha, left_beta, right_alpha, right_beta, p, q, r, s)

Return the nullity-one-or-two dimensionless direct-minus-exchange DP cofactor term from device-resident AO factors and null vectors.
Inputs are read-only, no allocation or synchronization occurs, and orbital bounds are guaranteed by the calling CUDA kernel.
"""
@inline function _dp_singular_two_electron_term_gpu(
    alpha_factor,
    beta_factor,
    pair_sum,
    left_alpha,
    left_beta,
    right_alpha,
    right_beta,
    p::Int,
    q::Int,
    r::Int,
    s::Int,
)
    first_pair_null = @inbounds left_alpha[p] * right_alpha[q] + left_alpha[q] * right_alpha[p] + left_beta[p] * right_beta[q] + left_beta[q] * right_beta[p]
    return @inbounds pair_sum[q, p] * (left_alpha[r] * right_alpha[s] + left_alpha[s] * right_alpha[r] + left_beta[r] * right_beta[s] + left_beta[s] * right_beta[r]) +
                     first_pair_null * pair_sum[s, r] -
                     right_alpha[p] * (alpha_factor[q, s] * left_alpha[r] + alpha_factor[q, r] * left_alpha[s]) -
                     right_beta[p] * (beta_factor[q, s] * left_beta[r] + beta_factor[q, r] * left_beta[s]) -
                     right_alpha[r] * (alpha_factor[p, s] * left_alpha[q] + alpha_factor[q, s] * left_alpha[p]) -
                     right_beta[r] * (beta_factor[p, s] * left_beta[q] + beta_factor[q, s] * left_beta[p]) -
                     right_alpha[q] * (alpha_factor[p, s] * left_alpha[r] + alpha_factor[p, r] * left_alpha[s]) -
                     right_beta[q] * (beta_factor[p, s] * left_beta[r] + beta_factor[p, r] * left_beta[s]) -
                     right_alpha[s] * (alpha_factor[p, r] * left_alpha[q] + alpha_factor[q, r] * left_alpha[p]) -
                     right_beta[s] * (beta_factor[p, r] * left_beta[q] + beta_factor[q, r] * left_beta[p])
end
