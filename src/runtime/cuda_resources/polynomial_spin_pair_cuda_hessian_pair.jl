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

# File: src/runtime/cuda_resources/polynomial_spin_pair_cuda_hessian_pair.jl
# Purpose: Prepare CUDA workspaces for polynomial spin-pair Hessian response.
# SPDX-License-Identifier: Apache-2.0

"""
    _polynomial_spin_pair_hessian_pair_gpu_workspace_bytes(T, n_active_occupied, batch_width)

Return exact reusable CUDA workspace bytes for the determinant-Hessian pair-space
polynomial cofactor circuit. Storage contains
`n_active_occupied^3 + 2*n_active_occupied^2 + n_active_occupied + 3` floating
entries per union; no allocation or CUDA initialization occurs.
"""
function _polynomial_spin_pair_hessian_pair_gpu_workspace_bytes(
    ::Type{T},
    n_active_occupied::Integer,
    batch_width::Integer,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    width = Int(batch_width)
    0 <= occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA occupied count must lie in 0:32"))
    width >= 0 || throw(ArgumentError("CUDA batch width must be nonnegative"))
    n_active_responses = Base.Checked.checked_mul(occupied, occupied)
    n_power_entries = Base.Checked.checked_mul(n_active_responses, occupied)
    entries_per_union = Base.Checked.checked_add(
        Base.Checked.checked_add(n_power_entries, Base.Checked.checked_mul(2, n_active_responses)),
        occupied + 3,
    )
    return estimate_array_bytes(T, entries_per_union, width)
end

"""
    _polynomial_spin_pair_hessian_pair_gpu_per_union_device_bytes(T, n_active_occupied)

Return transient per-union CUDA bytes excluding the shared state-pair Hessian-pair
response. The uploaded `n_active_occupied^2` UInt16 active-response index storage
is reused for determinant-Hessian row/column pair indices, so no additional pair-
index allocation is required.
"""
function _polynomial_spin_pair_hessian_pair_gpu_per_union_device_bytes(
    ::Type{T},
    n_active_occupied::Integer,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    0 <= occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA occupied count must lie in 0:32"))
    n_active_responses = Base.Checked.checked_mul(occupied, occupied)
    return Base.Checked.checked_add(
        _polynomial_spin_pair_hessian_pair_gpu_workspace_bytes(T, occupied, 1),
        Base.Checked.checked_add(
            estimate_array_bytes(T, n_active_responses),
            estimate_array_bytes(UInt16, n_active_responses),
        ),
    )
end

"""
    _hessian_pair_index_unchecked(first_active, second_active, n_active_orbitals)

Return the one-based lexicographic determinant-Hessian pair index for one already
validated unordered active-orbital pair. No bounds checking, allocation, floating-
point arithmetic, or CUDA state mutation occurs.
"""
@inline function _hessian_pair_index_unchecked(
    first_active::Int,
    second_active::Int,
    n_active_orbitals::Int,
)
    preceding_first = first_active - 1
    preceding_pairs = preceding_first * n_active_orbitals - div(preceding_first * first_active, 2)
    return preceding_pairs + (second_active - first_active)
end

"""
    _polynomial_spin_pair_hessian_pair_gpu_memory(T, preparation, n_union, feature_entries, n_active_occupied, batch_width)

Estimate retained host output, one state-pair Hessian-pair response, bounded host
staging, and peak device storage for one CUDA worker. Units are bytes; inputs are
read-only and no allocation or CUDA initialization occurs.
"""
function _polynomial_spin_pair_hessian_pair_gpu_memory(
    ::Type{T},
    preparation::AHPCStatePairPreparation{T},
    n_union::Integer,
    feature_entries::Integer,
    n_active_occupied::Integer,
    batch_width::Integer,
) where {T<:Union{Float32,Float64}}
    n_active_orbitals = size(preparation.effective_active_overlap, 1)
    n_state_pair_hessian_pairs = _hessian_pair_count(n_active_orbitals)
    state_pair_hessian_response_bytes = estimate_array_bytes(T, n_state_pair_hessian_pairs, n_state_pair_hessian_pairs)
    host_bytes = Base.Checked.checked_add(
        _polynomial_spin_pair_gpu_output_bytes(T, n_union, feature_entries),
        Base.Checked.checked_add(
            state_pair_hessian_response_bytes,
            _polynomial_spin_pair_gpu_host_staging_bytes(T, n_active_occupied, batch_width),
        ),
    )
    device_bytes = Base.Checked.checked_add(
        state_pair_hessian_response_bytes,
        Base.Checked.checked_mul(
            _polynomial_spin_pair_hessian_pair_gpu_per_union_device_bytes(T, n_active_occupied),
            Int(batch_width),
        ),
    )
    return MemoryEstimate(host_bytes, device_bytes)
end

"""
    _launch_polynomial_spin_pair_hessian_pair_gpu!(buffers, n_active_occupied, width)

Launch the polynomial determinant and complementary-cofactor CUDA evaluation for
one fixed occupied rank in 1:32. The production path uses one union-contiguous
direct-power backbone, the terminal Newton recurrence for determinant and first
complementary cofactors, and occupied-orbital-pair-space second complementary
cofactors when the occupied rank is at least two.
"""
function _launch_polynomial_spin_pair_hessian_pair_gpu!(
    buffers::Tuple,
    n_active_occupied_device::Int32,
    width::Int32,
)
    return _launch_polynomial_complementary_cofactors_gpu!(
        buffers,
        n_active_occupied_device,
        width,
    )
end
