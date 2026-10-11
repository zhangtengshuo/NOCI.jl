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

 File:            src/algorithms/ahpc_noci/implicit_spin_features.jl
 Purpose:         Union-support scalar channels and structural first cofactors without global response IDs.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    ImplicitSpinPairIntermediates

Store production AHPC union-support scalar channels and structural first-cofactor values without explicit global active-response IDs.
Feature values keep deterministic column-major local-response order, `feature_pairs` aliases the retained union support, all arrays are CPU-owned, and no CI weights, ERIs, fallback state, or CUDA storage are retained.
"""
struct ImplicitSpinPairIntermediates{T<:AbstractFloat}
    active_rank::Int
    cofactor_scalar::Vector{T}
    one_body_active::Vector{T}
    two_body_inactive_active::Vector{T}
    same_spin_two_body::Vector{T}
    feature_offsets::Vector{Int}
    first_cofactor_values::Vector{T}
    feature_pairs::Vector{CanonicalSpinStringPair}

    """
        ImplicitSpinPairIntermediates(active_rank, cofactor_scalar, one_body_active, two_body_inactive_active, same_spin_two_body, feature_offsets, first_cofactor_values, feature_pairs)

    Construct validated production union-support intermediates while retaining owned arrays without copying.
    Inputs are not mutated, values use CPU storage, no CUDA work occurs, and invalid ranks, offsets, bit masks, dimensions, or non-finite values raise an exception.
    """
    function ImplicitSpinPairIntermediates(
        active_rank::Integer,
        cofactor_scalar::Vector{T},
        one_body_active::Vector{T},
        two_body_inactive_active::Vector{T},
        same_spin_two_body::Vector{T},
        feature_offsets::Vector{Int},
        first_cofactor_values::Vector{T},
        feature_pairs::Vector{CanonicalSpinStringPair},
    ) where {T<:AbstractFloat}
        rank = Int(active_rank)
        0 <= rank <= 64 || throw(ArgumentError("active_rank must lie in 0:64"))
        n_union = length(cofactor_scalar)
        length(one_body_active) == n_union || throw(DimensionMismatch("one-body channel length differs from union support"))
        length(two_body_inactive_active) == n_union || throw(DimensionMismatch("inactive-active channel length differs from union support"))
        length(same_spin_two_body) == n_union || throw(DimensionMismatch("same-spin channel length differs from union support"))
        length(feature_pairs) == n_union || throw(DimensionMismatch("implicit feature-pair count differs from union support"))
        length(feature_offsets) == n_union + 1 || throw(DimensionMismatch("feature offset length must equal union support plus one"))
        first(feature_offsets) == 1 || throw(ArgumentError("first feature offset must be one"))
        last(feature_offsets) == length(first_cofactor_values) + 1 || throw(ArgumentError("final feature offset does not match first-cofactor value count"))
        all(index -> feature_offsets[index] <= feature_offsets[index + 1], 1:n_union) || throw(ArgumentError("feature offsets must be nondecreasing"))
        valid_mask = rank == 64 ? typemax(UInt64) : (UInt64(1) << rank) - UInt64(1)
        for union_id in 1:n_union
            pair = feature_pairs[union_id]
            pair.left_bits & ~valid_mask == 0 || throw(ArgumentError("left implicit feature spin string lies outside active_rank"))
            pair.right_bits & ~valid_mask == 0 || throw(ArgumentError("right implicit feature spin string lies outside active_rank"))
            q_left = count_ones(pair.left_bits)
            q_right = count_ones(pair.right_bits)
            q_left == q_right || throw(DimensionMismatch("implicit feature pair has unequal left/right occupied counts"))
            feature_offsets[union_id + 1] - feature_offsets[union_id] == q_left * q_left || throw(DimensionMismatch("implicit feature column width differs from occupied-rank square"))
        end
        all(isfinite, cofactor_scalar) && all(isfinite, one_body_active) && all(isfinite, two_body_inactive_active) && all(isfinite, same_spin_two_body) && all(isfinite, first_cofactor_values) || throw(ArgumentError("implicit spin-pair intermediates contain a non-finite value"))
        return new{T}(rank, cofactor_scalar, one_body_active, two_body_inactive_active, same_spin_two_body, feature_offsets, first_cofactor_values, feature_pairs)
    end

    """
        ImplicitSpinPairIntermediates(active_rank, cofactor_scalar, one_body_active, two_body_inactive_active, same_spin_two_body, feature_offsets, first_cofactor_values, feature_pairs, Val(:validated))

    Construct production intermediates from arrays already dimension-checked, domain-checked, and finite-checked by the internal builder.
    Inputs are retained without copying on CPU, no validation pass or CUDA work occurs, and this method is reserved for the trusted production construction path.
    """
    function ImplicitSpinPairIntermediates(
        active_rank::Int,
        cofactor_scalar::Vector{T},
        one_body_active::Vector{T},
        two_body_inactive_active::Vector{T},
        same_spin_two_body::Vector{T},
        feature_offsets::Vector{Int},
        first_cofactor_values::Vector{T},
        feature_pairs::Vector{CanonicalSpinStringPair},
        ::Val{:validated},
    ) where {T<:AbstractFloat}
        return new{T}(active_rank, cofactor_scalar, one_body_active, two_body_inactive_active, same_spin_two_body, feature_offsets, first_cofactor_values, feature_pairs)
    end
end

"""
    spin_pair_intermediate_memory(T, preparation, support)

Estimate retained scalar channels, structural first-cofactor values, offsets, one state-pair-global Hessian unordered-pair response table, and one reusable CPU polynomial cofactor workspace plus selected-overlap, response-map, and row/column pair-ID buffers per Julia worker.
The global response table has `binomial(active_rank,2)^2` floating entries and is allocated once when union support is nonempty; each worker retains only two `binomial(q,2)` integer pair maps instead of a local floating pair-response matrix. The union-support vector is shared, explicit retained response-ID payload and subset-plan storage are absent, returned units are bytes with zero device bytes, and checked overflow raises an exception.
"""
function spin_pair_intermediate_memory(
    ::Type{T},
    preparation::AHPCStatePairPreparation,
    support::SpinPairSupportDomain,
) where {T<:AbstractFloat}
    active_rank = size(preparation.effective_active_overlap, 1)
    maximum_occupied, feature_entries = _spin_pair_support_shape(support.union_support, active_rank)
    n_union = length(support.union_support)
    output_float_entries = Base.Checked.checked_add(Base.Checked.checked_mul(4, n_union), feature_entries)
    output_bytes = Base.Checked.checked_add(estimate_array_bytes(T, output_float_entries), estimate_array_bytes(Int, n_union + 1))

    global_pair_count = _hessian_pair_count(active_rank)
    global_pair_bytes = n_union == 0 ? 0 : estimate_array_bytes(T, global_pair_count, global_pair_count)
    local_response = Base.Checked.checked_mul(maximum_occupied, maximum_occupied)
    local_pair_count = _hessian_pair_count(maximum_occupied)
    selected_overlap_bytes = estimate_array_bytes(T, maximum_occupied, maximum_occupied)
    response_index_bytes = estimate_array_bytes(Int, local_response)
    pair_index_bytes = estimate_array_bytes(Int, Base.Checked.checked_mul(2, local_pair_count))
    local_bytes = Base.Checked.checked_add(
        selected_overlap_bytes,
        Base.Checked.checked_add(response_index_bytes, pair_index_bytes),
    )
    worker = combine_memory_estimates(
        MemoryEstimate(local_bytes, 0),
        polynomial_workspace_memory(T, maximum_occupied),
    )
    worker_count = min(Threads.nthreads(), n_union)
    worker_bytes = MemoryEstimate(Base.Checked.checked_mul(worker.host_bytes, worker_count), 0)
    return combine_memory_estimates(
        MemoryEstimate(Base.Checked.checked_add(output_bytes, global_pair_bytes), 0),
        worker_bytes,
    )
end

"""
    sparse_spin_pair_contraction_memory(T, sparse, intermediates, runtime)

Estimate aggregate reusable host blocks and peak per-device storage for production alpha-blocked CSR aggregation and Coulomb projection under the configured scratch budget.
GPU host scratch is divided across configured devices, inputs are read-only, returned units are bytes, no array or CUDA context is allocated, and invalid topology, scratch, or checked overflow raises an exception.
"""
function sparse_spin_pair_contraction_memory(
    ::Type{T},
    sparse::SparseSpinPairWeights,
    intermediates::ImplicitSpinPairIntermediates,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    n_response = Base.Checked.checked_mul(intermediates.active_rank, intermediates.active_rank)
    (n_response == 0 || sparse.n_lambda == 0) && return MemoryEstimate(0, 0)
    response_copies = runtime.use_gpu ? 2 : 3
    entries_per_column = Base.Checked.checked_add(4, Base.Checked.checked_mul(response_copies, n_response))
    bytes_per_column = Base.Checked.checked_mul(entries_per_column, sizeof(T))
    gpu_workers = runtime.use_gpu ? length(runtime.gpu_devices) : 1
    required_host = Base.Checked.checked_mul(bytes_per_column, gpu_workers)
    runtime.contraction_host_scratch_bytes >= required_host || return MemoryEstimate(required_host, 0)
    columns = min(sparse.n_kappa, runtime.contraction_host_scratch_bytes ÷ required_host)
    host_bytes = Base.Checked.checked_mul(Base.Checked.checked_mul(bytes_per_column, columns), gpu_workers)
    device_bytes = if runtime.use_gpu
        dense_bytes = estimate_array_bytes(T, n_response, Base.Checked.checked_mul(3, columns))
        offset_bytes = estimate_array_bytes(Int, columns + 1)
        pair_bytes = Base.Checked.checked_mul(sizeof(CanonicalSpinStringPair), columns)
        Base.Checked.checked_add(
            estimate_array_bytes(T, n_response, n_response),
            Base.Checked.checked_add(dense_bytes, Base.Checked.checked_add(offset_bytes, pair_bytes)),
        )
    else
        0
    end
    return MemoryEstimate(host_bytes, device_bytes)
end

"""
    _separable_spin_pair_matrix_element(preparation, coefficients, intermediates, kappa, lambda)

Return overlap, one-body, and all two-body terms except the opposite-spin Coulomb feature contraction for two union IDs.
Inputs are read-only, units follow `AHPCSpinPairMatrixElement`, no allocation or device work occurs, and invalid IDs raise `BoundsError`.
"""
function _separable_spin_pair_matrix_element(
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    intermediates::ImplicitSpinPairIntermediates{T},
    kappa::Int,
    lambda::Int,
) where {T<:AbstractFloat}
    checkbounds(intermediates.cofactor_scalar, kappa)
    checkbounds(intermediates.cofactor_scalar, lambda)
    kappa_scalar = intermediates.cofactor_scalar[kappa]
    lambda_scalar = intermediates.cofactor_scalar[lambda]
    product = kappa_scalar * lambda_scalar
    inactive_scale = preparation.inactive_overlap_determinant^2
    overlap = inactive_scale * product
    one_body = inactive_scale * (2 * coefficients.one_body_inactive * product + intermediates.one_body_active[kappa] * lambda_scalar + kappa_scalar * intermediates.one_body_active[lambda])
    two_body = inactive_scale * (coefficients.two_body_inactive_inactive * product + intermediates.two_body_inactive_active[kappa] * lambda_scalar + kappa_scalar * intermediates.two_body_inactive_active[lambda] + T(0.25) * lambda_scalar * intermediates.same_spin_two_body[kappa] + T(0.25) * kappa_scalar * intermediates.same_spin_two_body[lambda])
    return AHPCSpinPairMatrixElement(overlap, one_body, two_body)
end
