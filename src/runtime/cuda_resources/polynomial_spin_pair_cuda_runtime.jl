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

 File:            src/algorithms/ahpc_noci/polynomial_spin_pair_cuda_runtime.jl
 Purpose:         Bounded CPU staging and double host buffers for one polynomial Hessian-pair CUDA worker.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    PolynomialSpinPairGPUHostBuffers

Own bounded CPU staging for one polynomial Hessian-pair CUDA worker. Each column
stores one union's binary-scaled occupied overlap, compact state-pair active-
response indices, scaling exponent, and returned polynomial outputs. The value
owns no CUDA state and is reusable across bounded batches.
"""
struct PolynomialSpinPairGPUHostBuffers{T<:Union{Float32,Float64}}
    scaled_effective_active_overlap::Matrix{T}
    active_response_indices::Matrix{UInt16}
    binary_scaling_exponents::Vector{Int32}
    first_cofactor_output::Matrix{T}
    cofactor_scalar_and_contracted_second_output::Matrix{T}
end

function PolynomialSpinPairGPUHostBuffers(
    ::Type{T},
    n_active_occupied::Integer,
    batch_width::Integer,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    width = Int(batch_width)
    0 <= occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA polynomial staging occupied count must lie in 0:32"))
    width >= 1 || throw(ArgumentError("CUDA polynomial staging batch width must be positive"))
    n_local_active_responses = Base.Checked.checked_mul(occupied, occupied)
    return PolynomialSpinPairGPUHostBuffers(
        Matrix{T}(undef, n_local_active_responses, width),
        Matrix{UInt16}(undef, n_local_active_responses, width),
        Vector{Int32}(undef, width),
        Matrix{T}(undef, n_local_active_responses, width),
        Matrix{T}(undef, 2, width),
    )
end

"""
    _polynomial_spin_pair_gpu_per_union_device_bytes(T, n_active_occupied)

Return the conservative transient device-memory bound used to choose a batch
width for the Hessian-pair CUDA circuit. The numerical bound is intentionally
identical to the previously validated polynomial bound, so source cleanup cannot
silently increase the selected batch width.
"""
function _polynomial_spin_pair_gpu_per_union_device_bytes(
    ::Type{T},
    n_active_occupied::Integer,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    0 <= occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA polynomial occupied count must lie in 0:32"))
    n_local_active_responses = Base.Checked.checked_mul(occupied, occupied)
    n_matrix_power_entries = Base.Checked.checked_mul(n_local_active_responses, occupied)
    projection_entries = Base.Checked.checked_mul(n_local_active_responses, max(occupied - 1, 0))
    workspace_entries = Base.Checked.checked_add(
        Base.Checked.checked_add(Base.Checked.checked_mul(4, n_local_active_responses), n_matrix_power_entries),
        Base.Checked.checked_add(projection_entries, occupied + 3),
    )
    workspace_bytes = estimate_array_bytes(T, workspace_entries)
    return Base.Checked.checked_add(
        workspace_bytes,
        Base.Checked.checked_add(
            estimate_array_bytes(T, n_local_active_responses),
            estimate_array_bytes(UInt16, n_local_active_responses),
        ),
    )
end

function _polynomial_spin_pair_gpu_host_staging_bytes(
    ::Type{T},
    n_active_occupied::Integer,
    batch_width::Integer,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    width = Int(batch_width)
    0 <= occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA polynomial occupied count must lie in 0:32"))
    width >= 0 || throw(ArgumentError("CUDA polynomial batch width must be nonnegative"))
    n_local_active_responses = Base.Checked.checked_mul(occupied, occupied)
    per_union = Base.Checked.checked_add(
        Base.Checked.checked_add(
            estimate_array_bytes(T, Base.Checked.checked_mul(2, n_local_active_responses) + 2),
            estimate_array_bytes(UInt16, n_local_active_responses),
        ),
        sizeof(Int32),
    )
    return Base.Checked.checked_mul(per_union, width)
end

function _polynomial_spin_pair_gpu_output_bytes(
    ::Type{T},
    n_union::Integer,
    feature_entries::Integer,
) where {T<:Union{Float32,Float64}}
    unions = Int(n_union)
    features = Int(feature_entries)
    unions >= 0 || throw(ArgumentError("union count must be nonnegative"))
    features >= 0 || throw(ArgumentError("feature count must be nonnegative"))
    return Base.Checked.checked_add(
        estimate_array_bytes(T, 4, unions),
        Base.Checked.checked_add(
            estimate_array_bytes(Int, unions + 1),
            estimate_array_bytes(T, features),
        ),
    )
end

"""
    _fill_polynomial_spin_pair_gpu_host_union!(...)

Fill one fixed-occupied-count union in canonical column-major active-response
order. Binary scaling and traversal order are unchanged from the validated CUDA
implementation; only names describe the state-pair scientific objects directly.
"""
function _fill_polynomial_spin_pair_gpu_host_union!(
    cofactor_staging::PolynomialSpinPairGPUHostBuffers{T},
    support::SpinPairSupportDomain,
    preparation::AHPCStatePairPreparation{T},
    n_active_orbitals::Int,
    union_id::Int,
    local_union::Int,
    n_active_occupied::Int,
) where {T<:Union{Float32,Float64}}
    pair = @inbounds support.union_support[union_id]
    count_ones(pair.left_bits) == n_active_occupied && count_ones(pair.right_bits) == n_active_occupied ||
        throw(DimensionMismatch("CUDA polynomial fixed-count batch contains an unexpected occupied count"))
    n_local_active_responses = n_active_occupied * n_active_occupied
    slot = 1
    maximum_absolute_overlap = zero(T)
    right_bits = pair.right_bits
    while !iszero(right_bits)
        right_active = trailing_zeros(right_bits) + 1
        left_bits = pair.left_bits
        while !iszero(left_bits)
            left_active = trailing_zeros(left_bits) + 1
            value = preparation.effective_active_overlap[left_active, right_active]
            @inbounds cofactor_staging.scaled_effective_active_overlap[slot, local_union] = value
            state_pair_response = left_active + (right_active - 1) * n_active_orbitals
            state_pair_response <= typemax(UInt16) || throw(OverflowError("state-pair active response ID exceeds UInt16"))
            @inbounds cofactor_staging.active_response_indices[slot, local_union] = UInt16(state_pair_response)
            maximum_absolute_overlap = max(maximum_absolute_overlap, abs(value))
            slot += 1
            left_bits &= left_bits - UInt64(1)
        end
        right_bits &= right_bits - UInt64(1)
    end
    slot == n_local_active_responses + 1 ||
        throw(DimensionMismatch("CUDA polynomial fixed-count batch did not fill n_active_occupied^2 overlap entries"))
    exponent = maximum_absolute_overlap == zero(T) ? 0 : last(frexp(maximum_absolute_overlap))
    typemin(Int32) <= exponent <= typemax(Int32) || throw(OverflowError("binary scaling exponent exceeds Int32"))
    @inbounds cofactor_staging.binary_scaling_exponents[local_union] = Int32(exponent)
    @inbounds for response in 1:n_local_active_responses
        cofactor_staging.scaled_effective_active_overlap[response, local_union] =
            ldexp(cofactor_staging.scaled_effective_active_overlap[response, local_union], -exponent)
    end
    return nothing
end

function _fill_polynomial_spin_pair_gpu_host_batch!(
    cofactor_staging::PolynomialSpinPairGPUHostBuffers{T},
    support::SpinPairSupportDomain,
    preparation::AHPCStatePairPreparation{T},
    n_active_orbitals::Int,
    union_first::Int,
    union_last::Int,
    n_active_occupied::Int,
) where {T<:Union{Float32,Float64}}
    1 <= union_first <= union_last <= length(support.union_support) ||
        throw(BoundsError(support.union_support, union_first:union_last))
    width = union_last - union_first + 1
    width <= size(cofactor_staging.scaled_effective_active_overlap, 2) ||
        throw(DimensionMismatch("CUDA polynomial host staging is too small"))
    @inbounds for local_union in 1:width
        _fill_polynomial_spin_pair_gpu_host_union!(
            cofactor_staging,
            support,
            preparation,
            n_active_orbitals,
            union_first + local_union - 1,
            local_union,
            n_active_occupied,
        )
    end
    return width
end

function _finish_polynomial_spin_pair_gpu_host_union!(
    cofactor_scalar::Vector{T},
    one_body_active::Vector{T},
    two_body_inactive_active::Vector{T},
    same_spin_two_body::Vector{T},
    first_cofactor_values::Vector{T},
    cofactor_staging::PolynomialSpinPairGPUHostBuffers{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    feature_offsets::Vector{Int},
    union_id::Int,
    local_union::Int,
    n_active_occupied::Int,
) where {T<:Union{Float32,Float64}}
    n_local_active_responses = n_active_occupied * n_active_occupied
    exponent = Int(@inbounds cofactor_staging.binary_scaling_exponents[local_union])
    scalar_value = ldexp(
        @inbounds(cofactor_staging.cofactor_scalar_and_contracted_second_output[1, local_union]),
        exponent * n_active_occupied,
    )
    second_value = n_active_occupied < 2 ? zero(T) : ldexp(
        @inbounds(cofactor_staging.cofactor_scalar_and_contracted_second_output[2, local_union]),
        exponent * (n_active_occupied - 2),
    )
    isfinite(scalar_value) || throw(ArgumentError("CUDA polynomial complementary-cofactor scalar is non-finite"))
    isfinite(second_value) || throw(ArgumentError("CUDA polynomial contracted second complementary cofactor is non-finite"))
    first_exponent = exponent * max(n_active_occupied - 1, 0)
    feature_offset = @inbounds feature_offsets[union_id]
    one_body_value = zero(T)
    two_body_value = zero(T)
    @inbounds for local_response in 1:n_local_active_responses
        first_cofactor = ldexp(cofactor_staging.first_cofactor_output[local_response, local_union], first_exponent)
        isfinite(first_cofactor) || throw(ArgumentError("CUDA polynomial first complementary cofactor is non-finite"))
        state_pair_response = Int(cofactor_staging.active_response_indices[local_response, local_union])
        one_body_value += coefficients.one_body_active[state_pair_response] * first_cofactor
        two_body_value += coefficients.two_body_inactive_active[state_pair_response] * first_cofactor
        first_cofactor_values[feature_offset + local_response - 1] = first_cofactor
    end
    isfinite(one_body_value) || throw(ArgumentError("CUDA polynomial one-body active contraction is non-finite"))
    isfinite(two_body_value) || throw(ArgumentError("CUDA polynomial inactive-active contraction is non-finite"))
    @inbounds cofactor_scalar[union_id] = scalar_value
    @inbounds one_body_active[union_id] = one_body_value
    @inbounds two_body_inactive_active[union_id] = two_body_value
    @inbounds same_spin_two_body[union_id] = second_value
    return nothing
end

"""
    _build_trivial_spin_pair_intermediates(T, n_active_orbitals, support)

Construct the exact zero-occupied or empty-domain intermediate result without a
CUDA launch. This is the rank-zero endpoint of the same determinant/cofactor
formulation, not a fallback to another algorithm.
"""
function _build_trivial_spin_pair_intermediates(
    ::Type{T},
    n_active_orbitals::Int,
    support::SpinPairSupportDomain,
) where {T<:AbstractFloat}
    n_union = length(support.union_support)
    return ImplicitSpinPairIntermediates(
        n_active_orbitals,
        ones(T, n_union),
        zeros(T, n_union),
        zeros(T, n_union),
        zeros(T, n_union),
        ones(Int, n_union + 1),
        T[],
        support.union_support,
        Val(:validated),
    )
end
