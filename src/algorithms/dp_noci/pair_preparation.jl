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

 File:            src/algorithms/dp_noci/pair_preparation.jl
 Purpose:         Rank-aware full-occupied cross-overlap cofactors with separate alpha/beta SVDs and null modes.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
 Algorithm origin: The determinant-pair method originates from the GronOR program (T. P. Straatsma, R. Broer,
                   A. Sanchez-Mansilla, C. Sousa, and C. de Graaf, J. Chem. Theory Comput. 18, 3549-3565 (2022),
                   doi:10.1021/acs.jctc.2c00266; gitlab.com/gronor/gronor, Apache-2.0). NOCI.jl rewrites the method in
                   Julia with extensive single-node performance optimizations, keeping most functionality and settings
                   consistent with GronOR.
=#

"""
    DPFullOccupiedCofactors

Store rank-aware zeroth and first complementary cofactors of the full alpha/beta occupied cross-overlap for conventional DP-NOCI.
Values are dimensionless CPU arrays, `n_alpha` partitions the block-diagonal spin sectors, null modes retain at most the first two deficient directions needed by the later two-electron kernel, and no ERI or CUDA resource is stored.
"""
struct DPFullOccupiedCofactors{T<:AbstractFloat}
    n_alpha::Int
    rank_deficiency::Int
    overlap::T
    nonzero_cofactor_scale::T
    first_cofactor::Matrix{T}
    reduced_inverse_transpose::Matrix{T}
    left_null_modes::Matrix{T}
    right_null_modes::Matrix{T}
end

"""
    DPDeterminantPairPreparation

Store one conventional DP-NOCI determinant pair together with full-occupied cofactors and unnormalized alpha/beta one-particle transition densities.
All arrays are CPU-owned and dimensionless, source orbitals are retained through `pair`, no input is mutated, and no one- or two-electron integral is retained.
"""
struct DPDeterminantPairPreparation{T<:AbstractFloat}
    pair::PreparedDeterminantPair{T}
    cofactors::DPFullOccupiedCofactors{T}
    alpha_transition_density::Matrix{T}
    beta_transition_density::Matrix{T}
end

"""
    DPDeterminantPairOneBodyElement

Store one bare determinant-pair overlap and electronic one-body matrix element before CI weighting or nuclear repulsion.
Overlap is dimensionless, `electronic_one_body` follows the one-electron integral energy unit, and the immutable result owns no arrays or device resources.
"""
struct DPDeterminantPairOneBodyElement{T<:AbstractFloat}
    overlap::T
    electronic_one_body::T
end

"""
    dp_pair_preparation_memory(T, n_basis, n_alpha, n_beta)

Estimate a conservative peak host payload for full-occupied SVD/cofactor preparation and two AO transition-density outputs.
Counts are dimensionless, the result is in bytes with zero device storage, no array is allocated, and negative dimensions or checked overflow raise an exception.
"""
function dp_pair_preparation_memory(
    ::Type{T},
    n_basis::Integer,
    n_alpha::Integer,
    n_beta::Integer,
) where {T<:AbstractFloat}
    basis_count = Int(n_basis)
    alpha_count = Int(n_alpha)
    beta_count = Int(n_beta)
    basis_count >= 0 && alpha_count >= 0 && beta_count >= 0 || throw(ArgumentError("DP pair dimensions must be nonnegative"))
    occupied_count = Base.Checked.checked_add(alpha_count, beta_count)
    occupied_square = Base.Checked.checked_mul(occupied_count, occupied_count)
    basis_square = Base.Checked.checked_mul(basis_count, basis_count)
    basis_occupied = Base.Checked.checked_mul(basis_count, occupied_count)
    floating_entries = Base.Checked.checked_add(
        Base.Checked.checked_add(Base.Checked.checked_mul(8, occupied_square), Base.Checked.checked_mul(2, basis_square)),
        Base.Checked.checked_add(Base.Checked.checked_mul(4, basis_occupied), Base.Checked.checked_mul(5, occupied_count)),
    )
    return MemoryEstimate(estimate_array_bytes(T, floating_entries), 0)
end

"""
    _orthogonal_matrix_determinant_sign(matrix)

Return the numerical `+1` or `-1` determinant sign of a square orthogonal matrix.
The dimensionless input is read-only, no device work occurs, an empty matrix returns one, and a zero or non-finite determinant raises `ArgumentError`.
"""
function _orthogonal_matrix_determinant_sign(matrix::AbstractMatrix{T}) where {T<:AbstractFloat}
    size(matrix, 1) == size(matrix, 2) || throw(DimensionMismatch("orthogonal determinant sign requires a square matrix"))
    isempty(matrix) && return one(T)
    value = det(matrix)
    isfinite(value) && !iszero(value) || throw(ArgumentError("orthogonal determinant sign is zero or non-finite"))
    return value < zero(T) ? -one(T) : one(T)
end

"""
    _spin_cross_overlap_svd(cross_overlap)

Compute a full CPU SVD and its left/right parity product for one square spin-resolved occupied cross-overlap.
The dimensionless input is not mutated, returned arrays are independently owned, an empty block returns empty factors with positive parity, and nonsquare or non-finite inputs raise an exception.
"""
function _spin_cross_overlap_svd(cross_overlap::AbstractMatrix{T}) where {T<:AbstractFloat}
    occupied_count = size(cross_overlap, 1)
    size(cross_overlap, 2) == occupied_count || throw(DimensionMismatch("spin cross-overlap must be square"))
    all(isfinite, cross_overlap) || throw(ArgumentError("spin cross-overlap contains a non-finite value"))
    occupied_count == 0 && return (left=zeros(T, 0, 0), values=T[], right=zeros(T, 0, 0), parity=one(T))
    factorization = svd(Matrix{T}(cross_overlap); full=true)
    left = Matrix{T}(factorization.U)
    right = Matrix{T}(factorization.V)
    values = Vector{T}(factorization.S)
    parity = _orthogonal_matrix_determinant_sign(left) * _orthogonal_matrix_determinant_sign(right)
    return (left=left, values=values, right=right, parity=parity)
end

"""
    full_occupied_cofactors(alpha_cross_overlap, beta_cross_overlap; tau_singularity=1e-12)

Evaluate dimensionless full-occupied `C^(0)` and `C^(1)` from block-diagonal alpha/beta cross-overlaps using separate full SVDs and one common relative rank threshold.
Inputs are read-only, outputs are CPU-owned, nullity is exact under `tau_singularity*maximum(singular_value)`, no inverse or CUDA work occurs, and invalid shapes or thresholds raise an exception.
"""
function full_occupied_cofactors(
    alpha_cross_overlap::AbstractMatrix{T},
    beta_cross_overlap::AbstractMatrix{T};
    tau_singularity::Real=1.0e-12,
) where {T<:AbstractFloat}
    isfinite(tau_singularity) && tau_singularity >= 0 || throw(ArgumentError("tau_singularity must be finite and nonnegative"))
    alpha = _spin_cross_overlap_svd(alpha_cross_overlap)
    beta = _spin_cross_overlap_svd(beta_cross_overlap)
    n_alpha = length(alpha.values)
    n_beta = length(beta.values)
    occupied_count = n_alpha + n_beta
    occupied_count == 0 && return DPFullOccupiedCofactors(0, 0, one(T), one(T), zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0), zeros(T, 0, 0))
    left = zeros(T, occupied_count, occupied_count)
    right = zeros(T, occupied_count, occupied_count)
    singular_values = Vector{T}(undef, occupied_count)
    if n_alpha > 0
        left[1:n_alpha, 1:n_alpha] .= alpha.left
        right[1:n_alpha, 1:n_alpha] .= alpha.right
        singular_values[1:n_alpha] .= alpha.values
    end
    if n_beta > 0
        beta_range = (n_alpha + 1):occupied_count
        left[beta_range, beta_range] .= beta.left
        right[beta_range, beta_range] .= beta.right
        singular_values[beta_range] .= beta.values
    end
    cutoff = T(tau_singularity) * maximum(abs, singular_values)
    nonzero_indices = findall(value -> abs(value) > cutoff, singular_values)
    null_indices = findall(value -> abs(value) <= cutoff, singular_values)
    rank_deficiency = length(null_indices)
    nonzero_cofactor_scale = alpha.parity * beta.parity
    for index in nonzero_indices
        nonzero_cofactor_scale *= singular_values[index]
    end
    reduced_inverse_transpose = zeros(T, occupied_count, occupied_count)
    for index in nonzero_indices
        reduced_inverse_transpose .+= (one(T) / singular_values[index]) .* (left[:, index] * right[:, index]')
    end
    retained_nullity = min(rank_deficiency, 2)
    left_null_modes = retained_nullity == 0 ? zeros(T, occupied_count, 0) : Matrix(@view left[:, null_indices[1:retained_nullity]])
    right_null_modes = retained_nullity == 0 ? zeros(T, occupied_count, 0) : Matrix(@view right[:, null_indices[1:retained_nullity]])
    first_cofactor = zeros(T, occupied_count, occupied_count)
    overlap = zero(T)
    if rank_deficiency == 0
        overlap = nonzero_cofactor_scale
        first_cofactor .= overlap .* reduced_inverse_transpose
    elseif rank_deficiency == 1
        first_cofactor .= nonzero_cofactor_scale .* (left_null_modes[:, 1] * right_null_modes[:, 1]')
    end
    return DPFullOccupiedCofactors(
        n_alpha,
        rank_deficiency,
        overlap,
        nonzero_cofactor_scale,
        first_cofactor,
        reduced_inverse_transpose,
        left_null_modes,
        right_null_modes,
    )
end

"""
    prepare_dp_determinant_pair(pair; tau_singularity=1e-12, runtime=RuntimeOptions(), telemetry=nothing)

Prepare full-occupied complementary cofactors and unnormalized AO alpha/beta transition densities for one conventional DP-NOCI determinant pair.
The pair is read-only, returned arrays reside on CPU and are dimensionless, memory is preflighted before SVD/output allocation, optional telemetry records rank/SVD counts, and CUDA or ERI work does not occur.
"""
function prepare_dp_determinant_pair(
    pair::PreparedDeterminantPair{T};
    tau_singularity::Real=1.0e-12,
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    n_basis = size(pair.left.alpha, 1)
    size(pair.left.beta, 1) == n_basis && size(pair.right.alpha, 1) == n_basis && size(pair.right.beta, 1) == n_basis || throw(DimensionMismatch("occupied orbital AO dimensions differ"))
    n_alpha = size(pair.alpha_cross_overlap, 1)
    n_beta = size(pair.beta_cross_overlap, 1)
    estimate = dp_pair_preparation_memory(T, n_basis, n_alpha, n_beta)
    enforce_memory_limits(estimate, runtime)
    work = function ()
        cofactors = full_occupied_cofactors(pair.alpha_cross_overlap, pair.beta_cross_overlap; tau_singularity)
        alpha_first = @view cofactors.first_cofactor[1:n_alpha, 1:n_alpha]
        beta_range = (n_alpha + 1):(n_alpha + n_beta)
        beta_first = @view cofactors.first_cofactor[beta_range, beta_range]
        alpha_transition_density = pair.left.alpha * alpha_first * pair.right.alpha'
        beta_transition_density = pair.left.beta * beta_first * pair.right.beta'
        return DPDeterminantPairPreparation(pair, cofactors, alpha_transition_density, beta_transition_density)
    end
    preparation = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :prepare_dp_determinant_pair;
        estimate,
        metadata=Dict(:n_basis => n_basis, :n_alpha => n_alpha, :n_beta => n_beta),
        collect_allocations=runtime.collect_allocations,
    )
    if !isnothing(telemetry)
        increment_counter!(telemetry, :n_dp_pairs_prepared)
        increment_counter!(telemetry, :n_svd_calls, (n_alpha > 0) + (n_beta > 0))
        increment_counter!(telemetry, :n_rank_classifier_calls)
        bucket = min(preparation.cofactors.rank_deficiency, 3)
        increment_counter!(telemetry, Symbol("n_dp_rank_deficiency_", bucket))
    end
    return preparation
end

"""
    contract_dp_one_body(preparation, one_body_operator)

Contract one physical AO one-electron operator with the unnormalized alpha/beta transition densities and return bare DP-NOCI overlap and electronic one-body values.
Inputs are read-only, overlap is dimensionless and the operator/output share energy units, no allocation or CUDA work occurs, and AO dimension mismatches raise `DimensionMismatch`.
"""
function contract_dp_one_body(
    preparation::DPDeterminantPairPreparation{T},
    one_body_operator::AbstractMatrix{T},
) where {T<:AbstractFloat}
    n_basis = size(preparation.alpha_transition_density, 1)
    size(preparation.alpha_transition_density) == (n_basis, n_basis) || throw(DimensionMismatch("alpha transition density must be square"))
    size(preparation.beta_transition_density) == (n_basis, n_basis) || throw(DimensionMismatch("beta transition density dimension differs"))
    size(one_body_operator) == (n_basis, n_basis) || throw(DimensionMismatch("one-body operator dimension differs from transition densities"))
    electronic_one_body = zero(T)
    @inbounds for column in 1:n_basis, row in 1:n_basis
        electronic_one_body += one_body_operator[row, column] * (
            preparation.alpha_transition_density[row, column] + preparation.beta_transition_density[row, column]
        )
    end
    return DPDeterminantPairOneBodyElement(preparation.cofactors.overlap, electronic_one_body)
end
