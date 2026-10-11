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

 File:            src/algorithms/ahpc_noci/state_pair.jl
 Purpose:         Inactive/active overlap blocks, effective active overlap, and active-response space per MEBF pair.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    AHPCStatePairPreparation

Store the fixed inactive/active overlap blocks, inactive-overlap determinant and reciprocal condition estimate, Schur-reduced effective active overlap, and active-response space for one MEBF pair.
All matrices are CPU-owned, overlap quantities are dimensionless, and the object contains no determinant-dependent inverse, rank classification, SVD, ERI, or CUDA state.
"""
struct AHPCStatePairPreparation{T<:AbstractFloat}
    X_inactive_inactive::Matrix{T}
    X_inactive_active::Matrix{T}
    X_active_inactive::Matrix{T}
    X_active_active::Matrix{T}
    inactive_overlap_determinant::T
    inactive_reciprocal_condition::T
    effective_active_overlap::Matrix{T}
    response_space::ActiveResponseSpace{T}

    """
        AHPCStatePairPreparation(X_inactive_inactive, X_inactive_active, X_active_inactive, X_active_active, inactive_overlap_determinant, inactive_reciprocal_condition, effective_active_overlap, response_space)

    Construct a validated AHPC state-pair preparation and copy all overlap-block matrices.
    Inputs are not mutated, values are dimensionless and host-resident, and inconsistent dimensions, non-finite values, or an invalid reciprocal condition estimate raise an exception.
    """
    function AHPCStatePairPreparation(
        X_inactive_inactive::AbstractMatrix{T},
        X_inactive_active::AbstractMatrix{T},
        X_active_inactive::AbstractMatrix{T},
        X_active_active::AbstractMatrix{T},
        inactive_overlap_determinant::T,
        inactive_reciprocal_condition::T,
        effective_active_overlap::AbstractMatrix{T},
        response_space::ActiveResponseSpace{T},
    ) where {T<:AbstractFloat}
        n_inactive = size(X_inactive_inactive, 1)
        size(X_inactive_inactive, 2) == n_inactive || throw(DimensionMismatch("X_inactive_inactive must be square"))
        n_active = size(X_active_active, 1)
        size(X_active_active, 2) == n_active || throw(DimensionMismatch("X_active_active must be square"))
        size(X_inactive_active) == (n_inactive, n_active) || throw(DimensionMismatch("X_inactive_active dimensions are inconsistent"))
        size(X_active_inactive) == (n_active, n_inactive) || throw(DimensionMismatch("X_active_inactive dimensions are inconsistent"))
        size(effective_active_overlap) == (n_active, n_active) || throw(DimensionMismatch("effective_active_overlap dimensions are inconsistent"))
        size(response_space.left_active_response, 2) == n_active || throw(DimensionMismatch("active-response rank differs from the active overlap dimension"))
        for (name, values) in (
            (:X_inactive_inactive, X_inactive_inactive),
            (:X_inactive_active, X_inactive_active),
            (:X_active_inactive, X_active_inactive),
            (:X_active_active, X_active_active),
            (:effective_active_overlap, effective_active_overlap),
        )
            all(isfinite, values) || throw(ArgumentError("$name contains a non-finite value"))
        end
        isfinite(inactive_overlap_determinant) || throw(ArgumentError("inactive_overlap_determinant must be finite"))
        isfinite(inactive_reciprocal_condition) && zero(T) < inactive_reciprocal_condition <= one(T) || throw(ArgumentError(
            "inactive_reciprocal_condition must be finite and lie in (0,1]",
        ))
        return new{T}(
            Matrix{T}(X_inactive_inactive),
            Matrix{T}(X_inactive_active),
            Matrix{T}(X_active_inactive),
            Matrix{T}(X_active_active),
            inactive_overlap_determinant,
            inactive_reciprocal_condition,
            Matrix{T}(effective_active_overlap),
            response_space,
        )
    end
end

"""
    ahpc_state_pair_memory(T, left_basis, right_basis)

Estimate conservative peak host numerical storage for AHPC state-pair overlap preparation and return zero device bytes.
The estimate includes retained overlap/response matrices, constructor copies, LU factors, solves, and AO-overlap projection scratch; inputs are read-only and dimension mismatches or checked integer overflow raise an exception.
"""
function ahpc_state_pair_memory(
    ::Type{T},
    left_basis::MolecularBasisFunction,
    right_basis::MolecularBasisFunction,
) where {T<:AbstractFloat}
    n_basis = size(left_basis.orbital_coefficients, 1)
    size(right_basis.orbital_coefficients, 1) == n_basis || throw(DimensionMismatch("left/right MEBF basis dimensions differ"))
    n_inactive = left_basis.expansion.inactive_orbitals
    right_basis.expansion.inactive_orbitals == n_inactive || throw(DimensionMismatch("left/right inactive orbital counts differ"))
    n_active = active_orbital_count(left_basis.expansion)
    active_orbital_count(right_basis.expansion) == n_active || throw(DimensionMismatch("left/right active orbital counts differ"))
    size(left_basis.orbital_coefficients, 2) == n_inactive + n_active || throw(DimensionMismatch("left MEBF orbital columns differ from its expansion"))
    size(right_basis.orbital_coefficients, 2) == n_inactive + n_active || throw(DimensionMismatch("right MEBF orbital columns differ from its expansion"))
    inactive_square = Base.Checked.checked_mul(n_inactive, n_inactive)
    inactive_active = Base.Checked.checked_mul(n_inactive, n_active)
    active_square = Base.Checked.checked_mul(n_active, n_active)
    basis_square = Base.Checked.checked_mul(n_basis, n_basis)
    basis_active = Base.Checked.checked_mul(n_basis, n_active)
    basis_inactive = Base.Checked.checked_mul(n_basis, n_inactive)
    block_elements = Base.Checked.checked_add(
        Base.Checked.checked_add(inactive_square, Base.Checked.checked_mul(2, inactive_active)),
        Base.Checked.checked_mul(2, active_square),
    )
    response_elements = Base.Checked.checked_add(
        Base.Checked.checked_add(basis_square, Base.Checked.checked_mul(2, basis_active)),
        Base.Checked.checked_mul(2, basis_inactive),
    )
    projection_elements = Base.Checked.checked_add(basis_inactive, basis_active)
    factor_and_solve_elements = Base.Checked.checked_add(
        inactive_square,
        Base.Checked.checked_add(Base.Checked.checked_mul(2, inactive_active), basis_inactive),
    )
    total_elements = Base.Checked.checked_add(
        Base.Checked.checked_add(2 * block_elements, 2 * response_elements),
        Base.Checked.checked_add(
            Base.Checked.checked_add(projection_elements, factor_and_solve_elements),
            Base.Checked.checked_mul(4, n_inactive),
        ),
    )
    floating_bytes = estimate_array_bytes(T, total_elements)
    integer_work_bytes = estimate_array_bytes(LinearAlgebra.BlasInt, Base.Checked.checked_mul(2, n_inactive))
    return MemoryEstimate(Base.Checked.checked_add(floating_bytes, integer_work_bytes), 0)
end

"""
    _inactive_overlap_factorization(X_inactive_inactive, tau_singularity)

Factor the fixed inactive overlap with LU and return the factorization, determinant, and one-norm reciprocal condition estimate.
The input is dimensionless and not mutated, only CPU LU/condition-estimation scratch is allocated, and a singular or insufficiently conditioned block raises `ArgumentError` without rank classification or SVD.
"""
function _inactive_overlap_factorization(
    X_inactive_inactive::Matrix{T},
    tau_singularity::T,
) where {T<:Union{Float32,Float64}}
    n_inactive = size(X_inactive_inactive, 1)
    size(X_inactive_inactive, 2) == n_inactive || throw(DimensionMismatch("X_inactive_inactive must be square"))
    n_inactive == 0 && return nothing, one(T), one(T)
    norm_one = opnorm(X_inactive_inactive, 1)
    isfinite(norm_one) && norm_one > zero(T) || throw(ArgumentError("inactive overlap has zero or non-finite one-norm"))
    factorization = lu(X_inactive_inactive; check=false)
    issuccess(factorization) || throw(ArgumentError("inactive overlap is singular under LU factorization"))
    reciprocal_condition = LinearAlgebra.LAPACK.gecon!('1', factorization.factors, norm_one)
    isfinite(reciprocal_condition) && reciprocal_condition > tau_singularity || throw(ArgumentError(
        "inactive overlap reciprocal condition $reciprocal_condition does not exceed tau_singularity=$tau_singularity",
    ))
    determinant = T(det(factorization))
    isfinite(determinant) || throw(ArgumentError("inactive overlap determinant is non-finite"))
    return factorization, determinant, reciprocal_condition
end

"""
    _prepare_ahpc_state_pair(left_basis, right_basis, overlap, tau_singularity)

Form explicit inactive/active cross-overlap blocks, eliminate the fixed inactive block with LU solves, and construct the effective active overlap plus `ActiveResponseSpace` for one MEBF pair.
Inputs are read-only, overlaps are dimensionless, returned arrays are CPU-owned, and invalid dimensions, non-finite data, or a singular/ill-conditioned inactive block raise before ERI or CUDA work.
"""
function _prepare_ahpc_state_pair(
    left_basis::MolecularBasisFunction{T},
    right_basis::MolecularBasisFunction{T},
    overlap::AbstractMatrix{T},
    tau_singularity::T,
) where {T<:Union{Float32,Float64}}
    n_basis = size(overlap, 1)
    size(overlap, 2) == n_basis || throw(DimensionMismatch("AO overlap matrix must be square"))
    size(left_basis.orbital_coefficients, 1) == n_basis || throw(DimensionMismatch("left MEBF basis dimension differs from AO overlap"))
    size(right_basis.orbital_coefficients, 1) == n_basis || throw(DimensionMismatch("right MEBF basis dimension differs from AO overlap"))
    isfinite(tau_singularity) && zero(T) <= tau_singularity < one(T) || throw(ArgumentError("tau_singularity must be finite and lie in [0,1)"))
    all(isfinite, overlap) || throw(ArgumentError("AO overlap contains a non-finite value"))
    all(isfinite, left_basis.orbital_coefficients) || throw(ArgumentError("left MEBF orbitals contain a non-finite value"))
    all(isfinite, right_basis.orbital_coefficients) || throw(ArgumentError("right MEBF orbitals contain a non-finite value"))
    n_inactive = left_basis.expansion.inactive_orbitals
    right_basis.expansion.inactive_orbitals == n_inactive || throw(DimensionMismatch("left/right inactive orbital counts differ"))
    n_active = active_orbital_count(left_basis.expansion)
    active_orbital_count(right_basis.expansion) == n_active || throw(DimensionMismatch("left/right active orbital counts differ"))
    size(left_basis.orbital_coefficients, 2) == n_inactive + n_active || throw(DimensionMismatch("left MEBF orbital columns differ from its expansion"))
    size(right_basis.orbital_coefficients, 2) == n_inactive + n_active || throw(DimensionMismatch("right MEBF orbital columns differ from its expansion"))

    left_inactive = @view left_basis.orbital_coefficients[:, 1:n_inactive]
    left_active = @view left_basis.orbital_coefficients[:, (n_inactive + 1):(n_inactive + n_active)]
    right_inactive = @view right_basis.orbital_coefficients[:, 1:n_inactive]
    right_active = @view right_basis.orbital_coefficients[:, (n_inactive + 1):(n_inactive + n_active)]
    overlap_right_inactive = overlap * right_inactive
    overlap_right_active = overlap * right_active
    X_inactive_inactive = Matrix(left_inactive' * overlap_right_inactive)
    X_inactive_active = Matrix(left_inactive' * overlap_right_active)
    X_active_inactive = Matrix(left_active' * overlap_right_inactive)
    X_active_active = Matrix(left_active' * overlap_right_active)

    if n_inactive == 0
        effective_active_overlap = copy(X_active_active)
        inactive_density_kernel = zeros(T, n_basis, n_basis)
        inactive_density_left_factor = zeros(T, n_basis, 0)
        inactive_density_right_factor = zeros(T, n_basis, 0)
        left_active_response = Matrix(left_active)
        right_active_response = Matrix(right_active)
        response_space = ActiveResponseSpace(
            inactive_density_kernel,
            inactive_density_left_factor,
            inactive_density_right_factor,
            left_active_response,
            right_active_response,
        )
        return AHPCStatePairPreparation(
            X_inactive_inactive,
            X_inactive_active,
            X_active_inactive,
            X_active_active,
            one(T),
            one(T),
            effective_active_overlap,
            response_space,
        )
    end

    factorization, inactive_overlap_determinant, inactive_reciprocal_condition = _inactive_overlap_factorization(
        X_inactive_inactive,
        tau_singularity,
    )
    inactive_solve_active = factorization \ X_inactive_active
    inactive_transpose_solve_active = transpose(factorization) \ transpose(X_active_inactive)
    inactive_transpose_solve_right = transpose(factorization) \ transpose(right_inactive)
    effective_active_overlap = X_active_active - X_active_inactive * inactive_solve_active
    inactive_density_kernel = left_inactive * inactive_transpose_solve_right
    inactive_density_left_factor = Matrix(left_inactive)
    inactive_density_right_factor = Matrix(transpose(inactive_transpose_solve_right))
    left_active_response = left_active - left_inactive * inactive_transpose_solve_active
    right_active_response = right_active - right_inactive * inactive_solve_active
    response_space = ActiveResponseSpace(
        inactive_density_kernel,
        inactive_density_left_factor,
        inactive_density_right_factor,
        left_active_response,
        right_active_response,
    )
    return AHPCStatePairPreparation(
        X_inactive_inactive,
        X_inactive_active,
        X_active_inactive,
        X_active_active,
        inactive_overlap_determinant,
        inactive_reciprocal_condition,
        effective_active_overlap,
        response_space,
    )
end

"""
    prepare_ahpc_state_pair(left_basis, right_basis, overlap; tau_singularity, runtime=RuntimeOptions(), telemetry=nothing)

Prepare the fixed AHPC overlap factorization and active-response space for one MEBF pair using a nonsingular inactive-block LU solve.
Inputs are read-only, overlaps and `tau_singularity` are dimensionless, memory is preflighted before scientific allocation, optional telemetry records timing and zero inverse/rank/SVD calls, and no ERI or CUDA operation occurs.
"""
function prepare_ahpc_state_pair(
    left_basis::MolecularBasisFunction{T},
    right_basis::MolecularBasisFunction{T},
    overlap::AbstractMatrix{T};
    tau_singularity::T,
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:Union{Float32,Float64}}
    estimate = ahpc_state_pair_memory(T, left_basis, right_basis)
    enforce_memory_limits(estimate, runtime)
    work = () -> _prepare_ahpc_state_pair(left_basis, right_basis, overlap, tau_singularity)
    preparation = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :prepare_ahpc_state_pair;
        estimate,
        metadata=Dict(
            :n_inactive => left_basis.expansion.inactive_orbitals,
            :n_active => active_orbital_count(left_basis.expansion),
        ),
        collect_allocations=runtime.collect_allocations,
    )
    if !isnothing(telemetry)
        increment_counter!(telemetry, :n_state_pairs_prepared)
        increment_counter!(telemetry, :n_inactive_lu_factorizations, isempty(preparation.X_inactive_inactive) ? 0 : 1)
        increment_counter!(telemetry, :n_inverse_calls, 0)
        increment_counter!(telemetry, :n_rank_classifier_calls, 0)
        increment_counter!(telemetry, :n_svd_calls, 0)
    end
    return preparation
end
