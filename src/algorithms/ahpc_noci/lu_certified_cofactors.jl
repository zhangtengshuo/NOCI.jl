#==============================================================================#
#      ___           ___           ___                                         #
#     /\\__\\         /\\  \\         /\\  \\          ___                           #
#    /::|  |       /::\\  \\       /::\\  \\        /\\  \\                          #
#   /:|:|  |      /:/\\:\\  \\     /:/\\:\\  \\       \\:\\  \\         ___        ___  #
#  /:/|:|  |__   /:/  \\:\\  \\   /:/  \\:\\  \\      /::\\__\\       /\\  \\      /\\__\\ #
# /:/ |:| /\\__\\ /:/__/ \\:\\__\\ /:/__/ \\:\\__\\  __/:/\\/__/       \\:\\  \\    /:/  / #
# \\/__|:|/:/  / \\:\\  \\ /:/  / \\:\\  \\  \\/__/ /\\/:/  /      ___ /::\\__\\  /:/  /  #
#     |:/:/  /   \\:\\  /:/  /   \\:\\  \\       \\::/__/      /\\  /:/\\/__/ /:/__/   #
#     |::/  /     \\:\\/:/  /     \\:\\  \\       \\:\\__\\      \\:\\/:/  /    \\:\\  \\   #
#     /:/  /       \\::/  /       \\:\\__\\       \\/__/       \\::/  /      \\:\\  \\  #
#     \\/__/         \\/__/         \\/__/               ██   \\/__/        \\/__/  #
#                                                                              #
#==============================================================================#

#=
 NOCI.jl — Nonorthogonal Configuration Interaction in Julia

 File:            src/algorithms/ahpc_noci/lu_certified_cofactors.jl
 Purpose:         Certificate-guarded full-rank LU complementary cofactors with polynomial fallback signaling.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

const AHPC_LU_CERTIFICATE_RESIDUAL_TOLERANCE = 1.0e-4
const AHPC_LU_CERTIFICATE_CONDITION_LIMIT = 1.0e12

"""
    _lu_scaled_determinant(workspace)

Return the determinant of the already factorized, binary-scaled occupied-spin overlap from its LAPACK LU factors and pivot parity.
The workspace is read-only, the result has the scaled determinant units, no allocation or BLAS/CUDA work occurs, and the caller must have completed a square successful `getrf!` factorization.
"""
function _lu_scaled_determinant(
    workspace::PolynomialContractedCofactorWorkspace{T},
) where {T<:Union{Float32,Float64}}
    q = workspace.q
    length(workspace.lu_pivots) == q || throw(DimensionMismatch("LU pivot vector does not match occupied-spin rank"))
    determinant = one(T)
    @inbounds for occupied in 1:q
        determinant *= workspace.scaled[occupied, occupied]
        workspace.lu_pivots[occupied] != occupied && (determinant = -determinant)
    end
    return determinant
end

"""
    _scaled_inverse_certificate!(workspace, matrix, exponent)

Evaluate the infinity-norm residual and condition certificate for an inverse of the binary-scaled occupied-spin overlap already stored in `workspace.previous`.
The original dimensionless overlap and workspace are read-only except that `workspace.next` is overwritten with the residual product, no allocation or CUDA work occurs, and the returned pair is `(residual_inf, condition_inf)`.
"""
function _scaled_inverse_certificate!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    matrix::AbstractMatrix{T},
    exponent::Int,
) where {T<:Union{Float32,Float64}}
    q = workspace.q
    size(matrix) == (q, q) || throw(DimensionMismatch("LU certificate matrix/workspace rank mismatch"))
    q == 0 && return zero(T), one(T)

    mul!(workspace.next, matrix, workspace.previous)
    residual_inf = zero(T)
    scaled_matrix_inf = zero(T)
    inverse_inf = zero(T)
    @inbounds for row in 1:q
        residual_row_sum = zero(T)
        scaled_matrix_row_sum = zero(T)
        inverse_row_sum = zero(T)
        for column in 1:q
            residual_value = ldexp(workspace.next[row, column], -exponent)
            row == column && (residual_value -= one(T))
            residual_row_sum += abs(residual_value)
            scaled_matrix_row_sum += abs(ldexp(matrix[row, column], -exponent))
            inverse_row_sum += abs(workspace.previous[row, column])
        end
        residual_inf = max(residual_inf, residual_row_sum)
        scaled_matrix_inf = max(scaled_matrix_inf, scaled_matrix_row_sum)
        inverse_inf = max(inverse_inf, inverse_row_sum)
    end
    return residual_inf, scaled_matrix_inf * inverse_inf
end

"""
    _prepare_lu_full_rank_certificate!(workspace, matrix)

Binary-scale and LU-factor one finite occupied-spin overlap, solve for the scaled inverse with preallocated LAPACK pivots, and decide whether the inverse is safe for full-rank complementary-cofactor formulas.
The input is dimensionless and read-only; worker-owned `scaled`, `previous`, `next`, and pivot buffers are overwritten; the returned tuple is `(accepted, inverse_computed, determinant_scaled, exponent, residual_inf, condition_inf)`; exact LU singularity, scaled-determinant underflow to zero, non-finite inverse/certificate values, residual above `1e-4`, or infinity-norm condition above `1e12` reject the fast path without SVD or CUDA work.
"""
function _prepare_lu_full_rank_certificate!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    matrix::AbstractMatrix{T},
) where {T<:Union{Float32,Float64}}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("LU cofactor input must be square"))
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("LU-certified cofactor evaluation supports occupied rank 0:32"))
    all(isfinite, matrix) || throw(ArgumentError("LU cofactor input contains a non-finite value"))
    _resize_polynomial_contracted_workspace!(workspace, q)

    q == 0 && return true, false, one(T), 0, zero(T), one(T)

    exponent = _scale_polynomial_input!(workspace, matrix)
    _, _, info = LinearAlgebra.LAPACK.getrf!(workspace.scaled, workspace.lu_pivots; check=false)
    info == 0 || return false, false, zero(T), exponent, T(Inf), T(Inf)

    fill!(workspace.previous, zero(T))
    @inbounds for diagonal in 1:q
        workspace.previous[diagonal, diagonal] = one(T)
    end
    LinearAlgebra.LAPACK.getrs!('N', workspace.scaled, workspace.lu_pivots, workspace.previous)
    all(isfinite, workspace.previous) ||
        return false, true, zero(T), exponent, T(Inf), T(Inf)

    residual_inf, condition_inf = _scaled_inverse_certificate!(workspace, matrix, exponent)
    determinant_scaled = _lu_scaled_determinant(workspace)
    accepted =
        isfinite(determinant_scaled) &&
        !iszero(determinant_scaled) &&
        isfinite(residual_inf) &&
        isfinite(condition_inf) &&
        residual_inf <= T(AHPC_LU_CERTIFICATE_RESIDUAL_TOLERANCE) &&
        condition_inf <= T(AHPC_LU_CERTIFICATE_CONDITION_LIMIT)
    return accepted, true, determinant_scaled, exponent, residual_inf, condition_inf
end

"""
    _lu_certified_state_pair_hessian_contracted_complementary_cofactors!(workspace, matrix, state_pair_response, row_pair_indices, column_pair_indices)

Attempt the full-rank AHPC complementary-cofactor formulas after an LU residual/condition certificate and directly contract the second complementary cofactor against the state-pair determinant-Hessian response.
The dimensionless overlap and energy-valued response are read-only; worker-owned cofactor/LU buffers are mutated; a certified result costs `O(q^3)` factorization plus `O(q^4)` pair contraction; the return is `(result_or_nothing, inverse_computed)`, where `nothing` requires the caller to execute the rank-uniform polynomial PDDC path; no SVD or CUDA work occurs.
"""
function _lu_certified_state_pair_hessian_contracted_complementary_cofactors!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    matrix::AbstractMatrix{T},
    state_pair_response::AbstractMatrix{T},
    row_pair_indices::Vector{Int},
    column_pair_indices::Vector{Int},
) where {T<:Union{Float32,Float64}}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("LU cofactor input must be square"))
    n_local_pairs = _hessian_pair_count(q)
    length(row_pair_indices) == n_local_pairs || throw(DimensionMismatch("union Hessian row-pair index count mismatch"))
    length(column_pair_indices) == n_local_pairs || throw(DimensionMismatch("union Hessian column-pair index count mismatch"))
    size(state_pair_response, 1) == size(state_pair_response, 2) ||
        throw(DimensionMismatch("state-pair Hessian response must be square"))
    maximum_pair_index = size(state_pair_response, 1)
    all(index -> 1 <= index <= maximum_pair_index, row_pair_indices) ||
        throw(BoundsError(state_pair_response, row_pair_indices))
    all(index -> 1 <= index <= maximum_pair_index, column_pair_indices) ||
        throw(BoundsError(state_pair_response, column_pair_indices))

    accepted, inverse_computed, determinant_scaled, exponent, _, _ =
        _prepare_lu_full_rank_certificate!(workspace, matrix)
    accepted || return nothing, inverse_computed

    determinant = ldexp(determinant_scaled, exponent * q)
    first_exponent = exponent * max(q - 1, 0)
    @inbounds for column in 1:q, row in 1:q
        workspace.first_cofactor[row, column] =
            ldexp(determinant_scaled * workspace.previous[column, row], first_exponent)
    end

    second_scaled = zero(T)
    if q >= 2
        column_pair_id = 0
        @inbounds for first_column in 1:(q - 1)
            for second_column in (first_column + 1):q
                column_pair_id += 1
                state_pair_column_pair = column_pair_indices[column_pair_id]
                row_pair_id = 0
                for first_row in 1:(q - 1)
                    for second_row in (first_row + 1):q
                        row_pair_id += 1
                        state_pair_row_pair = row_pair_indices[row_pair_id]
                        inverse_wedge =
                            workspace.previous[first_column, first_row] * workspace.previous[second_column, second_row] -
                            workspace.previous[first_column, second_row] * workspace.previous[second_column, first_row]
                        second_scaled += state_pair_response[state_pair_row_pair, state_pair_column_pair] * inverse_wedge
                    end
                end
            end
        end
        # The state-pair Hessian response stores the antisymmetrized
        # unordered-pair projection with an explicit 1/2.  Contraction with
        # one canonical row/column pair therefore restores the two equivalent
        # ordered determinant-Hessian contributions.
        second_scaled *= T(2) * determinant_scaled
    end
    contracted_second = ldexp(second_scaled, exponent * max(q - 2, 0))

    isfinite(determinant) || return nothing, inverse_computed
    all(isfinite, workspace.first_cofactor) || return nothing, inverse_computed
    isfinite(contracted_second) || return nothing, inverse_computed
    return ComplementaryCofactorResult(
        determinant,
        workspace.first_cofactor,
        contracted_second,
        exponent,
    ), inverse_computed
end
