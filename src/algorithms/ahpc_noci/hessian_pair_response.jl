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

 File:            src/algorithms/ahpc_noci/hessian_pair_response.jl
 Purpose:         Index algebra and second-row layout for determinant-Hessian pairs at fixed first row.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _hessian_second_row_indices(first_row, first_column, second_column, n_active_occupied, identity_factor)

Return increasing second-row indices for determinant-Hessian pairs at fixed first row.
When one polynomial factor is the identity, only pairs sharing a local occupied
index with the column pair can contribute. Inputs are validated dimensionless
indices; this allocation-free CPU/CUDA helper changes no scientific array and
preserves lexicographic pair order without a numerical rank decision.
"""
@inline function _hessian_second_row_indices(first_row::Int, first_column::Int, second_column::Int, n_active_occupied::Int, identity_factor::Bool)
    if !identity_factor || first_row == first_column || first_row == second_column
        return (first_row + 1):1:n_active_occupied
    elseif first_row < first_column
        return first_column:(second_column - first_column):second_column
    elseif first_row < second_column
        return second_column:1:second_column
    end
    return (n_active_occupied + 1):1:n_active_occupied
end

"""
    _occupied_hessian_pair_index(first_row, second_row, n_active_occupied)

Return the lexicographic ID of an already validated occupied row pair.
Inputs are dimensionless with 1 <= first_row < second_row <= n_active_occupied;
no allocation, mutation, rank test or device state is involved.
"""
@inline function _occupied_hessian_pair_index(first_row::Int, second_row::Int, n_active_occupied::Int)
    return (first_row - 1) * n_active_occupied - div((first_row - 1) * first_row, 2) + second_row - first_row
end

"""
    _hessian_pair_count(n_pair_indices)

Return the number of canonical unordered index pairs `i<k` for an occupied or active-orbital pair space.
The count is dimensionless, allocates no storage, and accepts `0:64` because a state-pair response can span 42 active orbitals even when the occupied-spin rank is at most 32.
"""
@inline function _hessian_pair_count(n_pair_indices::Int)
    0 <= n_pair_indices <= 64 || throw(ArgumentError("pair-space dimension must lie in 0:64"))
    return Base.Checked.checked_mul(n_pair_indices, n_pair_indices - 1) ÷ 2
end

@inline function _coulomb_exchange_response_element(
    coefficients::ActiveHamiltonianCoefficients{T},
    first_state_pair_response::Int,
    second_state_pair_response::Int,
) where {T<:AbstractFloat}
    return coefficients.coulomb_response[first_state_pair_response, second_state_pair_response] -
           coefficients.exchange_response[first_state_pair_response, second_state_pair_response]
end

"""
    _fill_hessian_pair_response!(hessian_pair_response, response_indices, coefficients)

Project an arbitrary local Coulomb-minus-exchange response directly into the
independent row-pair/column-pair representation seen by the determinant Hessian.
For canonical `i<k` and `j<l`, the stored value is

`(K_ij,kl + K_kl,ij - K_il,kj - K_kj,il)/2`.

The factor includes the determinant-Hessian pair-exchange projection and the
factor of two obtained from equality of the coefficient-derivative and
matrix-power Frechet polynomial contributions. The destination is overwritten,
inputs are read-only, pair order is lexicographic, and no inverse, determinant
threshold, rank classifier, SVD, subset fallback, allocation, or CUDA work occurs.
"""
function _fill_hessian_pair_response!(
    hessian_pair_response::Matrix{T},
    response_indices::Vector{Int},
    coefficients::ActiveHamiltonianCoefficients{T},
) where {T<:AbstractFloat}
    n_local_response = length(response_indices)
    n_active_occupied = isqrt(n_local_response)
    n_active_occupied * n_active_occupied == n_local_response ||
        throw(DimensionMismatch("local response count must be an occupied-count square"))
    n_hessian_pairs = _hessian_pair_count(n_active_occupied)
    size(hessian_pair_response) == (n_hessian_pairs, n_hessian_pairs) ||
        throw(DimensionMismatch("Hessian pair response must have size (binomial(n_active_occupied,2),binomial(n_active_occupied,2))"))
    n_active_occupied < 2 && return hessian_pair_response

    half = T(0.5)
    column_pair_id = 0
    @inbounds for first_column in 1:(n_active_occupied - 1)
        for second_column in (first_column + 1):n_active_occupied
            column_pair_id += 1
            row_pair_id = 0
            for first_row in 1:(n_active_occupied - 1)
                for second_row in (first_row + 1):n_active_occupied
                    row_pair_id += 1
                    response_ij = first_row + (first_column - 1) * n_active_occupied
                    response_kl = second_row + (second_column - 1) * n_active_occupied
                    response_il = first_row + (second_column - 1) * n_active_occupied
                    response_kj = second_row + (first_column - 1) * n_active_occupied

                    state_pair_response_ij = response_indices[response_ij]
                    state_pair_response_kl = response_indices[response_kl]
                    state_pair_response_il = response_indices[response_il]
                    state_pair_response_kj = response_indices[response_kj]

                    pair_value = half * (
                        _coulomb_exchange_response_element(coefficients, state_pair_response_ij, state_pair_response_kl) +
                        _coulomb_exchange_response_element(coefficients, state_pair_response_kl, state_pair_response_ij) -
                        _coulomb_exchange_response_element(coefficients, state_pair_response_il, state_pair_response_kj) -
                        _coulomb_exchange_response_element(coefficients, state_pair_response_kj, state_pair_response_il)
                    )
                    isfinite(pair_value) || throw(ArgumentError("Hessian pair response contains a non-finite value"))
                    hessian_pair_response[row_pair_id, column_pair_id] = pair_value
                end
            end
        end
    end
    return hessian_pair_response
end

"""
    _contracted_second_from_hessian_pair_response!(workspace, hessian_pair_response)

Evaluate the complete rank-uniform determinant-Hessian contraction from the
independent Hessian pair response and the coefficient-derivative polynomial
family. Prepared matrix powers and elementary symmetric coefficients are reused;
only the worker Newton matrix is overwritten. The algorithm is division-free,
uses `O(n_active_occupied^5)` arithmetic with `binomial(n_active_occupied,2)^2`
response entries per polynomial order, and has no inverse, rank branch, SVD,
subset fallback, projection buffer access, or CUDA work.
"""
function _contracted_second_from_hessian_pair_response!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    hessian_pair_response::AbstractMatrix{T},
) where {T<:AbstractFloat}
    n_active_occupied = workspace.q
    n_hessian_pairs = _hessian_pair_count(n_active_occupied)
    size(hessian_pair_response) == (n_hessian_pairs, n_hessian_pairs) ||
        throw(DimensionMismatch("Hessian pair response dimension mismatch"))
    n_active_occupied < 2 && return zero(T)

    contracted_second = zero(T)
    @inbounds for power in 0:(n_active_occupied - 2)
        elementary_order = n_active_occupied - 1 - power
        sign = isodd(power) ? -one(T) : one(T)
        _newton_transformation_workspace!(workspace, elementary_order - 1)
        left_power = workspace.powers[power + 1]
        identity_factor = n_active_occupied < 32 && (power == 0 || power == n_active_occupied - 2)
        pair_contraction = zero(T)
        column_pair_id = 0
        for first_column in 1:(n_active_occupied - 1)
            for second_column in (first_column + 1):n_active_occupied
                column_pair_id += 1
                row_pair_id = 0
                for first_row in 1:(n_active_occupied - 1)
                    for second_row in _hessian_second_row_indices(first_row, first_column, second_column, n_active_occupied, identity_factor)
                        row_pair_id = _occupied_hessian_pair_index(first_row, second_row, n_active_occupied)
                        wedge_bilinear =
                            left_power[first_column, first_row] * workspace.newton[second_column, second_row] -
                            left_power[second_column, first_row] * workspace.newton[first_column, second_row] -
                            left_power[first_column, second_row] * workspace.newton[second_column, first_row] +
                            left_power[second_column, second_row] * workspace.newton[first_column, first_row]
                        pair_contraction += hessian_pair_response[row_pair_id, column_pair_id] * wedge_bilinear
                    end
                end
            end
        end
        contracted_second += sign * pair_contraction
    end
    return contracted_second
end

"""
    _polynomial_hessian_pair_contracted_complementary_cofactors!(workspace, matrix, hessian_pair_response)

Compute determinant, complete first complementary cofactors, and the exact
rank-uniform contracted second determinant derivative from a preprojected Hessian
pair response. Binary scaling and first-cofactor ordering match the canonical
polynomial implementation; the routine uses no determinant inverse, rank
classifier, threshold, SVD, subset fallback, response projection buffer, or CUDA.
"""
function _polynomial_hessian_pair_contracted_complementary_cofactors!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    matrix::AbstractMatrix{T},
    hessian_pair_response::AbstractMatrix{T},
) where {T<:AbstractFloat}
    n_active_occupied = size(matrix, 1)
    size(matrix, 2) == n_active_occupied || throw(DimensionMismatch("polynomial cofactor input must be square"))
    0 <= n_active_occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("polynomial cofactor engine supports occupied count 0:32"))
    n_hessian_pairs = _hessian_pair_count(n_active_occupied)
    size(hessian_pair_response) == (n_hessian_pairs, n_hessian_pairs) ||
        throw(DimensionMismatch("Hessian pair response dimension mismatch"))
    _resize_polynomial_contracted_workspace!(workspace, n_active_occupied)

    exponent = _scale_polynomial_input!(workspace, matrix)
    _elementary_symmetrics_scaled!(workspace)
    _matrix_powers_scaled!(workspace)

    determinant_scaled = workspace.elementary[n_active_occupied + 1]
    determinant = ldexp(determinant_scaled, exponent * n_active_occupied)
    first_exponent = exponent * max(n_active_occupied - 1, 0)
    if n_active_occupied > 0
        _newton_transformation_workspace!(workspace, n_active_occupied - 1)
        @inbounds for column in 1:n_active_occupied, row in 1:n_active_occupied
            workspace.first_cofactor[row, column] = ldexp(workspace.newton[column, row], first_exponent)
        end
    end

    second_scaled = _contracted_second_from_hessian_pair_response!(workspace, hessian_pair_response)
    contracted_second = ldexp(second_scaled, exponent * max(n_active_occupied - 2, 0))
    isfinite(determinant) || throw(ArgumentError("polynomial determinant is non-finite"))
    all(isfinite, workspace.first_cofactor) || throw(ArgumentError("polynomial first cofactors contain a non-finite value"))
    isfinite(contracted_second) || throw(ArgumentError("polynomial contracted Hessian-pair second cofactor is non-finite"))
    return ComplementaryCofactorResult(
        determinant,
        workspace.first_cofactor,
        contracted_second,
        exponent,
    )
end

