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

 File:            src/algorithms/ahpc_noci/state_pair_hessian_response.jl
 Purpose:         Canonical unordered active-orbital-pair indexing for determinant-Hessian response channels.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _hessian_pair_index(first_active, second_active, n_active_orbitals)

Map one canonical unordered active-orbital pair `first_active < second_active` to
one-based lexicographic determinant-Hessian pair order, with the first active
index varying slowest. The map is allocation-free and rejects noncanonical or
out-of-range pairs.
"""
@inline function _hessian_pair_index(
    first_active::Int,
    second_active::Int,
    n_active_orbitals::Int,
)
    0 <= n_active_orbitals <= 64 || throw(ArgumentError("active-orbital count must lie in 0:64"))
    1 <= first_active < second_active <= n_active_orbitals ||
        throw(ArgumentError("Hessian pair indices must satisfy 1 <= first < second <= n_active_orbitals"))
    preceding_first = first_active - 1
    preceding_pairs = preceding_first * n_active_orbitals - div(preceding_first * first_active, 2)
    return preceding_pairs + (second_active - first_active)
end

"""
    state_pair_hessian_response(coefficients, n_active_orbitals)

Project one NOCI state pair's ordered Coulomb-minus-exchange active response onto
the determinant Hessian's unordered active-orbital-pair basis. For `i<k` and
`j<l`, the returned entry is
`(K_ij,kl + K_kl,ij - K_il,kj - K_kj,il)/2`, in the same lexicographic pair order
used by `_hessian_pair_index`.

The coefficient matrices are read-only, the returned CPU matrix owns
`binomial(n_active_orbitals,2)^2` energy-valued entries, and no numerical-rank
classification, threshold, inverse, SVD, subset state, or CUDA work is used.
"""
function state_pair_hessian_response(
    coefficients::ActiveHamiltonianCoefficients{T},
    n_active_orbitals::Int,
) where {T<:AbstractFloat}
    0 <= n_active_orbitals <= 64 || throw(ArgumentError("active-orbital count must lie in 0:64"))
    n_response = Base.Checked.checked_mul(n_active_orbitals, n_active_orbitals)
    size(coefficients.coulomb_response) == (n_response, n_response) ||
        throw(DimensionMismatch("Coulomb response dimension mismatch"))
    size(coefficients.exchange_response) == (n_response, n_response) ||
        throw(DimensionMismatch("exchange response dimension mismatch"))
    all(isfinite, coefficients.coulomb_response) ||
        throw(ArgumentError("Coulomb response contains a non-finite value"))
    all(isfinite, coefficients.exchange_response) ||
        throw(ArgumentError("exchange response contains a non-finite value"))

    n_hessian_pairs = _hessian_pair_count(n_active_orbitals)
    response = Matrix{T}(undef, n_hessian_pairs, n_hessian_pairs)
    n_active_orbitals < 2 && return response
    half = T(0.5)

    column_pair_id = 0
    @inbounds for first_column in 1:(n_active_orbitals - 1)
        for second_column in (first_column + 1):n_active_orbitals
            column_pair_id += 1
            row_pair_id = 0
            for first_row in 1:(n_active_orbitals - 1)
                for second_row in (first_row + 1):n_active_orbitals
                    row_pair_id += 1
                    state_pair_response_ij = first_row + (first_column - 1) * n_active_orbitals
                    state_pair_response_kl = second_row + (second_column - 1) * n_active_orbitals
                    state_pair_response_il = first_row + (second_column - 1) * n_active_orbitals
                    state_pair_response_kj = second_row + (first_column - 1) * n_active_orbitals
                    pair_value = half * (
                        _coulomb_exchange_response_element(coefficients, state_pair_response_ij, state_pair_response_kl) +
                        _coulomb_exchange_response_element(coefficients, state_pair_response_kl, state_pair_response_ij) -
                        _coulomb_exchange_response_element(coefficients, state_pair_response_il, state_pair_response_kj) -
                        _coulomb_exchange_response_element(coefficients, state_pair_response_kj, state_pair_response_il)
                    )
                    isfinite(pair_value) ||
                        throw(ArgumentError("state-pair Hessian response contains a non-finite value"))
                    response[row_pair_id, column_pair_id] = pair_value
                end
            end
        end
    end
    return response
end

"""
    _fill_union_hessian_pair_indices!(row_pair_indices, column_pair_indices, response_indices, n_active_orbitals)

Map one union's ordered active-response grid into the state-pair determinant-
Hessian unordered-pair basis. Destinations are overwritten without allocation;
response indices are read-only and must represent the canonical column-major
Cartesian occupied grid.
"""
function _fill_union_hessian_pair_indices!(
    row_pair_indices::Vector{Int},
    column_pair_indices::Vector{Int},
    response_indices::Vector{Int},
    n_active_orbitals::Int,
)
    0 <= n_active_orbitals <= 64 || throw(ArgumentError("active-orbital count must lie in 0:64"))
    n_local_response = length(response_indices)
    n_active_occupied = isqrt(n_local_response)
    n_active_occupied * n_active_occupied == n_local_response ||
        throw(DimensionMismatch("local response count must be an occupied-count square"))
    n_hessian_pairs = _hessian_pair_count(n_active_occupied)
    length(row_pair_indices) == n_hessian_pairs ||
        throw(DimensionMismatch("union Hessian row-pair index count mismatch"))
    length(column_pair_indices) == n_hessian_pairs ||
        throw(DimensionMismatch("union Hessian column-pair index count mismatch"))
    n_active_occupied < 2 && return row_pair_indices, column_pair_indices

    pair_id = 0
    @inbounds for first_local in 1:(n_active_occupied - 1)
        first_response = response_indices[first_local]
        first_active = (first_response - 1) % n_active_orbitals + 1
        for second_local in (first_local + 1):n_active_occupied
            second_response = response_indices[second_local]
            second_active = (second_response - 1) % n_active_orbitals + 1
            pair_id += 1
            row_pair_indices[pair_id] = _hessian_pair_index(first_active, second_active, n_active_orbitals)
        end
    end

    pair_id = 0
    @inbounds for first_local in 1:(n_active_occupied - 1)
        first_response = response_indices[1 + (first_local - 1) * n_active_occupied]
        first_active = div(first_response - 1, n_active_orbitals) + 1
        for second_local in (first_local + 1):n_active_occupied
            second_response = response_indices[1 + (second_local - 1) * n_active_occupied]
            second_active = div(second_response - 1, n_active_orbitals) + 1
            pair_id += 1
            column_pair_indices[pair_id] = _hessian_pair_index(first_active, second_active, n_active_orbitals)
        end
    end
    return row_pair_indices, column_pair_indices
end

"""
    _contracted_second_from_state_pair_hessian_response!(workspace, state_pair_response, row_pair_indices, column_pair_indices)

Evaluate the complete contracted second complementary cofactor directly against
one state-pair determinant-Hessian response and one union's row/column Hessian-
pair index maps. Arithmetic remains rank-uniform `O(n_active_occupied^5)` and no
inverse, numerical-rank branch, threshold, SVD, subset fallback, allocation, or
CUDA work occurs.
"""
function _contracted_second_from_state_pair_hessian_response!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    state_pair_response::AbstractMatrix{T},
    row_pair_indices::Vector{Int},
    column_pair_indices::Vector{Int},
) where {T<:AbstractFloat}
    n_active_occupied = workspace.q
    n_local_pairs = _hessian_pair_count(n_active_occupied)
    length(row_pair_indices) == n_local_pairs ||
        throw(DimensionMismatch("union Hessian row-pair index count mismatch"))
    length(column_pair_indices) == n_local_pairs ||
        throw(DimensionMismatch("union Hessian column-pair index count mismatch"))
    size(state_pair_response, 1) == size(state_pair_response, 2) ||
        throw(DimensionMismatch("state-pair Hessian response must be square"))
    n_active_occupied < 2 && return zero(T)

    maximum_pair_index = size(state_pair_response, 1)
    all(index -> 1 <= index <= maximum_pair_index, row_pair_indices) ||
        throw(BoundsError(state_pair_response, row_pair_indices))
    all(index -> 1 <= index <= maximum_pair_index, column_pair_indices) ||
        throw(BoundsError(state_pair_response, column_pair_indices))

    contracted_second_cofactor = zero(T)
    @inbounds for power in 0:(n_active_occupied - 2)
        elementary_order = n_active_occupied - 1 - power
        sign = isodd(power) ? -one(T) : one(T)
        _newton_transformation_workspace!(workspace, elementary_order - 1)
        left_power = workspace.powers[power + 1]
        identity_factor = n_active_occupied < 32 && (power == 0 || power == n_active_occupied - 2)
        hessian_pair_contraction = zero(T)
        column_pair_id = 0
        for first_column in 1:(n_active_occupied - 1)
            for second_column in (first_column + 1):n_active_occupied
                column_pair_id += 1
                state_pair_column_pair = column_pair_indices[column_pair_id]
                row_pair_id = 0
                for first_row in 1:(n_active_occupied - 1)
                    for second_row in _hessian_second_row_indices(first_row, first_column, second_column, n_active_occupied, identity_factor)
                        row_pair_id = _occupied_hessian_pair_index(first_row, second_row, n_active_occupied)
                        wedge_bilinear =
                            left_power[first_column, first_row] * workspace.newton[second_column, second_row] -
                            left_power[second_column, first_row] * workspace.newton[first_column, second_row] -
                            left_power[first_column, second_row] * workspace.newton[second_column, first_row] +
                            left_power[second_column, second_row] * workspace.newton[first_column, first_row]
                        state_pair_row_pair = row_pair_indices[row_pair_id]
                        hessian_pair_contraction += state_pair_response[state_pair_row_pair, state_pair_column_pair] * wedge_bilinear
                    end
                end
            end
        end
        contracted_second_cofactor += sign * hessian_pair_contraction
    end
    return contracted_second_cofactor
end

"""
    _polynomial_state_pair_hessian_contracted_complementary_cofactors!(workspace, matrix, state_pair_response, row_pair_indices, column_pair_indices)

Compute determinant, complete first complementary cofactors, and the contracted
second complementary cofactor using one state-pair determinant-Hessian response
plus union-local Hessian-pair index maps. Binary scaling and accumulation order
match the validated pair-space implementation.
"""
function _polynomial_state_pair_hessian_contracted_complementary_cofactors!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    matrix::AbstractMatrix{T},
    state_pair_response::AbstractMatrix{T},
    row_pair_indices::Vector{Int},
    column_pair_indices::Vector{Int},
) where {T<:AbstractFloat}
    n_active_occupied = size(matrix, 1)
    size(matrix, 2) == n_active_occupied || throw(DimensionMismatch("polynomial cofactor input must be square"))
    0 <= n_active_occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("polynomial cofactor engine supports occupied count 0:32"))
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

    second_scaled = _contracted_second_from_state_pair_hessian_response!(
        workspace,
        state_pair_response,
        row_pair_indices,
        column_pair_indices,
    )
    contracted_second_cofactor = ldexp(second_scaled, exponent * max(n_active_occupied - 2, 0))
    isfinite(determinant) || throw(ArgumentError("polynomial determinant is non-finite"))
    all(isfinite, workspace.first_cofactor) ||
        throw(ArgumentError("polynomial first cofactors contain a non-finite value"))
    isfinite(contracted_second_cofactor) ||
        throw(ArgumentError("polynomial contracted state-pair Hessian second cofactor is non-finite"))
    return ComplementaryCofactorResult(
        determinant,
        workspace.first_cofactor,
        contracted_second_cofactor,
        exponent,
    )
end

