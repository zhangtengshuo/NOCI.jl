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

 File:            src/algorithms/ahpc_noci/polynomial_cofactors.jl
 Purpose:         Singular-safe polynomial determinant and first cofactors with exact binary scaling exponents.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    PolynomialFirstCofactorResult

Store the determinant, complete first complementary-cofactor matrix, and exact binary scaling exponent produced by the singular-safe polynomial cofactor prototype.
The determinant and first cofactors have the units implied by the input overlap matrix, arrays are CPU-owned, no inverse/rank classifier/SVD is used, and the result contains no CUDA state.
"""
struct PolynomialFirstCofactorResult{T<:AbstractFloat}
    cofactor_scalar::T
    first_cofactor::Matrix{T}
    binary_scaling_exponent::Int
end

"""
    polynomial_first_cofactor_memory(T, n_active_occupied)

Return the host numerical payload needed by the current Faddeev-LeVerrier determinant/first-cofactor prototype for one occupied-spin dimension.
The estimate counts four dense `q x q` floating-point matrices, reports zero device bytes, allocates no scientific storage, and accepts dimensions in `0:32`.
"""
function polynomial_first_cofactor_memory(
    ::Type{T},
    n_active_occupied::Integer,
) where {T<:AbstractFloat}
    q = Int(n_active_occupied)
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("polynomial cofactor prototype supports occupied rank 0:32"))
    return MemoryEstimate(estimate_array_bytes(T, 4, q, q), 0)
end

"""
    _faddeev_leverrier_scaled_det_adjugate(matrix)

Evaluate the determinant and ordinary adjugate of a finite square matrix with the Faddeev-LeVerrier polynomial recurrence.
The input is read-only and already expected to be numerically scaled; the routine allocates bounded `O(q^2)` CPU matrices, divides only by positive integer recurrence indices, performs no inverse/rank classification/SVD/CUDA work, and accepts singular matrices without a fallback branch.
"""
function _faddeev_leverrier_scaled_det_adjugate(
    matrix::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("polynomial cofactor input must be square"))
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("polynomial cofactor prototype supports occupied rank 0:32"))
    all(isfinite, matrix) || throw(ArgumentError("polynomial cofactor input contains a non-finite value"))

    if q == 0
        return one(T), zeros(T, 0, 0)
    elseif q == 1
        return matrix[1, 1], ones(T, 1, 1)
    end

    previous = Matrix{T}(I, q, q)
    next = zeros(T, q, q)
    adjugate = zeros(T, q, q)
    final_characteristic_coefficient = zero(T)

    @inbounds for order in 1:q
        mul!(next, matrix, previous)
        coefficient = -tr(next) / T(order)
        if order == q
            final_characteristic_coefficient = coefficient
            break
        end
        for diagonal in 1:q
            next[diagonal, diagonal] += coefficient
        end
        if order == q - 1
            sign = isodd(q - 1) ? -one(T) : one(T)
            for column in 1:q, row in 1:q
                adjugate[row, column] = sign * next[row, column]
            end
        end
        previous, next = next, previous
    end

    determinant_sign = isodd(q) ? -one(T) : one(T)
    determinant = determinant_sign * final_characteristic_coefficient
    return determinant, adjugate
end

"""
    polynomial_first_cofactors(matrix; runtime=RuntimeOptions())

Compute the determinant and complete first complementary cofactors of one occupied-spin overlap matrix with a rank-uniform Faddeev-LeVerrier polynomial circuit.
The input is read-only, outputs are CPU-owned, exact power-of-two scaling limits intermediate magnitude growth, host memory is preflighted, no inverse/rank classifier/SVD/fallback/CUDA work is performed, and singular matrices are valid inputs.
"""
function polynomial_first_cofactors(
    matrix::AbstractMatrix{T};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("polynomial cofactor input must be square"))
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("polynomial cofactor prototype supports occupied rank 0:32"))
    all(isfinite, matrix) || throw(ArgumentError("polynomial cofactor input contains a non-finite value"))
    enforce_memory_limits(polynomial_first_cofactor_memory(T, q), runtime)

    if q == 0
        return PolynomialFirstCofactorResult(one(T), zeros(T, 0, 0), 0)
    end

    maximum_absolute = zero(T)
    @inbounds for value in matrix
        maximum_absolute = max(maximum_absolute, abs(value))
    end
    exponent = maximum_absolute == zero(T) ? 0 : last(frexp(maximum_absolute))
    scaled = Matrix{T}(undef, q, q)
    @inbounds for column in 1:q, row in 1:q
        scaled[row, column] = ldexp(matrix[row, column], -exponent)
    end

    determinant_scaled, adjugate_scaled = _faddeev_leverrier_scaled_det_adjugate(scaled)
    determinant = ldexp(determinant_scaled, exponent * q)
    first_exponent = exponent * max(q - 1, 0)
    first_cofactor = Matrix{T}(undef, q, q)
    @inbounds for column in 1:q, row in 1:q
        first_cofactor[row, column] = ldexp(adjugate_scaled[column, row], first_exponent)
    end

    isfinite(determinant) || throw(ArgumentError("polynomial determinant is non-finite"))
    all(isfinite, first_cofactor) || throw(ArgumentError("polynomial first cofactors contain a non-finite value"))
    return PolynomialFirstCofactorResult(determinant, first_cofactor, exponent)
end

"""
    active_pair_count(n_active_occupied)

Return the number of strictly ordered occupied-index pairs `(i,k)` with `i < k` for one occupied-spin rank.
The argument is dimensionless, no allocation occurs, and dimensions outside `0:32` raise `ArgumentError`.
"""
function active_pair_count(n_active_occupied::Integer)
    q = Int(n_active_occupied)
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("occupied rank must lie in 0:32"))
    return div(q * (q - 1), 2)
end

"""
    active_pair_indices(n_active_occupied)

Return lexicographically ordered one-based occupied-index pairs `(i,k)` with `i < k` used by the planned second-higher-adjugate pair space.
The returned CPU vector owns only dimensionless index tuples, no scientific matrix is allocated, and dimensions outside `0:32` raise `ArgumentError`.
"""
function active_pair_indices(n_active_occupied::Integer)
    q = Int(n_active_occupied)
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("occupied rank must lie in 0:32"))
    pairs = Vector{NTuple{2,Int}}(undef, active_pair_count(q))
    position = 0
    for first in 1:(q - 1), second in (first + 1):q
        position += 1
        pairs[position] = (first, second)
    end
    return pairs
end
