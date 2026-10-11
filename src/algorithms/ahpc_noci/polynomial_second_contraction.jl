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

 File:            src/algorithms/ahpc_noci/polynomial_second_contraction.jl
 Purpose:         Memory estimates and direct contracted second-cofactor engine at occupied rank q.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    polynomial_contracted_cofactor_memory(T, n_active_occupied)

Return a conservative host numerical-payload estimate for the polynomial determinant/first-cofactor plus direct contracted-second engine at occupied rank `q`.
The estimate includes retained first cofactors, scaled/polynomial matrices, all powers through `A^(q-1)`, the batched trace-power kernel projection, and small response vectors while excluding the caller-owned response kernel; it reports zero device bytes and accepts `q` in `0:32`.
"""
function polynomial_contracted_cofactor_memory(
    ::Type{T},
    n_active_occupied::Integer,
) where {T<:AbstractFloat}
    q = Int(n_active_occupied)
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("polynomial contracted cofactor engine supports occupied rank 0:32"))
    square_matrices = estimate_array_bytes(T, 10, q, q)
    cubic_work = estimate_array_bytes(T, 2, q, q, q)
    response_vectors = estimate_array_bytes(T, 3, q, q)
    return MemoryEstimate(
        Base.Checked.checked_add(
            Base.Checked.checked_add(square_matrices, cubic_work),
            response_vectors,
        ),
        0,
    )
end

"""
    _elementary_symmetrics_scaled(matrix)

Compute elementary symmetric polynomials `e_0...e_q` of a finite scaled square matrix with the Faddeev-LeVerrier recurrence.
The input is read-only, the returned CPU vector is dimensionless relative to the scaled matrix, only `O(q^2)` scratch is allocated, divisions are by positive integer recurrence indices only, and no inverse/rank classifier/SVD/CUDA work occurs.
"""
function _elementary_symmetrics_scaled(
    matrix::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("elementary-symmetric input must be square"))
    all(isfinite, matrix) || throw(ArgumentError("elementary-symmetric input contains a non-finite value"))
    elementary = zeros(T, q + 1)
    elementary[1] = one(T)
    q == 0 && return elementary
    previous = Matrix{T}(I, q, q)
    next = zeros(T, q, q)
    @inbounds for order in 1:q
        mul!(next, matrix, previous)
        characteristic_coefficient = -tr(next) / T(order)
        elementary[order + 1] = isodd(order) ? -characteristic_coefficient : characteristic_coefficient
        order == q && break
        for diagonal in 1:q
            next[diagonal, diagonal] += characteristic_coefficient
        end
        previous, next = next, previous
    end
    return elementary
end

"""
    _matrix_powers_scaled(matrix)

Return CPU-owned matrix powers `[I, A, ..., A^(q-1)]` for one finite scaled `q x q` matrix.
The input is read-only, storage is `O(q^3)`, multiplication order is deterministic at the matrix-product level, no inverse/rank classification/SVD/CUDA work occurs, and nonsquare inputs raise `DimensionMismatch`.
"""
function _matrix_powers_scaled(
    matrix::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("matrix-power input must be square"))
    all(isfinite, matrix) || throw(ArgumentError("matrix-power input contains a non-finite value"))
    q == 0 && return Matrix{T}[]
    powers = Vector{Matrix{T}}(undef, q)
    powers[1] = Matrix{T}(I, q, q)
    for exponent in 1:(q - 1)
        powers[exponent + 1] = powers[exponent] * matrix
    end
    return powers
end

"""
    _newton_transformation!(destination, powers, elementary, order)

Overwrite `destination` with the Newton transformation `T_order = sum_{j=0}^order (-1)^j e_{order-j} A^j` for one scaled matrix polynomial basis.
Destination/powers/elementary inputs are CPU arrays, only `destination` is mutated, no allocation or CUDA work occurs, and invalid polynomial orders raise `BoundsError`.
"""
function _newton_transformation!(
    destination::Matrix{T},
    powers::Vector{Matrix{T}},
    elementary::Vector{T},
    order::Int,
) where {T<:AbstractFloat}
    q = size(destination, 1)
    size(destination, 2) == q || throw(DimensionMismatch("Newton transformation destination must be square"))
    0 <= order <= length(powers) - 1 || throw(BoundsError(0:(length(powers) - 1), order))
    fill!(destination, zero(T))
    @inbounds for power in 0:order
        coefficient = elementary[order - power + 1]
        isodd(power) && (coefficient = -coefficient)
        source = powers[power + 1]
        for column in 1:q, row in 1:q
            destination[row, column] += coefficient * source[row, column]
        end
    end
    return destination
end

"""
    _trace_basis_kernel_bilinear(response_kernel, left_matrix, right_matrix)

Contract a response-space kernel with `tr(left_matrix E_a) * tr(right_matrix E_b)` over column-major matrix-element basis directions `E_(i,j)`.
Inputs are finite read-only CPU arrays, accumulation follows fixed response-column then response-row order, no allocation/CUDA work occurs, and shape mismatches raise `DimensionMismatch`.
"""
function _trace_basis_kernel_bilinear(
    response_kernel::AbstractMatrix{T},
    left_matrix::AbstractMatrix{T},
    right_matrix::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = size(left_matrix, 1)
    size(left_matrix, 2) == q && size(right_matrix) == (q, q) || throw(DimensionMismatch("trace-basis matrices must be equal square matrices"))
    n_response = q * q
    size(response_kernel) == (n_response, n_response) || throw(DimensionMismatch("response kernel must have size (q^2,q^2)"))
    value = zero(T)
    @inbounds for second_column in 1:q, second_row in 1:q
        second_response = second_row + (second_column - 1) * q
        right_trace = right_matrix[second_column, second_row]
        for first_column in 1:q, first_row in 1:q
            first_response = first_row + (first_column - 1) * q
            left_trace = left_matrix[first_column, first_row]
            value += response_kernel[first_response, second_response] * left_trace * right_trace
        end
    end
    return value
end

"""
    _trace_power_kernel_contraction(response_kernel, left_power, right_power)

Contract a response-space kernel with the bilinear trace `tr(left_power E_b right_power E_a)` required by the Fréchet derivative of a matrix power.
Inputs are finite read-only CPU arrays, the four active indices are traversed in fixed order, no Hessian or temporary tensor is allocated, no CUDA work occurs, and incompatible dimensions raise `DimensionMismatch`.
This scalar form is retained as a small-rank characterization primitive; the production polynomial contraction batches all right powers instead of calling it for every split.
"""
function _trace_power_kernel_contraction(
    response_kernel::AbstractMatrix{T},
    left_power::AbstractMatrix{T},
    right_power::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = size(left_power, 1)
    size(left_power, 2) == q && size(right_power) == (q, q) || throw(DimensionMismatch("power matrices must be equal square matrices"))
    n_response = q * q
    size(response_kernel) == (n_response, n_response) || throw(DimensionMismatch("response kernel must have size (q^2,q^2)"))
    value = zero(T)
    @inbounds for second_column in 1:q, second_row in 1:q
        second_response = second_row + (second_column - 1) * q
        for first_column in 1:q, first_row in 1:q
            first_response = first_row + (first_column - 1) * q
            value += response_kernel[first_response, second_response] *
                     left_power[first_column, second_row] *
                     right_power[second_column, first_row]
        end
    end
    return value
end

"""
    _project_trace_power_kernel!(projection, response_kernel, powers, n_powers)

Project the dense response kernel against right matrix powers `A^0...A^(n_powers-1)` once, storing `Y_t[j,k] = sum_{i,l} K[(i,j),(k,l)] A^t[l,i]` in column-major `(j,k)` rows.
The destination is overwritten, kernel/powers are read-only CPU arrays, storage is `O(q^3)`, arithmetic is `O(n_powers*q^4)`, no four-index temporary/Hessian/CUDA state is formed, and invalid dimensions raise before traversal.
"""
function _project_trace_power_kernel!(
    projection::Matrix{T},
    response_kernel::AbstractMatrix{T},
    powers::Vector{Matrix{T}},
    n_powers::Int,
) where {T<:AbstractFloat}
    isempty(powers) && n_powers == 0 && return projection
    q = size(first(powers), 1)
    size(first(powers), 2) == q || throw(DimensionMismatch("matrix powers must be square"))
    0 <= n_powers <= length(powers) || throw(BoundsError(powers, n_powers))
    n_response = q * q
    size(response_kernel) == (n_response, n_response) || throw(DimensionMismatch("response kernel must have size (q^2,q^2)"))
    size(projection) == (n_response, n_powers) || throw(DimensionMismatch("trace-power projection has incorrect dimensions"))
    fill!(projection, zero(T))
    @inbounds for right_exponent in 0:(n_powers - 1)
        right_power = powers[right_exponent + 1]
        for second_row in 1:q, first_column in 1:q
            value = zero(T)
            for second_column in 1:q, first_row in 1:q
                first_response = first_row + (first_column - 1) * q
                second_response = second_row + (second_column - 1) * q
                value += response_kernel[first_response, second_response] * right_power[second_column, first_row]
            end
            projected_response = first_column + (second_row - 1) * q
            projection[projected_response, right_exponent + 1] = value
        end
    end
    return projection
end

"""
    _contracted_second_determinant_scaled(matrix, response_kernel)

Evaluate `sum_{a,b} K[a,b] * d^2 det(A)/(dA_a dA_b)` for one finite scaled matrix by differentiating the Newton/Faddeev adjugate polynomial and batching the matrix-power Fréchet contractions.
The matrix and kernel are read-only CPU inputs, only `O(q^3)` transient power/projection storage is retained beyond the caller-owned `O(q^4)` kernel, no `q^2 x q^2` Hessian is formed, arithmetic is `O(q^5)`, and no inverse/rank classifier/SVD/fallback/CUDA work occurs; singular matrices follow exactly the same polynomial path.
"""
function _contracted_second_determinant_scaled(
    matrix::AbstractMatrix{T},
    response_kernel::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("contracted-second matrix must be square"))
    n_response = q * q
    size(response_kernel) == (n_response, n_response) || throw(DimensionMismatch("response kernel must have size (q^2,q^2)"))
    all(isfinite, matrix) || throw(ArgumentError("contracted-second matrix contains a non-finite value"))
    all(isfinite, response_kernel) || throw(ArgumentError("response kernel contains a non-finite value"))
    q < 2 && return zero(T)

    elementary = _elementary_symmetrics_scaled(matrix)
    powers = _matrix_powers_scaled(matrix)
    newton = zeros(T, q, q)
    contracted = zero(T)

    # Derivatives of the elementary-symmetric coefficients.  There are O(q)
    # dense q^4 kernel contractions, so this part is O(q^5).
    for power in 0:(q - 1)
        elementary_order = q - 1 - power
        sign = isodd(power) ? -one(T) : one(T)
        if elementary_order > 0
            _newton_transformation!(newton, powers, elementary, elementary_order - 1)
            contracted += sign * _trace_basis_kernel_bilinear(
                response_kernel,
                powers[power + 1],
                newton,
            )
        end
    end

    # Matrix-power Fréchet terms originally require O(q^2) split contractions,
    # each costing O(q^4).  Project K against every needed right power once:
    #   Y_t[j,k] = sum_{i,l} K[(i,j),(k,l)] (A^t)[l,i].
    # Then C(A^s,A^t) is only dot(A^s, Y_t), reducing this entire part to
    # O(q^5) projection + O(q^4) triangular dot products.
    n_projected_powers = q - 1
    projection = Matrix{T}(undef, n_response, n_projected_powers)
    _project_trace_power_kernel!(projection, response_kernel, powers, n_projected_powers)
    @inbounds for total_degree in 0:(q - 2)
        power = total_degree + 1
        elementary_order = q - 1 - power
        sign = isodd(power) ? -one(T) : one(T)
        coefficient = sign * elementary[elementary_order + 1]
        iszero(coefficient) && continue
        for left_exponent in 0:total_degree
            right_exponent = total_degree - left_exponent
            left_power = powers[left_exponent + 1]
            split_value = zero(T)
            for second_row in 1:q, first_column in 1:q
                projected_response = first_column + (second_row - 1) * q
                split_value += left_power[first_column, second_row] * projection[projected_response, right_exponent + 1]
            end
            contracted += coefficient * split_value
        end
    end
    return contracted
end

"""
    polynomial_contracted_complementary_cofactors(matrix, response_kernel; runtime=RuntimeOptions())

Compute determinant, complete first complementary cofactors, and direct response-kernel-contracted second determinant derivative with one singular-safe polynomial AHPC engine.
Inputs are read-only, outputs are CPU-owned, exact power-of-two scaling matches the established recurrence convention, memory is preflighted, the contracted-second arithmetic is `O(q^5)` with `O(q^3)` transient storage beyond the caller-owned kernel, and no inverse/rank classifier/SVD/fallback/CUDA work occurs.
"""
function polynomial_contracted_complementary_cofactors(
    matrix::AbstractMatrix{T},
    response_kernel::AbstractMatrix{T};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("polynomial contracted cofactor input must be square"))
    n_response = q * q
    size(response_kernel) == (n_response, n_response) || throw(DimensionMismatch("response kernel must have size (q^2,q^2)"))
    all(isfinite, matrix) || throw(ArgumentError("polynomial contracted cofactor input contains a non-finite value"))
    all(isfinite, response_kernel) || throw(ArgumentError("response kernel contains a non-finite value"))
    enforce_memory_limits(polynomial_contracted_cofactor_memory(T, q), runtime)

    first_result = polynomial_first_cofactors(matrix; runtime)
    q < 2 && return ComplementaryCofactorResult(
        first_result.cofactor_scalar,
        first_result.first_cofactor,
        zero(T),
        first_result.binary_scaling_exponent,
    )

    exponent = first_result.binary_scaling_exponent
    scaled = Matrix{T}(undef, q, q)
    @inbounds for column in 1:q, row in 1:q
        scaled[row, column] = ldexp(matrix[row, column], -exponent)
    end
    second_scaled = _contracted_second_determinant_scaled(scaled, response_kernel)
    contracted_second = ldexp(second_scaled, exponent * max(q - 2, 0))
    isfinite(contracted_second) || throw(ArgumentError("polynomial contracted second cofactor is non-finite"))
    return ComplementaryCofactorResult(
        first_result.cofactor_scalar,
        first_result.first_cofactor,
        contracted_second,
        exponent,
    )
end
