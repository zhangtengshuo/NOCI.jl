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

 File:            src/algorithms/ahpc_noci/polynomial_workspace.jl
 Purpose:         One CPU worker's reusable polynomial determinant, first-cofactor, and contracted second buffers.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    PolynomialContractedCofactorWorkspace{T}()

Own one CPU worker's reusable rank-uniform polynomial determinant, first-cofactor, and directly contracted second-cofactor buffers.
The workspace starts empty, grows only through `_resize_polynomial_contracted_workspace!`, retains `O(q^3)` storage beyond the caller-owned response kernel, contains no subset plan or CUDA state, and is never shared between Julia workers.
"""
mutable struct PolynomialContractedCofactorWorkspace{T<:AbstractFloat}
    q::Int
    scaled::Matrix{T}
    previous::Matrix{T}
    next::Matrix{T}
    elementary::Vector{T}
    powers::Vector{Matrix{T}}
    newton::Matrix{T}
    projection::Matrix{T}
    first_cofactor::Matrix{T}
    lu_pivots::Vector{LinearAlgebra.BlasInt}
end

function PolynomialContractedCofactorWorkspace{T}() where {T<:AbstractFloat}
    return PolynomialContractedCofactorWorkspace{T}(
        -1,
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        T[],
        Matrix{T}[],
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        LinearAlgebra.BlasInt[],
    )
end

"""
    polynomial_workspace_memory(T, n_active_occupied)

Return the exact retained numerical payload for one reusable production polynomial cofactor workspace at occupied-spin rank `q`.
The estimate counts five dense `q x q` matrices, `q` matrix powers, one `q^2 x max(q-1,0)` kernel projection, `q+1` elementary-symmetric scalars, and one preallocated LAPACK pivot vector; it reports zero device bytes and accepts `q` in `0:32`.
"""
function polynomial_workspace_memory(
    ::Type{T},
    n_active_occupied::Integer,
) where {T<:AbstractFloat}
    q = Int(n_active_occupied)
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("production polynomial workspace supports occupied rank 0:32"))
    square_bytes = estimate_array_bytes(T, 5, q, q)
    power_bytes = estimate_array_bytes(T, q, q, q)
    projection_bytes = estimate_array_bytes(T, q * q, max(q - 1, 0))
    elementary_bytes = estimate_array_bytes(T, q + 1)
    pivot_bytes = estimate_array_bytes(LinearAlgebra.BlasInt, q)
    return MemoryEstimate(
        Base.Checked.checked_add(
            Base.Checked.checked_add(square_bytes, power_bytes),
            Base.Checked.checked_add(Base.Checked.checked_add(projection_bytes, elementary_bytes), pivot_bytes),
        ),
        0,
    )
end

"""
    _resize_polynomial_contracted_workspace!(workspace, q)

Resize one worker-owned production polynomial workspace to occupied-spin rank `q` and return it.
Existing buffers are retained unchanged when the rank already matches; otherwise all CPU buffers are replaced together after validating `q in 0:32`, no subset state or CUDA allocation is created, and prior scientific contents are discarded.
"""
function _resize_polynomial_contracted_workspace!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    q::Int,
) where {T<:AbstractFloat}
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("production polynomial workspace supports occupied rank 0:32"))
    workspace.q == q && return workspace
    workspace.q = q
    workspace.scaled = zeros(T, q, q)
    workspace.previous = zeros(T, q, q)
    workspace.next = zeros(T, q, q)
    workspace.elementary = zeros(T, q + 1)
    workspace.powers = [zeros(T, q, q) for _ in 1:q]
    workspace.newton = zeros(T, q, q)
    workspace.projection = zeros(T, q * q, max(q - 1, 0))
    workspace.first_cofactor = zeros(T, q, q)
    workspace.lu_pivots = zeros(LinearAlgebra.BlasInt, q)
    return workspace
end

"""
    _scale_polynomial_input!(workspace, matrix)

Copy one finite square overlap matrix into the worker workspace using the exact common binary scaling `A_scaled = 2^-e A` and return `e`.
Only `workspace.scaled` is overwritten, the input is read-only, no allocation or CUDA work occurs, and matrix/workspace rank mismatches or non-finite values raise before polynomial evaluation.
"""
function _scale_polynomial_input!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    matrix::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = workspace.q
    size(matrix) == (q, q) || throw(DimensionMismatch("polynomial workspace/input rank mismatch"))
    maximum_absolute = zero(T)
    @inbounds for value in matrix
        isfinite(value) || throw(ArgumentError("polynomial cofactor input contains a non-finite value"))
        maximum_absolute = max(maximum_absolute, abs(value))
    end
    exponent = maximum_absolute == zero(T) ? 0 : last(frexp(maximum_absolute))
    @inbounds for column in 1:q, row in 1:q
        workspace.scaled[row, column] = ldexp(matrix[row, column], -exponent)
    end
    return exponent
end

"""
    _elementary_symmetrics_scaled!(workspace)

Overwrite the worker's elementary-symmetric vector for its already scaled matrix using the rank-uniform Faddeev-LeVerrier polynomial recurrence.
Only worker-owned CPU buffers are mutated, arithmetic uses `O(q^4)` work and `O(q^2)` scratch, divisions are by positive integer recurrence orders only, no inverse/rank classifier/SVD/subset fallback/CUDA work occurs, and singular matrices follow the same path.
"""
function _elementary_symmetrics_scaled!(workspace::PolynomialContractedCofactorWorkspace{T}) where {T<:AbstractFloat}
    q = workspace.q
    fill!(workspace.elementary, zero(T))
    workspace.elementary[1] = one(T)
    q == 0 && return workspace.elementary

    fill!(workspace.previous, zero(T))
    @inbounds for diagonal in 1:q
        workspace.previous[diagonal, diagonal] = one(T)
    end
    fill!(workspace.next, zero(T))

    @inbounds for order in 1:q
        if q < 32 && order == 1
            copyto!(workspace.next, workspace.scaled)
        elseif q < 32 && order == q
            for diagonal in 1:q
                diagonal_product = zero(T)
                for occupied in 1:q
                    diagonal_product += workspace.scaled[diagonal, occupied] * workspace.previous[occupied, diagonal]
                end
                workspace.next[diagonal, diagonal] = diagonal_product
            end
        else
            _polynomial_matrix_product!(workspace.next, workspace.scaled, workspace.previous)
        end
        characteristic_coefficient = -tr(workspace.next) / T(order)
        workspace.elementary[order + 1] = isodd(order) ? -characteristic_coefficient : characteristic_coefficient
        order == q && break
        for diagonal in 1:q
            workspace.next[diagonal, diagonal] += characteristic_coefficient
        end
        workspace.previous, workspace.next = workspace.next, workspace.previous
    end
    return workspace.elementary
end

"""
    _matrix_powers_scaled!(workspace)

Overwrite worker-owned powers with `[I,A,...,A^(q-1)]` for the already scaled matrix.
Only the workspace is mutated, multiplication order is deterministic at the dense-matrix level, storage is `O(q^3)`, no allocation/subset state/CUDA work occurs after sizing, and `q=0` is valid.
"""
function _matrix_powers_scaled!(workspace::PolynomialContractedCofactorWorkspace{T}) where {T<:AbstractFloat}
    q = workspace.q
    q == 0 && return workspace.powers
    fill!(workspace.powers[1], zero(T))
    @inbounds for diagonal in 1:q
        workspace.powers[1][diagonal, diagonal] = one(T)
    end
    @inbounds for exponent in 1:(q - 1)
        if q < 32 && exponent == 1
            copyto!(workspace.powers[2], workspace.scaled)
        else
            _polynomial_matrix_product!(workspace.powers[exponent + 1], workspace.powers[exponent], workspace.scaled)
        end
    end
    return workspace.powers
end

"""
    _newton_transformation_workspace!(workspace, order)

Overwrite the worker's Newton-transformation matrix `T_order = sum_j (-1)^j e_(order-j) A^j` from prepared powers and elementary symmetrics.
Only `workspace.newton` is mutated, no allocation or CUDA work occurs, and polynomial orders outside `0:q-1` raise `BoundsError`.
"""
function _newton_transformation_workspace!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    order::Int,
) where {T<:AbstractFloat}
    q = workspace.q
    0 <= order <= q - 1 || throw(BoundsError(0:(q - 1), order))
    fill!(workspace.newton, zero(T))
    @inbounds for power in 0:order
        coefficient = workspace.elementary[order - power + 1]
        isodd(power) && (coefficient = -coefficient)
        source = workspace.powers[power + 1]
        for column in 1:q, row in 1:q
            workspace.newton[row, column] += coefficient * source[row, column]
        end
    end
    return workspace.newton
end

"""
    _project_trace_power_kernel_workspace!(workspace, response_kernel)

Project the dense response kernel once against every required right power `A^0...A^(q-2)` into the worker-owned `q^2 x (q-1)` buffer.
The projection is overwritten, kernel and powers are read-only, arithmetic is `O(q^5)`, no four-index temporary/full Hessian/subset state/CUDA work is formed, and shape or non-finite kernel errors raise before use.
"""
function _project_trace_power_kernel_workspace!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    response_kernel::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = workspace.q
    n_response = q * q
    size(response_kernel) == (n_response, n_response) || throw(DimensionMismatch("response kernel must have size (q^2,q^2)"))
    all(isfinite, response_kernel) || throw(ArgumentError("response kernel contains a non-finite value"))
    fill!(workspace.projection, zero(T))
    q < 2 && return workspace.projection
    @inbounds for right_exponent in 0:(q - 2)
        right_power = workspace.powers[right_exponent + 1]
        for second_row in 1:q, first_column in 1:q
            value = zero(T)
            for second_column in 1:q, first_row in 1:q
                first_response = first_row + (first_column - 1) * q
                second_response = second_row + (second_column - 1) * q
                value += response_kernel[first_response, second_response] * right_power[second_column, first_row]
            end
            projected_response = first_column + (second_row - 1) * q
            workspace.projection[projected_response, right_exponent + 1] = value
        end
    end
    return workspace.projection
end

"""
    _trace_basis_kernel_bilinear_workspace(response_kernel, left_matrix, right_matrix)

Return the dense response-kernel contraction with `tr(left_matrix E_a) tr(right_matrix E_b)` using deterministic column-major response traversal.
All CPU inputs are read-only, no allocation or CUDA work occurs, arithmetic is `O(q^4)`, and incompatible dimensions raise `DimensionMismatch`.
"""
function _trace_basis_kernel_bilinear_workspace(
    response_kernel::AbstractMatrix{T},
    left_matrix::AbstractMatrix{T},
    right_matrix::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = size(left_matrix, 1)
    size(left_matrix, 2) == q && size(right_matrix) == (q, q) || throw(DimensionMismatch("trace-basis matrices must be equal square matrices"))
    size(response_kernel) == (q * q, q * q) || throw(DimensionMismatch("response kernel must have size (q^2,q^2)"))
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
    _contracted_second_from_polynomial_basis!(workspace, response_kernel)

Return the direct kernel-contracted second determinant derivative from already prepared elementary symmetrics and matrix powers.
Worker projection/Newton buffers are reused in place, transient storage is `O(q^3)` beyond the caller-owned kernel, arithmetic is `O(q^5)`, no `q^2 x q^2` Hessian/inverse/rank classifier/SVD/subset fallback/CUDA work is used, and singular matrices take the identical circuit.
"""
function _contracted_second_from_polynomial_basis!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    response_kernel::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = workspace.q
    q < 2 && return zero(T)
    contracted = zero(T)

    @inbounds for power in 0:(q - 1)
        elementary_order = q - 1 - power
        sign = isodd(power) ? -one(T) : one(T)
        if elementary_order > 0
            _newton_transformation_workspace!(workspace, elementary_order - 1)
            contracted += sign * _trace_basis_kernel_bilinear_workspace(
                response_kernel,
                workspace.powers[power + 1],
                workspace.newton,
            )
        end
    end

    _project_trace_power_kernel_workspace!(workspace, response_kernel)
    @inbounds for total_degree in 0:(q - 2)
        power = total_degree + 1
        elementary_order = q - 1 - power
        sign = isodd(power) ? -one(T) : one(T)
        coefficient = sign * workspace.elementary[elementary_order + 1]
        iszero(coefficient) && continue
        for left_exponent in 0:total_degree
            right_exponent = total_degree - left_exponent
            left_power = workspace.powers[left_exponent + 1]
            split_value = zero(T)
            for second_row in 1:q, first_column in 1:q
                projected_response = first_column + (second_row - 1) * q
                split_value += left_power[first_column, second_row] * workspace.projection[projected_response, right_exponent + 1]
            end
            contracted += coefficient * split_value
        end
    end
    return contracted
end

"""
    _polynomial_contracted_complementary_cofactors!(workspace, matrix, response_kernel)

Compute determinant, every first complementary cofactor, and the direct response-kernel-contracted second determinant derivative in one reusable production polynomial workspace.
The matrix/kernel are read-only, the returned first-cofactor matrix aliases worker storage until the next workspace call, exact power-of-two scaling is applied uniformly, arithmetic is rank-uniform and `O(q^5)`, and no inverse/rank classifier/SVD/subset recurrence/fallback/CUDA work occurs.
"""
function _polynomial_contracted_complementary_cofactors!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    matrix::AbstractMatrix{T},
    response_kernel::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("polynomial cofactor input must be square"))
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("production polynomial cofactor engine supports occupied rank 0:32"))
    size(response_kernel) == (q * q, q * q) || throw(DimensionMismatch("response kernel must have size (q^2,q^2)"))
    _resize_polynomial_contracted_workspace!(workspace, q)

    exponent = _scale_polynomial_input!(workspace, matrix)
    _elementary_symmetrics_scaled!(workspace)
    _matrix_powers_scaled!(workspace)

    determinant_scaled = workspace.elementary[q + 1]
    determinant = ldexp(determinant_scaled, exponent * q)
    first_exponent = exponent * max(q - 1, 0)
    if q > 0
        _newton_transformation_workspace!(workspace, q - 1)
        @inbounds for column in 1:q, row in 1:q
            workspace.first_cofactor[row, column] = ldexp(workspace.newton[column, row], first_exponent)
        end
    end

    second_scaled = _contracted_second_from_polynomial_basis!(workspace, response_kernel)
    contracted_second = ldexp(second_scaled, exponent * max(q - 2, 0))
    isfinite(determinant) || throw(ArgumentError("polynomial determinant is non-finite"))
    all(isfinite, workspace.first_cofactor) || throw(ArgumentError("polynomial first cofactors contain a non-finite value"))
    isfinite(contracted_second) || throw(ArgumentError("polynomial contracted second cofactor is non-finite"))
    return ComplementaryCofactorResult(
        determinant,
        workspace.first_cofactor,
        contracted_second,
        exponent,
    )
end

"""
    _polynomial_matrix_product!(output, left, right)

Overwrite one worker-owned square matrix with the polynomial recurrence product and return it in dimensionless overlap units.
For occupied ranks through 20, the CPU kernel avoids shared BLAS dispatch on very small matrices; ranks 21 through 32 retain the dense BLAS implementation, inputs remain read-only, output must not alias either input, and callers validate matching square dimensions before union traversal.
"""
@inline function _polynomial_matrix_product!(output::Matrix{T}, left::Matrix{T}, right::Matrix{T}) where {T<:AbstractFloat}
    if size(output, 1) <= 20
        return _ordinary_polynomial_matrix_product!(output, left, right)
    end
    return mul!(output, left, right)
end

