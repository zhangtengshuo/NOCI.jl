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

 File:            src/core/matrix_utils.jl
 Purpose:         Symmetric-matrix assignment and shared host matrix utilities.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _assign_symmetric!(matrix, left, right, value)

Assign one scalar to both symmetric positions `(left,right)` and `(right,left)` of a square host matrix and return the matrix.
The destination is mutated, units are inherited from `value`, no allocation or device work occurs, and invalid indices raise `BoundsError`.
"""
function _assign_symmetric!(matrix::AbstractMatrix{T}, left::Int, right::Int, value::T) where {T}
    checkbounds(matrix, left, right)
    checkbounds(matrix, right, left)
    matrix[left, right] = value
    matrix[right, left] = value
    return matrix
end

"""
    active_spin_occupied_counts(active_occupation)

Return the alpha and beta occupied-active counts represented by spatial occupation codes `-1`, `0`, `1`, and `2`.
The dimensionless occupation vector is read-only, no arrays or CUDA state are allocated, and unsupported codes raise `ArgumentError`.
"""
function active_spin_occupied_counts(active_occupation::AbstractVector{<:Integer})
    n_alpha = 0
    n_beta = 0
    for value in active_occupation
        value == 2 && (n_alpha += 1; n_beta += 1)
        value == 1 && (n_alpha += 1)
        value == -1 && (n_beta += 1)
        value in (-1, 0, 1, 2) || throw(ArgumentError("unsupported active occupation value $value"))
    end
    return n_alpha, n_beta
end

"""
    _mebf_active_spin_counts(basis_function)

Return the invariant occupied-active alpha/beta counts shared by every determinant in one MEBF.
The MEBF is read-only, occupation counts are dimensionless, no molecular-scale storage or CUDA work occurs, and inconsistent determinants raise `DimensionMismatch`.
"""
function _mebf_active_spin_counts(basis_function::MolecularBasisFunction)
    expansion = basis_function.expansion
    determinant_count(expansion) >= 1 || throw(ArgumentError("MEBF determinant expansion must not be empty"))
    expected = active_spin_occupied_counts(@view expansion.active_occupations[:, 1])
    for determinant in 2:determinant_count(expansion)
        actual = active_spin_occupied_counts(@view expansion.active_occupations[:, determinant])
        actual == expected || throw(DimensionMismatch("MEBF determinants do not share occupied-active alpha/beta counts"))
    end
    return expected
end
