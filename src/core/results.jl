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

 File:            src/core/results.jl
 Purpose:         Raw NOCI overlap and Hamiltonian component matrix containers before shifts or eigensolution.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    RawNOCIMatrices

Store symmetric raw NOCI overlap and Hamiltonian component matrices before normalization, correlation shifts, or generalized eigensolution.
Overlap is dimensionless, Hamiltonian components use the integral energy unit, labels and host matrices are owned copies, and no CUDA or solver state is retained.
"""
struct RawNOCIMatrices{T<:AbstractFloat}
    labels::Vector{String}
    overlap::Matrix{T}
    electronic_one_body::Matrix{T}
    nuclear_repulsion::Matrix{T}
    one_body_with_nuclear_repulsion::Matrix{T}
    two_body::Matrix{T}
    hamiltonian::Matrix{T}
    determinant_pair_counts::Matrix{Int}

    """
        RawNOCIMatrices(labels, overlap, electronic_one_body, nuclear_repulsion, one_body_with_nuclear_repulsion, two_body, hamiltonian, determinant_pair_counts)

    Construct validated raw NOCI matrices and copy every label and array into CPU-owned storage.
    Inputs are not mutated, matrix dimensions and finite/symmetric values are checked, and inconsistent component sums or negative pair counts raise an exception.
    """
    function RawNOCIMatrices(
        labels::AbstractVector{<:AbstractString},
        overlap::AbstractMatrix{T},
        electronic_one_body::AbstractMatrix{T},
        nuclear_repulsion::AbstractMatrix{T},
        one_body_with_nuclear_repulsion::AbstractMatrix{T},
        two_body::AbstractMatrix{T},
        hamiltonian::AbstractMatrix{T},
        determinant_pair_counts::AbstractMatrix{<:Integer},
    ) where {T<:AbstractFloat}
        owned_labels = String.(labels)
        n_basis_functions = length(owned_labels)
        length(unique(owned_labels)) == n_basis_functions || throw(ArgumentError("NOCI matrix labels must be unique"))
        all(!isempty, owned_labels) || throw(ArgumentError("NOCI matrix labels must not be empty"))
        expected_size = (n_basis_functions, n_basis_functions)
        matrices = (
            (:overlap, overlap),
            (:electronic_one_body, electronic_one_body),
            (:nuclear_repulsion, nuclear_repulsion),
            (:one_body_with_nuclear_repulsion, one_body_with_nuclear_repulsion),
            (:two_body, two_body),
            (:hamiltonian, hamiltonian),
        )
        for (name, matrix) in matrices
            size(matrix) == expected_size || throw(DimensionMismatch("$name matrix dimension differs from label count"))
            all(isfinite, matrix) || throw(ArgumentError("$name matrix contains a non-finite value"))
            issymmetric(matrix) || throw(ArgumentError("$name matrix must be symmetric"))
        end
        size(determinant_pair_counts) == expected_size || throw(DimensionMismatch("determinant_pair_counts dimension differs from label count"))
        all(>=(0), determinant_pair_counts) || throw(ArgumentError("determinant_pair_counts must be nonnegative"))
        one_body_with_nuclear_repulsion == electronic_one_body + nuclear_repulsion || throw(ArgumentError("one-body-plus-nuclear matrix is inconsistent"))
        hamiltonian == one_body_with_nuclear_repulsion + two_body || throw(ArgumentError("Hamiltonian matrix is inconsistent with its components"))
        return new{T}(
            owned_labels,
            Matrix{T}(overlap),
            Matrix{T}(electronic_one_body),
            Matrix{T}(nuclear_repulsion),
            Matrix{T}(one_body_with_nuclear_repulsion),
            Matrix{T}(two_body),
            Matrix{T}(hamiltonian),
            Matrix{Int}(determinant_pair_counts),
        )
    end
end

"""
    raw_noci_matrices_memory(T, n_basis_functions)

Estimate conservative peak host payload for six raw floating-point matrices and one integer pair-count matrix plus their validated constructor copies.
The basis-function count is dimensionless, device bytes are zero, no arrays are allocated, and negative dimensions or checked overflow raise an exception.
"""
function raw_noci_matrices_memory(::Type{T}, n_basis_functions::Integer) where {T<:AbstractFloat}
    n = Int(n_basis_functions)
    n >= 0 || throw(ArgumentError("n_basis_functions must be nonnegative"))
    matrix_elements = Base.Checked.checked_mul(n, n)
    floating_bytes = estimate_array_bytes(T, Base.Checked.checked_mul(12, matrix_elements))
    integer_bytes = estimate_array_bytes(Int, Base.Checked.checked_mul(2, matrix_elements))
    return MemoryEstimate(Base.Checked.checked_add(floating_bytes, integer_bytes), 0)
end
