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

 File:            src/core/couplings.jl
 Purpose:         Pairwise and full multistate Loewdin scalar electronic couplings over normalized matrices.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _normalized_coupling_matrices(hamiltonian, overlap)

Return diagonal-overlap-normalized Hamiltonian and overlap matrices for scalar electronic-coupling evaluation.
The inputs are read-only, Hamiltonian values retain their energy unit while overlap is dimensionless, two dense CPU matrices are allocated, and nonsquare, nonfinite, nonsymmetric, or nonpositive diagonal-overlap inputs raise an exception.
"""
function _normalized_coupling_matrices(
    hamiltonian::AbstractMatrix{T},
    overlap::AbstractMatrix{T},
) where {T<:AbstractFloat}
    n_states = size(hamiltonian, 1)
    size(hamiltonian, 2) == n_states || throw(DimensionMismatch("Hamiltonian must be square"))
    size(overlap) == (n_states, n_states) || throw(DimensionMismatch("Hamiltonian and overlap dimensions differ"))
    all(isfinite, hamiltonian) || throw(ArgumentError("Hamiltonian matrix must be finite"))
    all(isfinite, overlap) || throw(ArgumentError("overlap matrix must be finite"))
    issymmetric(hamiltonian) || throw(ArgumentError("Hamiltonian matrix must be symmetric"))
    issymmetric(overlap) || throw(ArgumentError("overlap matrix must be symmetric"))

    scales = Vector{T}(undef, n_states)
    for state in 1:n_states
        diagonal_overlap = overlap[state, state]
        diagonal_overlap > zero(T) || throw(ArgumentError("diagonal overlaps must be positive"))
        scales[state] = inv(sqrt(diagonal_overlap))
    end

    normalized_hamiltonian = Matrix{T}(undef, n_states, n_states)
    normalized_overlap = Matrix{T}(undef, n_states, n_states)
    @inbounds for right in 1:n_states, left in 1:right
        scale = scales[left] * scales[right]
        h_value = hamiltonian[left, right] * scale
        s_value = overlap[left, right] * scale
        normalized_hamiltonian[left, right] = h_value
        normalized_hamiltonian[right, left] = h_value
        normalized_overlap[left, right] = s_value
        normalized_overlap[right, left] = s_value
    end
    return normalized_hamiltonian, normalized_overlap
end

"""
    pairwise_lowdin_couplings(hamiltonian, overlap)

Return the symmetric matrix of two-state Löwdin effective electronic couplings with zero diagonal.
For each normalized state pair `(i,j)`, the coupling is `(H_ij - 0.5*(H_ii + H_jj)*S_ij) / (1 - S_ij^2)`; inputs are read-only, Hamiltonian energy units are preserved, and a nonpositive two-state Löwdin denominator raises `ArgumentError`.
"""
function pairwise_lowdin_couplings(
    hamiltonian::AbstractMatrix{T},
    overlap::AbstractMatrix{T},
) where {T<:AbstractFloat}
    normalized_hamiltonian, normalized_overlap = _normalized_coupling_matrices(hamiltonian, overlap)
    n_states = size(normalized_hamiltonian, 1)
    couplings = zeros(T, n_states, n_states)
    @inbounds for right in 2:n_states, left in 1:(right - 1)
        s_ij = normalized_overlap[left, right]
        denominator = one(T) - s_ij * s_ij
        denominator > zero(T) || throw(ArgumentError("pairwise Löwdin denominator must be positive for states $left and $right"))
        numerator = normalized_hamiltonian[left, right] -
            T(0.5) * (normalized_hamiltonian[left, left] + normalized_hamiltonian[right, right]) * s_ij
        value = numerator / denominator
        couplings[left, right] = value
        couplings[right, left] = value
    end
    return couplings
end

"""
    full_multistate_lowdin_couplings(hamiltonian, overlap)

Return the off-diagonal electronic couplings obtained by symmetric Löwdin orthogonalization of the complete supplied state space, with the diagonal set to zero.
Inputs are read-only, Hamiltonian energy units are preserved, dense CPU eigendecomposition/matrix products are used through `lowdin_orthogonalization`, and a non-positive-definite overlap raises an exception.
"""
function full_multistate_lowdin_couplings(
    hamiltonian::AbstractMatrix{T},
    overlap::AbstractMatrix{T},
) where {T<:AbstractFloat}
    normalized_hamiltonian, normalized_overlap = _normalized_coupling_matrices(hamiltonian, overlap)
    lowdin = lowdin_orthogonalization(normalized_overlap)
    orthogonal_hamiltonian = transpose(lowdin) * normalized_hamiltonian * lowdin
    couplings = Matrix{T}(Symmetric(orthogonal_hamiltonian))
    for state in axes(couplings, 1)
        couplings[state, state] = zero(T)
    end
    return couplings
end
