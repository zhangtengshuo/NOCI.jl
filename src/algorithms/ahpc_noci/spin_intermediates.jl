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

 File:            src/algorithms/ahpc_noci/spin_intermediates.jl
 Purpose:         CPU worker workspace for occupied-spin overlaps, response maps, and union intermediates.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    SpinPairIntermediateWorkspace{T}()

Own one CPU worker's selected occupied-spin overlap, global response map,
union-local Hessian row/column pair IDs, and reusable polynomial cofactor buffers
for production union-support evaluation.
The workspace starts empty, is not shared between threads, contains no local
Hessian pair-response matrix, full `q^2 x q^2` same-spin response kernel,
ERI/subset-plan/CUDA storage, and grows only after memory preflight.
"""
mutable struct SpinPairIntermediateWorkspace{T<:AbstractFloat}
    cofactors::PolynomialContractedCofactorWorkspace{T}
    selected_overlap::Matrix{T}
    response_indices::Vector{Int}
    hessian_row_pair_indices::Vector{Int}
    hessian_column_pair_indices::Vector{Int}
end

"""
    SpinPairIntermediateWorkspace{T}()

Construct an empty production spin-pair workspace of floating-point type `T`.
No molecular-scale CPU array, subset recurrence state, device state, or external
input is allocated or mutated.
"""
function SpinPairIntermediateWorkspace{T}() where {T<:AbstractFloat}
    return SpinPairIntermediateWorkspace{T}(
        PolynomialContractedCofactorWorkspace{T}(),
        zeros(T, 0, 0),
        Int[],
        Int[],
        Int[],
    )
end

"""
    _spin_pair_support_shape(support, active_rank)

Return the maximum occupied-spin rank and exact structural first-cofactor entry
count over a canonical union support.
The support is read-only, counts are dimensionless, no arrays or device state
are allocated, and unequal left/right populations, invalid bits, or checked
overflow raise an exception.
"""
function _spin_pair_support_shape(support::AbstractVector{CanonicalSpinStringPair}, active_rank::Int)
    0 <= active_rank <= 64 || throw(ArgumentError("production active_rank must lie in 0:64"))
    valid_mask = active_rank == 64 ? typemax(UInt64) : (UInt64(1) << active_rank) - UInt64(1)
    maximum_occupied = 0
    feature_entries = 0
    for pair in support
        pair.left_bits & ~valid_mask == 0 || throw(ArgumentError("left spin string contains an orbital outside active_rank"))
        pair.right_bits & ~valid_mask == 0 || throw(ArgumentError("right spin string contains an orbital outside active_rank"))
        left_occupied = count_ones(pair.left_bits)
        right_occupied = count_ones(pair.right_bits)
        left_occupied == right_occupied || throw(DimensionMismatch("left/right canonical spin strings have different occupied counts"))
        maximum_occupied = max(maximum_occupied, left_occupied)
        feature_entries = Base.Checked.checked_add(feature_entries, Base.Checked.checked_mul(left_occupied, left_occupied))
    end
    return maximum_occupied, feature_entries
end

"""
    _resize_spin_pair_intermediate_workspace!(workspace, n_active_occupied, runtime)

Resize one worker's selected overlap, response map, local Hessian row/column pair
ID maps, and polynomial cofactor buffers for one occupied-spin rank and return
the workspace.
The CPU workspace is mutated, prior scientific contents may be discarded,
runtime active-rank limits are enforced before growth, only two
`binomial(q,2)` integer pair maps are retained instead of a
`binomial(q,2)^2` floating response matrix, no subset plan or CUDA work occurs,
and invalid ranks raise `ArgumentError`.
"""
function _resize_spin_pair_intermediate_workspace!(
    workspace::SpinPairIntermediateWorkspace{T},
    n_active_occupied::Int,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    0 <= n_active_occupied <= runtime.max_active_occupied || throw(ArgumentError("spin-string occupied count exceeds max_active_occupied"))
    n_local_response = Base.Checked.checked_mul(n_active_occupied, n_active_occupied)
    n_hessian_pairs = _hessian_pair_count(n_active_occupied)
    size(workspace.selected_overlap) == (n_active_occupied, n_active_occupied) ||
        (workspace.selected_overlap = zeros(T, n_active_occupied, n_active_occupied))
    length(workspace.response_indices) == n_local_response || resize!(workspace.response_indices, n_local_response)
    length(workspace.hessian_row_pair_indices) == n_hessian_pairs || resize!(workspace.hessian_row_pair_indices, n_hessian_pairs)
    length(workspace.hessian_column_pair_indices) == n_hessian_pairs || resize!(workspace.hessian_column_pair_indices, n_hessian_pairs)
    _resize_polynomial_contracted_workspace!(workspace.cofactors, n_active_occupied)
    return workspace
end
