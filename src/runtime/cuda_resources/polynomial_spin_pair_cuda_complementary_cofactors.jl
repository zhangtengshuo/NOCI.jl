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

 File:            src/algorithms/ahpc_noci/polynomial_spin_pair_cuda_complementary_cofactors.jl
 Purpose:         Shared direct-power terminal first complementary cofactor and occupied-orbital-pair-space second complementary cofactor evaluation.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

const AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_UNION = 8
const AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_BLOCK = 128
const AHPC_SECOND_COMPLEMENTARY_COFACTOR_MATRIX_STAGE_BUDGET_BYTES = 40 * 1024
const AHPC_MAX_OCCUPIED_ORBITAL_PAIRS = 496 # binomial(32, 2)

"""
    _second_complementary_cofactor_blocked_stage_groups(T, n_active_occupied)

Return the number of dense grouped-power matrix pairs staged together by the
blocked shared-memory second-cofactor CUDA kernel. The fixed 40 KiB matrix
budget is divided by two `q`-by-`q` matrices of element type `T`; the result is
dimensionless, inputs are not mutated, and no CUDA state or memory is touched.
Supported element types are Float32 and Float64 and occupied ranks must lie in
0:32; invalid values throw `ArgumentError` before device work.
"""
function _second_complementary_cofactor_blocked_stage_groups(
    ::Type{T},
    n_active_occupied::Integer,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    0 <= occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA occupied count must lie in 0:32"))
    highest_grouped_power = max(0, (occupied - 2) ÷ 2)
    highest_grouped_power == 0 && return 0
    bytes_per_group = Base.Checked.checked_mul(2 * occupied * occupied, sizeof(T))
    return max(
        1,
        min(
            highest_grouped_power,
            fld(
                AHPC_SECOND_COMPLEMENTARY_COFACTOR_MATRIX_STAGE_BUDGET_BYTES,
                bytes_per_group,
            ),
        ),
    )
end

"""
    _second_complementary_cofactor_blocked_shared_bytes(T, n_active_occupied)

Return dynamic shared-memory bytes for the staged grouped-power matrices and
the 128-thread block reduction in the production second-cofactor CUDA kernel.
The value is a device-allocation size in bytes; inputs are read-only, no memory
is allocated, and invalid element types or occupied ranks are rejected by the
stage-group resource rule.
"""
function _second_complementary_cofactor_blocked_shared_bytes(
    ::Type{T},
    n_active_occupied::Integer,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    stage_groups = _second_complementary_cofactor_blocked_stage_groups(T, occupied)
    matrix_groups = max(stage_groups, 1)
    entries = Base.Checked.checked_add(
        Base.Checked.checked_mul(2 * matrix_groups, occupied * occupied),
        AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_BLOCK,
    )
    return Base.Checked.checked_mul(entries, sizeof(T))
end

"""
    _occupied_orbital_pair_from_index(pair_index, n_active_occupied)

Return the one-based lexicographic occupied-orbital pair associated with one
index in the unordered occupied-orbital-pair space.
"""
@inline function _occupied_orbital_pair_from_index(
    pair_index::Int,
    n_active_occupied::Int,
)
    first_occupied = 1
    remaining = pair_index
    n_pairs_with_first = n_active_occupied - 1
    while remaining > n_pairs_with_first
        remaining -= n_pairs_with_first
        first_occupied += 1
        n_pairs_with_first -= 1
    end
    return first_occupied, first_occupied + remaining
end


"""
    _overlapping_occupied_orbital_pair_from_index(overlap_index, first_column, second_column, n_active_occupied)

Return the lexicographically ordered occupied-orbital pair at position
`overlap_index` among the `2n-3` pairs sharing at least one occupied index
with the column pair `(first_column, second_column)`. The returned third value
is the corresponding index in the complete unordered occupied-orbital-pair
space. This is the exact structural support of the endpoint second-cofactor
terms containing an identity matrix.
"""
@inline function _overlapping_occupied_orbital_pair_from_index(
    overlap_index::Int,
    first_column::Int,
    second_column::Int,
    n_active_occupied::Int,
)
    first_region = 2 * (first_column - 1)
    second_region = n_active_occupied - first_column
    third_region = second_column - first_column - 1

    first_row = 0
    second_row = 0
    local_index = overlap_index

    if local_index <= first_region
        first_row = (local_index + 1) ÷ 2
        second_row = isodd(local_index) ? first_column : second_column
    else
        local_index -= first_region
        if local_index <= second_region
            first_row = first_column
            second_row = first_column + local_index
        else
            local_index -= second_region
            if local_index <= third_region
                first_row = first_column + local_index
                second_row = second_column
            else
                local_index -= third_region
                first_row = second_column
                second_row = second_column + local_index
            end
        end
    end

    preceding_pairs =
        (first_row - 1) * n_active_occupied -
        ((first_row - 1) * first_row) ÷ 2
    row_pair_index =
        preceding_pairs + (second_row - first_row)
    return first_row, second_row, row_pair_index
end

"""
    _second_complementary_cofactor_identity_pair_term(...)

Evaluate one endpoint occupied-orbital-pair contribution in the common
left-identity orientation. `factor_matrix` stores the nonidentity factor and
`factor_offset` is its zero-based leading-dimension offset, so the same
formula serves both the power-zero Newton endpoint and the highest-power
matrix-power endpoint. Only the exact overlapping-pair support is intended to
call this routine; terms multiplied by zero identity elements are omitted
analytically. Inputs are read-only GPU-compatible arrays, the return value is
one scalar response-weighted wedge contribution, and no allocation or
synchronization is performed.
"""
@inline function _second_complementary_cofactor_identity_pair_term(
    factor_matrix,
    hessian_row_pair_indices,
    state_pair_hessian_response_device,
    union_id::Int,
    factor_offset::Int,
    first_column::Int,
    second_column::Int,
    first_row::Int,
    second_row::Int,
    row_pair_index::Int,
    state_pair_column_pair::Int,
    n_active_occupied::Int,
)
    wedge_bilinear = zero(eltype(factor_matrix))

    if first_column == first_row
        wedge_bilinear += @inbounds factor_matrix[
            factor_offset +
            second_column + (second_row - 1) * n_active_occupied,
            union_id,
        ]
    end
    if second_column == first_row
        wedge_bilinear -= @inbounds factor_matrix[
            factor_offset +
            first_column + (second_row - 1) * n_active_occupied,
            union_id,
        ]
    end
    if first_column == second_row
        wedge_bilinear -= @inbounds factor_matrix[
            factor_offset +
            second_column + (first_row - 1) * n_active_occupied,
            union_id,
        ]
    end
    if second_column == second_row
        wedge_bilinear += @inbounds factor_matrix[
            factor_offset +
            first_column + (first_row - 1) * n_active_occupied,
            union_id,
        ]
    end

    state_pair_row_pair =
        Int(@inbounds hessian_row_pair_indices[row_pair_index, union_id])
    return @inbounds state_pair_hessian_response_device[
        state_pair_row_pair,
        state_pair_column_pair,
    ] * wedge_bilinear
end
