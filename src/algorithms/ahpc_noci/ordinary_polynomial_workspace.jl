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

 File:            src/algorithms/ahpc_noci/ordinary_polynomial_workspace.jl
 Purpose:         Allocation-free CPU matrix products for ordinary occupied ranks up to 20.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _ordinary_polynomial_matrix_product!(output, left, right)

Multiply validated square overlap matrices for ordinary occupied ranks up to 20 with allocation-free Julia arithmetic, returning and mutating only `output`.
Inputs use dimensionless overlap units and must not alias the output; a SIMD reduction over the occupied index avoids tiny BLAS calls and their interworker synchronization without inverse, numerical-rank classification, GPU work, or a change to the polynomial formula.
"""
function _ordinary_polynomial_matrix_product!(output::Matrix{T}, left::Matrix{T}, right::Matrix{T}) where {T<:AbstractFloat}
    q = size(output, 1)
    @inbounds for column in 1:q, row in 1:q
        value = zero(T)
        @simd for occupied in 1:q
            value += left[row, occupied] * right[occupied, column]
        end
        output[row, column] = value
    end
    return output
end
