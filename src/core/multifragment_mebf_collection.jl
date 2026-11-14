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

 File:            src/core/multifragment_mebf_collection.jl
 Purpose:         Collection-wide CI pre-pruning and normalization of assembled multifragment MEBF collections.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _reduce_multifragment_basis_collection(basis_functions, tau_ci)

Apply collection-wide CI pre-pruning to normalized multifragment MEBFs, preserving their order.
With `Cmax = maximum(abs(C))`, retain coefficients satisfying the strict condition
`abs(C_i) > tau_ci/Cmax`. Any removed coefficient cannot satisfy the downstream
pair criterion `abs(C_i*C_j) > tau_ci` with the same collection.
Retained coefficients are not renormalized, preserving determinant-pair weights.
Empty collections or complete screening raise `ArgumentError`.
"""
function _reduce_multifragment_basis_collection(
    basis_functions::AbstractVector{MolecularBasisFunction{T}},
    tau_ci::T,
) where {T<:AbstractFloat}
    isempty(basis_functions) && throw(ArgumentError("MEBF collection must not be empty"))
    isfinite(tau_ci) && tau_ci >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    maximum_coefficient = maximum(maximum(abs, basis.expansion.coefficients) for basis in basis_functions)
    maximum_coefficient > zero(T) || throw(ArgumentError("MEBF collection has zero coefficient norm"))

    # Pair-threshold-implied pre-pruning preserves the original CI weights.
    cutoff = tau_ci / maximum_coefficient
    reduced = Vector{MolecularBasisFunction{T}}(undef, length(basis_functions))
    for index in eachindex(basis_functions)
        basis = basis_functions[index]
        keep = findall(coefficient -> abs(coefficient) > cutoff, basis.expansion.coefficients)
        isempty(keep) && throw(ArgumentError("MEBF $(basis.label) lost every determinant at cutoff $cutoff"))
        reduced_coefficients = basis.expansion.coefficients[keep]
        reduced_occupations = basis.expansion.active_occupations[:, keep]
        reduced_expansion = DeterminantExpansion(
            reduced_coefficients,
            reduced_occupations,
            basis.expansion.inactive_orbitals,
            Val(:owned),
        )
        reduced[index] = MolecularBasisFunction(
            basis.label,
            basis.fragment_state_indices,
            basis.orbital_coefficients,
            reduced_expansion,
            basis.correlation_energy_shift,
        )
    end
    return reduced
end

"""
    _build_screened_multifragment_molecular_basis_functions(problem; screening=ScreeningOptions(), runtime=RuntimeOptions())

Build arbitrary-fragment candidate MEBFs through the exact lazy 64-active product engine and then apply the same collection-wide CI pre-pruning used by the established production dimer builder.
The loaded problem is read-only, preliminary molecular products are exact-count memory-preflighted, final retained coefficients are not renormalized after the shared cutoff, result order follows `problem.mebfs`, no CUDA work occurs, and this internal characterization path remains separate from production until the production migration gates pass.

The earlier fragment-product `tau_ci` screening and this later collection-wide `tau_ci/Cmax` reduction are intentionally documented as separate operations.  Their exact historical relationship to GRONOR is not resolved in this implementation step; current numerical behavior is preserved unchanged pending a future screening-semantics audit.
"""
function _build_screened_multifragment_molecular_basis_functions(
    problem::NOCIProblemDefinition{T};
    screening::ScreeningOptions{T}=ScreeningOptions(T(1.0e-6), T(1.0e-5), T(1.0e-12)),
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    preliminary = _build_multifragment_molecular_basis_functions(
        problem;
        screening=screening,
        runtime=runtime,
    )
    return _reduce_multifragment_basis_collection(preliminary, screening.tau_ci)
end
