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

 File:            src/runtime/coupling_output.jl
 Purpose:         Scalar electronic-coupling output preparation in Hartree-to-meV units.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

const NOCI_HARTREE_TO_MEV = 1000 * NOCI_HARTREE_TO_EV

"""
    _scalar_coupling_matrices(hamiltonian, overlap)

Evaluate both supported scalar electronic-coupling definitions for one Hamiltonian/overlap pair and return them as CPU matrices.
Inputs are read-only, Hamiltonian energy units are preserved, no spatial coupling quantity is constructed, and numerical validation errors propagate from the coupling routines.
"""
function _scalar_coupling_matrices(
    hamiltonian::AbstractMatrix{T},
    overlap::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return (
        pairwise=pairwise_lowdin_couplings(hamiltonian, overlap),
        full_multistate=full_multistate_lowdin_couplings(hamiltonian, overlap),
    )
end

"""
    _scalar_coupling_json(couplings)

Convert pairwise and full-multistate scalar coupling matrices into explicit Hartree and meV JSON payloads.
The matrices are read-only, only reporting containers and scaled CPU matrices are allocated, and no spatial or determinant-pair decomposition data are emitted.
"""
function _scalar_coupling_json(couplings)
    return Dict{String,Any}(
        "pairwise_lowdin" => Dict{String,Any}(
            "hartree" => _matrix_json(couplings.pairwise),
            "mev" => _matrix_json(couplings.pairwise .* NOCI_HARTREE_TO_MEV),
        ),
        "full_multistate_lowdin" => Dict{String,Any}(
            "hartree" => _matrix_json(couplings.full_multistate),
            "mev" => _matrix_json(couplings.full_multistate .* NOCI_HARTREE_TO_MEV),
        ),
    )
end

"""
    _write_scalar_coupling_summary!(logger, result)

Append unshifted and baseline-CASPT2-shifted scalar coupling matrices to the production `.out` report using both pairwise and complete-state-space Löwdin definitions.
The production result is read-only, reported matrix values are meV with zero diagonal, host-only dense linear algebra is used, and no spatial coupling analysis is enabled.
"""
function _write_scalar_coupling_summary!(logger::ProgressLogger, result::ProductionRunResult)
    io = logger.out_io
    normalized = result.evaluation.normalized
    labels = normalized.labels
    unshifted = _scalar_coupling_matrices(normalized.hamiltonian, normalized.overlap)
    shifted = _scalar_coupling_matrices(result.shifted_hamiltonian, normalized.overlap)

    _write_section!(io, "SCALAR ELECTRONIC COUPLINGS")
    println(io, "units = meV")
    println(io, "diagonal = zero by definition")
    println(io, "pairwise_lowdin = [H_ij - 0.5*(H_ii + H_jj)*S_ij] / [1 - S_ij^2] after diagonal-overlap normalization")
    println(io, "full_multistate_lowdin = off-diagonal elements of S^(-1/2) * H * S^(-1/2) in the complete supplied MEBF space")
    println(io, "note = scalar reporting only; coupling densities, cubes, decompositions, and other spatial outputs remain disabled")

    for (name, matrix) in (
        ("Unshifted pairwise Löwdin couplings / meV", unshifted.pairwise),
        ("Unshifted full-multistate Löwdin couplings / meV", unshifted.full_multistate),
        ("Baseline CASPT2-shifted pairwise Löwdin couplings / meV", shifted.pairwise),
        ("Baseline CASPT2-shifted full-multistate Löwdin couplings / meV", shifted.full_multistate),
    )
        _write_named_matrix!(io, name, matrix .* NOCI_HARTREE_TO_MEV; row_labels=labels)
    end
    flush(io)
    return nothing
end

"""
    _attach_scalar_coupling_json!(data, result)

Attach explicit scalar-coupling blocks and historical pairwise `couplings_mev` aliases to an already constructed production JSON mapping.
The result and existing scientific arrays are read-only, `data` is mutated only by adding reporting fields, no files or CUDA state are touched, and spatial coupling output remains absent.
"""
function _attach_scalar_coupling_json!(
    data::Dict{String,Any},
    result::ProductionRunResult,
)
    normalized = result.evaluation.normalized
    unshifted = _scalar_coupling_matrices(normalized.hamiltonian, normalized.overlap)
    shifted = _scalar_coupling_matrices(result.shifted_hamiltonian, normalized.overlap)

    data["scalar_couplings"] = Dict{String,Any}(
        "representation" => "state-pair scalar matrices",
        "state_labels" => copy(normalized.labels),
        "diagonal" => "zero_by_definition",
        "pairwise_definition" => "[H_ij - 0.5*(H_ii + H_jj)*S_ij] / [1 - S_ij^2] after diagonal-overlap normalization",
        "full_multistate_definition" => "off-diagonal elements of S^(-1/2) * H * S^(-1/2) in the complete supplied MEBF space",
        "unshifted" => _scalar_coupling_json(unshifted),
        "baseline_caspt2_shifted" => _scalar_coupling_json(shifted),
    )

    unshifted_solution = data["unshifted_solution"]
    unshifted_solution["coupling_definition"] = "pairwise_lowdin"
    unshifted_solution["couplings_hartree"] = _matrix_json(unshifted.pairwise)
    unshifted_solution["couplings_mev"] = _matrix_json(unshifted.pairwise .* NOCI_HARTREE_TO_MEV)
    unshifted_solution["full_multistate_couplings_hartree"] = _matrix_json(unshifted.full_multistate)
    unshifted_solution["full_multistate_couplings_mev"] = _matrix_json(unshifted.full_multistate .* NOCI_HARTREE_TO_MEV)

    shifted_solution = data["baseline_caspt2"]["shifted_solution"]
    shifted_solution["coupling_definition"] = "pairwise_lowdin"
    shifted_solution["couplings_hartree"] = _matrix_json(shifted.pairwise)
    shifted_solution["couplings_mev"] = _matrix_json(shifted.pairwise .* NOCI_HARTREE_TO_MEV)
    shifted_solution["full_multistate_couplings_hartree"] = _matrix_json(shifted.full_multistate)
    shifted_solution["full_multistate_couplings_mev"] = _matrix_json(shifted.full_multistate .* NOCI_HARTREE_TO_MEV)
    return data
end
