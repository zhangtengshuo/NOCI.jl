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

 File:            src/runtime/snapshot.jl
 Purpose:         Matrix snapshot serialization with SHA-256 integrity digests for reference gates.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

const NOCI_MATRIX_SNAPSHOT_VERSION = 1

"""
    _sha256_array(array)

Return a SHA-256 hexadecimal digest over the exact in-memory bytes of one dense bits array.
The array is read-only, units are irrelevant to byte identity, no copy or CUDA operation is requested, and non-bits element types raise `ArgumentError`.
"""
function _sha256_array(array::Array{T}) where {T}
    isbitstype(T) || throw(ArgumentError("snapshot array checksums require bits element types"))
    return bytes2hex(SHA.sha256(reinterpret(UInt8, vec(array))))
end

"""
    _fragment_energy_metadata(input, T)

Read configured determinant files and return ordered fragment energies plus MEBF mappings expanded with the same complete-spin-manifold semantics used by problem loading.
Only determinant sources are read, CASSCF/CASPT2 values use Hartree while spin multiplicities are dimensionless internal metadata, returned vectors are CPU-owned, no orbital/integral file or CUDA state is touched, and malformed or inconsistent spin data raise through the determinant/parser and manifold validators.
"""
function _fragment_energy_metadata(input::NOCIInput, ::Type{T}) where {T<:AbstractFloat}
    fragment_labels = String[]
    casscf_energies = T[]
    caspt2_energies = T[]
    spin_multiplicities = Int[]
    for specification in input.fragment_states
        determinant = read_determinant_file(specification.determinant_path; T)
        push!(fragment_labels, specification.label)
        push!(casscf_energies, determinant.casscf_energy)
        push!(caspt2_energies, determinant.caspt2_energy)
        push!(spin_multiplicities, determinant.spin_multiplicity)
    end
    fragment_index_by_label = Dict(
        label => index
        for (index, label) in pairs(fragment_labels)
    )
    mebf_labels = String[]
    mebf_states = Vector{Vector{String}}()
    mebf_manual_shifts = T[]
    for definition in input.mebfs
        local_multiplicities = [
            spin_multiplicities[fragment_index_by_label[label]]
            for label in definition.state_labels
        ]
        descriptors = _spin_coupled_basis_descriptors(
            definition.label,
            local_multiplicities,
            input.spin_multiplicity,
            definition.coupling_multiplicities,
        )
        for descriptor in descriptors
            push!(mebf_labels, descriptor.label)
            push!(mebf_states, copy(definition.state_labels))
            push!(mebf_manual_shifts, T(definition.diagonal_energy_shift))
        end
    end
    allunique(mebf_labels) || throw(ArgumentError("resolved spin-coupled MEBF energy labels must be unique"))
    return (
        fragment_labels=fragment_labels,
        casscf_energies=casscf_energies,
        caspt2_energies=caspt2_energies,
        mebf_labels=mebf_labels,
        mebf_states=mebf_states,
        mebf_manual_shifts=mebf_manual_shifts,
    )
end

"""
    _correlation_shifts(metadata, caspt2_energies)

Build one label-preserving MEBF shift from fragment CASPT2-minus-CASSCF energies and explicit manual MEBF shifts.
All energies use Hartree and inputs are read-only, the returned vector is CPU-owned, and label-set, dimension, duplicate, or finite-value errors raise explicitly.
"""
function _correlation_shifts(metadata, caspt2_energies::AbstractVector{T}) where {T<:AbstractFloat}
    n_fragments = length(metadata.fragment_labels)
    length(metadata.casscf_energies) == n_fragments || throw(DimensionMismatch("fragment CASSCF energy count differs from labels"))
    length(caspt2_energies) == n_fragments || throw(DimensionMismatch("fragment CASPT2 energy count differs from labels"))
    allunique(metadata.fragment_labels) || throw(ArgumentError("fragment-state labels must be unique"))
    all(isfinite, metadata.casscf_energies) && all(isfinite, caspt2_energies) || throw(ArgumentError("fragment energies must be finite"))
    length(metadata.mebf_labels) == length(metadata.mebf_states) == length(metadata.mebf_manual_shifts) || throw(DimensionMismatch("MEBF energy metadata dimensions differ"))
    correlation_by_label = Dict(metadata.fragment_labels[index] => caspt2_energies[index] - metadata.casscf_energies[index] for index in eachindex(metadata.fragment_labels))
    shifts = collect(T, metadata.mebf_manual_shifts)
    for mebf in eachindex(shifts)
        for label in metadata.mebf_states[mebf]
            haskey(correlation_by_label, label) || throw(ArgumentError("MEBF $(metadata.mebf_labels[mebf]) references unknown fragment-state label $label"))
            shifts[mebf] += correlation_by_label[label]
        end
    end
    return shifts
end

"""
    _snapshot_checksum(values)

Return a SHA-256 digest covering snapshot identity, ordered metadata, and exact bytes of every stored scientific array.
The mapping is read-only, no file or CUDA I/O occurs, and missing required snapshot fields raise `KeyError`.
"""
function _snapshot_checksum(values::AbstractDict{<:AbstractString,<:Any})
    array_keys = (
        "raw_overlap",
        "raw_electronic_one_body",
        "raw_nuclear_repulsion",
        "raw_one_body_with_nuclear_repulsion",
        "raw_two_body",
        "raw_hamiltonian",
        "determinant_pair_counts",
        "normalized_overlap",
        "normalized_electronic_one_body",
        "normalized_nuclear_repulsion",
        "normalized_one_body_with_nuclear_repulsion",
        "normalized_two_body",
        "normalized_hamiltonian",
        "unshifted_energies",
        "unshifted_vectors",
        "unshifted_residual_norms",
        "fragment_casscf_energies",
        "fragment_caspt2_energies",
        "mebf_manual_shifts",
        "baseline_correlation_shifts",
        "baseline_shifted_hamiltonian",
        "baseline_shifted_energies",
        "baseline_shifted_vectors",
        "baseline_shifted_residual_norms",
    )
    array_digests = Tuple((key, _sha256_array(values[key])) for key in array_keys)
    metadata = (
        Int(values["version"]),
        String(values["scientific_signature"]),
        String(values["execution_signature"]),
        String(values["label"]),
        String(values["algorithm"]),
        String(values["branch"]),
        String(values["backend"]),
        Tuple(String.(values["mebf_labels"])),
        Tuple(String.(values["fragment_labels"])),
        Tuple(Tuple(String.(states)) for states in values["mebf_states"]),
        Tuple(
            haskey(record, "sha256") ?
                (Int(record["index"]), String(record["path"]), Int(record["bytes"]), String(record["sha256"])) :
                (Int(record["index"]), String(record["path"]), Int(record["bytes"]), Int64(record["mtime_ns"]), String(record["identity_kind"]))
            for record in values["input_manifest"]
        ),
    )
    return _sha256_value((metadata, array_digests))
end

"""
    save_matrix_snapshot(path, input, evaluation, metadata, shifts, shifted_hamiltonian, shifted_solution, input_manifest, scientific_signature, execution_signature)

Atomically save a complete self-contained unshifted NOCI matrix snapshot with baseline scalar CASPT2 postprocessing and SHA-256 integrity metadata.
Scientific arrays and input metadata are read-only, only the destination host file is changed, no integrals or CUDA objects are serialized, and validation or filesystem failures preserve the previous valid snapshot.
"""
function save_matrix_snapshot(
    path::AbstractString,
    input::NOCIInput,
    evaluation::NOCIEvaluationResult{T},
    metadata,
    shifts::AbstractVector{T},
    shifted_hamiltonian::AbstractMatrix{T},
    shifted_solution::NOCISolution{T},
    input_manifest::AbstractVector,
    scientific_signature::AbstractString,
    execution_signature::AbstractString,
) where {T<:AbstractFloat}
    raw = evaluation.raw
    normalized = evaluation.normalized
    metadata.mebf_labels == raw.labels || throw(ArgumentError("snapshot MEBF energy metadata labels differ from evaluated MEBF labels"))
    length(shifts) == length(raw.labels) || throw(DimensionMismatch("snapshot shift count differs from evaluated MEBF count"))
    values = Dict{String,Any}(
        "version" => NOCI_MATRIX_SNAPSHOT_VERSION,
        "scientific_signature" => String(scientific_signature),
        "execution_signature" => String(execution_signature),
        "input_manifest" => deepcopy(input_manifest),
        "written_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "matrix_stage" => "raw_and_normalized_unshifted_complete",
        "label" => evaluation.label,
        "algorithm" => evaluation.algorithm,
        "branch" => evaluation.branch,
        "backend" => evaluation.backend,
        "eri_factor_provenance" => Dict{String,Any}(String(key) => value for phase in evaluation.telemetry.phases if phase.name == :load_eri for (key, value) in phase.metadata),
        "mebf_labels" => copy(raw.labels),
        "raw_overlap" => copy(raw.overlap),
        "raw_electronic_one_body" => copy(raw.electronic_one_body),
        "raw_nuclear_repulsion" => copy(raw.nuclear_repulsion),
        "raw_one_body_with_nuclear_repulsion" => copy(raw.one_body_with_nuclear_repulsion),
        "raw_two_body" => copy(raw.two_body),
        "raw_hamiltonian" => copy(raw.hamiltonian),
        "determinant_pair_counts" => copy(raw.determinant_pair_counts),
        "normalized_overlap" => copy(normalized.overlap),
        "normalized_electronic_one_body" => copy(normalized.electronic_one_body),
        "normalized_nuclear_repulsion" => copy(normalized.nuclear_repulsion),
        "normalized_one_body_with_nuclear_repulsion" => copy(normalized.one_body_with_nuclear_repulsion),
        "normalized_two_body" => copy(normalized.two_body),
        "normalized_hamiltonian" => copy(normalized.hamiltonian),
        "unshifted_energies" => copy(evaluation.solution.energies),
        "unshifted_vectors" => copy(evaluation.solution.vectors),
        "unshifted_residual_norms" => copy(evaluation.solution.residual_norms),
        "fragment_labels" => copy(metadata.fragment_labels),
        "fragment_casscf_energies" => copy(metadata.casscf_energies),
        "fragment_caspt2_energies" => copy(metadata.caspt2_energies),
        "mebf_states" => deepcopy(metadata.mebf_states),
        "mebf_manual_shifts" => copy(metadata.mebf_manual_shifts),
        "baseline_correlation_shifts" => collect(shifts),
        "baseline_shift_weighting" => "identity",
        "baseline_shifted_hamiltonian" => Matrix{T}(shifted_hamiltonian),
        "baseline_shifted_energies" => copy(shifted_solution.energies),
        "baseline_shifted_vectors" => copy(shifted_solution.vectors),
        "baseline_shifted_residual_norms" => copy(shifted_solution.residual_norms),
    )
    values["snapshot_sha256"] = _snapshot_checksum(values)
    return _atomic_jld2_write(path, values)
end

"""
    load_matrix_snapshot(path)

Load and integrity-check one complete self-contained NOCI matrix snapshot for postprocessing.
Only the snapshot file is opened, returned arrays are CPU-owned, CUDA and original determinant/orbital/integral paths are never touched, and format, checksum, dimension, or component inconsistencies raise explicitly.
"""
function load_matrix_snapshot(path::AbstractString)
    isfile(path) || throw(ArgumentError("matrix snapshot not found: $path"))
    values = JLD2.load(path)
    Int(values["version"]) == NOCI_MATRIX_SNAPSHOT_VERSION || throw(ArgumentError("unsupported matrix snapshot version in $path"))
    String(values["matrix_stage"]) == "raw_and_normalized_unshifted_complete" || throw(ArgumentError("matrix snapshot is not marked complete: $path"))
    String(values["snapshot_sha256"]) == _snapshot_checksum(values) || throw(ArgumentError("matrix snapshot checksum mismatch: $path"))
    return values
end
