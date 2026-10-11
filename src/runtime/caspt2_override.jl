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

 File:            src/runtime/caspt2_override.jl
 Purpose:         Snapshot-only CASPT2 energy override with exact-label shifts and SHA-256 provenance.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _validate_override_method(value)

Return an owned CASPT2 method identifier restricted to letters, digits, dot, underscore, and dash.
The input is read-only and dimensionless, no files or CUDA state are touched, and whitespace or unsafe filename characters raise `ArgumentError`.
"""
function _validate_override_method(value)
    value isa AbstractString || throw(ArgumentError("CASPT2 override method must be a string"))
    method = String(value)
    isempty(method) && throw(ArgumentError("CASPT2 override method must not be empty"))
    strip(method) == method || throw(ArgumentError("CASPT2 override method must not contain surrounding whitespace"))
    occursin(r"^[A-Za-z0-9_.-]+$", method) || throw(ArgumentError("CASPT2 override method contains unsupported characters"))
    return method
end

"""
    _parse_override_weighting(value)

Map one explicit YAML shift-weighting name to the supported Löwdin-basis scalar-shift selector.
The value is read-only, no CPU scientific array or CUDA state is allocated, and unsupported names raise `ArgumentError`.
"""
function _parse_override_weighting(value)
    value isa AbstractString || throw(ArgumentError("CASPT2 shift_weighting must be a string"))
    normalized = lowercase(replace(strip(String(value)), "-" => "_", "/" => "_"))
    normalized == "identity" && return :identity
    normalized in ("gallup_norbeck_fortran", "gn_fortran") && return :gallup_norbeck_fortran
    normalized in ("gallup_norbeck_paper", "gn_paper") && return :gallup_norbeck_paper
    throw(ArgumentError("shift_weighting must be identity, gallup_norbeck_fortran, or gallup_norbeck_paper"))
end

"""
    _parse_override_energy(T, label, value)

Convert one YAML CASPT2 energy to finite floating type `T` and return an owned label-energy pair.
Energy units are Hartree, inputs are read-only, no file or CUDA side effect occurs, and empty labels, booleans, nonnumeric values, or nonfinite results raise `ArgumentError`.
"""
function _parse_override_energy(::Type{T}, label, value) where {T<:AbstractFloat}
    key = String(label)
    isempty(strip(key)) && throw(ArgumentError("CASPT2 override energy label must not be empty"))
    value isa Bool && throw(ArgumentError("CASPT2 override energy for $key must be numeric"))
    energy = if value isa Real
        T(value)
    elseif value isa AbstractString
        parse(T, value)
    else
        throw(ArgumentError("CASPT2 override energy for $key must be numeric"))
    end
    isfinite(energy) || throw(ArgumentError("CASPT2 override energy for $key must be finite"))
    return key => energy
end

"""
    _load_strict_yaml(path)

Parse one YAML document while rejecting duplicate keys at every mapping level.
The file is read-only, returned values are CPU metadata, no CUDA state is touched, and YAML syntax or duplicate-key errors propagate as explicit parser exceptions.
"""
function _load_strict_yaml(path::AbstractString)
    mapping_constructor = (constructor, node) -> YAML.construct_mapping(Dict{Any,Any}, constructor, node; strict_unique_keys=true)
    constructors = Dict{String,Function}("tag:yaml.org,2002:map" => mapping_constructor)
    return YAML.load_file(path, constructors)
end

"""
    load_caspt2_override(path, T)

Read and strictly validate one label-preserving scalar CASPT2 override YAML and return small immutable-style metadata.
The YAML file is read-only, all energies are finite Hartree values, no scientific source or CUDA state is touched, and missing, extra, or malformed schema fields raise `ArgumentError`.
"""
function load_caspt2_override(path::AbstractString, ::Type{T}) where {T<:AbstractFloat}
    isfile(path) || throw(ArgumentError("CASPT2 override YAML not found: $path"))
    raw = _load_strict_yaml(path)
    raw isa AbstractDict || throw(ArgumentError("CASPT2 override YAML must contain a mapping"))
    allowed = Set(("schema_version", "method", "source", "units", "state_mapping", "shift_weighting", "energies"))
    all(key -> String(key) in allowed, keys(raw)) || throw(ArgumentError("CASPT2 override YAML contains an unsupported key"))
    required = ("schema_version", "method", "units", "state_mapping", "shift_weighting", "energies")
    all(key -> haskey(raw, key), required) || throw(ArgumentError("CASPT2 override YAML is missing a required key"))
    raw["schema_version"] isa Integer && !(raw["schema_version"] isa Bool) && Int(raw["schema_version"]) == 1 || throw(ArgumentError("CASPT2 override schema_version must equal 1"))
    method = _validate_override_method(raw["method"])
    source = haskey(raw, "source") ? String(raw["source"]) : ""
    lowercase(strip(String(raw["units"]))) == "hartree" || throw(ArgumentError("CASPT2 override units must be Hartree"))
    lowercase(strip(String(raw["state_mapping"]))) == "label_preserving" || throw(ArgumentError("CASPT2 override state_mapping must be label_preserving"))
    weighting = _parse_override_weighting(raw["shift_weighting"])
    raw_energies = raw["energies"]
    raw_energies isa AbstractDict || throw(ArgumentError("CASPT2 override energies must be a mapping"))
    isempty(raw_energies) && throw(ArgumentError("CASPT2 override energies must not be empty"))
    energies = Dict{String,T}()
    for (label, value) in raw_energies
        key, energy = _parse_override_energy(T, label, value)
        haskey(energies, key) && throw(ArgumentError("duplicate CASPT2 override label $key"))
        energies[key] = energy
    end
    return (
        method=method,
        source=source,
        weighting=weighting,
        weighting_label=String(weighting),
        path=abspath(path),
        sha256=_sha256_file(path),
        energies=energies,
    )
end

"""
    _snapshot_energy_metadata(snapshot, T)

Extract ordered fragment and MEBF energy metadata from a validated self-contained snapshot.
Snapshot values are read-only, all energies are copied as type `T` in Hartree, no file or CUDA I/O occurs, and inconsistent dimensions raise through downstream validation.
"""
function _snapshot_energy_metadata(snapshot::AbstractDict, ::Type{T}) where {T<:AbstractFloat}
    return (
        fragment_labels=String.(snapshot["fragment_labels"]),
        casscf_energies=T.(snapshot["fragment_casscf_energies"]),
        caspt2_energies=T.(snapshot["fragment_caspt2_energies"]),
        mebf_labels=String.(snapshot["mebf_labels"]),
        mebf_states=[String.(states) for states in snapshot["mebf_states"]],
        mebf_manual_shifts=T.(snapshot["mebf_manual_shifts"]),
    )
end

"""
    _normalized_from_snapshot(snapshot, T)

Reconstruct validated normalized unshifted matrix components from a self-contained snapshot.
All snapshot arrays are read-only and copied to CPU type `T`, energy and overlap units are preserved, and malformed dimensions or values raise during construction or solve.
"""
function _normalized_from_snapshot(snapshot::AbstractDict, ::Type{T}) where {T<:AbstractFloat}
    return NormalizedNOCIMatrices(
        String.(snapshot["mebf_labels"]),
        Matrix{T}(snapshot["normalized_overlap"]),
        Matrix{T}(snapshot["normalized_electronic_one_body"]),
        Matrix{T}(snapshot["normalized_nuclear_repulsion"]),
        Matrix{T}(snapshot["normalized_one_body_with_nuclear_repulsion"]),
        Matrix{T}(snapshot["normalized_two_body"]),
        Matrix{T}(snapshot["normalized_hamiltonian"]),
    )
end

"""
    apply_caspt2_override(snapshot_path, override_path; output=nothing)

Apply a strict label-preserving scalar CASPT2 energy override using only one completed matrix snapshot and YAML file, then write independent `.day/.out/.json` postprocessing outputs.
Original determinant, orbital, integral, checkpoint, and CUDA resources are never opened or initialized, unshifted snapshot data remain unchanged, energies use Hartree, and schema, label-set, checksum, or Löwdin-solve failures raise explicitly.
"""
function apply_caspt2_override(
    snapshot_path::AbstractString,
    override_path::AbstractString;
    output::Union{Nothing,RunOutputOptions}=nothing,
)
    snapshot = load_matrix_snapshot(snapshot_path)
    T = eltype(snapshot["normalized_hamiltonian"])
    T <: AbstractFloat || throw(ArgumentError("snapshot Hamiltonian must use a floating-point type"))
    override = load_caspt2_override(override_path, T)
    metadata = _snapshot_energy_metadata(snapshot, T)
    expected_labels = Set(metadata.fragment_labels)
    provided_labels = Set(keys(override.energies))
    expected_labels == provided_labels || begin
        missing = sort!(collect(setdiff(expected_labels, provided_labels)))
        extra = sort!(collect(setdiff(provided_labels, expected_labels)))
        throw(ArgumentError("CASPT2 override label set mismatch; missing=$(missing), extra=$(extra)"))
    end
    override_energies = T[override.energies[label] for label in metadata.fragment_labels]
    shifts = _correlation_shifts(metadata, override_energies)
    normalized = _normalized_from_snapshot(snapshot, T)
    shifted_hamiltonian = apply_diagonal_shifts(normalized.hamiltonian, normalized.overlap, shifts; weighting=override.weighting)
    shifted_solution = solve_generalized_noci(_normalized_with_hamiltonian(normalized, shifted_hamiltonian))
    selected_output = if isnothing(output)
        directory = joinpath(dirname(abspath(snapshot_path)), "overrides", string(override.method, "_", first(override.sha256, 12)))
        RunOutputOptions(directory, splitext(basename(snapshot_path))[1])
    else
        output
    end
    mkpath(selected_output.directory)
    day_path = abspath(joinpath(selected_output.directory, string(selected_output.root, ".day")))
    out_path = abspath(joinpath(selected_output.directory, string(selected_output.root, ".out")))
    json_path = abspath(joinpath(selected_output.directory, string(selected_output.root, ".json")))
    isfile(json_path) && throw(ArgumentError("refusing to overwrite completed CASPT2 override JSON: $json_path"))
    logger = open_progress_logger(day_path, out_path)
    result = Caspt2OverrideResult(
        override.method,
        override.source,
        override.weighting_label,
        override.sha256,
        shifts,
        shifted_hamiltonian,
        shifted_solution,
        json_path,
    )
    try
        progress_event!(logger, :caspt2_override_started; method=override.method, snapshot=abspath(snapshot_path), override_sha256=override.sha256)
        println(logger.out_io, "CASPT2 snapshot-only scalar energy override")
        println(logger.out_io, "method = ", override.method)
        println(logger.out_io, "source = ", override.source)
        println(logger.out_io, "state_mapping = label_preserving")
        println(logger.out_io, "shift_weighting = ", override.weighting_label)
        println(logger.out_io, "snapshot = ", abspath(snapshot_path))
        println(logger.out_io, "snapshot_sha256 = ", snapshot["snapshot_sha256"])
        println(logger.out_io, "override_sha256 = ", override.sha256)
        println(logger.out_io, "MEBF shifts (Hartree) = ", shifts)
        println(logger.out_io, "Unshifted energies (Hartree) = ", snapshot["unshifted_energies"])
        println(logger.out_io, "Shifted energies (Hartree) = ", shifted_solution.energies)
        flush(logger.out_io)
        data = Dict{String,Any}(
            "schema_version" => 1,
            "operation" => "caspt2_energy_override",
            "units" => "Hartree",
            "method" => override.method,
            "source" => override.source,
            "state_mapping" => "label_preserving",
            "shift_weighting" => override.weighting_label,
            "snapshot_path" => abspath(snapshot_path),
            "snapshot_sha256" => String(snapshot["snapshot_sha256"]),
            "override_path" => override.path,
            "override_sha256" => override.sha256,
            "fragment_labels" => metadata.fragment_labels,
            "original_caspt2_energies" => metadata.caspt2_energies,
            "override_caspt2_energies" => override_energies,
            "mebf_labels" => metadata.mebf_labels,
            "mebf_shifts" => shifts,
            "unshifted_overlap" => _matrix_json(normalized.overlap),
            "unshifted_hamiltonian" => _matrix_json(normalized.hamiltonian),
            "unshifted_energies" => collect(snapshot["unshifted_energies"]),
            "shifted_hamiltonian" => _matrix_json(shifted_hamiltonian),
            "shifted_solution" => _solution_json(shifted_solution),
        )
        _atomic_json_write(json_path, data)
        progress_event!(logger, :caspt2_override_complete; json=json_path, ground_energy=first(shifted_solution.energies))
    finally
        close_progress_logger!(logger)
    end
    return result
end
