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

 File:            src/core/input.jl
 Purpose:         Typed scientific input schema: system, MEBFs, screening, ERI backends, and runtime options.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    ScreeningOptions(; tau_ci=1e-6, tau_ci_offdiagonal=1e-5, tau_singularity=1e-12)

Store scientific screening thresholds separately from algorithm identity and runtime resources.
All thresholds are dimensionless nonnegative finite values; construction performs no file I/O or CPU/GPU workspace allocation and rejects invalid values.
"""
struct ScreeningOptions{T<:AbstractFloat}
    tau_ci::T
    tau_ci_offdiagonal::T
    tau_singularity::T

    """
        ScreeningOptions(tau_ci, tau_ci_offdiagonal, tau_singularity)

    Construct validated screening thresholds of one floating-point type and return an immutable value.
    Inputs are not mutated, no device state is touched, and negative or non-finite values raise `ArgumentError`.
    """
    function ScreeningOptions(
        tau_ci::T,
        tau_ci_offdiagonal::T,
        tau_singularity::T,
    ) where {T<:AbstractFloat}
        all(isfinite, (tau_ci, tau_ci_offdiagonal, tau_singularity)) || throw(ArgumentError("screening thresholds must be finite"))
        all(value -> value >= zero(T), (tau_ci, tau_ci_offdiagonal, tau_singularity)) || throw(ArgumentError("screening thresholds must be nonnegative"))
        return new{T}(tau_ci, tau_ci_offdiagonal, tau_singularity)
    end
end

"""
    ScreeningOptions(; tau_ci=1e-6, tau_ci_offdiagonal=1e-5, tau_singularity=1e-12)

Construct `Float64` or explicitly typed screening thresholds from keyword arguments.
Values are dimensionless, inputs are not mutated, no CPU/GPU scientific storage is allocated, and invalid thresholds raise `ArgumentError`.
"""
function ScreeningOptions(;
    tau_ci::Real=1.0e-6,
    tau_ci_offdiagonal::Real=1.0e-5,
    tau_singularity::Real=1.0e-12,
)
    any(value -> value isa Bool, (tau_ci, tau_ci_offdiagonal, tau_singularity)) && throw(ArgumentError("screening thresholds must be real numbers, not booleans"))
    values = float.(promote(tau_ci, tau_ci_offdiagonal, tau_singularity))
    return ScreeningOptions(values...)
end

"""
    FragmentStateSpecification(label, fragment, state, determinant_path, orbital_path)

Describe one named electronic state belonging to one ordered fragment role and its repository-relative `.det/.vec` sources.
Construction owns normalized path strings, performs no file I/O or CUDA work, and rejects empty identifiers or paths.
"""
struct FragmentStateSpecification
    label::String
    fragment::String
    state::String
    determinant_path::String
    orbital_path::String

    """
        FragmentStateSpecification(label, fragment, state, determinant_path, orbital_path)

    Construct a validated fragment-state specification and copy all string-like inputs.
    Inputs are not mutated, files are not opened, and empty labels, role names, state names, or paths raise `ArgumentError`.
    """
    function FragmentStateSpecification(
        label::AbstractString,
        fragment::AbstractString,
        state::AbstractString,
        determinant_path::AbstractString,
        orbital_path::AbstractString,
    )
        values = (label, fragment, state, determinant_path, orbital_path)
        all(value -> !isempty(strip(value)), values) || throw(ArgumentError("fragment-state labels and paths must not be empty"))
        return new(String(label), String(fragment), String(state), normpath(String(determinant_path)), normpath(String(orbital_path)))
    end
end

"""
    MEBFDefinition(label, state_labels; coupling_multiplicities=[], diagonal_energy_shift=0.0)

Describe one electronic MEBF configuration as an ordered direct product of fragment-state labels with optional legacy explicit intermediate spin-coupling multiplicities and an explicit post-matrix diagonal energy shift.
An empty coupling vector requests the complete target-spin manifold after fragment wavefunctions are loaded; construction copies inputs, stores the shift in Hartree as finite `Float64`, performs no CUDA work, and rejects empty or duplicate state labels, nonpositive explicit coupling multiplicities, or invalid shifts.
"""
struct MEBFDefinition
    label::String
    state_labels::Vector{String}
    coupling_multiplicities::Vector{Int}
    diagonal_energy_shift::Float64

    """
        MEBFDefinition(label, state_labels; coupling_multiplicities=[], diagonal_energy_shift=0.0)

    Construct a validated MEBF definition whose state-label order defines the fragment-role order.
    Inputs are not mutated, the returned vectors are owned, and malformed labels or coupling multiplicities raise `ArgumentError`.
    """
    function MEBFDefinition(
        label::AbstractString,
        state_labels::AbstractVector{<:AbstractString};
        coupling_multiplicities::AbstractVector{<:Integer}=Int[],
        diagonal_energy_shift::Real=0.0,
    )
        isempty(strip(label)) && throw(ArgumentError("MEBF label must not be empty"))
        isempty(state_labels) && throw(ArgumentError("MEBF state_labels must not be empty"))
        labels = String.(state_labels)
        all(value -> !isempty(strip(value)), labels) || throw(ArgumentError("MEBF state labels must not be empty"))
        allunique(labels) || throw(ArgumentError("MEBF state labels must be unique"))
        couplings = Int.(coupling_multiplicities)
        all(>(0), couplings) || throw(ArgumentError("MEBF coupling multiplicities must be positive"))
        diagonal_energy_shift isa Bool && throw(ArgumentError("MEBF diagonal_energy_shift must be a real number"))
        shift = Float64(diagonal_energy_shift)
        isfinite(shift) || throw(ArgumentError("MEBF diagonal_energy_shift must be finite"))
        return new(String(label), labels, couplings, shift)
    end
end

"""
    NOCIInput

Store one fully validated but not yet numerically loaded NOCI configuration.
The value owns algorithm, backend, screening, runtime, fragment-state, and MEBF metadata; paths are resolved against the YAML location before construction and no scientific files or CUDA state are loaded.
"""
struct NOCIInput{A<:NOCIAlgorithm,B<:ERIBackend,T<:AbstractFloat}
    label::String
    spin_multiplicity::Int
    algorithm::A
    backend::B
    screening::ScreeningOptions{T}
    runtime::RuntimeOptions
    fragment_states::Vector{FragmentStateSpecification}
    mebfs::Vector{MEBFDefinition}

    """
        NOCIInput(label, spin_multiplicity, algorithm, backend, screening, runtime, fragment_states, mebfs)

    Construct a validated unloaded NOCI input with unique state/MEBF labels, consistent fragment roles, legal spin-coupling metadata, and a supported algorithm/backend pair.
    Inputs are not mutated, metadata vectors are copied, no scientific files or CUDA state are touched, and structural inconsistencies raise `ArgumentError` or `UnsupportedERIBackend`.
    """
    function NOCIInput(
        label::AbstractString,
        spin_multiplicity::Integer,
        algorithm::A,
        backend::B,
        screening::ScreeningOptions{T},
        runtime::RuntimeOptions,
        fragment_states::AbstractVector{FragmentStateSpecification},
        mebfs::AbstractVector{MEBFDefinition},
    ) where {A<:NOCIAlgorithm,B<:ERIBackend,T<:AbstractFloat}
        _validate_input_structure(label, spin_multiplicity, algorithm, backend, fragment_states, mebfs)
        return new{A,B,T}(
            String(label),
            Int(spin_multiplicity),
            algorithm,
            backend,
            screening,
            runtime,
            collect(fragment_states),
            collect(mebfs),
        )
    end
end

"""
    FragmentStateEntry(specification, wavefunction)

Pair a fragment-state specification with its loaded and validated wavefunction.
The immutable entry retains both inputs without copying numerical arrays and performs no additional CPU/GPU allocation.
"""
struct FragmentStateEntry{T<:AbstractFloat}
    specification::FragmentStateSpecification
    wavefunction::FragmentWavefunction{T}
end

"""
    MEBFComposition(label, fragment_state_indices, coupling_multiplicities)

Store one resolved MEBF composition using one-based indices into a `NOCIProblemDefinition` fragment-state pool.
Vectors are owned and validated as nonempty, positive, and duplicate-free; construction allocates only metadata and performs no CUDA work.
"""
struct MEBFComposition
    label::String
    fragment_state_indices::Vector{Int}
    coupling_multiplicities::Vector{Int}

    """
        MEBFComposition(label, fragment_state_indices, coupling_multiplicities)

    Construct a resolved MEBF composition from validated one-based state-pool indices.
    Inputs are not mutated, returned vectors are copied, and invalid labels, indices, duplicates, or coupling multiplicities raise `ArgumentError`.
    """
    function MEBFComposition(
        label::AbstractString,
        fragment_state_indices::AbstractVector{<:Integer},
        coupling_multiplicities::AbstractVector{<:Integer},
    )
        isempty(strip(label)) && throw(ArgumentError("MEBF composition label must not be empty"))
        indices = Int.(fragment_state_indices)
        isempty(indices) && throw(ArgumentError("MEBF composition must reference at least one fragment state"))
        all(>(0), indices) || throw(ArgumentError("fragment-state indices must be positive"))
        allunique(indices) || throw(ArgumentError("fragment-state indices within one MEBF must be unique"))
        couplings = Int.(coupling_multiplicities)
        all(>(0), couplings) || throw(ArgumentError("MEBF coupling multiplicities must be positive"))
        return new(String(label), indices, couplings)
    end
end

"""
    NOCIProblemDefinition

Store the loaded fragment-state pool and resolved MEBF compositions before molecular determinant-product assembly.
The value records the common orbital basis dimension and common total electron count, retains loaded CPU arrays, and contains no CUDA allocation or integral values.
"""
struct NOCIProblemDefinition{T<:AbstractFloat}
    label::String
    spin_multiplicity::Int
    fragment_order::Vector{String}
    fragment_states::Vector{FragmentStateEntry{T}}
    state_index_by_label::Dict{String,Int}
    mebfs::Vector{MEBFComposition}
    fragment_basis_dimensions::Vector{Int}
    n_basis::Int
    n_electrons::Int
end

"""
    _required_yaml(mapping, key, location)

Return a required YAML value or raise `ArgumentError` naming its structural location.
The mapping is read-only, the value is not copied, and no scientific CPU/GPU memory is allocated.
"""
function _required_yaml(mapping::AbstractDict, key::AbstractString, location::AbstractString)
    haskey(mapping, key) || throw(ArgumentError("missing YAML key $location.$key"))
    return mapping[key]
end

"""
    _yaml_mapping(value, location)

Return `value` after requiring a YAML mapping at the named location.
The value is not mutated or copied and malformed input raises `ArgumentError` without device side effects.
"""
function _yaml_mapping(value, location::AbstractString)
    value isa AbstractDict || throw(ArgumentError("$location must be a mapping"))
    return value
end

"""
    _yaml_sequence(value, location)

Return `value` after requiring a YAML sequence at the named location.
The value is not mutated or copied and malformed input raises `ArgumentError` without device side effects.
"""
function _yaml_sequence(value, location::AbstractString)
    value isa AbstractVector || throw(ArgumentError("$location must be a sequence"))
    return value
end

"""
    _yaml_string(value, location)

Convert a YAML scalar string to an owned nonempty `String`.
The input is not mutated, no scientific storage is allocated, and non-string or empty values raise `ArgumentError`.
"""
function _yaml_string(value, location::AbstractString)
    value isa AbstractString || throw(ArgumentError("$location must be a string"))
    isempty(strip(value)) && throw(ArgumentError("$location must not be empty"))
    return String(value)
end

"""
    _yaml_integer(value, location)

Convert an integer YAML scalar to `Int` without accepting booleans or floating-point truncation.
The input is not mutated, no CPU/GPU workspace is allocated, and invalid or overflowing values raise `ArgumentError` or `InexactError`.
"""
function _yaml_integer(value, location::AbstractString)
    value isa Integer && !(value isa Bool) || throw(ArgumentError("$location must be an integer"))
    return Int(value)
end

"""
    _yaml_boolean(value, location)

Return a YAML boolean scalar without accepting string aliases or integer coercion.
The input is not mutated, no CPU/GPU workspace is allocated, and non-boolean values raise `ArgumentError`.
"""
function _yaml_boolean(value, location::AbstractString)
    value isa Bool || throw(ArgumentError("$location must be a boolean"))
    return value
end

"""
    _yaml_float(T, value, location)

Convert a real YAML scalar to floating-point type `T` and require a finite result.
The input is not mutated, no device storage is allocated, and non-real, overflowing, or non-finite values raise an exception.
"""
function _yaml_float(::Type{T}, value, location::AbstractString) where {T<:AbstractFloat}
    value isa Real && !(value isa Bool) || throw(ArgumentError("$location must be a real number"))
    result = T(value)
    isfinite(result) || throw(ArgumentError("$location must be finite"))
    return result
end

"""
    _resolve_config_path(config_directory, value, location)

Resolve a nonempty YAML path relative to its configuration directory while preserving explicit absolute paths.
The filesystem is not accessed, inputs are not mutated, no CUDA work occurs, and malformed path scalars raise `ArgumentError`.
"""
function _resolve_config_path(config_directory::AbstractString, value, location::AbstractString)
    path = _yaml_string(value, location)
    return normpath(isabspath(path) ? path : joinpath(config_directory, path))
end

"""
    _parse_noci_algorithm(mapping)

Parse exactly `dp-noci` or production `ahpc-noci` plus its typed support lookup and occupied-spin cofactor evaluation from a YAML algorithm mapping.
The mapping is read-only, construction allocates only small immutable selectors, no CUDA work occurs, and unknown or inapplicable options raise `ArgumentError`.
"""
function _parse_noci_algorithm(mapping::AbstractDict)
    name = _yaml_string(_required_yaml(mapping, "name", "algorithm"), "algorithm.name")
    if name == "dp-noci"
        length(mapping) == 1 || throw(ArgumentError("dp-noci accepts no AHPC branch or acceleration options"))
        return DPNOCI()
    end
    name == "ahpc-noci" || throw(ArgumentError("algorithm.name must be dp-noci or ahpc-noci"))
    branch_name_value = _yaml_string(get(mapping, "branch", "spin_string"), "algorithm.branch")
    branch_name_value == "spin_string" || throw(ArgumentError("algorithm.branch must be spin_string"))
    lookup_name = _yaml_string(get(mapping, "support_lookup", "adaptive"), "algorithm.support_lookup")
    support_lookup = lookup_name == "dict_presized" ? DictPresizedLookup :
                     lookup_name == "flat_hash_presized" ? FlatHashPresizedLookup :
                     lookup_name == "adaptive" ? AdaptiveLookup :
                     throw(ArgumentError("unsupported algorithm.support_lookup: $lookup_name"))
    cofactor_name = _yaml_string(get(mapping, "cofactor_evaluation", "polynomial"), "algorithm.cofactor_evaluation")
    cofactor_evaluation = cofactor_name == "polynomial" ? PolynomialCofactorEvaluation :
                          cofactor_name == "lu_certified" ? LUCertifiedCofactorEvaluation :
                          throw(ArgumentError("unsupported algorithm.cofactor_evaluation: $cofactor_name"))
    all(key -> key in ("name", "branch", "support_lookup", "cofactor_evaluation"), keys(mapping)) || throw(ArgumentError("algorithm contains an unsupported option"))
    return AHPCNOCI(branch=AHPCSpinStringBranch(
        acceleration=AHPCAccelerationOptions(; support_lookup, cofactor_evaluation),
    ))
end

"""
    _parse_eri_backend(mapping, config_directory)

Parse one conventional packed `.two`, ERI factor-bundle, or embedded SEWARD/Julia-MOTRA backend and resolve all paths against the YAML directory.
The mapping is read-only, files, OpenMolcas, and CUDA are not touched, and missing, empty, unknown, or backend-inapplicable keys raise `ArgumentError`.
"""
function _parse_eri_backend(mapping::AbstractDict, config_directory::AbstractString)
    backend = _yaml_string(_required_yaml(mapping, "backend", "integrals"), "integrals.backend")
    one_path = _resolve_config_path(config_directory, _required_yaml(mapping, "one_electron", "integrals"), "integrals.one_electron")
    if backend == "packed_two"
        paths = _yaml_sequence(_required_yaml(mapping, "two_electron", "integrals"), "integrals.two_electron")
        isempty(paths) && throw(ArgumentError("integrals.two_electron must not be empty"))
        all(key -> key in ("backend", "one_electron", "two_electron"), keys(mapping)) || throw(ArgumentError("packed_two integrals contain an unsupported option"))
        return PackedTwoERI(one_path, [_resolve_config_path(config_directory, value, "integrals.two_electron[$index]") for (index, value) in pairs(paths)])
    end
    if backend in ("factorized", "cholesky")
        source_name = lowercase(_yaml_string(get(mapping, "source", haskey(mapping, "factor_bundle") ? "factor_bundle" : "seward"), "integrals.source"))
        if source_name == "factor_bundle"
            factor_path = _resolve_config_path(config_directory, _required_yaml(mapping, "factor_bundle", "integrals"), "integrals.factor_bundle")
            all(key -> key in ("backend", "one_electron", "source", "factor_bundle"), keys(mapping)) || throw(ArgumentError("factor-bundle factorized integrals contain an unsupported option"))
            return FactorizedERI(one_path, factor_path)
        end
        source_name in ("seward", "seward_mpi_shards") || throw(ArgumentError("integrals.source must be factor_bundle, seward, or seward_mpi_shards"))
        allowed = (
            "backend", "one_electron", "source", "snapshot", "project", "ranks",
            "commonorb", "n_frozen", "n_deleted", "batch_size", "transform_backend",
            "gpu_device", "scratch_parent", "molcas_root", "omc_root",
        )
        all(key -> key in allowed, keys(mapping)) || throw(ArgumentError("SEWARD factorized integrals contain an unsupported option"))
        snapshot = _resolve_config_path(config_directory, _required_yaml(mapping, "snapshot", "integrals"), "integrals.snapshot")
        commonorb = _resolve_config_path(config_directory, _required_yaml(mapping, "commonorb", "integrals"), "integrals.commonorb")
        scratch_parent = _resolve_config_path(config_directory, get(mapping, "scratch_parent", dirname(snapshot)), "integrals.scratch_parent")
        molcas_value = get(mapping, "molcas_root", get(ENV, "MOLCAS", ""))
        omc_value = get(mapping, "omc_root", get(ENV, "OMC", ""))
        isempty(strip(String(molcas_value))) && throw(ArgumentError("set integrals.molcas_root or the MOLCAS environment variable"))
        isempty(strip(String(omc_value))) && throw(ArgumentError("set integrals.omc_root or the OMC environment variable"))
        molcas_root = _resolve_config_path(config_directory, molcas_value, "integrals.molcas_root")
        omc_root = _resolve_config_path(config_directory, omc_value, "integrals.omc_root")
        transform_backend = Symbol(lowercase(_yaml_string(get(mapping, "transform_backend", "cpu"), "integrals.transform_backend")))
        source = SewardMOTRASource(
            snapshot=snapshot,
            project=_yaml_string(_required_yaml(mapping, "project", "integrals"), "integrals.project"),
            ranks=_yaml_integer(_required_yaml(mapping, "ranks", "integrals"), "integrals.ranks"),
            commonorb=commonorb,
            n_frozen=_yaml_integer(get(mapping, "n_frozen", 0), "integrals.n_frozen"),
            n_deleted=_yaml_integer(_required_yaml(mapping, "n_deleted", "integrals"), "integrals.n_deleted"),
            batch_size=_yaml_integer(get(mapping, "batch_size", 8), "integrals.batch_size"),
            transform_backend=transform_backend,
            gpu_device=_yaml_integer(get(mapping, "gpu_device", 0), "integrals.gpu_device"),
            scratch_parent=scratch_parent,
            molcas_root=molcas_root,
            omc_root=omc_root,
        )
        return FactorizedERI(one_path, source)
    end
    throw(ArgumentError("integrals.backend must be packed_two or factorized"))
end

"""
    _parse_screening_options(T, mapping)

Parse dimensionless CI, off-diagonal CI, and singularity thresholds with stable defaults.
The mapping is read-only, no scientific storage or CUDA state is allocated, and unknown, negative, or non-finite values raise `ArgumentError`.
"""
function _parse_screening_options(::Type{T}, mapping::AbstractDict) where {T<:AbstractFloat}
    all(key -> key in ("tau_ci", "tau_ci_offdiagonal", "tau_singularity"), keys(mapping)) || throw(ArgumentError("screening contains an unsupported option"))
    return ScreeningOptions(
        _yaml_float(T, get(mapping, "tau_ci", 1.0e-6), "screening.tau_ci"),
        _yaml_float(T, get(mapping, "tau_ci_offdiagonal", 1.0e-5), "screening.tau_ci_offdiagonal"),
        _yaml_float(T, get(mapping, "tau_singularity", 1.0e-12), "screening.tau_singularity"),
    )
end

"""
    _parse_runtime_options(mapping)

Parse resource-only YAML runtime settings into `RuntimeOptions` without initializing CUDA.
The mapping is read-only, byte values use exact integers, unknown keys or invalid types raise `ArgumentError`, and construction allocates no molecular-scale storage.
"""
function _parse_runtime_options(mapping::AbstractDict)
    allowed = ("use_gpu", "gpu_device", "gpu_devices", "max_host_bytes", "max_device_bytes", "scratch_bytes", "gpu_intermediate_scratch_bytes_per_device", "contraction_host_scratch_bytes", "max_active_occupied", "collect_allocations")
    all(key -> key in allowed, keys(mapping)) || throw(ArgumentError("runtime contains an unsupported option"))
    max_host_bytes = haskey(mapping, "max_host_bytes") ? _yaml_integer(mapping["max_host_bytes"], "runtime.max_host_bytes") : nothing
    max_device_bytes = haskey(mapping, "max_device_bytes") ? _yaml_integer(mapping["max_device_bytes"], "runtime.max_device_bytes") : nothing
    gpu_device = haskey(mapping, "gpu_device") ? _yaml_integer(mapping["gpu_device"], "runtime.gpu_device") : nothing
    gpu_devices = haskey(mapping, "gpu_devices") ? Tuple(
        _yaml_integer(value, "runtime.gpu_devices[$index]")
        for (index, value) in pairs(_yaml_sequence(mapping["gpu_devices"], "runtime.gpu_devices"))
    ) : nothing
    return RuntimeOptions(
        use_gpu=_yaml_boolean(get(mapping, "use_gpu", false), "runtime.use_gpu"),
        gpu_device=gpu_device,
        gpu_devices=gpu_devices,
        max_host_bytes=max_host_bytes,
        max_device_bytes=max_device_bytes,
        scratch_bytes=_yaml_integer(get(mapping, "scratch_bytes", 256 * 1024^2), "runtime.scratch_bytes"),
        gpu_intermediate_scratch_bytes_per_device=haskey(mapping, "gpu_intermediate_scratch_bytes_per_device") ? _yaml_integer(mapping["gpu_intermediate_scratch_bytes_per_device"], "runtime.gpu_intermediate_scratch_bytes_per_device") : nothing,
        contraction_host_scratch_bytes=haskey(mapping, "contraction_host_scratch_bytes") ? _yaml_integer(mapping["contraction_host_scratch_bytes"], "runtime.contraction_host_scratch_bytes") : nothing,
        max_active_occupied=_yaml_integer(get(mapping, "max_active_occupied", AHPC_MAX_OCCUPIED_RANK), "runtime.max_active_occupied"),
        collect_allocations=_yaml_boolean(get(mapping, "collect_allocations", true), "runtime.collect_allocations"),
    )
end

"""
    _validate_input_structure(label, spin_multiplicity, algorithm, backend, fragment_states, mebfs)

Validate unique labels, fragment-role ordering, spin-coupling metadata, and algorithm/backend compatibility and return `nothing`.
Inputs are read-only, no scientific files or CUDA state are touched, and structural inconsistencies raise `ArgumentError` or `UnsupportedERIBackend`.
"""
function _validate_input_structure(
    label::AbstractString,
    spin_multiplicity::Integer,
    algorithm::A,
    backend::B,
    fragment_states::AbstractVector{FragmentStateSpecification},
    mebfs::AbstractVector{MEBFDefinition},
) where {A<:NOCIAlgorithm,B<:ERIBackend}
    isempty(strip(label)) && throw(ArgumentError("system.label must not be empty"))
    spin_multiplicity >= 1 || throw(ArgumentError("system.spin_multiplicity must be positive"))
    isempty(fragment_states) && throw(ArgumentError("system.fragment_states must not be empty"))
    isempty(mebfs) && throw(ArgumentError("system.mebfs must not be empty"))
    state_labels = [state.label for state in fragment_states]
    allunique(state_labels) || throw(ArgumentError("fragment-state labels must be unique"))
    mebf_labels = [mebf.label for mebf in mebfs]
    allunique(mebf_labels) || throw(ArgumentError("MEBF labels must be unique"))
    state_by_label = Dict(state.label => state for state in fragment_states)
    fragment_order = unique([state.fragment for state in fragment_states])
    n_fragments = length(fragment_order)
    for mebf in mebfs
        length(mebf.state_labels) == n_fragments || throw(ArgumentError("MEBF $(mebf.label) must contain exactly $n_fragments fragment states"))
        for (position, state_label) in pairs(mebf.state_labels)
            haskey(state_by_label, state_label) || throw(ArgumentError("MEBF $(mebf.label) references unknown fragment state $state_label"))
            state_by_label[state_label].fragment == fragment_order[position] || throw(ArgumentError("MEBF $(mebf.label) position $position must reference fragment $(fragment_order[position])"))
        end
        if !isempty(mebf.coupling_multiplicities)
            n_fragments >= 2 || throw(ArgumentError("MEBF $(mebf.label) explicit coupling multiplicities require at least two fragments"))
            allowed_coupling_counts = (n_fragments - 2, n_fragments - 1)
            length(mebf.coupling_multiplicities) in allowed_coupling_counts || throw(ArgumentError("MEBF $(mebf.label) has an invalid number of coupling multiplicities"))
            last(mebf.coupling_multiplicities) != spin_multiplicity && length(mebf.coupling_multiplicities) == n_fragments - 1 && throw(ArgumentError("MEBF $(mebf.label) final coupling multiplicity must match system.spin_multiplicity"))
        end
    end
    validate_backend(algorithm, backend)
    return nothing
end

"""
    read_noci_input(path; T=Float64)

Read the clean repository-relative YAML schema and return a validated `NOCIInput` without loading scientific data files.
The YAML file is read-only, relative paths resolve against its directory, CUDA and OpenMolcas are never initialized during parsing, and malformed schemas or unsupported DP-NOCI/direct-factor-bundle combinations raise before numerical allocation.
"""
function read_noci_input(path::AbstractString; T::Type{S}=Float64) where {S<:AbstractFloat}
    isfile(path) || throw(ArgumentError("NOCI YAML input not found: $path"))
    raw = YAML.load_file(path)
    root = _yaml_mapping(raw, "root")
    all(key -> key in ("algorithm", "system", "integrals", "screening", "runtime"), keys(root)) || throw(ArgumentError("root YAML mapping contains an unsupported section"))
    algorithm_mapping = _yaml_mapping(_required_yaml(root, "algorithm", "root"), "algorithm")
    system = _yaml_mapping(_required_yaml(root, "system", "root"), "system")
    integrals = _yaml_mapping(_required_yaml(root, "integrals", "root"), "integrals")
    screening_mapping = _yaml_mapping(get(root, "screening", Dict{String,Any}()), "screening")
    runtime_mapping = _yaml_mapping(get(root, "runtime", Dict{String,Any}()), "runtime")
    all(key -> key in ("label", "spin_multiplicity", "fragment_states", "mebfs"), keys(system)) || throw(ArgumentError("system contains an unsupported option"))
    config_directory = dirname(abspath(path))
    fragment_state_values = _yaml_sequence(_required_yaml(system, "fragment_states", "system"), "system.fragment_states")
    fragment_states = FragmentStateSpecification[]
    for (index, value) in pairs(fragment_state_values)
        location = "system.fragment_states[$index]"
        mapping = _yaml_mapping(value, location)
        all(key -> key in ("label", "fragment", "state", "determinants", "orbitals"), keys(mapping)) || throw(ArgumentError("$location contains an unsupported option"))
        push!(
            fragment_states,
            FragmentStateSpecification(
                _yaml_string(_required_yaml(mapping, "label", location), "$location.label"),
                _yaml_string(_required_yaml(mapping, "fragment", location), "$location.fragment"),
                _yaml_string(_required_yaml(mapping, "state", location), "$location.state"),
                _resolve_config_path(config_directory, _required_yaml(mapping, "determinants", location), "$location.determinants"),
                _resolve_config_path(config_directory, _required_yaml(mapping, "orbitals", location), "$location.orbitals"),
            ),
        )
    end
    mebf_values = _yaml_sequence(_required_yaml(system, "mebfs", "system"), "system.mebfs")
    mebfs = MEBFDefinition[]
    for (index, value) in pairs(mebf_values)
        location = "system.mebfs[$index]"
        mapping = _yaml_mapping(value, location)
        all(key -> key in ("label", "states", "coupling_multiplicities", "diagonal_energy_shift"), keys(mapping)) || throw(ArgumentError("$location contains an unsupported option"))
        state_values = _yaml_sequence(_required_yaml(mapping, "states", location), "$location.states")
        coupling_values = _yaml_sequence(get(mapping, "coupling_multiplicities", Any[]), "$location.coupling_multiplicities")
        push!(
            mebfs,
            MEBFDefinition(
                _yaml_string(_required_yaml(mapping, "label", location), "$location.label"),
                [_yaml_string(state, "$location.states[$state_index]") for (state_index, state) in pairs(state_values)];
                coupling_multiplicities=[_yaml_integer(value, "$location.coupling_multiplicities[$coupling_index]") for (coupling_index, value) in pairs(coupling_values)],
                diagonal_energy_shift=_yaml_float(S, get(mapping, "diagonal_energy_shift", 0.0), "$location.diagonal_energy_shift"),
            ),
        )
    end
    algorithm = _parse_noci_algorithm(algorithm_mapping)
    backend = _parse_eri_backend(integrals, config_directory)
    screening = _parse_screening_options(S, screening_mapping)
    runtime = _parse_runtime_options(runtime_mapping)
    label = _yaml_string(_required_yaml(system, "label", "system"), "system.label")
    spin_multiplicity = _yaml_integer(_required_yaml(system, "spin_multiplicity", "system"), "system.spin_multiplicity")
    return NOCIInput(label, spin_multiplicity, algorithm, backend, screening, runtime, fragment_states, mebfs)
end

"""
    _orbital_dimension(path)

Read only the first line of a text `.vec` file and return its positive square orbital dimension.
The file is read-only, no matrix or CUDA storage is allocated, and missing, empty, or malformed headers raise `ArgumentError`.
"""
function _orbital_dimension(path::AbstractString)
    isfile(path) || throw(ArgumentError("orbital coefficient file not found: $path"))
    return open(path, "r") do io
        eof(io) && throw(ArgumentError("empty orbital coefficient file: $path"))
        fields = split(strip(readline(io)))
        length(fields) == 1 || throw(ArgumentError("orbital coefficient header must contain exactly one dimension in $path"))
        n_orbitals = parse(Int, fields[1])
        n_orbitals >= 1 || throw(ArgumentError("orbital coefficient dimension must be positive in $path"))
        return n_orbitals
    end
end

"""
    _determinant_file_shape(path)

Read only a determinant-file header and its first nonempty determinant record, returning the declared record count and active occupation width.
The source is read-only, no coefficient or occupation collection is materialized, CUDA is not touched, and missing, empty, malformed, or record-free files raise `ArgumentError`.
"""
function _determinant_file_shape(path::AbstractString)
    isfile(path) || throw(ArgumentError("determinant file not found: $path"))
    return open(path, "r") do io
        eof(io) && throw(ArgumentError("empty determinant file: $path"))
        header = split(strip(readline(io)))
        length(header) >= 8 || throw(ArgumentError("invalid determinant header in $path"))
        n_records = parse(Int, header[3])
        n_records >= 1 || throw(ArgumentError("determinant record count must be positive in $path"))
        while !eof(io)
            text = strip(readline(io))
            isempty(text) && continue
            fields = split(text)
            length(fields) >= 2 || throw(ArgumentError("invalid determinant record in $path"))
            return n_records, length(fields[end])
        end
        throw(ArgumentError("no determinant records found in $path"))
    end
end

"""
    problem_definition_memory(input; T=Float64)

Estimate peak host numerical payload in bytes for retaining every configured fragment wavefunction while all configured states may parse concurrently; device bytes are zero.
The function reads only `.det` headers/first records and `.vec` headers, does not load scientific matrices or initialize CUDA, and propagates malformed-file and integer-overflow errors.
"""
function problem_definition_memory(input::NOCIInput; T::Type{S}=Float64) where {S<:AbstractFloat}
    retained_bytes = 0
    concurrent_orbital_copy_bytes = 0
    for specification in input.fragment_states
        n_records, _ = _determinant_file_shape(specification.determinant_path)
        retained_bytes = Base.Checked.checked_add(retained_bytes, estimate_array_bytes(S, n_records))
        retained_bytes = Base.Checked.checked_add(retained_bytes, estimate_array_bytes(UInt64, 2, n_records))
        n_orbitals = _orbital_dimension(specification.orbital_path)
        orbital_bytes = estimate_array_bytes(S, n_orbitals, n_orbitals)
        retained_bytes = Base.Checked.checked_add(retained_bytes, orbital_bytes)
        concurrent_orbital_copy_bytes = Base.Checked.checked_add(concurrent_orbital_copy_bytes, orbital_bytes)
    end
    return MemoryEstimate(Base.Checked.checked_add(retained_bytes, concurrent_orbital_copy_bytes), 0)
end

"""
    _load_fragment_state_entries(input, T, runtime)

Load all configured fragment wavefunctions concurrently on Julia threads and return entries plus their deterministic label-to-index mapping in input order.
Scientific files are read-only, each task owns its parser storage, CUDA is never initialized, and any task failure propagates after all launched tasks are joined.
"""
function _load_fragment_state_entries(
    input::NOCIInput,
    ::Type{T},
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    entries = Vector{FragmentStateEntry{T}}(undef, length(input.fragment_states))
    @sync for (index, specification) in pairs(input.fragment_states)
        Threads.@spawn begin
            wavefunction = read_fragment_wavefunction(
                specification.determinant_path,
                specification.orbital_path;
                T=T,
                runtime=runtime,
            )
            entries[index] = FragmentStateEntry(specification, wavefunction)
        end
    end
    state_index_by_label = Dict{String,Int}(entry.specification.label => index for (index, entry) in pairs(entries))
    return entries, state_index_by_label
end

"""
    _validate_loaded_fragment_states(input, entries, state_index_by_label)

Resolve MEBF state labels, expand empty coupling specifications into complete target-spin manifolds, and validate determinant state names, per-fragment basis/domain consistency, molecular basis size, and conserved total electron count.
Inputs and numerical arrays are read-only, only small spin-coupling/composition metadata is allocated, CUDA is not touched, and inconsistent scientific data raise `ArgumentError` or `DimensionMismatch`.
"""
function _validate_loaded_fragment_states(
    input::NOCIInput,
    entries::AbstractVector{FragmentStateEntry{T}},
    state_index_by_label::AbstractDict{String,Int},
) where {T<:AbstractFloat}
    fragment_order = unique([entry.specification.fragment for entry in entries])
    basis_dimension_by_fragment = Dict{String,Int}()
    orbital_domain_by_fragment = Dict{String,Tuple{Int,Int}}()
    for entry in entries
        specification = entry.specification
        wavefunction = entry.wavefunction
        wavefunction.determinants.label == specification.state || throw(ArgumentError("fragment state $(specification.label) expected .det state $(specification.state), got $(wavefunction.determinants.label)"))
        n_fragment_basis = size(wavefunction.orbitals.coefficients, 1)
        size(wavefunction.orbitals.coefficients, 2) == n_fragment_basis || throw(DimensionMismatch("fragment state $(specification.label) orbital coefficients are not square"))
        previous_basis = get(basis_dimension_by_fragment, specification.fragment, n_fragment_basis)
        previous_basis == n_fragment_basis || throw(DimensionMismatch("fragment $(specification.fragment) states use inconsistent basis dimensions"))
        basis_dimension_by_fragment[specification.fragment] = n_fragment_basis
        expansion = wavefunction.determinants.expansion
        domain = (expansion.inactive_orbitals, active_orbital_count(expansion))
        previous = get(orbital_domain_by_fragment, specification.fragment, domain)
        previous == domain || throw(DimensionMismatch("fragment $(specification.fragment) states use inconsistent inactive/active orbital domains"))
        orbital_domain_by_fragment[specification.fragment] = domain
    end
    compositions = MEBFComposition[]
    total_electrons = nothing
    for definition in input.mebfs
        indices = [state_index_by_label[label] for label in definition.state_labels]
        electron_count = sum(entries[index].wavefunction.determinants.n_electrons for index in indices)
        if total_electrons === nothing
            total_electrons = electron_count
        else
            electron_count == total_electrons || throw(ArgumentError("MEBF $(definition.label) has total electron count $electron_count, expected $total_electrons"))
        end
        fragment_spin_multiplicities = [
            entries[index].wavefunction.determinants.spin_multiplicity
            for index in indices
        ]
        descriptors = _spin_coupled_basis_descriptors(
            definition.label,
            fragment_spin_multiplicities,
            input.spin_multiplicity,
            definition.coupling_multiplicities,
        )
        for descriptor in descriptors
            push!(
                compositions,
                MEBFComposition(
                    descriptor.label,
                    indices,
                    descriptor.coupling_multiplicities,
                ),
            )
        end
    end
    composition_labels = [composition.label for composition in compositions]
    allunique(composition_labels) || throw(ArgumentError("resolved spin-coupled MEBF labels must be unique"))
    fragment_basis_dimensions = [basis_dimension_by_fragment[fragment] for fragment in fragment_order]
    n_basis = sum(fragment_basis_dimensions)
    return fragment_order, compositions, fragment_basis_dimensions, n_basis, something(total_electrons)
end

"""
    load_problem_definition(input; T=Float64, runtime=input.runtime, telemetry=nothing)

Load configured `.det/.vec` fragment states concurrently, resolve MEBF compositions in input order, and return a validated CPU `NOCIProblemDefinition` ready for molecular determinant assembly.
The operation is read-only on source files, performs a concurrent whole-pool numerical memory preflight before orbital loading, optionally records one telemetry phase, never initializes CUDA, and raises before return on resource or scientific-consistency failures.
"""
function load_problem_definition(
    input::NOCIInput;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=input.runtime,
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {S<:AbstractFloat}
    estimate = problem_definition_memory(input; T=S)
    enforce_memory_limits(estimate, runtime)
    work = function ()
        entries, state_index_by_label = _load_fragment_state_entries(input, S, runtime)
        fragment_order, compositions, fragment_basis_dimensions, n_basis, n_electrons = _validate_loaded_fragment_states(input, entries, state_index_by_label)
        return NOCIProblemDefinition(
            input.label,
            input.spin_multiplicity,
            fragment_order,
            entries,
            state_index_by_label,
            compositions,
            fragment_basis_dimensions,
            n_basis,
            n_electrons,
        )
    end
    isnothing(telemetry) && return work()
    return record_phase!(
        work,
        telemetry,
        :load_problem_definition;
        estimate=estimate,
        metadata=Dict(:n_fragment_states => length(input.fragment_states), :n_mebfs => length(input.mebfs), :fragment_load_tasks => length(input.fragment_states), :julia_threads => Threads.nthreads()),
        collect_allocations=runtime.collect_allocations,
    )
end
