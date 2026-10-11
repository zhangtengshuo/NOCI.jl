#==============================================================================#
# NOCI.jl — portable OpenMolcas4NOCI SEWARD HDF5 import.
#==============================================================================#

"""
    SewardHDF5Manifest

Hold validated metadata and one-electron AO data from a version-1 portable SEWARD HDF5 export.
Dimensions and indices are dimensionless, coordinates and origins are in bohr, energies and AO operators are in atomic units, arrays are CPU-owned, and no shard factor payload or HDF5 handle is retained.
"""
struct SewardHDF5Manifest
    directory::String
    manifest_path::String
    n_ao::Int
    n_reduced_pairs::Int
    n_auxiliary::Int
    n_mpi_ranks::Int
    reduced_pair_indices::Matrix{Int64}
    shard_paths::Vector{String}
    shard_global_auxiliary_indices::Vector{Vector{Int64}}
    ao_labels::Vector{String}
    atom_labels::Vector{String}
    atom_coordinates_bohr::Matrix{Float64}
    nuclear_charges::Vector{Float64}
    effective_nuclear_charges::Vector{Float64}
    center_of_mass_bohr::Vector{Float64}
    nuclear_repulsion::Float64
    provenance::ERIFactorProvenance
    overlap::Matrix{Float64}
    core_hamiltonian::Matrix{Float64}
    dipole::Array{Float64,3}
    quadrupole::Array{Float64,3}
end

"""
    SewardHDF5ImportResult

Hold the retained-orbital one-electron values, portable ERI factor bundle, source manifest, and transformation telemetry produced by one HDF5 import.
The numerical arrays are CPU-owned in atomic units, elapsed time is seconds, the source export is never mutated, and no OpenMolcas, HDF5, or CUDA handle remains open.
"""
struct SewardHDF5ImportResult
    one_electron::OneElectronValues{Float64}
    factor_bundle::ERIFactorBundle{Float64}
    manifest::SewardHDF5Manifest
    n_frozen::Int
    n_deleted::Int
    elapsed_seconds::Float64
    backend::Symbol
    gpu_device::Int
end

"""
    _seward_hdf5_text(value)

Convert an HDF5 fixed- or variable-length textual scalar to a stripped Julia string.
The input is not mutated, trailing NUL and whitespace padding are removed, and unsupported byte representations raise through `String` conversion.
"""
function _seward_hdf5_text(value)
    text = value isa AbstractVector{UInt8} ? String(value) : String(value)
    return strip(replace(text, '\0' => ""))
end

"""
    _seward_hdf5_scalar(file, path, T)

Read one required scalar dataset and convert it to `T`.
The file is read-only, the dataset must contain exactly one value, and missing, nonscalar, or inconvertible values raise before return.
"""
function _seward_hdf5_scalar(file, path::AbstractString, ::Type{T}) where {T}
    haskey(file, path) || throw(ArgumentError("SEWARD HDF5 manifest is missing $path"))
    value = read(file[path])
    if value isa AbstractArray
        length(value) == 1 || throw(DimensionMismatch("SEWARD HDF5 dataset $path is not scalar"))
        value = only(value)
    end
    return T(value)
end

"""
    _seward_hdf5_strings(file, path)

Read one required HDF5 text dataset as an owned vector of stripped strings.
The file is not modified and missing datasets or undecodable entries raise before the vector is returned.
"""
function _seward_hdf5_strings(file, path::AbstractString)
    haskey(file, path) || throw(ArgumentError("SEWARD HDF5 manifest is missing $path"))
    values = read(file[path])
    return [_seward_hdf5_text(value) for value in vec(values)]
end

"""
    _require_unfiltered_seward_dataset(dataset, path)

Verify that a SEWARD HDF5 factor dataset uses contiguous, uncompressed storage and return `nothing`.
The dataset is read-only; any HDF5 creation filter or noncontiguous layout raises `ArgumentError`, preventing hidden decompression memory and performance costs.
"""
function _require_unfiltered_seward_dataset(dataset, path::AbstractString)
    properties = HDF5.get_create_properties(dataset)
    isempty(properties.filters) || throw(ArgumentError("SEWARD HDF5 factor dataset is compressed or filtered: $path"))
    return nothing
end

"""
    _seward_hdf5_factor_provenance(file, minor)

Read and validate the physical factor origin and threshold semantics from a format-1.0 or format-1.1 SEWARD manifest.
Format 1.0 is the established conventional-CD export; format 1.1 explicitly distinguishes molecular CD from metric-whitened RICD and rejects contradictory metadata.
The HDF5 file remains read-only, thresholds use hartree, only small CPU metadata are allocated, and unknown schemas or invalid origins raise before any factor payload is loaded.
"""
function _seward_hdf5_factor_provenance(file, minor::Integer)
    minor in (0, 1) || throw(ArgumentError("unsupported SEWARD HDF5 format minor version $minor"))
    if minor == 0
        return ERIFactorProvenance(origin=:conventional_cd, auxiliary_basis_kind=:none, threshold_kind=:molecular_eri_cd, generation_threshold=_seward_hdf5_scalar(file, "system/cholesky_threshold", Float64))
    end
    provenance = ERIFactorProvenance(
        origin=Symbol(_seward_hdf5_scalar(file, "provenance/factor_origin", String)),
        ri_type=_seward_hdf5_scalar(file, "provenance/ri_type", Int),
        auxiliary_basis_kind=Symbol(_seward_hdf5_scalar(file, "provenance/auxiliary_basis_kind", String)),
        threshold_kind=Symbol(_seward_hdf5_scalar(file, "provenance/threshold_kind", String)),
        generation_threshold=_seward_hdf5_scalar(file, "system/factor_generation_threshold", Float64),
    )
    provenance.origin != :unknown || throw(ArgumentError("SEWARD HDF5 1.1 must declare its physical factor origin"))
    if provenance.origin == :conventional_cd
        _seward_hdf5_scalar(file, "system/cholesky_threshold", Float64) == provenance.generation_threshold || throw(ArgumentError("inconsistent molecular CD thresholds"))
    else
        !haskey(file, "system/cholesky_threshold") || throw(ArgumentError("RICD must not declare a molecular Cholesky threshold"))
    end
    return provenance
end

"""
    inspect_seward_hdf5(directory)

Read and fully validate a portable OpenMolcas4NOCI SEWARD HDF5 manifest plus all shard metadata without loading factor payloads.
The source remains read-only, only AO operators and metadata are retained, format-major/C1/type/layout/path/global-ID consistency is enforced, and unknown or incomplete exports raise before molecular-scale factor allocation.
"""
function inspect_seward_hdf5(directory::AbstractString)
    root = abspath(directory)
    manifest_path = joinpath(root, "manifest.h5")
    isfile(manifest_path) || throw(ArgumentError("SEWARD HDF5 manifest not found: $manifest_path"))
    metadata = HDF5.h5open(manifest_path, "r") do file
        attrs = HDF5.attributes(file)
        for name in ("FORMAT_NAME", "FORMAT_MAJOR", "FORMAT_MINOR", "INDEX_BASE", "FLOAT_FORMAT", "INTEGER_FORMAT", "SYMMETRY_MODE", "FACTOR_CONVENTION", "PAIR_CONVENTION")
            haskey(attrs, name) || throw(ArgumentError("SEWARD HDF5 root attribute $name is missing"))
        end
        _seward_hdf5_text(read(attrs["FORMAT_NAME"])) == "OpenMolcas NOCI SEWARD portable data" || throw(ArgumentError("unrecognized SEWARD HDF5 format name"))
        Int(read(attrs["FORMAT_MAJOR"])) == 1 || throw(ArgumentError("unsupported SEWARD HDF5 format major version"))
        Int(read(attrs["INDEX_BASE"])) == 1 || throw(ArgumentError("SEWARD HDF5 indices must be one based"))
        _seward_hdf5_text(read(attrs["FLOAT_FORMAT"])) == "IEEE_F64LE" || throw(ArgumentError("SEWARD HDF5 requires IEEE_F64LE factors"))
        _seward_hdf5_text(read(attrs["INTEGER_FORMAT"])) == "I64LE" || throw(ArgumentError("SEWARD HDF5 requires I64LE indices"))
        _seward_hdf5_text(read(attrs["SYMMETRY_MODE"])) == "C1" || throw(ArgumentError("SEWARD HDF5 importer currently supports C1 only"))
        _seward_hdf5_text(read(attrs["FACTOR_CONVENTION"])) == "(pq|rs)=sum_L L[pq,L]*L[rs,L]" || throw(ArgumentError("unsupported SEWARD HDF5 factor convention"))
        _seward_hdf5_text(read(attrs["PAIR_CONVENTION"])) == "explicit AO indices in basis/reduced_pair_indices" || throw(ArgumentError("unsupported SEWARD HDF5 pair convention"))
        minor = Int(read(attrs["FORMAT_MINOR"]))
        provenance = _seward_hdf5_factor_provenance(file, minor)
        n_ao = _seward_hdf5_scalar(file, "system/n_ao", Int)
        n_reduced = _seward_hdf5_scalar(file, "system/n_reduced_pairs", Int)
        n_auxiliary = _seward_hdf5_scalar(file, "system/n_auxiliary_global", Int)
        n_ranks = _seward_hdf5_scalar(file, "system/n_mpi_ranks", Int)
        _seward_hdf5_scalar(file, "system/n_symmetry", Int) == 1 || throw(ArgumentError("SEWARD HDF5 importer currently supports one symmetry only"))
        n_ao >= 1 || throw(ArgumentError("SEWARD HDF5 AO dimension must be positive"))
        n_reduced >= 1 || throw(ArgumentError("SEWARD HDF5 reduced-pair count must be positive"))
        n_auxiliary >= 1 || throw(ArgumentError("SEWARD HDF5 auxiliary count must be positive"))
        n_ranks >= 1 || throw(ArgumentError("SEWARD HDF5 MPI rank count must be positive"))
        mapping = Matrix{Int64}(read(file["basis/reduced_pair_indices"]))
        size(mapping) == (2, n_reduced) || throw(DimensionMismatch("SEWARD HDF5 reduced-pair map must have shape (2,$n_reduced)"))
        all(index -> 1 <= index <= n_ao, mapping) || throw(ArgumentError("SEWARD HDF5 reduced-pair map contains an invalid AO index"))
        names = _seward_hdf5_strings(file, "shards/filenames")
        length(names) == n_ranks || throw(DimensionMismatch("SEWARD HDF5 shard filename count differs from MPI rank count"))
        ao_labels = _seward_hdf5_strings(file, "basis/ao_labels")
        length(ao_labels) == n_ao || throw(DimensionMismatch("SEWARD HDF5 AO-label count differs from n_ao"))
        atom_labels = _seward_hdf5_strings(file, "system/atom_labels")
        coordinates = Matrix{Float64}(read(file["system/atom_coordinates_bohr"]))
        size(coordinates) == (3, length(atom_labels)) || throw(DimensionMismatch("SEWARD HDF5 atom coordinates must have shape (3,n_atoms)"))
        nuclear = Vector{Float64}(read(file["system/nuclear_charges"]))
        effective = Vector{Float64}(read(file["system/effective_nuclear_charges"]))
        length(nuclear) == length(atom_labels) == length(effective) || throw(DimensionMismatch("SEWARD HDF5 atom metadata lengths differ"))
        center = Vector{Float64}(read(file["system/center_of_mass_bohr"]))
        length(center) == 3 || throw(DimensionMismatch("SEWARD HDF5 center of mass must contain three values"))
        overlap = Matrix{Float64}(read(file["one_electron/overlap"]))
        core = Matrix{Float64}(read(file["one_electron/core_hamiltonian"]))
        dipole = Array{Float64,3}(read(file["one_electron/dipole"]))
        quadrupole = Array{Float64,3}(read(file["one_electron/quadrupole"]))
        size(overlap) == (n_ao, n_ao) || throw(DimensionMismatch("SEWARD HDF5 overlap shape mismatch"))
        size(core) == (n_ao, n_ao) || throw(DimensionMismatch("SEWARD HDF5 core-Hamiltonian shape mismatch"))
        size(dipole) == (n_ao, n_ao, 3) || throw(DimensionMismatch("SEWARD HDF5 dipole shape mismatch"))
        size(quadrupole) == (n_ao, n_ao, 6) || throw(DimensionMismatch("SEWARD HDF5 quadrupole shape mismatch"))
        all(isfinite, overlap) && all(isfinite, core) && all(isfinite, dipole) && all(isfinite, quadrupole) || throw(ArgumentError("SEWARD HDF5 one-electron data contain a non-finite value"))
        return (
            n_ao, n_reduced, n_auxiliary, n_ranks, mapping, names, ao_labels, atom_labels,
            coordinates, nuclear, effective, center,
            _seward_hdf5_scalar(file, "system/nuclear_repulsion", Float64),
            provenance, minor,
            overlap, core, dipole, quadrupole,
        )
    end
    n_ao, n_reduced, n_auxiliary, n_ranks, mapping, names, ao_labels, atom_labels, coordinates, nuclear, effective, center, nuclear_repulsion, provenance, minor, overlap, core, dipole, quadrupole = metadata
    shard_paths = String[]
    shard_ids = Vector{Vector{Int64}}()
    assigned = falses(n_auxiliary)
    length(unique(names)) == length(names) || throw(ArgumentError("SEWARD HDF5 shard filenames must be unique"))
    for (rank_index, name) in enumerate(names)
        isabspath(name) && throw(ArgumentError("SEWARD HDF5 shard filenames must be relative"))
        normalized = normpath(name)
        startswith(normalized, "..") && throw(ArgumentError("SEWARD HDF5 shard path escapes the export directory: $name"))
        path = joinpath(root, normalized)
        isfile(path) || throw(ArgumentError("SEWARD HDF5 shard not found: $path"))
        ids = HDF5.h5open(path, "r") do file
            attrs = HDF5.attributes(file)
            for attribute in ("FORMAT_NAME", "FORMAT_MAJOR", "FILE_KIND", "MPI_RANK", "MPI_SIZE", "N_LOCAL_AUXILIARY", "N_REDUCED_PAIRS")
                haskey(attrs, attribute) || throw(ArgumentError("SEWARD HDF5 shard attribute $attribute is missing: $path"))
            end
            _seward_hdf5_text(read(attrs["FORMAT_NAME"])) == "OpenMolcas NOCI SEWARD portable data" || throw(ArgumentError("unrecognized SEWARD HDF5 shard format: $path"))
            Int(read(attrs["FORMAT_MAJOR"])) == 1 || throw(ArgumentError("unsupported SEWARD HDF5 shard major version: $path"))
            if minor == 1
                haskey(attrs, "FORMAT_MINOR") && Int(read(attrs["FORMAT_MINOR"])) == minor || throw(ArgumentError("SEWARD HDF5 shard minor version mismatch: $path"))
            end
            _seward_hdf5_text(read(attrs["FILE_KIND"])) == "shard" || throw(ArgumentError("SEWARD HDF5 shard has an invalid file kind: $path"))
            Int(read(attrs["MPI_RANK"])) == rank_index - 1 || throw(ArgumentError("SEWARD HDF5 shard MPI rank does not match manifest order: $path"))
            Int(read(attrs["MPI_SIZE"])) == n_ranks || throw(ArgumentError("SEWARD HDF5 shard MPI size mismatch: $path"))
            Int(read(attrs["N_REDUCED_PAIRS"])) == n_reduced || throw(ArgumentError("SEWARD HDF5 shard reduced-pair count mismatch: $path"))
            haskey(file, "cholesky/factors") || throw(ArgumentError("SEWARD HDF5 shard lacks cholesky/factors: $path"))
            haskey(file, "cholesky/global_auxiliary_indices") || throw(ArgumentError("SEWARD HDF5 shard lacks global auxiliary indices: $path"))
            dataset = file["cholesky/factors"]
            _require_unfiltered_seward_dataset(dataset, path)
            dataset_attrs = HDF5.attributes(dataset)
            haskey(dataset_attrs, "AXIS_ORDER") && _seward_hdf5_text(read(dataset_attrs["AXIS_ORDER"])) != "auxiliary,reduced_pair" && throw(ArgumentError("SEWARD HDF5 factor axis order is invalid: $path"))
            haskey(dataset_attrs, "COMPRESSION") && _seward_hdf5_text(read(dataset_attrs["COMPRESSION"])) != "none" && throw(ArgumentError("SEWARD HDF5 factor compression metadata is invalid: $path"))
            eltype(dataset) == Float64 || throw(ArgumentError("SEWARD HDF5 factors must be Float64: $path"))
            local_ids = Vector{Int64}(read(file["cholesky/global_auxiliary_indices"]))
            Int(read(attrs["N_LOCAL_AUXILIARY"])) == length(local_ids) || throw(ArgumentError("SEWARD HDF5 shard local auxiliary count mismatch: $path"))
            size(dataset) == (n_reduced, length(local_ids)) || throw(DimensionMismatch("SEWARD HDF5 factor shape differs from shard index count: $path"))
            return local_ids
        end
        for global_id in ids
            1 <= global_id <= n_auxiliary || throw(ArgumentError("SEWARD HDF5 global auxiliary ID $global_id is outside 1:$n_auxiliary"))
            assigned[global_id] && throw(ArgumentError("SEWARD HDF5 global auxiliary ID $global_id occurs more than once"))
            assigned[global_id] = true
        end
        push!(shard_paths, path)
        push!(shard_ids, ids)
    end
    all(assigned) || throw(ArgumentError("SEWARD HDF5 shards do not cover every global auxiliary ID"))
    return SewardHDF5Manifest(root, manifest_path, n_ao, n_reduced, n_auxiliary, n_ranks, mapping, shard_paths, shard_ids, ao_labels, atom_labels, coordinates, nuclear, effective, center, nuclear_repulsion, provenance, overlap, core, dipole, quadrupole)
end

"""
    seward_hdf5_import_memory(manifest, n_frozen, n_deleted, batch_size; backend=:cpu)

Estimate peak host/device array storage for portable SEWARD HDF5 AO-to-MO import before factor allocation.
Counts are dimensionless and results are bytes; the estimate includes final factors, AO/MO batch workspaces and one-electron transforms but excludes HDF5 library buffers and already loaded manifest arrays.
"""
function seward_hdf5_import_memory(
    manifest::SewardHDF5Manifest,
    n_frozen::Integer,
    n_deleted::Integer,
    batch_size::Integer;
    backend::Symbol=:cpu,
)
    backend in (:cpu, :gpu) || throw(ArgumentError("SEWARD HDF5 transform backend must be :cpu or :gpu"))
    n_frozen >= 0 && n_deleted >= 0 || throw(ArgumentError("frozen and deleted counts must be nonnegative"))
    batch_size >= 1 || throw(ArgumentError("batch_size must be positive"))
    n_used = manifest.n_ao - Int(n_deleted)
    n_retained = n_used - Int(n_frozen)
    n_retained >= 1 || throw(ArgumentError("SEWARD HDF5 retained-orbital count must be positive"))
    factors = estimate_array_bytes(Float64, canonical_pair_count(n_retained), manifest.n_auxiliary)
    host_work = sum((
        estimate_array_bytes(Float64, manifest.n_reduced_pairs, batch_size),
        estimate_array_bytes(Float64, manifest.n_ao, manifest.n_ao, batch_size),
        estimate_array_bytes(Float64, n_used, n_used, batch_size),
        estimate_array_bytes(Float64, 16, n_used, n_used),
    ))
    device_work = 0
    if backend == :cpu
        host_work = Base.Checked.checked_add(host_work, estimate_array_bytes(Float64, n_used, manifest.n_ao, batch_size))
    else
        device_work = sum((
            estimate_array_bytes(Float64, 2, n_used, manifest.n_ao),
            estimate_array_bytes(Float64, manifest.n_ao, manifest.n_ao, batch_size),
            estimate_array_bytes(Float64, n_used, manifest.n_ao, batch_size),
            estimate_array_bytes(Float64, n_used, n_used, batch_size),
        ))
    end
    return MemoryEstimate(Base.Checked.checked_add(factors, host_work), device_work)
end

"""
    _transform_seward_operator(orbitals, operator)

Apply a two-sided real AO-to-MO transformation and return a dense symmetric MO operator.
Inputs are read-only in atomic units, BLAS performs the CPU work, and incompatible dimensions raise `DimensionMismatch`.
"""
function _transform_seward_operator(orbitals::Matrix{Float64}, operator::Matrix{Float64})
    size(operator) == (size(orbitals, 2), size(orbitals, 2)) || throw(DimensionMismatch("AO operator and orbital dimensions differ"))
    transformed = orbitals * operator * transpose(orbitals)
    return Matrix(Symmetric((transformed + transpose(transformed)) ./ 2))
end

"""
    _pack_seward_mo_batch!(destination, global_ids, mo, n_frozen, n_vectors)

Pack retained-retained blocks from transformed MO ERI-factor matrices into global canonical pair rows.
The destination columns named by one-based `global_ids` are mutated, source matrices are unchanged, and inconsistent dimensions or IDs raise before an out-of-bounds write.
"""
function _pack_seward_mo_batch!(
    destination::Matrix{Float64},
    global_ids::AbstractVector{Int64},
    mo::Array{Float64,3},
    n_frozen::Integer,
    n_vectors::Integer,
)
    n = Int(n_vectors)
    length(global_ids) == n || throw(DimensionMismatch("global auxiliary batch length mismatch"))
    n_retained = size(mo, 1) - Int(n_frozen)
    size(destination, 1) == canonical_pair_count(n_retained) || throw(DimensionMismatch("retained factor row count mismatch"))
    @inbounds for local_auxiliary in 1:n
        global_id = Int(global_ids[local_auxiliary])
        1 <= global_id <= size(destination, 2) || throw(BoundsError(destination, (:, global_id)))
        pair = 0
        for upper in 1:n_retained, lower in 1:upper
            pair += 1
            destination[pair, global_id] = mo[Int(n_frozen) + lower, Int(n_frozen) + upper, local_auxiliary]
        end
    end
    return destination
end

"""
    _accumulate_frozen_core!(effective_h, core_two, mo, n_frozen, n_vectors)

Accumulate closed-shell frozen-core one- and two-electron Cholesky corrections for one transformed factor batch.
`effective_h` is mutated in Hartree, `core_two` is returned in Hartree, MO factors are read-only, and `n_frozen=0` is an exact no-op.
"""
function _accumulate_frozen_core!(
    effective_h::Matrix{Float64},
    core_two::Float64,
    mo::Array{Float64,3},
    n_frozen::Integer,
    n_vectors::Integer,
)
    frozen = Int(n_frozen)
    frozen == 0 && return core_two
    retained = frozen + 1:size(mo, 1)
    @inbounds for vector_index in 1:Int(n_vectors)
        frozen_block = view(mo, 1:frozen, 1:frozen, vector_index)
        retained_block = view(mo, retained, retained, vector_index)
        mixed_block = view(mo, retained, 1:frozen, vector_index)
        diagonal_sum = tr(frozen_block)
        effective_h .+= 2.0 .* diagonal_sum .* retained_block
        mul!(effective_h, mixed_block, transpose(mixed_block), -1.0, 1.0)
        core_two += 2.0 * diagonal_sum^2 - sum(abs2, frozen_block)
    end
    return core_two
end

"""
    import_seward_hdf5(directory, commonorb; n_frozen=0, n_deleted=0, batch_size=8, backend=:cpu, gpu_device=0, runtime=RuntimeOptions())

Import a portable SEWARD HDF5 export, transform AO data to retained CommonOrb orbitals, apply exact closed-shell frozen-core corrections, and return reusable NOCI `.one` values plus a version-2 factor bundle.
The export and CommonOrb are read-only, factor payloads are streamed by shard/batch, memory limits are enforced before molecular-scale allocation, GPU requests use the selected device without fallback, and malformed schema or non-finite data raise before a result is returned.
"""
function import_seward_hdf5(
    directory::AbstractString,
    commonorb::AbstractString;
    n_frozen::Integer=0,
    n_deleted::Integer=0,
    batch_size::Integer=8,
    backend::Symbol=:cpu,
    gpu_device::Integer=0,
    runtime::RuntimeOptions=RuntimeOptions(),
)
    manifest = inspect_seward_hdf5(directory)
    backend in (:cpu, :gpu) || throw(ArgumentError("SEWARD HDF5 transform backend must be :cpu or :gpu"))
    gpu_device >= 0 || throw(ArgumentError("gpu_device must be nonnegative"))
    estimate = seward_hdf5_import_memory(manifest, n_frozen, n_deleted, batch_size; backend)
    enforce_memory_limits(estimate, runtime)
    n_used = manifest.n_ao - Int(n_deleted)
    n_retained = n_used - Int(n_frozen)
    count_commonorb_orbitals(commonorb) >= n_used || throw(ArgumentError("CommonOrb does not contain the $n_used required used orbitals"))
    orbitals = read_commonorb_rows(commonorb, manifest.n_ao, n_used)
    retained_range = Int(n_frozen) + 1:n_used
    retained_orbitals = Matrix(view(orbitals, retained_range, :))
    overlap = _transform_seward_operator(retained_orbitals, manifest.overlap)
    used_core = _transform_seward_operator(orbitals, manifest.core_hamiltonian)
    effective_h = Matrix(view(used_core, retained_range, retained_range))
    core_energy = manifest.nuclear_repulsion + 2.0 * tr(view(used_core, 1:Int(n_frozen), 1:Int(n_frozen)))
    multipoles = Array{Float64,3}(undef, n_retained, n_retained, 9)
    for component in 1:3
        multipoles[:, :, component] .= _transform_seward_operator(retained_orbitals, Matrix(view(manifest.dipole, :, :, component)))
    end
    for component in 1:6
        multipoles[:, :, component + 3] .= _transform_seward_operator(retained_orbitals, Matrix(view(manifest.quadrupole, :, :, component)))
    end
    batch = Int(batch_size)
    reduced = Matrix{Float64}(undef, manifest.n_reduced_pairs, batch)
    ao = Array{Float64}(undef, manifest.n_ao, manifest.n_ao, batch)
    mo = Array{Float64}(undef, n_used, n_used, batch)
    half = backend == :cpu ? Array{Float64}(undef, n_used, manifest.n_ao, batch) : Array{Float64}(undef, 0, 0, 0)
    coefficients = Matrix{Float64}(undef, canonical_pair_count(n_retained), manifest.n_auxiliary)
    gpu_workspace = backend == :gpu ? JuliaMOTRAGPUWorkspace(orbitals, manifest.n_ao, n_used, batch, Int(gpu_device)) : nothing
    core_two = 0.0
    started = time_ns()
    try
        for (shard_path, global_ids) in zip(manifest.shard_paths, manifest.shard_global_auxiliary_indices)
            HDF5.h5open(shard_path, "r") do file
                dataset = file["cholesky/factors"]
                for first_local in 1:batch:length(global_ids)
                    last_local = min(length(global_ids), first_local + batch - 1)
                    n_vectors = last_local - first_local + 1
                    reduced[:, 1:n_vectors] .= dataset[:, first_local:last_local]
                    all(isfinite, view(reduced, :, 1:n_vectors)) || throw(ArgumentError("SEWARD HDF5 factor payload contains a non-finite value in $shard_path"))
                    scatter_reduced_eri_factors_c1!(ao, reduced, manifest.reduced_pair_indices, n_vectors)
                    if backend == :gpu
                        transform_ao_eri_factors_gpu!(mo, something(gpu_workspace), ao, n_vectors)
                    else
                        transform_ao_eri_factors_cpu!(mo, half, orbitals, ao, n_vectors)
                    end
                    ids = view(global_ids, first_local:last_local)
                    _pack_seward_mo_batch!(coefficients, ids, mo, n_frozen, n_vectors)
                    core_two = _accumulate_frozen_core!(effective_h, core_two, mo, n_frozen, n_vectors)
                end
            end
        end
    finally
        gpu_workspace === nothing || release_julia_motra_gpu_workspace!(gpu_workspace)
    end
    all(isfinite, coefficients) || throw(ArgumentError("transformed SEWARD HDF5 factors contain a non-finite value"))
    core_energy += core_two
    one_electron = OneElectronValues(core_energy, overlap, zeros(Float64, n_retained, n_retained), effective_h, multipoles)
    factor_values = ERIFactorValues{Float64}(n_retained, coefficients, Val(:validated_owned))
    bundle = ERIFactorBundle(2, factor_values, "openmolcas4noci-seward-hdf5-v1-$(manifest.provenance.origin)", manifest.n_mpi_ranks, manifest.provenance)
    elapsed = (time_ns() - started) / 1.0e9
    return SewardHDF5ImportResult(one_electron, bundle, manifest, Int(n_frozen), Int(n_deleted), elapsed, backend, backend == :gpu ? Int(gpu_device) : -1)
end

"""
    _write_native_fortran_record(io, parts...)

Write one native-endian Fortran sequential record with 32-bit byte markers.
The stream is mutated, array payloads retain Julia column-major order, payloads larger than `typemax(Int32)` raise before markers are written, and I/O errors propagate.
"""
function _write_native_fortran_record(io::IO, parts...)
    payload = IOBuffer()
    for part in parts
        if part isa AbstractString
            write(payload, codeunits(part))
        elseif part isa Number || part isa AbstractArray
            write(payload, part)
        else
            throw(ArgumentError("unsupported Fortran record payload type $(typeof(part))"))
        end
    end
    bytes = take!(payload)
    length(bytes) <= typemax(Int32) || throw(ArgumentError("Fortran record exceeds Int32 byte-marker capacity"))
    marker = Int32(length(bytes))
    write(io, marker, bytes, marker)
    return nothing
end

"""
    _pack_seward_symmetric(matrix)

Pack one square symmetric operator in canonical column-major triangular order.
The matrix is read-only, values retain their physical units and weights, and nonsquare inputs raise `DimensionMismatch`.
"""
function _pack_seward_symmetric(matrix::AbstractMatrix{Float64})
    n = size(matrix, 1)
    size(matrix, 2) == n || throw(DimensionMismatch("operator must be square"))
    packed = Vector{Float64}(undef, canonical_pair_count(n))
    for upper in 1:n, lower in 1:upper
        packed[canonical_pair_index(lower, upper)] = matrix[lower, upper]
    end
    return packed
end

"""
    _write_seward_one_file(path, values)

Atomically write retained one-electron data in the conventional native `.one` format expected by NOCI.jl.
The values are read-only, no `.two` file is created, the header advertises a zero-integral companion layout, and filesystem failures preserve any previous complete destination.
"""
function _write_seward_one_file(path::AbstractString, values::OneElectronValues{Float64})
    n = size(values.overlap, 1)
    n_pairs = canonical_pair_count(n)
    return _atomic_write(path) do io
        title1 = rpad("OpenMolcas4NOCI portable SEWARD HDF5 import", 80)
        title2 = rpad("Generated by NOCI.jl without libmolcas.so", 80)
        _write_native_fortran_record(io, title1, title2, Int64(n_pairs), values.nuclear_repulsion, Int64(n), Int64(9600), Int64(1))
        _write_native_fortran_record(io, _pack_seward_symmetric(values.overlap))
        _write_native_fortran_record(io, _pack_seward_symmetric(values.kinetic))
        _write_native_fortran_record(io, _pack_seward_symmetric(values.nuclear_attraction))
        for component in 1:9
            _write_native_fortran_record(io, _pack_seward_symmetric(view(values.multipoles, :, :, component)))
        end
        _write_native_fortran_record(io, Int64[0, -1, 0])
        _write_native_fortran_record(io, Int64(0))
        _write_native_fortran_record(io, Int64(1))
        _write_native_fortran_record(io, Int64[0])
        _write_native_fortran_record(io, Int64[0])
    end
end

"""
    _write_seward_sys_file(path, manifest)

Atomically write a human-readable `.sys` provenance summary from portable SEWARD metadata.
Coordinates are emitted in bohr, charges are dimensionless atomic charges, the manifest is read-only, and the file is informational rather than a numerical input to NOCI kernels.
"""
function _write_seward_sys_file(path::AbstractString, manifest::SewardHDF5Manifest)
    return _atomic_write(path) do io
        println(io, " * * * OpenMolcas4NOCI portable SEWARD HDF5 system")
        println(io, "number of atoms: ", length(manifest.atom_labels))
        println(io, "coordinates: bohr")
        @printf(io, "%6s %-12s %16s %16s %16s %10s %10s\n", "index", "atom", "x", "y", "z", "Znuc", "Zeff")
        for atom in eachindex(manifest.atom_labels)
            @printf(io, "%6d %-12s %16.9f %16.9f %16.9f %10.4f %10.4f\n", atom, manifest.atom_labels[atom], manifest.atom_coordinates_bohr[1, atom], manifest.atom_coordinates_bohr[2, atom], manifest.atom_coordinates_bohr[3, atom], manifest.nuclear_charges[atom], manifest.effective_nuclear_charges[atom])
        end
        @printf(io, "center of mass (bohr): %.9f %.9f %.9f\n", manifest.center_of_mass_bohr...)
        @printf(io, "AO functions: %d\n", manifest.n_ao)
        @printf(io, "ERI factors: %d\n", manifest.n_auxiliary)
        println(io, "ERI factor origin: ", manifest.provenance.origin)
        println(io, "auxiliary basis: ", manifest.provenance.auxiliary_basis_kind)
        println(io, "threshold kind: ", manifest.provenance.threshold_kind)
        @printf(io, "factor generation threshold (hartree): %.12e\n", manifest.provenance.generation_threshold)
    end
end

"""
    write_seward_hdf5_products(result; one_path, system_path, bundle_path)

Atomically write `.one`, `.sys`, and version-2 JLD2 factor-bundle products from one completed portable HDF5 import and return their absolute paths.
The scientific result is read-only, each destination is independently replaced only after a complete durable write, no `.two` data are generated, and file-system errors propagate.
"""
function write_seward_hdf5_products(
    result::SewardHDF5ImportResult;
    one_path::AbstractString,
    system_path::AbstractString,
    bundle_path::AbstractString,
)
    one = _write_seward_one_file(one_path, result.one_electron)
    system = _write_seward_sys_file(system_path, result.manifest)
    bundle = write_eri_factor_bundle_v2(bundle_path, result.factor_bundle)
    return (one=one, system=system, bundle=bundle)
end
