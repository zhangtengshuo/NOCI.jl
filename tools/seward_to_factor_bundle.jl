#!/usr/bin/env julia
# File: tools/seward_to_factor_bundle.jl
# Purpose: Convert SEWARD factor exports into a validated NOCI factor bundle.
# SPDX-License-Identifier: Apache-2.0

using NOCI
using YAML

"""
    usage(io=stdout)

Print the SEWARD-to-factor-bundle command syntax to `io` and return `nothing`.
The operation writes only text, allocates no scientific arrays, and does not initialize OpenMolcas or CUDA.
"""
function usage(io::IO=stdout)
    println(io, "usage: julia --project=. tools/seward_to_factor_bundle.jl SEWARD_SOURCE.yaml OUTPUT.jld2")
    println(io, "The input may be a focused `seward:` conversion file or a complete NOCI input using a SEWARD source.")
    return nothing
end

"""
    _required(mapping, key, location)

Return a required conversion-setting value without mutating the YAML mapping.
No scientific data or CUDA state is touched, and a missing key raises `ArgumentError` naming its location.
"""
function _required(mapping::AbstractDict, key::AbstractString, location::AbstractString)
    haskey(mapping, key) || throw(ArgumentError("missing YAML key $location.$key"))
    return mapping[key]
end

"""
    _source_path(directory, value, location)

Resolve a nonempty conversion path against the directory containing the YAML file.
The filesystem is not modified or inspected, CUDA is not initialized, and non-string or empty values raise `ArgumentError`.
"""
function _source_path(directory::AbstractString, value, location::AbstractString)
    value isa AbstractString || throw(ArgumentError("$location must be a string"))
    isempty(strip(value)) && throw(ArgumentError("$location must not be empty"))
    return normpath(isabspath(value) ? String(value) : joinpath(directory, value))
end

"""
    _source_integer(mapping, key, default, location)

Read one integer conversion setting, using `default` when the key is absent.
The mapping remains read-only, no scientific storage is allocated, and booleans or nonintegers raise `ArgumentError`.
"""
function _source_integer(mapping::AbstractDict, key::AbstractString, default, location::AbstractString)
    value = get(mapping, key, default)
    value isa Integer && !(value isa Bool) || throw(ArgumentError("$location.$key must be an integer"))
    return Int(value)
end

"""
    _focused_request(path, document)

Construct a SEWARD source and resource settings from the focused public conversion schema.
Paths are resolved relative to `path`, the YAML document is not mutated, no source file or CUDA context is opened, and malformed or incomplete settings raise `ArgumentError`.
"""
function _focused_request(path::AbstractString, document::AbstractDict)
    mapping = _required(document, "seward", "root")
    mapping isa AbstractDict || throw(ArgumentError("seward must be a mapping"))
    directory = dirname(path)
    backend_text = lowercase(String(get(mapping, "transform_backend", "cpu")))
    backend = Symbol(backend_text)
    gpu_device = _source_integer(mapping, "gpu_device", 0, "seward")
    molcas_value = get(mapping, "molcas_root", get(ENV, "MOLCAS", ""))
    omc_value = get(mapping, "omc_root", get(ENV, "OMC", ""))
    isempty(strip(String(molcas_value))) && throw(ArgumentError("set seward.molcas_root or the MOLCAS environment variable"))
    isempty(strip(String(omc_value))) && throw(ArgumentError("set seward.omc_root or the OMC environment variable"))
    source = SewardMOTRASource(
        snapshot=_source_path(directory, _required(mapping, "snapshot", "seward"), "seward.snapshot"),
        project=String(_required(mapping, "project", "seward")),
        ranks=_source_integer(mapping, "ranks", nothing, "seward"),
        commonorb=_source_path(directory, _required(mapping, "commonorb", "seward"), "seward.commonorb"),
        n_frozen=_source_integer(mapping, "n_frozen", 0, "seward"),
        n_deleted=_source_integer(mapping, "n_deleted", nothing, "seward"),
        batch_size=_source_integer(mapping, "batch_size", 8, "seward"),
        transform_backend=backend,
        gpu_device=gpu_device,
        scratch_parent=_source_path(directory, get(mapping, "scratch_parent", "."), "seward.scratch_parent"),
        molcas_root=_source_path(directory, molcas_value, "seward.molcas_root"),
        omc_root=_source_path(directory, omc_value, "seward.omc_root"),
    )
    runtime_mapping = get(document, "runtime", Dict{String,Any}())
    runtime_mapping isa AbstractDict || throw(ArgumentError("runtime must be a mapping"))
    runtime = RuntimeOptions(
        use_gpu=backend == :gpu,
        gpu_device=gpu_device,
        max_host_bytes=haskey(runtime_mapping, "max_host_bytes") ? _source_integer(runtime_mapping, "max_host_bytes", nothing, "runtime") : nothing,
        max_device_bytes=haskey(runtime_mapping, "max_device_bytes") ? _source_integer(runtime_mapping, "max_device_bytes", nothing, "runtime") : nothing,
        scratch_bytes=_source_integer(runtime_mapping, "scratch_bytes", 256 * 1024^2, "runtime"),
        collect_allocations=false,
    )
    return source, runtime
end

"""
    _bundle_request(path)

Read either the focused public conversion schema or a complete NOCI YAML and return its SEWARD source plus runtime options.
Only YAML metadata is loaded, scientific sources remain read-only, CUDA is not initialized, and non-SEWARD configurations raise before conversion.
"""
function _bundle_request(path::AbstractString)
    document = YAML.load_file(path)
    document isa AbstractDict || throw(ArgumentError("conversion YAML root must be a mapping"))
    haskey(document, "seward") && return _focused_request(path, document)
    input = read_noci_input(path)
    input.backend isa FactorizedERI{SewardMOTRASource} || throw(ArgumentError("input does not select an embedded SEWARD Cholesky source"))
    return input.backend.source, input.runtime
end

"""
    main(args=ARGS)

Load one NOCI YAML file, execute its embedded serial/MPI SEWARD Julia-MOTRA transform, and atomically write a portable version-2 JLD2 factor bundle.
Scientific sources remain read-only, the output is replaced only by the atomic bundle writer, CPU/GPU and memory settings come from the typed input, and non-SEWARD backends raise before OpenMolcas initialization.
"""
function main(args::Vector{String}=ARGS)
    length(args) == 2 || (usage(stderr); return 2)
    input_path = abspath(args[1])
    output_path = abspath(args[2])
    source, runtime = _bundle_request(input_path)
    product = transform_seward_eri_factors(source; runtime)
    source_kind = "OpenMolcas_SEWARD_$(product.source_ranks == 1 ? "SERIAL" : "MPI_SHARDS")_JuliaMOTRA_$(uppercase(String(product.transform_backend)))"
    bundle = ERIFactorBundle(2, product.factors, source_kind, product.source_ranks)
    written = write_eri_factor_bundle_v2(output_path, bundle)
    println("factor_bundle = ", written)
    println("n_orbitals = ", product.factors.n_orbitals)
    println("n_orbital_pairs = ", size(product.factors.coefficients, 1))
    println("n_auxiliary = ", size(product.factors.coefficients, 2))
    println("source_ranks = ", product.source_ranks)
    println("transform_backend = ", product.transform_backend)
    println("transform_seconds = ", product.transform_seconds)
    return 0
end

exit(main())
