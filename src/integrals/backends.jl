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

 File:            src/integrals/backends.jl
 Purpose:         Abstract and concrete ERI backend types for packed .two and ERI factor storage.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    ERIBackend

Abstract supertype for electron-repulsion-integral storage backends.
Backend values contain file identities but do not load data during construction.
"""
abstract type ERIBackend end

"""
    PackedTwoERI(one_path, two_paths)

Describe conventional one-electron and packed `.two` integral files.
Paths are copied to owned strings; construction performs no file I/O and no CPU/GPU allocation beyond path storage.
"""
struct PackedTwoERI <: ERIBackend
    one_path::String
    two_paths::Vector{String}
end

"""
    ERIFactorSource

Abstract source descriptor for physical metric-whitened ERI factors consumed by `FactorizedERI`.
Sources contain only paths and transformation settings; constructing them performs no OpenMolcas, host-array, or CUDA work.
"""
abstract type ERIFactorSource end

"""
    ERIFactorBundleSource(factor_path)

Describe a directly loadable versioned JLD2 ERI-factor bundle.
The path is copied to owned storage, no file is opened, and validation is deferred to the factor loader.
"""
struct ERIFactorBundleSource <: ERIFactorSource
    factor_path::String
end

"""
    ERIFactorBundleSource(factor_path)

Construct a factor-bundle source from a path-like value.
The argument is not mutated, the returned path is owned, and the operation allocates no scientific or device arrays.
"""
function ERIFactorBundleSource(factor_path::AbstractString)
    return ERIFactorBundleSource(String(factor_path))
end

"""
    SewardMOTRASource(; snapshot, project, ranks, commonorb, n_frozen=0, n_deleted, batch_size=8, transform_backend=:cpu, gpu_device=0, scratch_parent=dirname(snapshot), molcas_root=get(ENV, "MOLCAS", ""), omc_root=get(ENV, "OMC", ""))

Describe a serial or MPI-sharded SEWARD AO ERI-factor source and its Julia-MOTRA AO-to-retained-MO transformation.
Counts are dimensionless, GPU indices are zero based, paths are copied, and invalid ranks, dimensions, batch sizes, or backend names raise `ArgumentError` without opening OpenMolcas or CUDA.
"""
struct SewardMOTRASource <: ERIFactorSource
    snapshot::String
    project::String
    ranks::Int
    commonorb::String
    n_frozen::Int
    n_deleted::Int
    batch_size::Int
    transform_backend::Symbol
    gpu_device::Int
    scratch_parent::String
    molcas_root::String
    omc_root::String
end

"""
    SewardMOTRASource(; kwargs...)

Construct a validated SEWARD/Julia-MOTRA source descriptor.
The constructor owns all strings, performs no file-system or CUDA checks, and rejects configurations outside C1 serial/MPI transformation resource semantics.
"""
function SewardMOTRASource(;
    snapshot::AbstractString,
    project::AbstractString,
    ranks::Integer,
    commonorb::AbstractString,
    n_frozen::Integer=0,
    n_deleted::Integer,
    batch_size::Integer=8,
    transform_backend::Symbol=:cpu,
    gpu_device::Integer=0,
    scratch_parent::AbstractString=dirname(abspath(snapshot)),
    molcas_root::AbstractString=get(ENV, "MOLCAS", ""),
    omc_root::AbstractString=get(ENV, "OMC", ""),
)
    ranks >= 1 || throw(ArgumentError("SEWARD ranks must be positive"))
    n_frozen >= 0 || throw(ArgumentError("SEWARD n_frozen must be nonnegative"))
    n_deleted >= 0 || throw(ArgumentError("SEWARD n_deleted must be nonnegative"))
    batch_size >= 1 || throw(ArgumentError("SEWARD batch_size must be positive"))
    transform_backend in (:cpu, :gpu) || throw(ArgumentError("SEWARD transform_backend must be :cpu or :gpu"))
    gpu_device >= 0 || throw(ArgumentError("SEWARD gpu_device must be nonnegative"))
    isempty(strip(project)) && throw(ArgumentError("SEWARD project must not be empty"))
    return SewardMOTRASource(
        String(snapshot),
        String(project),
        Int(ranks),
        String(commonorb),
        Int(n_frozen),
        Int(n_deleted),
        Int(batch_size),
        transform_backend,
        Int(gpu_device),
        String(scratch_parent),
        String(molcas_root),
        String(omc_root),
    )
end

"""
    PackedTwoERI(one_path, two_path)

Construct a conventional ERI backend containing one packed `.two` file.
Arguments are path-like strings, the return value owns normalized `String` copies, and missing-file validation is deferred to the loader.
"""
function PackedTwoERI(one_path::AbstractString, two_path::AbstractString)
    return PackedTwoERI(String(one_path), [String(two_path)])
end

"""
    PackedTwoERI(one_path, two_paths)

Construct a conventional ERI backend from multiple packed `.two` file paths.
The input collection is not mutated; construction copies every path and performs no file I/O.
"""
function PackedTwoERI(one_path::AbstractString, two_paths::AbstractVector{<:AbstractString})
    return PackedTwoERI(String(one_path), String.(two_paths))
end

"""
    FactorizedERI(one_path, source)

Describe one-electron data and a typed factor-bundle or SEWARD/Julia-MOTRA Cholesky source.
Construction copies the one-electron path, retains the immutable source, and performs no file loading, OpenMolcas initialization, or device allocation.
"""
struct FactorizedERI{S<:ERIFactorSource} <: ERIBackend
    one_path::String
    source::S
end

"""
    FactorizedERI(one_path, factor_path)

Construct a Cholesky ERI backend from path-like strings.
The return value owns `String` copies, does not mutate inputs, and defers file-format validation to the loader.
"""
function FactorizedERI(one_path::AbstractString, factor_path::AbstractString)
    return FactorizedERI(String(one_path), ERIFactorBundleSource(factor_path))
end

"""
    FactorizedERI(one_path, source::SewardMOTRASource)

Construct a Cholesky backend that transforms a serial or MPI-sharded SEWARD source during NOCI input loading.
Arguments are retained as immutable configuration, no source file or CUDA context is opened, and scientific allocation is deferred to evaluation.
"""
function FactorizedERI(one_path::AbstractString, source::SewardMOTRASource)
    return FactorizedERI(String(one_path), source)
end

"""
    FactorizedERI(one_path::String, source::SewardMOTRASource)

Resolve the concrete-string constructor without ambiguity against the parametric inner constructor.
Inputs are retained without scientific allocation, no file is opened, and the immutable source is not mutated.
"""
function FactorizedERI(one_path::String, source::SewardMOTRASource)
    return FactorizedERI{SewardMOTRASource}(one_path, source)
end

"""
    backend_name(backend)

Return the stable CLI/report identifier for an ERI backend.
This query does not inspect files or allocate numerical workspaces.
"""
backend_name(::PackedTwoERI) = "packed-two-eri"

"""
    backend_name(backend::FactorizedERI)

Return `"factorized-eri"` for a factorized-ERI backend without loading its data.
"""
backend_name(::FactorizedERI{ERIFactorBundleSource}) = "factorized-eri"

"""
    backend_name(backend::FactorizedERI{SewardMOTRASource})

Return the stable `"seward-factorized-eri"` telemetry name without inspecting the snapshot or loading OpenMolcas.
"""
backend_name(::FactorizedERI{SewardMOTRASource}) = "seward-factorized-eri"
