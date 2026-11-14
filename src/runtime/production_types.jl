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

 File:            src/runtime/production_types.jl
 Purpose:         RunOutputOptions and shared production run configuration types.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    RunOutputOptions(directory, root; geometry_path=nothing)

Describe the directory and filename root owned by one production run together with optional report-only molecular-geometry provenance.
Construction normalizes and owns strings, performs no file or CUDA I/O, and rejects empty values, path separators in the root, or an empty geometry path.
"""
struct RunOutputOptions
    directory::String
    root::String
    geometry_path::Union{Nothing,String}

    """
        RunOutputOptions(directory, root, geometry_path)

    Construct validated production-output metadata and optional geometry-report path.
    Geometry is reporting provenance only and does not alter the numerical NOCI problem, no files are opened, and malformed paths raise `ArgumentError`.
    """
    function RunOutputOptions(
        directory::AbstractString,
        root::AbstractString,
        geometry_path::Union{Nothing,AbstractString},
    )
        isempty(strip(directory)) && throw(ArgumentError("output directory must not be empty"))
        isempty(strip(root)) && throw(ArgumentError("output root must not be empty"))
        basename(root) == root || throw(ArgumentError("output root must not contain path separators"))
        if !isnothing(geometry_path)
            isempty(strip(geometry_path)) && throw(ArgumentError("geometry path must not be empty"))
        end
        geometry = isnothing(geometry_path) ? nothing : normpath(String(geometry_path))
        return new(normpath(String(directory)), String(root), geometry)
    end
end

"""
    RunOutputOptions(directory, root; geometry_path=nothing)

Construct production-output metadata while preserving the historical two-positional-argument API.
The optional geometry path is report provenance only; construction performs no filesystem access or scientific allocation.
"""
function RunOutputOptions(
    directory::AbstractString,
    root::AbstractString;
    geometry_path::Union{Nothing,AbstractString}=nothing,
)
    return RunOutputOptions(directory, root, geometry_path)
end

"""
    CheckpointOptions(; enabled=true, directory, gronor_noci_checkpoint_pairs=8192, ahpc_checkpoint_blocks=32)

Configure restart-file placement and write batching independently from NOCI scientific algorithms.
Pair and block intervals are positive dimensionless I/O frequencies, construction performs no file or CUDA I/O, and changing `gronor_noci_checkpoint_pairs` must never change G-NOCI traversal or reduction order.
"""
struct CheckpointOptions
    enabled::Bool
    directory::String
    gronor_noci_checkpoint_pairs::Int
    ahpc_checkpoint_blocks::Int

    """
        CheckpointOptions(enabled, directory, gronor_noci_checkpoint_pairs, ahpc_checkpoint_blocks)

    Construct validated checkpoint options and own the normalized directory string.
    No files are created, values are not used as scientific selectors, and empty paths or nonpositive intervals raise `ArgumentError`.
    """
    function CheckpointOptions(
        enabled::Bool,
        directory::AbstractString,
        gronor_noci_checkpoint_pairs::Integer,
        ahpc_checkpoint_blocks::Integer,
    )
        isempty(strip(directory)) && throw(ArgumentError("checkpoint directory must not be empty"))
        gronor_noci_checkpoint_pairs > 0 || throw(ArgumentError("gronor_noci_checkpoint_pairs must be positive"))
        ahpc_checkpoint_blocks > 0 || throw(ArgumentError("ahpc_checkpoint_blocks must be positive"))
        return new(enabled, normpath(String(directory)), Int(gronor_noci_checkpoint_pairs), Int(ahpc_checkpoint_blocks))
    end
end

"""
    NOCIEvaluationResult

Store one completed unshifted NOCI evaluation as raw matrices, normalized matrices, generalized solution, and execution telemetry.
Scientific arrays remain CPU-owned through their component objects, energies follow the integral unit, and no coupling analysis, output-file handle, or CUDA resource is retained.
"""
struct NOCIEvaluationResult{T<:AbstractFloat}
    label::String
    algorithm::String
    branch::String
    backend::String
    raw::RawNOCIMatrices{T}
    normalized::NormalizedNOCIMatrices{T}
    solution::NOCISolution{T}
    telemetry::ExecutionTelemetry
end

"""
    CheckpointOptions(; enabled=true, directory, gronor_noci_checkpoint_pairs=8192, ahpc_checkpoint_blocks=32)

Construct production checkpoint options using explicit keyword configuration.
Inputs are read-only, no file or device state is touched, and validation failures raise `ArgumentError`.
"""
function CheckpointOptions(;
    enabled::Bool=true,
    directory::AbstractString,
    gronor_noci_checkpoint_pairs::Integer=8192,
    ahpc_checkpoint_blocks::Integer=32,
)
    return CheckpointOptions(enabled, directory, gronor_noci_checkpoint_pairs, ahpc_checkpoint_blocks)
end

"""
    ProductionRunResult

Store an unshifted NOCI evaluation, baseline CASPT2 shifts, shifted normalized Hamiltonian and shifted generalized solution together with committed output paths.
Scientific arrays are CPU-owned in the integral energy unit, paths and signatures are owned strings, and no open files or CUDA resources are retained.
"""
struct ProductionRunResult{T<:AbstractFloat}
    evaluation::NOCIEvaluationResult{T}
    correlation_shifts::Vector{T}
    shifted_hamiltonian::Matrix{T}
    shifted_solution::NOCISolution{T}
    input_manifest::Vector{Dict{String,Any}}
    scientific_signature::String
    execution_signature::String
    sec_path::String
    out_path::String
    json_path::String
    snapshot_path::String
end

"""
    Caspt2OverrideResult

Store one snapshot-only CASPT2 scalar-energy override and its independently solved shifted result.
Energies are in Hartree, arrays and provenance strings are CPU-owned, and construction or retention performs no scientific-input I/O or CUDA initialization.
"""
struct Caspt2OverrideResult{T<:AbstractFloat}
    method::String
    source::String
    weighting::String
    override_sha256::String
    correlation_shifts::Vector{T}
    shifted_hamiltonian::Matrix{T}
    shifted_solution::NOCISolution{T}
    json_path::String
end

"""
    ProgressLogger

Own open append streams for the real-time `.sec` progress record and scientific `.out` report during one production operation.
The mutable object records monotonic start time plus whether extra test-mode progress is enabled; file writes are host side effects and it contains no scientific arrays or CUDA resources.
"""
mutable struct ProgressLogger
    sec_io::IOStream
    out_io::IOStream
    start_ns::UInt64
    test_mode::Bool
end

"""
    CheckpointSession

Own one process-scoped advisory lock and validated identities for a checkpoint directory.
The session may write only small scalar restart payloads, its lock stream must be closed after use, and it never owns scientific tensors or CUDA objects.
"""
mutable struct CheckpointSession
    directory::String
    scientific_signature::String
    execution_signature::String
    gronor_noci_checkpoint_pairs::Int
    ahpc_checkpoint_blocks::Int
    lock_io::IOStream
    commit_hook::Function
end
