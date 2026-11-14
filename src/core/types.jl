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

 File:            src/core/types.jl
 Purpose:         Core abstract types shared by both public NOCI algorithms.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    NOCIAlgorithm

Abstract supertype for the two public NOCI algorithms.
"""
abstract type NOCIAlgorithm end

"""
    GronorNOCI()

Select GronOR NOCI with conventional packed `.two` electron-repulsion integrals.
The selector stores no mutable state and does not itself allocate host or device memory.
"""
struct GronorNOCI <: NOCIAlgorithm end

"""
    AHPCBranch

Abstract supertype for contraction branches inside AHPC-NOCI.
Every branch starts from the same state-pair active-Hamiltonian precontraction.
"""
abstract type AHPCBranch end

"""
    UnsupportedERIBackend(algorithm, backend)

Error raised before large allocation when an algorithm does not support an ERI backend.
The stored names are stable user-facing identifiers.
"""
struct UnsupportedERIBackend <: Exception
    algorithm::String
    backend::String
end

"""
    showerror(io, error::UnsupportedERIBackend)

Render a concise unsupported algorithm/backend diagnostic to `io` and return `nothing`.
This method does not mutate the error object or allocate scientific workspaces.
"""
function Base.showerror(io::IO, error::UnsupportedERIBackend)
    print(io, error.algorithm, " does not support ", error.backend)
    return nothing
end

"""
    AlgorithmNotImplemented(algorithm, branch)

Temporary framework error raised when a valid algorithm/backend combination reaches a numerical branch that has not yet been implemented.
"""
struct AlgorithmNotImplemented <: Exception
    algorithm::String
    branch::String
end

"""
    showerror(io, error::AlgorithmNotImplemented)

Render the missing numerical branch diagnostic to `io` and return `nothing`.
This method has no CPU/GPU side effects.
"""
function Base.showerror(io::IO, error::AlgorithmNotImplemented)
    print(io, error.algorithm, " branch ", error.branch, " is not implemented yet")
    return nothing
end

"""
    ResourceLimitError(resource, required_bytes, limit_bytes)

Error raised by memory preflight before an allocation would exceed its configured byte limit.
All byte counts are nonnegative host integers.
"""
struct ResourceLimitError <: Exception
    resource::Symbol
    required_bytes::Int
    limit_bytes::Int
end

"""
    showerror(io, error::ResourceLimitError)

Render a byte-accurate memory-limit diagnostic to `io` and return `nothing`.
This method does not modify runtime configuration.
"""
function Base.showerror(io::IO, error::ResourceLimitError)
    print(
        io,
        error.resource,
        " memory estimate ",
        error.required_bytes,
        " bytes exceeds limit ",
        error.limit_bytes,
        " bytes",
    )
    return nothing
end

"""
    GPUUnavailableError(device, reason)

Error raised before GPU scientific allocation when the requested zero-based CUDA device cannot execute the selected path.
The stored device index is dimensionless and the reason is a stable diagnostic string; construction has no CUDA side effects.
"""
struct GPUUnavailableError <: Exception
    device::Int
    reason::String
end

"""
    showerror(io, error::GPUUnavailableError)

Render a concise CUDA-device diagnostic to `io` and return `nothing`.
This method does not mutate the exception, initialize CUDA, or allocate scientific CPU/GPU workspaces.
"""
function Base.showerror(io::IO, error::GPUUnavailableError)
    print(io, "CUDA device ", error.device, " is unavailable: ", error.reason)
    return nothing
end
