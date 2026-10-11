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

 File:            src/integrals/fortran_records.jl
 Purpose:         Fortran sequential record-marker detection and fixed-record .two stream reading.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    detect_fortran_record_marker(path)

Detect native-endian `Int32` or `Int64` record markers in a Fortran sequential file and return the marker type.
The function performs read-only host I/O, allocates only a small path/record probe, and raises `ArgumentError` for a missing, too-small, or unrecognized file.
"""
function detect_fortran_record_marker(path::AbstractString)
    isfile(path) || throw(ArgumentError("Fortran sequential file not found: $path"))
    n_bytes = filesize(path)
    n_bytes >= 12 || throw(ArgumentError("File is too small to be a Fortran sequential file: $path"))
    for marker_type in (Int32, Int64)
        marker_bytes = sizeof(marker_type)
        n_bytes >= 2 * marker_bytes || continue
        candidate = open(path, "r") do io
            leading = Int(read(io, marker_type))
            0 < leading <= n_bytes - 2 * marker_bytes || return false
            seek(io, marker_bytes + leading)
            trailing = Int(read(io, marker_type))
            return trailing == leading
        end
        candidate && return marker_type
    end
    throw(ArgumentError("Unable to detect native-endian Fortran record markers in $path"))
end

"""
    read_fortran_record(io, marker_type)

Read one Fortran sequential record, verify matching nonnegative leading/trailing byte counts, and return its payload bytes.
The stream advances past exactly one record; truncated data raise `EOFError` and inconsistent markers raise `ArgumentError`.
"""
function read_fortran_record(io::IO, ::Type{M}) where {M<:Integer}
    leading = Int(read(io, M))
    leading >= 0 || throw(ArgumentError("Negative Fortran record length $leading"))
    payload = read(io, leading)
    length(payload) == leading || throw(EOFError())
    trailing = Int(read(io, M))
    trailing == leading || throw(ArgumentError("Fortran record marker mismatch: $leading != $trailing"))
    return payload
end

"""
    _reinterpret_record_values(T, bytes)

Copy native-endian record bytes into a dense vector of values of type `T`.
The byte count must be divisible by `sizeof(T)`; the input bytes are not mutated.
"""
function _reinterpret_record_values(::Type{T}, bytes::AbstractVector{UInt8}) where {T}
    length(bytes) % sizeof(T) == 0 || throw(ArgumentError("Byte count is not a multiple of $(sizeof(T))"))
    owned = bytes isa Vector{UInt8} ? bytes : collect(bytes)
    return copy(reinterpret(T, owned))
end

"""
    _reinterpret_record_scalar(T, bytes)

Decode exactly one native-endian scalar of type `T` from record bytes.
The input is not mutated; any element count other than one raises `ArgumentError`.
"""
function _reinterpret_record_scalar(::Type{T}, bytes::AbstractVector{UInt8}) where {T}
    values = _reinterpret_record_values(T, bytes)
    length(values) == 1 || throw(ArgumentError("Expected one $(T) value, got $(length(values))"))
    return values[1]
end

"""
    _decode_fortran_string(bytes)

Copy a fixed-width Fortran character field into a Julia `String` and remove trailing whitespace.
The byte input is not mutated and no character-set conversion beyond Julia string construction is attempted.
"""
function _decode_fortran_string(bytes::AbstractVector{UInt8})
    return rstrip(String(copy(bytes)))
end
