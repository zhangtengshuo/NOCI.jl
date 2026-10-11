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

 File:            src/runtime/atomic_io.jl
 Purpose:         Atomic file replacement and fsync-synchronized durable output helpers.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _fsync_stream(io)

Flush and synchronize one open host file stream to stable storage and return `nothing`.
The stream is mutated only through flushing, units are bytes at the operating-system boundary, and an OS synchronization failure raises `SystemError`.
"""
function _fsync_stream(io::IOStream)
    flush(io)
    status = ccall(:fsync, Cint, (Cint,), Base.fd(io))
    status == 0 || throw(SystemError("fsync", Libc.errno()))
    return nothing
end

"""
    _fsync_directory(path)

Synchronize directory metadata after an atomic rename and return `nothing`.
The named directory is opened read-only without changing contents, no CUDA state is touched, and OS open/fsync errors raise `SystemError`.
"""
function _fsync_directory(path::AbstractString)
    descriptor = ccall(:open, Cint, (Cstring, Cint), path, 0)
    descriptor >= 0 || throw(SystemError("open directory", Libc.errno()))
    try
        status = ccall(:fsync, Cint, (Cint,), descriptor)
        status == 0 || throw(SystemError("fsync directory", Libc.errno()))
    finally
        ccall(:close, Cint, (Cint,), descriptor)
    end
    return nothing
end

"""
    _atomic_write(writer, path)

Write one replacement file through a same-directory temporary path, fsync it, atomically rename it, and fsync the containing directory.
`writer` exclusively owns the temporary stream during its call, existing destination content is replaced only after a complete write, and callback or filesystem errors propagate while leaving the prior destination valid.
"""
function _atomic_write(writer::F, path::AbstractString) where {F<:Function}
    destination = abspath(path)
    directory = dirname(destination)
    mkpath(directory)
    temporary = string(destination, ".tmp.", getpid(), ".", time_ns())
    completed = false
    try
        open(temporary, "w") do io
            writer(io)
            _fsync_stream(io)
        end
        mv(temporary, destination; force=true)
        _fsync_directory(directory)
        completed = true
    finally
        !completed && isfile(temporary) && Base.Filesystem.unlink(temporary)
    end
    return destination
end

"""
    _atomic_jld2_write(path, values)

Atomically persist a string-keyed mapping as one JLD2 file and return its absolute path.
Values are read-only and serialized by JLD2 on CPU, no CUDA resources are retained, and unsupported values or filesystem failures propagate without replacing the prior valid file.
"""
function _atomic_jld2_write(path::AbstractString, values::AbstractDict{<:AbstractString,<:Any})
    destination = abspath(path)
    directory = dirname(destination)
    mkpath(directory)
    temporary = string(destination, ".tmp.", getpid(), ".", time_ns())
    completed = false
    try
        JLD2.jldopen(temporary, "w") do file
            for key in sort!(String.(collect(keys(values))))
                file[key] = values[key]
            end
        end
        open(temporary, "r") do io
            _fsync_stream(io)
        end
        mv(temporary, destination; force=true)
        _fsync_directory(directory)
        completed = true
    finally
        !completed && isfile(temporary) && Base.Filesystem.unlink(temporary)
    end
    return destination
end

"""
    _json_escape(value)

Return a JSON-escaped owned string for one input string without surrounding quotes.
The input is read-only, only ordinary host string storage is allocated, and every control character is encoded deterministically.
"""
function _json_escape(value::AbstractString)
    io = IOBuffer()
    for character in value
        if character == '"'
            print(io, "\\\"")
        elseif character == '\\'
            print(io, "\\\\")
        elseif character == '\b'
            print(io, "\\b")
        elseif character == '\f'
            print(io, "\\f")
        elseif character == '\n'
            print(io, "\\n")
        elseif character == '\r'
            print(io, "\\r")
        elseif character == '\t'
            print(io, "\\t")
        elseif Int(character) < 0x20
            @printf(io, "\\u%04x", Int(character))
        else
            print(io, character)
        end
    end
    return String(take!(io))
end

"""
    _write_json(io, value)

Write a deterministic JSON representation of supported scalar, mapping, tuple, vector, or matrix data to `io`.
Inputs are read-only, matrix rows retain Julia row order, non-finite floats and unsupported types raise `ArgumentError`, and no CUDA state is touched.
"""
function _write_json(io::IO, value)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Integer
        print(io, value)
    elseif value isa AbstractFloat
        isfinite(value) || throw(ArgumentError("JSON scientific values must be finite"))
        print(io, repr(value))
    elseif value isa AbstractString || value isa Symbol
        print(io, '"', _json_escape(string(value)), '"')
    elseif value isa AbstractDict
        print(io, '{')
        ordered_keys = sort!(collect(keys(value)); by=string)
        for (index, key) in pairs(ordered_keys)
            index > 1 && print(io, ',')
            _write_json(io, string(key))
            print(io, ':')
            _write_json(io, value[key])
        end
        print(io, '}')
    elseif value isa AbstractMatrix
        print(io, '[')
        for row in axes(value, 1)
            row != first(axes(value, 1)) && print(io, ',')
            _write_json(io, collect(@view value[row, :]))
        end
        print(io, ']')
    elseif value isa AbstractVector || value isa Tuple
        print(io, '[')
        for (index, item) in pairs(value)
            index != firstindex(value) && print(io, ',')
            _write_json(io, item)
        end
        print(io, ']')
    else
        throw(ArgumentError("unsupported JSON value type $(typeof(value))"))
    end
    return nothing
end

"""
    _atomic_json_write(path, value)

Atomically write deterministic UTF-8 JSON followed by a newline and return the absolute destination path.
The data are read-only, non-finite scientific floats are rejected, and the previous destination remains valid if serialization or filesystem synchronization fails.
"""
function _atomic_json_write(path::AbstractString, value)
    return _atomic_write(path) do io
        _write_json(io, value)
        write(io, '\n')
    end
end
