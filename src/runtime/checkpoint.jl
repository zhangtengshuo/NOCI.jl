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

 File:            src/runtime/checkpoint.jl
 Purpose:         Versioned DP cumulative-prefix and two-level AHPC resume checkpoints.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

const NOCI_CHECKPOINT_VERSION = 2

"""
    _sha256_file(path)

Stream one file through SHA-256 and return its lowercase hexadecimal content digest.
The source file is read-only, memory use is bounded by the SHA implementation, no CUDA state is touched, and missing or unreadable files raise filesystem errors.
"""
function _sha256_file(path::AbstractString)
    return bytes2hex(open(SHA.sha256, path))
end

"""
    _sha256_value(value)

Return the SHA-256 digest of a deterministic textual representation of small metadata.
The value is read-only, output is a lowercase hexadecimal string, and callers must not use unordered mappings in the represented scientific identity.
"""
function _sha256_value(value)
    return bytes2hex(SHA.sha256(codeunits(repr(value))))
end

"""
    _input_file_paths(input)

Return all determinant, orbital, one-electron, and ERI source paths in stable configuration order.
Input metadata is read-only, returned strings are owned, no paths are opened, and unsupported backend types raise `ArgumentError`.
"""
function _input_file_paths(input::NOCIInput)
    paths = String[]
    for state in input.fragment_states
        push!(paths, state.determinant_path, state.orbital_path)
    end
    push!(paths, input.backend.one_path)
    if input.backend isa PackedTwoERI
        append!(paths, input.backend.two_paths)
    elseif input.backend isa FactorizedERI{ERIFactorBundleSource}
        push!(paths, input.backend.source.factor_path)
    elseif input.backend isa FactorizedERI{SewardMOTRASource}
        source = input.backend.source
        push!(paths, source.commonorb)
        for suffix in ("RunFile", "OneInt", "ChRed", "ChMap")
            push!(paths, _seward_source_file(source, 0, suffix))
        end
        for rank in 0:(source.ranks - 1), suffix in ("ChRst", "ChVec1")
            push!(paths, _seward_source_file(source, rank, suffix))
        end
    else
        throw(ArgumentError("unsupported ERI backend $(typeof(input.backend))"))
    end
    return paths
end

"""
    build_input_manifest(input)

Stat every scientific source and return ordered absolute-path, byte-size, and nanosecond modification-time provenance records.
Source contents are never read, returned mappings are CPU-owned diagnostics, no CUDA state is initialized, and missing or unreadable files raise before calculation.
"""
function build_input_manifest(input::NOCIInput)
    records = Vector{Dict{String,Any}}()
    for (index, path) in pairs(_input_file_paths(input))
        absolute_path = abspath(path)
        status = stat(absolute_path)
        push!(records, Dict{String,Any}(
            "index" => index,
            "path" => absolute_path,
            "bytes" => status.size,
            "mtime_ns" => round(Int64, status.mtime * 1.0e9),
            "identity_kind" => "filesystem-stat-v1",
        ))
    end
    return records
end

"""
    build_scientific_signature(input, T; input_manifest=nothing)

Hash NOCI algorithm semantics, ordered MEBF metadata, screening values, floating type, and lightweight filesystem identities of every scientific source file.
Inputs and file contents are read-only, the result is independent of checkpoint write frequency, no CUDA state is initialized, and missing scientific files raise before a run starts without a full-file content pass.
"""
function build_scientific_signature(
    input::NOCIInput,
    ::Type{T};
    input_manifest::Union{Nothing,AbstractVector}=nothing,
) where {T<:AbstractFloat}
    manifest = isnothing(input_manifest) ? build_input_manifest(input) : input_manifest
    files = Tuple((Int(record["index"]), String(record["path"]), Int(record["bytes"]), Int64(record["mtime_ns"])) for record in manifest)
    mebfs = Tuple((definition.label, Tuple(definition.state_labels), Tuple(definition.coupling_multiplicities), definition.diagonal_energy_shift) for definition in input.mebfs)
    fragments = Tuple((state.label, state.fragment, state.state) for state in input.fragment_states)
    branch = input.algorithm isa AHPCNOCI ? branch_name(input.algorithm.branch) : "direct-determinant-pair"
    lookup = input.algorithm isa AHPCNOCI ? string(input.algorithm.branch.acceleration.support_lookup) : "none"
    integral_source = if input.backend isa FactorizedERI{SewardMOTRASource}
        source = input.backend.source
        (
            source=:seward,
            project=source.project,
            ranks=source.ranks,
            n_frozen=source.n_frozen,
            n_deleted=source.n_deleted,
            batch_size=source.batch_size,
            transform_backend=source.transform_backend,
            gpu_device=source.gpu_device,
        )
    else
        (source=:files,)
    end
    payload = (
        checkpoint_version=NOCI_CHECKPOINT_VERSION,
        algorithm=algorithm_name(input.algorithm),
        branch=branch,
        backend=backend_name(input.backend),
        floating_type=string(T),
        screening=(input.screening.tau_ci, input.screening.tau_ci_offdiagonal, input.screening.tau_singularity),
        fragments=fragments,
        mebfs=mebfs,
        support_lookup=lookup,
        integral_source=integral_source,
        files=files,
    )
    return _sha256_value(payload)
end

"""
    build_execution_signature(runtime, T)

Hash resource settings that select CPU/CUDA work and floating reduction structure while excluding checkpoint write frequencies.
The runtime is read-only, no CUDA query or file I/O occurs, and the returned digest is stable for equal typed settings.
"""
function build_execution_signature(runtime::RuntimeOptions, ::Type{T}) where {T<:AbstractFloat}
    payload = (
        floating_type=string(T),
        use_gpu=runtime.use_gpu,
        gpu_device=runtime.gpu_device,
        gpu_devices=runtime.gpu_devices,
        scratch_bytes=runtime.scratch_bytes,
        julia_threads=Threads.nthreads(),
    )
    return _sha256_value(payload)
end

"""
    _checkpoint_payload_digest(values)

Return a SHA-256 digest over sorted small checkpoint fields except an existing `payload_sha256` entry.
The mapping is read-only, scientific scalars retain their textual bit-precise representation, and unsupported nondeterministic payloads must be rejected by callers.
"""
function _checkpoint_payload_digest(values::AbstractDict{<:AbstractString,<:Any})
    ordered = Tuple((String(key), values[key]) for key in sort!(collect(keys(values)); by=string) if String(key) != "payload_sha256")
    return _sha256_value(ordered)
end

"""
    _write_checkpoint_payload(path, values)

Add a deterministic payload SHA-256 field and atomically commit one small JLD2 checkpoint file.
The input mapping is copied and not mutated, no scientific tensor or CUDA object is accepted by design, and serialization or filesystem errors preserve the previous valid payload.
"""
function _write_checkpoint_payload(path::AbstractString, values::AbstractDict{<:AbstractString,<:Any})
    payload = Dict{String,Any}(String(key) => value for (key, value) in values)
    payload["payload_sha256"] = _checkpoint_payload_digest(payload)
    return _atomic_jld2_write(path, payload)
end

"""
    _load_checkpoint_payload(path, session)

Load and validate one versioned JLD2 payload against session scientific/execution signatures and its SHA-256 field.
The file and session are read-only, returned data are CPU-owned small values, and corruption, schema mismatch, or identity mismatch raises `ArgumentError`.
"""
function _load_checkpoint_payload(path::AbstractString, session::CheckpointSession)
    data = JLD2.load(path)
    Int(data["version"]) == NOCI_CHECKPOINT_VERSION || throw(ArgumentError("unsupported checkpoint version in $path"))
    String(data["scientific_signature"]) == session.scientific_signature || throw(ArgumentError("checkpoint scientific signature mismatch in $path"))
    String(data["execution_signature"]) == session.execution_signature || throw(ArgumentError("checkpoint execution signature mismatch in $path"))
    expected = String(data["payload_sha256"])
    actual = _checkpoint_payload_digest(data)
    expected == actual || throw(ArgumentError("checkpoint payload checksum mismatch in $path"))
    return data
end

"""
    open_checkpoint_session(options, scientific_signature, execution_signature; commit_hook=(args...)->nothing)

Create or validate a checkpoint directory, acquire its nonblocking process advisory lock, and return an owning session.
Only small metadata files are written, no scientific or CUDA arrays are retained, and an active writer or incompatible metadata causes an immediate error.
"""
function open_checkpoint_session(
    options::CheckpointOptions,
    scientific_signature::AbstractString,
    execution_signature::AbstractString,
    ;
    commit_hook::Function=(args...)->nothing,
)
    options.enabled || throw(ArgumentError("cannot open a disabled checkpoint session"))
    directory = abspath(options.directory)
    mkpath(joinpath(directory, "dp"))
    mkpath(joinpath(directory, "ahpc"))
    lock_io = open(joinpath(directory, "writer.lock"), "a+")
    lock_status = ccall(:flock, Cint, (Cint, Cint), Base.fd(lock_io), 6)
    if lock_status != 0
        close(lock_io)
        throw(ArgumentError("checkpoint directory already has an active writer: $directory"))
    end
    session = CheckpointSession(
        directory,
        String(scientific_signature),
        String(execution_signature),
        options.dp_checkpoint_pairs,
        options.ahpc_checkpoint_blocks,
        lock_io,
        commit_hook,
    )
    meta_path = joinpath(directory, "checkpoint_meta.jld2")
    try
        if isfile(meta_path)
            _load_checkpoint_payload(meta_path, session)
        else
            _write_checkpoint_payload(meta_path, Dict{String,Any}(
                "version" => NOCI_CHECKPOINT_VERSION,
                "scientific_signature" => session.scientific_signature,
                "execution_signature" => session.execution_signature,
                "created_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
                "kind" => "noci-checkpoint-metadata",
            ))
        end
    catch
        close(lock_io)
        rethrow()
    end
    return session
end

"""
    _notify_checkpoint_commit!(session, kind, algorithm, left_mebf, right_mebf, next_index, path)

Invoke the session's post-fsync checkpoint hook after one payload is durably committed and return `nothing`.
Production sessions use a no-op hook, tests may deliberately throw to emulate process interruption, no scientific state is changed by this wrapper, and hook failures propagate after the valid checkpoint is on disk.
"""
function _notify_checkpoint_commit!(
    session::CheckpointSession,
    kind::Symbol,
    algorithm::Symbol,
    left_mebf::Integer,
    right_mebf::Integer,
    next_index::Integer,
    path::AbstractString,
)
    session.commit_hook(kind, algorithm, Int(left_mebf), Int(right_mebf), Int(next_index), String(path))
    return nothing
end

"""
    close_checkpoint_session!(session)

Release the advisory writer lock and close the lock stream owned by a checkpoint session.
No checkpoint payload or scientific data are deleted, no CUDA state is touched, and repeated closure follows Julia stream error behavior.
"""
function close_checkpoint_session!(session::CheckpointSession)
    close(session.lock_io)
    return nothing
end

"""
    _state_pair_checkpoint_path(session, algorithm, left_mebf, right_mebf; inflight=false)

Return the stable absolute checkpoint filename for one lower-triangular state pair.
Arguments are dimensionless metadata, no file is created, and invalid algorithm selectors or nonpositive indices raise `ArgumentError`.
"""
function _state_pair_checkpoint_path(
    session::CheckpointSession,
    algorithm::Symbol,
    left_mebf::Integer,
    right_mebf::Integer;
    inflight::Bool=false,
)
    algorithm in (:dp, :ahpc) || throw(ArgumentError("checkpoint algorithm must be :dp or :ahpc"))
    left_mebf >= 1 && right_mebf >= 1 || throw(ArgumentError("state-pair indices must be positive"))
    prefix = inflight ? "inflight" : "complete"
    return joinpath(session.directory, String(algorithm), @sprintf("%s_%04d_%04d.jld2", prefix, left_mebf, right_mebf))
end

"""
    load_state_pair_checkpoint(session, algorithm, left_mebf, right_mebf)

Load a completed state-pair scalar checkpoint or return `nothing` when no file exists.
The session and file are read-only, returned values are small CPU data, and malformed or incompatible payloads raise rather than silently falling back.
"""
function load_state_pair_checkpoint(
    session::CheckpointSession,
    algorithm::Symbol,
    left_mebf::Integer,
    right_mebf::Integer,
)
    path = _state_pair_checkpoint_path(session, algorithm, left_mebf, right_mebf)
    isfile(path) || return nothing
    data = _load_checkpoint_payload(path, session)
    String(data["kind"]) == "completed-state-pair" || throw(ArgumentError("unexpected checkpoint kind in $path"))
    Int(data["left_mebf"]) == left_mebf || throw(ArgumentError("checkpoint left state mismatch in $path"))
    Int(data["right_mebf"]) == right_mebf || throw(ArgumentError("checkpoint right state mismatch in $path"))
    return data
end

"""
    save_state_pair_checkpoint!(session, algorithm, left_mebf, right_mebf, overlap, one_body, two_body, pair_count; metadata=Dict())

Atomically commit one completed state-pair scalar result and return its path.
All scientific values are read-only and retain their units, only KiB-scale host metadata are written, and non-finite scalars or negative counts raise before replacement.
"""
function save_state_pair_checkpoint!(
    session::CheckpointSession,
    algorithm::Symbol,
    left_mebf::Integer,
    right_mebf::Integer,
    overlap::T,
    one_body::T,
    two_body::T,
    pair_count::Integer;
    metadata::AbstractDict{<:AbstractString,<:Any}=Dict{String,Any}(),
) where {T<:AbstractFloat}
    all(isfinite, (overlap, one_body, two_body)) || throw(ArgumentError("state-pair checkpoint scalars must be finite"))
    pair_count >= 0 || throw(ArgumentError("state-pair checkpoint count must be nonnegative"))
    payload = Dict{String,Any}(String(key) => value for (key, value) in metadata)
    merge!(payload, Dict{String,Any}(
        "version" => NOCI_CHECKPOINT_VERSION,
        "scientific_signature" => session.scientific_signature,
        "execution_signature" => session.execution_signature,
        "kind" => "completed-state-pair",
        "algorithm" => String(algorithm),
        "left_mebf" => Int(left_mebf),
        "right_mebf" => Int(right_mebf),
        "overlap" => overlap,
        "one_body" => one_body,
        "two_body" => two_body,
        "pair_count" => Int(pair_count),
        "completed_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    ))
    path = _write_checkpoint_payload(_state_pair_checkpoint_path(session, algorithm, left_mebf, right_mebf), payload)
    _notify_checkpoint_commit!(session, :complete, algorithm, left_mebf, right_mebf, pair_count + 1, path)
    return path
end

"""
    load_inflight_checkpoint(session, algorithm, left_mebf, right_mebf)

Load the current cumulative-prefix checkpoint for one state pair or return `nothing` when absent.
The payload is small CPU metadata, no scientific tensors or CUDA state are restored, and incompatible or corrupted files raise explicitly.
"""
function load_inflight_checkpoint(
    session::CheckpointSession,
    algorithm::Symbol,
    left_mebf::Integer,
    right_mebf::Integer,
)
    path = _state_pair_checkpoint_path(session, algorithm, left_mebf, right_mebf; inflight=true)
    isfile(path) || return nothing
    data = _load_checkpoint_payload(path, session)
    String(data["kind"]) == "cumulative-prefix" || throw(ArgumentError("unexpected inflight checkpoint kind in $path"))
    Int(data["left_mebf"]) == left_mebf || throw(ArgumentError("inflight left state mismatch in $path"))
    Int(data["right_mebf"]) == right_mebf || throw(ArgumentError("inflight right state mismatch in $path"))
    return data
end

"""
    save_inflight_checkpoint!(session, algorithm, left_mebf, right_mebf, next_index, total_count, overlap, one_body, two_body; metadata=Dict())

Atomically overwrite one state-pair cumulative prefix after a configured I/O batching boundary.
Accumulator values and indices are read-only, only scalar host metadata are written, and invalid indices or non-finite sums raise without changing the prior valid prefix.
"""
function save_inflight_checkpoint!(
    session::CheckpointSession,
    algorithm::Symbol,
    left_mebf::Integer,
    right_mebf::Integer,
    next_index::Integer,
    total_count::Integer,
    overlap::T,
    one_body::T,
    two_body::T;
    metadata::AbstractDict{<:AbstractString,<:Any}=Dict{String,Any}(),
) where {T<:AbstractFloat}
    1 <= next_index <= total_count + 1 || throw(ArgumentError("cumulative-prefix next index is outside 1:total+1"))
    total_count >= 0 || throw(ArgumentError("cumulative-prefix total count must be nonnegative"))
    all(isfinite, (overlap, one_body, two_body)) || throw(ArgumentError("cumulative-prefix sums must be finite"))
    payload = Dict{String,Any}(String(key) => value for (key, value) in metadata)
    merge!(payload, Dict{String,Any}(
        "version" => NOCI_CHECKPOINT_VERSION,
        "scientific_signature" => session.scientific_signature,
        "execution_signature" => session.execution_signature,
        "kind" => "cumulative-prefix",
        "algorithm" => String(algorithm),
        "left_mebf" => Int(left_mebf),
        "right_mebf" => Int(right_mebf),
        "next_index" => Int(next_index),
        "total_count" => Int(total_count),
        "overlap" => overlap,
        "one_body" => one_body,
        "two_body" => two_body,
        "updated_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    ))
    path = _write_checkpoint_payload(_state_pair_checkpoint_path(session, algorithm, left_mebf, right_mebf; inflight=true), payload)
    _notify_checkpoint_commit!(session, :prefix, algorithm, left_mebf, right_mebf, next_index, path)
    return path
end
