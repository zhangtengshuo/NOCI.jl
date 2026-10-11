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

 File:            src/runtime/progress.jl
 Purpose:         Process memory sampling and real-time production progress logging.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _process_memory_bytes()

Read current and peak resident-set sizes for this process from Linux procfs and return `(rss_bytes, hwm_bytes)`.
Values are byte counts, no scientific or CUDA allocation is inspected, and unavailable fields return zero rather than interrupting a calculation.
"""
function _process_memory_bytes()
    rss_bytes = 0
    hwm_bytes = 0
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmRSS:")
                rss_bytes = parse(Int, split(line)[2]) * 1024
            elseif startswith(line, "VmHWM:")
                hwm_bytes = parse(Int, split(line)[2]) * 1024
            end
        end
    catch
        return 0, 0
    end
    return rss_bytes, hwm_bytes
end

"""
    open_progress_logger(sec_path, out_path; append=false, test_mode=false)

Open the real-time `.sec` progress stream and scientific `.out` stream and return their owning `ProgressLogger`.
The function creates parent directories and host files, does not initialize CUDA, and propagates path or permission failures before returning a partially owned logger.
"""
function open_progress_logger(
    sec_path::AbstractString,
    out_path::AbstractString;
    append::Bool=false,
    test_mode::Bool=false,
)
    mkpath(dirname(abspath(sec_path)))
    mode = append ? "a" : "w"
    sec_io = open(sec_path, mode)
    try
        out_io = open(out_path, mode)
        return ProgressLogger(sec_io, out_io, UInt64(time_ns()), test_mode)
    catch
        close(sec_io)
        rethrow()
    end
end

"""
    close_progress_logger!(logger)

Flush, fsync, and close both streams owned by a production progress logger.
The logger becomes unusable after return, no scientific arrays or CUDA resources are changed, and synchronization errors propagate after best-effort closure.
"""
function close_progress_logger!(logger::ProgressLogger)
    first_error = nothing
    for io in (logger.sec_io, logger.out_io)
        try
            _fsync_stream(io)
        catch error
            first_error === nothing && (first_error = error)
        finally
            close(io)
        end
    end
    first_error === nothing || throw(first_error)
    return nothing
end

"""
    progress_event!(logger, event; fields...)

Append one timestamped, parseable runtime event to `.sec` and stdout and immediately flush both streams.
The scientific `.out` stream is intentionally not modified: it remains a clean report while `.sec` owns real-time progress and resource diagnostics.
"""
function progress_event!(logger::ProgressLogger, event::Symbol; fields...)
    rss_bytes, hwm_bytes = _process_memory_bytes()
    elapsed_seconds = Float64(time_ns() - logger.start_ns) / 1.0e9
    pieces = String[
        Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS.sss"),
        @sprintf("elapsed_seconds=%.6f", elapsed_seconds),
        "event=$(event)",
        "rss_bytes=$(rss_bytes)",
        "hwm_bytes=$(hwm_bytes)",
    ]
    for key in sort!(collect(keys(fields)); by=string)
        push!(pieces, "$(key)=$(repr(fields[key]))")
    end
    line = join(pieces, ' ')
    println(logger.sec_io, line)
    println(stdout, line)
    flush(logger.sec_io)
    flush(stdout)
    return line
end

"""
    progress_event!(nothing, event; fields...)

Discard one optional progress event and return `nothing` for pure in-memory evaluation paths.
No input is mutated, no host file or CUDA side effect occurs, and all field values are ignored.
"""
function progress_event!(::Nothing, ::Symbol; fields...)
    return nothing
end

"""
    test_progress_event!(progress, event; fields...)

Emit one additional real-time progress event only when a production progress logger is running in test mode.
The observation path writes only `.sec` and stdout metadata, never changes scientific arrays, CUDA execution, screening, accumulation order, or checkpoint contents, and returns `nothing` when test mode is disabled.
"""
function test_progress_event!(logger::ProgressLogger, event::Symbol; fields...)
    logger.test_mode || return nothing
    progress_event!(logger, event; fields...)
    return nothing
end

function test_progress_event!(::Nothing, ::Symbol; fields...)
    return nothing
end

"""
    test_progress_enabled(progress)

Return whether the supplied progress context has explicit test-mode observation enabled.
The query is allocation-free, reads host metadata only, and never initializes CUDA or changes calculation state.
"""
test_progress_enabled(logger::ProgressLogger) = logger.test_mode
test_progress_enabled(::Nothing) = false

"""
    with_test_progress(work, progress, event; interval_seconds=10.0, fields...)

Execute `work` unchanged while test mode emits start, periodic running, and terminal success/failure records to `.sec` and stdout.
Heartbeat fields are diagnostic host metadata only; the timer never reads or mutates scientific arrays or CUDA buffers, normal mode executes `work` directly with no timer, and scientific exceptions are rethrown unchanged after a best-effort failed-phase record.
"""
function with_test_progress(
    work::F,
    logger::ProgressLogger,
    event::Symbol;
    interval_seconds::Real=10.0,
    fields...,
) where {F<:Function}
    logger.test_mode || return work()
    interval = Float64(interval_seconds)
    isfinite(interval) && interval > 0.0 || throw(ArgumentError("test progress interval must be finite and positive"))
    progress_event!(logger, event; status=:started, fields...)
    heartbeat_timer = Timer(interval; interval=interval) do _
        try
            progress_event!(logger, event; status=:running, fields...)
        catch
        end
    end
    try
        result = work()
        progress_event!(logger, event; status=:finished, fields...)
        return result
    catch error
        try
            progress_event!(logger, event; status=:failed, error=sprint(showerror, error), fields...)
        catch
        end
        rethrow()
    finally
        close(heartbeat_timer)
    end
end

function with_test_progress(
    work::F,
    ::Nothing,
    ::Symbol;
    interval_seconds::Real=10.0,
    fields...,
) where {F<:Function}
    return work()
end
