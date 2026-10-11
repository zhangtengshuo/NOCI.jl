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

 File:            src/runtime/telemetry.jl
 Purpose:         Common telemetry: algorithm identity, phase timing, memory estimates, and preflight outcomes.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    AHPC_MAX_OCCUPIED_RANK

Maximum supported occupied-spin rank `q` for the AHPC polynomial CPU/CUDA numerical kernels; this does not limit the active-orbital bit-mask rank.
"""
const AHPC_MAX_OCCUPIED_RANK = 32

"""
    RuntimeOptions(; use_gpu=false, gpu_device=nothing, gpu_devices=nothing, max_host_bytes=nothing, max_device_bytes=nothing, scratch_bytes=256 * 1024^2, max_active_occupied=32, collect_allocations=true)

Configure execution resources and telemetry independently of scientific algorithm identity.
`gpu_devices` is an ordered tuple of unique zero-based CUDA devices whose first entry is the primary device, byte limits and generic/intermediate/contraction scratch sizes are nonnegative integers, `max_active_occupied` is the production occupied-spin limit in `0:32` for CPU and CUDA, and active-orbital bit masks retain their separate 64-orbital representation limit.
"""
struct RuntimeOptions
    use_gpu::Bool
    gpu_device::Int
    gpu_devices::Tuple{Vararg{Int}}
    max_host_bytes::Union{Nothing,Int}
    max_device_bytes::Union{Nothing,Int}
    scratch_bytes::Int
    gpu_intermediate_scratch_bytes_per_device::Int
    contraction_host_scratch_bytes::Int
    max_active_occupied::Int
    collect_allocations::Bool
end

"""
    RuntimeOptions(; use_gpu=false, gpu_device=nothing, gpu_devices=nothing, max_host_bytes=nothing, max_device_bytes=nothing, scratch_bytes=256 * 1024^2, max_active_occupied=32, collect_allocations=true)

Construct validated runtime settings.
Limits and all scratch capacities are measured in bytes, device collections are copied into an immutable tuple, inputs are not mutated, the occupied-spin limit accepts `0:32`, and no CUDA runtime is initialized.
"""
function RuntimeOptions(;
    use_gpu::Bool=false,
    gpu_device::Union{Nothing,Integer}=nothing,
    gpu_devices=nothing,
    max_host_bytes::Union{Nothing,Integer}=nothing,
    max_device_bytes::Union{Nothing,Integer}=nothing,
    scratch_bytes::Integer=256 * 1024^2,
    gpu_intermediate_scratch_bytes_per_device::Union{Nothing,Integer}=nothing,
    contraction_host_scratch_bytes::Union{Nothing,Integer}=nothing,
    max_active_occupied::Integer=AHPC_MAX_OCCUPIED_RANK,
    collect_allocations::Bool=true,
)
    devices = if isnothing(gpu_devices)
        (Int(something(gpu_device, 0)),)
    else
        values = Tuple(Int(device) for device in gpu_devices)
        isempty(values) && throw(ArgumentError("gpu_devices must not be empty"))
        !isnothing(gpu_device) && Int(gpu_device) != first(values) && throw(ArgumentError("gpu_device must equal the first gpu_devices entry"))
        values
    end
    all(device -> device >= 0, devices) || throw(ArgumentError("GPU device indices must be nonnegative"))
    length(unique(devices)) == length(devices) || throw(ArgumentError("gpu_devices must contain unique indices"))
    isnothing(max_host_bytes) || max_host_bytes >= 0 || throw(ArgumentError("max_host_bytes must be nonnegative"))
    isnothing(max_device_bytes) || max_device_bytes >= 0 || throw(ArgumentError("max_device_bytes must be nonnegative"))
    scratch_bytes >= 0 || throw(ArgumentError("scratch_bytes must be nonnegative"))
    intermediate_scratch = isnothing(gpu_intermediate_scratch_bytes_per_device) ? Int(scratch_bytes) : Int(gpu_intermediate_scratch_bytes_per_device)
    contraction_scratch = isnothing(contraction_host_scratch_bytes) ? Int(scratch_bytes) : Int(contraction_host_scratch_bytes)
    intermediate_scratch >= 0 || throw(ArgumentError("gpu_intermediate_scratch_bytes_per_device must be nonnegative"))
    contraction_scratch >= 0 || throw(ArgumentError("contraction_host_scratch_bytes must be nonnegative"))
    0 <= max_active_occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("max_active_occupied must lie in 0:32"))
    return RuntimeOptions(
        use_gpu,
        first(devices),
        devices,
        isnothing(max_host_bytes) ? nothing : Int(max_host_bytes),
        isnothing(max_device_bytes) ? nothing : Int(max_device_bytes),
        Int(scratch_bytes),
        intermediate_scratch,
        contraction_scratch,
        Int(max_active_occupied),
        collect_allocations,
    )
end

"""
    ahpc_threads_per_gpu(runtime)

Validate the AHPC CPU/GPU topology and return the number of Julia threads assigned to each configured GPU.
CPU-only execution returns all Julia threads, GPU execution requires at least one device and an integer thread-to-device ratio, no CUDA context is initialized, and invalid asymmetric resource requests raise `ArgumentError` before scientific loading.
"""
function ahpc_threads_per_gpu(runtime::RuntimeOptions)
    runtime.use_gpu || return Threads.nthreads()
    gpu_count = length(runtime.gpu_devices)
    gpu_count >= 1 || throw(ArgumentError("AHPC GPU execution requires at least one configured GPU"))
    thread_count = Threads.nthreads()
    thread_count % gpu_count == 0 || throw(ArgumentError("AHPC requires Julia thread count ($thread_count) to be an integer multiple of GPU count ($gpu_count)"))
    return div(thread_count, gpu_count)
end

"""
    MemoryEstimate(host_bytes=0, device_bytes=0)

Represent a pre-allocation host/device memory estimate in bytes.
Both counts must be nonnegative and the value contains no live allocation handles.
"""
struct MemoryEstimate
    host_bytes::Int
    device_bytes::Int

    """
        MemoryEstimate(host_bytes=0, device_bytes=0)

    Construct a validated byte estimate, rejecting negative counts before any scientific allocation.
    """
    function MemoryEstimate(host_bytes::Integer=0, device_bytes::Integer=0)
        host_bytes >= 0 || throw(ArgumentError("host_bytes must be nonnegative"))
        device_bytes >= 0 || throw(ArgumentError("device_bytes must be nonnegative"))
        return new(Int(host_bytes), Int(device_bytes))
    end
end

"""
    MemoryPreflight(estimate, passed, limiting_resource)

Record a memory-limit decision before large host or device allocation.
`limiting_resource` is `nothing`, `:host`, or `:device`.
"""
struct MemoryPreflight
    estimate::MemoryEstimate
    passed::Bool
    limiting_resource::Union{Nothing,Symbol}
end

"""
    PhaseTelemetry

Record one execution phase using seconds and bytes.
Metadata is diagnostic only and must not influence scientific results.
"""
struct PhaseTelemetry
    name::Symbol
    wall_seconds::Float64
    estimated_host_bytes::Int
    estimated_device_bytes::Int
    allocated_bytes::Union{Nothing,Int}
    metadata::Dict{Symbol,Any}
end

"""
    ExecutionTelemetry(algorithm, backend)

Create mutable telemetry for one evaluation.
The constructor copies algorithm/backend identifiers and starts with no phases or counters.
"""
mutable struct ExecutionTelemetry
    algorithm::String
    backend::String
    phases::Vector{PhaseTelemetry}
    counters::Dict{Symbol,Int}
    preflight_passed::Bool
end

"""
    ExecutionTelemetry(algorithm, backend)

Construct empty execution telemetry from string-like stable identifiers.
No CPU/GPU scientific workspace is allocated.
"""
function ExecutionTelemetry(algorithm::AbstractString, backend::AbstractString)
    return ExecutionTelemetry(String(algorithm), String(backend), PhaseTelemetry[], Dict{Symbol,Int}(), true)
end

"""
    estimate_array_bytes(T, dimensions...)

Return the exact storage in bytes for a dense array of element type `T` and nonnegative dimensions.
The function allocates no array and raises `ArgumentError` for negative dimensions or `OverflowError` when the byte product exceeds `Int`.
"""
function estimate_array_bytes(::Type{T}, dimensions::Integer...) where {T}
    elements = 1
    for dimension in dimensions
        dimension >= 0 || throw(ArgumentError("array dimensions must be nonnegative"))
        elements = Base.Checked.checked_mul(elements, Int(dimension))
    end
    return Base.Checked.checked_mul(elements, sizeof(T))
end

"""
    combine_memory_estimates(estimates...)

Add host and device byte estimates with overflow checking.
Inputs are immutable and are not modified; `OverflowError` is raised if either total exceeds `Int`.
"""
function combine_memory_estimates(estimates::MemoryEstimate...)
    host_bytes = 0
    device_bytes = 0
    for estimate in estimates
        host_bytes = Base.Checked.checked_add(host_bytes, estimate.host_bytes)
        device_bytes = Base.Checked.checked_add(device_bytes, estimate.device_bytes)
    end
    return MemoryEstimate(host_bytes, device_bytes)
end

"""
    check_memory_limits(estimate, runtime)

Compare a memory estimate against configured host/device byte limits without allocating the estimated storage.
Return a `MemoryPreflight` result; this query never throws `ResourceLimitError` and does not initialize CUDA.
"""
function check_memory_limits(estimate::MemoryEstimate, runtime::RuntimeOptions)
    if !isnothing(runtime.max_host_bytes) && estimate.host_bytes > runtime.max_host_bytes
        return MemoryPreflight(estimate, false, :host)
    end
    if !isnothing(runtime.max_device_bytes) && estimate.device_bytes > runtime.max_device_bytes
        return MemoryPreflight(estimate, false, :device)
    end
    return MemoryPreflight(estimate, true, nothing)
end

"""
    enforce_memory_limits(estimate, runtime)

Return a passing `MemoryPreflight` or raise `ResourceLimitError` before large allocation.
Required and limiting values are measured in bytes; the function has no device side effects.
"""
function enforce_memory_limits(estimate::MemoryEstimate, runtime::RuntimeOptions)
    preflight = check_memory_limits(estimate, runtime)
    preflight.passed && return preflight
    limit = preflight.limiting_resource === :host ? runtime.max_host_bytes : runtime.max_device_bytes
    required = preflight.limiting_resource === :host ? estimate.host_bytes : estimate.device_bytes
    throw(ResourceLimitError(preflight.limiting_resource, required, something(limit)))
end

"""
    record_phase!(work, telemetry, name; estimate=MemoryEstimate(), metadata=Dict())

Run `work` exactly once, measure wall time with the monotonic `time_ns()` clock, optionally observe Julia allocations, append phase telemetry, and return the work result.
Estimated sizes are bytes, metadata is copied, and scientific inputs are modified only if `work` itself mutates them.
"""
function record_phase!(
    work::F,
    telemetry::ExecutionTelemetry,
    name::Symbol;
    estimate::MemoryEstimate=MemoryEstimate(),
    metadata::AbstractDict{Symbol,<:Any}=Dict{Symbol,Any}(),
    collect_allocations::Bool=true,
) where {F<:Function}
    start_ns = time_ns()
    if collect_allocations
        timed = @timed work()
        value = timed.value
        allocated_bytes = timed.bytes
    else
        value = work()
        allocated_bytes = nothing
    end
    wall_seconds = Float64(time_ns() - start_ns) / 1.0e9
    push!(
        telemetry.phases,
        PhaseTelemetry(
            name,
            wall_seconds,
            estimate.host_bytes,
            estimate.device_bytes,
            allocated_bytes,
            Dict{Symbol,Any}(metadata),
        ),
    )
    return value
end

"""
    set_counter!(telemetry, name, value)

Set a nonnegative integer telemetry counter and return the stored value.
The function mutates only `telemetry.counters` and raises `ArgumentError` for negative values.
"""
function set_counter!(telemetry::ExecutionTelemetry, name::Symbol, value::Integer)
    value >= 0 || throw(ArgumentError("telemetry counters must be nonnegative"))
    telemetry.counters[name] = Int(value)
    return telemetry.counters[name]
end

"""
    increment_counter!(telemetry, name, increment=1)

Increase a telemetry counter by a nonnegative integer and return the new value.
The function mutates only `telemetry.counters` and uses checked integer addition.
"""
function increment_counter!(telemetry::ExecutionTelemetry, name::Symbol, increment::Integer=1)
    increment >= 0 || throw(ArgumentError("telemetry increments must be nonnegative"))
    current = get(telemetry.counters, name, 0)
    telemetry.counters[name] = Base.Checked.checked_add(current, Int(increment))
    return telemetry.counters[name]
end
