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

 File:            src/runtime/cuda_resources/polynomial_spin_pair_cuda_multidevice.jl
 Purpose:         Deterministic binding of configured CUDA devices to contiguous union ranges.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    PolynomialSpinPairGPUAssignment

Bind one configured CUDA device to one deterministic contiguous union range for
polynomial spin-pair intermediate construction. The immutable value contains
scheduling metadata only; scientific arrays and CUDA state remain external.
"""
struct PolynomialSpinPairGPUAssignment
    device::Int
    union_range::UnitRange{Int}
end

"""
    _polynomial_spin_pair_gpu_assignments(n_union, gpu_devices)

Partition the union axis into deterministic contiguous balanced ranges assigned
to configured CUDA devices. Inputs are read-only, no CUDA context is initialized,
and invalid or duplicate device indices raise `ArgumentError`.
"""
function _polynomial_spin_pair_gpu_assignments(n_union::Integer, gpu_devices)
    n_unions = Int(n_union)
    n_unions >= 0 || throw(ArgumentError("union count must be nonnegative"))
    devices = Tuple(Int(device) for device in gpu_devices)
    isempty(devices) && throw(ArgumentError("at least one GPU device must be configured"))
    all(device -> device >= 0, devices) || throw(ArgumentError("GPU device indices must be nonnegative"))
    length(unique(devices)) == length(devices) || throw(ArgumentError("GPU device indices must be unique"))
    n_unions == 0 && return PolynomialSpinPairGPUAssignment[]
    worker_count = min(n_unions, length(devices))
    ranges = _balanced_auxiliary_ranges(n_unions, worker_count)
    return [PolynomialSpinPairGPUAssignment(devices[worker], ranges[worker]) for worker in 1:worker_count]
end

"""
    _polynomial_runtime_for_gpu_device(runtime, device)

Derive a one-device runtime from validated multi-device configuration while
preserving memory limits, scratch budgets, occupied-count bounds, and allocation
telemetry. No CUDA context is initialized and `device` must belong to the input
runtime device set.
"""
function _polynomial_runtime_for_gpu_device(runtime::RuntimeOptions, device::Integer)
    selected_device = Int(device)
    selected_device in runtime.gpu_devices || throw(ArgumentError("polynomial CUDA worker device is not present in runtime.gpu_devices"))
    return RuntimeOptions(
        use_gpu=true,
        gpu_devices=(selected_device,),
        max_host_bytes=runtime.max_host_bytes,
        max_device_bytes=runtime.max_device_bytes,
        scratch_bytes=runtime.scratch_bytes,
        gpu_intermediate_scratch_bytes_per_device=runtime.gpu_intermediate_scratch_bytes_per_device,
        contraction_host_scratch_bytes=runtime.contraction_host_scratch_bytes,
        max_active_occupied=runtime.max_active_occupied,
        collect_allocations=runtime.collect_allocations,
    )
end

"""
    _polynomial_spin_pair_gpu_configured_width(T, n_active_occupied, n_union, n_state_pair_active_responses, runtime)

Return the deterministic batch-width ceiling implied by configured per-device
intermediate scratch and optional device-memory limits. This preflight helper
initializes no CUDA context; an executing worker may choose a smaller width from
current free device memory.
"""
function _polynomial_spin_pair_gpu_configured_width(
    ::Type{T},
    n_active_occupied::Integer,
    n_union::Integer,
    n_state_pair_active_responses::Integer,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    n_unions = Int(n_union)
    n_active_responses = Int(n_state_pair_active_responses)
    0 <= occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA polynomial occupied count must lie in 0:32"))
    n_unions >= 1 || throw(ArgumentError("configured polynomial batch width requires at least one union"))
    n_active_responses >= 0 || throw(ArgumentError("state-pair active-response count must be nonnegative"))
    per_union_bytes = _polynomial_spin_pair_gpu_per_union_device_bytes(T, occupied)
    fixed_device_bytes = estimate_array_bytes(T, n_active_responses, n_active_responses)
    scratch_width = runtime.gpu_intermediate_scratch_bytes_per_device ÷ per_union_bytes
    scratch_width >= 1 || throw(ResourceLimitError(:device, per_union_bytes, runtime.gpu_intermediate_scratch_bytes_per_device))
    width = min(n_unions, scratch_width)
    if !isnothing(runtime.max_device_bytes)
        required_one = Base.Checked.checked_add(fixed_device_bytes, per_union_bytes)
        required_one <= runtime.max_device_bytes || throw(ResourceLimitError(:device, required_one, runtime.max_device_bytes))
        width = min(width, (runtime.max_device_bytes - fixed_device_bytes) ÷ per_union_bytes)
    end
    width >= 1 || throw(ResourceLimitError(
        :device,
        Base.Checked.checked_add(fixed_device_bytes, per_union_bytes),
        something(runtime.max_device_bytes, runtime.gpu_intermediate_scratch_bytes_per_device),
    ))
    return width
end
