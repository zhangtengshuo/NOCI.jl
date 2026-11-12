
"""
    _polynomial_spin_pair_gpu_batch_width(T, n_active_occupied, n_union, n_state_pair_active_responses, runtime)

Choose one CUDA worker's bounded batch width from the retained conservative
per-union memory estimate, configured limits, and current free device memory.
No scientific array is allocated and inability to hold one union raises rather
than selecting another numerical algorithm.
"""
function _polynomial_spin_pair_gpu_batch_width(
    ::Type{T},
    n_active_occupied::Integer,
    n_union::Integer,
    n_state_pair_active_responses::Integer,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    unions = Int(n_union)
    n_active_responses = Int(n_state_pair_active_responses)
    unions >= 1 || throw(ArgumentError("CUDA polynomial batch selection requires at least one union"))
    runtime.use_gpu || throw(ArgumentError("CUDA polynomial batch selection requires runtime.use_gpu=true"))
    length(runtime.gpu_devices) == 1 || throw(ArgumentError("one CUDA polynomial worker requires exactly one configured GPU"))
    _activate_cuda_device(runtime)
    per_union = _polynomial_spin_pair_gpu_per_union_device_bytes(T, occupied)
    fixed_device = estimate_array_bytes(T, n_active_responses, n_active_responses)
    scratch_width = runtime.gpu_intermediate_scratch_bytes_per_device ÷ per_union
    scratch_width >= 1 || throw(ResourceLimitError(:device, per_union, runtime.gpu_intermediate_scratch_bytes_per_device))

    configured_width = unions
    if !isnothing(runtime.max_device_bytes)
        runtime.max_device_bytes > fixed_device || throw(ResourceLimitError(
            :device,
            Base.Checked.checked_add(fixed_device, per_union),
            runtime.max_device_bytes,
        ))
        configured_width = min(configured_width, (runtime.max_device_bytes - fixed_device) ÷ per_union)
    end
    configured_width = min(configured_width, scratch_width)

    available = Int(CUDA.available_memory())
    fixed_with_reserve = Base.Checked.checked_add(fixed_device, AHPC_SPIN_PAIR_GPU_SAFETY_RESERVE_BYTES)
    available > fixed_with_reserve || throw(ResourceLimitError(
        :device,
        Base.Checked.checked_add(fixed_with_reserve, per_union),
        available,
    ))
    free_width = (available - fixed_with_reserve) ÷ per_union
    width = min(configured_width, free_width)
    width >= 1 || throw(ResourceLimitError(
        :device,
        Base.Checked.checked_add(fixed_with_reserve, per_union),
        available,
    ))
    return width, available
end
