# File: ext/gpu_implicit_spin_feature_contraction.jl
# Purpose: Implement optional CUDA kernels for implicit spin-feature contraction.
# SPDX-License-Identifier: Apache-2.0


"""
    _scatter_spin_pair_alpha_first_cofactor_response_kernel!(features, packed_values, feature_offsets, feature_pairs, block_value_first, width, rank)

Scatter one packed alpha-support first-cofactor block into its dense active-response columns on the GPU.
Every CUDA thread owns one output column, packed values and structural masks are read-only, no floating-point reduction occurs, and launch dimensions are supplied by the caller.
"""
function _scatter_spin_pair_alpha_first_cofactor_response_kernel!(
    features,
    packed_values,
    feature_offsets,
    feature_pairs,
    block_value_first::Int64,
    width::Int32,
    rank::Int32,
)
    local_kappa = (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) + Int(CUDA.threadIdx().x)
    if local_kappa <= Int(width)
        pair = @inbounds feature_pairs[local_kappa]
        feature_slot = Int(@inbounds(feature_offsets[local_kappa])) - Int(block_value_first) + 1
        expected_end = Int(@inbounds(feature_offsets[local_kappa + 1])) - Int(block_value_first) + 1
        right_bits = pair.right_bits
        while !iszero(right_bits)
            right_active = trailing_zeros(right_bits) + 1
            left_bits = pair.left_bits
            while !iszero(left_bits)
                left_active = trailing_zeros(left_bits) + 1
                response = left_active + (right_active - 1) * Int(rank)
                @inbounds features[response, local_kappa] = packed_values[feature_slot]
                feature_slot += 1
                left_bits &= left_bits - UInt32(1)
            end
            right_bits &= right_bits - UInt32(1)
        end
        feature_slot == expected_end || return nothing
    end
    return nothing
end

"""
    _project_spin_pair_alpha_first_cofactor_response_gpu_packed!(coulomb_projected_alpha_response, intermediates, kappa_first, kappa_last, coulomb_device, alpha_first_cofactor_response_device, coulomb_projected_alpha_response_device, packed_value_device, feature_offset_device, feature_pair_device)

Upload one contiguous packed alpha first-cofactor slice and compact structural metadata, scatter it directly into reusable CUDA feature storage, project with cuBLAS, and copy only the Coulomb-projected alpha-response block to CPU.
Scientific inputs are read-only, no floating-point reduction is introduced before the existing GEMM, device buffers are reused without CPU fallback, and invalid ranges or undersized buffers raise before launch.
When phase timing is requested, reusable CUDA events measure the packed scatter stage and GEMM on the device timeline and are read after the blocking D2H copy completes; return-copy wall time includes its necessary device wait, avoiding an extra event synchronization without changing scientific arithmetic or queue order.
"""
function _project_spin_pair_alpha_first_cofactor_response_gpu_packed!(
    coulomb_projected_alpha_response::AbstractMatrix{T},
    intermediates::ImplicitSpinPairIntermediates{T},
    kappa_first::Int,
    kappa_last::Int,
    coulomb_device,
    alpha_first_cofactor_response_device_storage,
    coulomb_projected_alpha_response_device_storage,
    packed_value_device_storage,
    feature_offset_device_storage,
    feature_pair_device_storage;
    phase_timing::Union{Nothing,AHPCContractionPhaseTiming}=nothing,
    cuda_scatter_start_event=nothing,
    cuda_scatter_stop_event=nothing,
    cuda_gemm_stop_event=nothing,
) where {T<:AbstractFloat}
    1 <= kappa_first <= kappa_last <= length(intermediates.feature_pairs) || throw(BoundsError(intermediates.feature_pairs, kappa_first:kappa_last))
    width = kappa_last - kappa_first + 1
    n_response = Base.Checked.checked_mul(intermediates.active_rank, intermediates.active_rank)
    size(coulomb_projected_alpha_response) == (n_response, width) || throw(DimensionMismatch("Coulomb-projected alpha-response block has incorrect dimensions"))
    size(alpha_first_cofactor_response_device_storage, 1) == n_response && size(alpha_first_cofactor_response_device_storage, 2) >= width || throw(DimensionMismatch("CUDA alpha-feature storage is too small"))
    size(coulomb_projected_alpha_response_device_storage) == size(alpha_first_cofactor_response_device_storage) || throw(DimensionMismatch("CUDA Coulomb-projected alpha-response storage differs from source storage"))
    length(feature_offset_device_storage) >= width + 1 || throw(DimensionMismatch("CUDA alpha-feature offset storage is too small"))
    length(feature_pair_device_storage) >= width || throw(DimensionMismatch("CUDA alpha-feature pair storage is too small"))
    block_value_first = intermediates.feature_offsets[kappa_first]
    block_value_end = intermediates.feature_offsets[kappa_last + 1]
    packed_count = block_value_end - block_value_first
    packed_count >= 0 || throw(ArgumentError("alpha packed feature offsets are decreasing"))
    length(packed_value_device_storage) >= packed_count || throw(DimensionMismatch("CUDA packed alpha-feature value storage is too small"))

    timing_enabled = !isnothing(phase_timing)
    if timing_enabled
        isnothing(cuda_scatter_start_event) && throw(ArgumentError("CUDA scatter start event is required when AHPC contraction phase timing is enabled"))
        isnothing(cuda_scatter_stop_event) && throw(ArgumentError("CUDA scatter stop event is required when AHPC contraction phase timing is enabled"))
        isnothing(cuda_gemm_stop_event) && throw(ArgumentError("CUDA GEMM stop event is required when AHPC contraction phase timing is enabled"))
    end

    alpha_first_cofactor_response_device = @view alpha_first_cofactor_response_device_storage[:, 1:width]
    coulomb_projected_alpha_response_device = @view coulomb_projected_alpha_response_device_storage[:, 1:width]
    timing_enabled && CUDA.record(cuda_scatter_start_event)
    fill!(alpha_first_cofactor_response_device, zero(T))
    packed_count > 0 && Base.unsafe_copyto!(packed_value_device_storage, 1, intermediates.first_cofactor_values, block_value_first, packed_count)
    Base.unsafe_copyto!(feature_offset_device_storage, 1, intermediates.feature_offsets, kappa_first, width + 1)
    Base.unsafe_copyto!(feature_pair_device_storage, 1, intermediates.feature_pairs, kappa_first, width)
    threads = 256
    blocks = cld(width, threads)
    CUDA.@cuda threads=threads blocks=blocks _scatter_spin_pair_alpha_first_cofactor_response_kernel!(
        alpha_first_cofactor_response_device,
        packed_value_device_storage,
        feature_offset_device_storage,
        feature_pair_device_storage,
        Int64(block_value_first),
        Int32(width),
        Int32(intermediates.active_rank),
    )
    timing_enabled && CUDA.record(cuda_scatter_stop_event)
    mul!(coulomb_projected_alpha_response_device, transpose(coulomb_device), alpha_first_cofactor_response_device)
    timing_enabled && CUDA.record(cuda_gemm_stop_event)
    destination = parent(coulomb_projected_alpha_response)
    length(destination) >= length(coulomb_projected_alpha_response_device) || throw(DimensionMismatch("Coulomb-projected alpha-response host storage is smaller than the CUDA block"))
    pointer(coulomb_projected_alpha_response) == pointer(destination) || throw(ArgumentError("Coulomb-projected alpha-response CUDA destination must begin at the first element of its parent host matrix"))
    if timing_enabled
        d2h_started_ns = time_ns()
        Base.unsafe_copyto!(destination, 1, coulomb_projected_alpha_response_device, 1, length(coulomb_projected_alpha_response_device))
        phase_timing.d2h_seconds += Float64(time_ns() - d2h_started_ns) / 1.0e9
        phase_timing.cuda_scatter_seconds += Float64(CUDA.elapsed(cuda_scatter_start_event, cuda_scatter_stop_event))
        phase_timing.cuda_gemm_seconds += Float64(CUDA.elapsed(cuda_scatter_stop_event, cuda_gemm_stop_event))
    else
        Base.unsafe_copyto!(destination, 1, coulomb_projected_alpha_response_device, 1, length(coulomb_projected_alpha_response_device))
    end
    return coulomb_projected_alpha_response
end
