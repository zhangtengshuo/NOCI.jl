
"""
    LUCertifiedHybridGPUWorkspace

Own one CUDA worker's bounded arrays for inverse-based second complementary
cofactor contraction. LU factorization, inverse construction, certification,
determinant restoration, and first cofactors remain CPU operations.
"""
mutable struct LUCertifiedHybridGPUWorkspace{T<:Union{Float32,Float64}}
    device::Int
    n_active_occupied::Int
    batch_width::Int
    scaled_inverse::CUDA.CuMatrix{T}
    active_response_or_hessian_pair_indices::CUDA.CuMatrix{UInt16}
    determinant_scaled::CUDA.CuVector{T}
    accepted::CUDA.CuVector{UInt8}
    contracted_second_scaled::CUDA.CuVector{T}
    local_pair_first::CUDA.CuVector{UInt8}
    local_pair_second::CUDA.CuVector{UInt8}
    closed::Bool
end

function _lu_certified_hybrid_batch_width(
    ::Type{T},
    n_active_occupied::Integer,
    n_union::Integer,
    state_pair_hessian_response_bytes::Integer,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    q = Int(n_active_occupied)
    unions = Int(n_union)
    fixed_device = Int(state_pair_hessian_response_bytes)
    1 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("hybrid LU-certified occupied count must lie in 1:32"))
    unions >= 1 || throw(ArgumentError("hybrid LU-certified batch selection requires at least one union"))
    runtime.use_gpu || throw(ArgumentError("hybrid LU-certified batch selection requires runtime.use_gpu=true"))
    length(runtime.gpu_devices) == 1 || throw(ArgumentError("one hybrid CUDA worker requires exactly one configured GPU"))
    _activate_cuda_device(runtime)

    per_union = _lu_certified_hybrid_per_union_device_bytes(T, q)
    scratch_width = runtime.gpu_intermediate_scratch_bytes_per_device ÷ per_union
    scratch_width >= 1 || throw(ResourceLimitError(:device, per_union, runtime.gpu_intermediate_scratch_bytes_per_device))
    configured_width = min(unions, scratch_width)

    if !isnothing(runtime.max_device_bytes)
        required_one = Base.Checked.checked_add(fixed_device, per_union)
        required_one <= runtime.max_device_bytes || throw(ResourceLimitError(:device, required_one, runtime.max_device_bytes))
        configured_width = min(configured_width, (runtime.max_device_bytes - fixed_device) ÷ per_union)
    end

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

function LUCertifiedHybridGPUWorkspace(
    ::Type{T},
    n_active_occupied::Integer,
    batch_width::Integer,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    q = Int(n_active_occupied)
    width = Int(batch_width)
    1 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("hybrid LU-certified occupied count must lie in 1:32"))
    width >= 1 || throw(ArgumentError("hybrid LU-certified batch width must be positive"))
    runtime.use_gpu || throw(ArgumentError("hybrid LU-certified workspace requires runtime.use_gpu=true"))
    length(runtime.gpu_devices) == 1 || throw(ArgumentError("hybrid LU-certified workspace requires exactly one GPU"))
    _activate_cuda_device(runtime)
    q2 = Base.Checked.checked_mul(q, q)
    pair_first_host, pair_second_host = _lu_certified_hybrid_local_pairs(q)
    allocated = Any[]
    try
        scaled_inverse = CUDA.CuArray{T}(undef, q2, width); push!(allocated, scaled_inverse)
        indices = CUDA.CuArray{UInt16}(undef, q2, width); push!(allocated, indices)
        determinant_scaled = CUDA.CuArray{T}(undef, width); push!(allocated, determinant_scaled)
        accepted = CUDA.CuArray{UInt8}(undef, width); push!(allocated, accepted)
        contracted_second_scaled = CUDA.CuArray{T}(undef, width); push!(allocated, contracted_second_scaled)
        local_pair_first = CUDA.CuArray(pair_first_host); push!(allocated, local_pair_first)
        local_pair_second = CUDA.CuArray(pair_second_host); push!(allocated, local_pair_second)
        return LUCertifiedHybridGPUWorkspace(
            runtime.gpu_device,
            q,
            width,
            scaled_inverse,
            indices,
            determinant_scaled,
            accepted,
            contracted_second_scaled,
            local_pair_first,
            local_pair_second,
            false,
        )
    catch
        for array in Iterators.reverse(allocated)
            CUDA.unsafe_free!(array)
        end
        rethrow()
    end
end

function _close_lu_certified_hybrid_gpu_workspace!(workspace::LUCertifiedHybridGPUWorkspace)
    workspace.closed && return nothing
    CUDA.device!(workspace.device)
    for array in (
        workspace.local_pair_second,
        workspace.local_pair_first,
        workspace.contracted_second_scaled,
        workspace.accepted,
        workspace.determinant_scaled,
        workspace.active_response_or_hessian_pair_indices,
        workspace.scaled_inverse,
    )
        CUDA.unsafe_free!(array)
    end
    workspace.closed = true
    return nothing
end

"""
    _lu_certified_hybrid_second_cofactor_kernel!(...)

Contract the inverse-based second complementary cofactor on CUDA with one block
per union. Pair-pair terms are independent across threads; each thread accumulates
a strided subset and the block reduces to one scaled contracted-second scalar.
The determinant-Hessian response uses the same unordered-pair convention as the
CPU LU-certified path, including the required factor of two.
"""
function _lu_certified_hybrid_second_cofactor_kernel!(
    contracted_second_scaled,
    scaled_inverse,
    determinant_scaled,
    accepted,
    hessian_row_pair_indices,
    hessian_column_pair_indices,
    state_pair_hessian_response,
    local_pair_first,
    local_pair_second,
    n_active_occupied_device::Int32,
    n_hessian_pairs_device::Int32,
    batch_count::Int32,
)
    union_id = Int(CUDA.blockIdx().x)
    thread_id = Int(CUDA.threadIdx().x)
    q = Int(n_active_occupied_device)
    n_pairs = Int(n_hessian_pairs_device)
    T = eltype(scaled_inverse)
    partial = zero(T)

    if union_id <= Int(batch_count) && @inbounds(accepted[union_id]) == UInt8(1) && n_pairs > 0
        pair_pair_count = n_pairs * n_pairs
        linear_pair = thread_id
        stride = Int(CUDA.blockDim().x)
        @inbounds while linear_pair <= pair_pair_count
            row_pair_id = (linear_pair - 1) % n_pairs + 1
            column_pair_id = div(linear_pair - 1, n_pairs) + 1

            first_row = Int(local_pair_first[row_pair_id])
            second_row = Int(local_pair_second[row_pair_id])
            first_column = Int(local_pair_first[column_pair_id])
            second_column = Int(local_pair_second[column_pair_id])

            inverse_wedge =
                scaled_inverse[first_column + (first_row - 1) * q, union_id] *
                scaled_inverse[second_column + (second_row - 1) * q, union_id] -
                scaled_inverse[first_column + (second_row - 1) * q, union_id] *
                scaled_inverse[second_column + (first_row - 1) * q, union_id]

            state_pair_row_pair = Int(hessian_row_pair_indices[row_pair_id, union_id])
            state_pair_column_pair = Int(hessian_column_pair_indices[column_pair_id, union_id])
            partial += state_pair_hessian_response[state_pair_row_pair, state_pair_column_pair] * inverse_wedge
            linear_pair += stride
        end
    end

    shared = CUDA.CuStaticSharedArray(T, AHPC_LU_HYBRID_SECOND_THREADS)
    @inbounds shared[thread_id] = partial
    CUDA.sync_threads()

    reduction_stride = AHPC_LU_HYBRID_SECOND_THREADS ÷ 2
    while reduction_stride >= 1
        if thread_id <= reduction_stride
            @inbounds shared[thread_id] += shared[thread_id + reduction_stride]
        end
        CUDA.sync_threads()
        reduction_stride ÷= 2
    end

    if thread_id == 1 && union_id <= Int(batch_count)
        if @inbounds(accepted[union_id]) == UInt8(1) && n_pairs > 0
            @inbounds contracted_second_scaled[union_id] =
                T(2) * determinant_scaled[union_id] * shared[1]
        else
            @inbounds contracted_second_scaled[union_id] = zero(T)
        end
    end
    return nothing
end

function _launch_lu_certified_hybrid_second_cofactor!(
    workspace::LUCertifiedHybridGPUWorkspace,
    row_pair_indices_device,
    column_pair_indices_device,
    state_pair_hessian_response_device,
    n_active_occupied::Int,
    width::Int,
)
    n_pairs = _hessian_pair_count(n_active_occupied)
    CUDA.@cuda threads=AHPC_LU_HYBRID_SECOND_THREADS blocks=width _lu_certified_hybrid_second_cofactor_kernel!(
        workspace.contracted_second_scaled,
        workspace.scaled_inverse,
        workspace.determinant_scaled,
        workspace.accepted,
        row_pair_indices_device,
        column_pair_indices_device,
        state_pair_hessian_response_device,
        workspace.local_pair_first,
        workspace.local_pair_second,
        Int32(n_active_occupied),
        Int32(n_pairs),
        Int32(width),
    )
    return nothing
end

function _build_lu_certified_hybrid_gpu_worker!(
    cofactor_scalar::Vector{T},
    one_body_active::Vector{T},
    two_body_inactive_active::Vector{T},
    same_spin_two_body::Vector{T},
    first_cofactor_values::Vector{T},
    state_pair_hessian_response::Matrix{T},
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    support::SpinPairSupportDomain,
    feature_offsets::Vector{Int},
    assignment::PolynomialSpinPairGPUAssignment,
    n_active_occupied::Int,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    worker_started_ns = time_ns()
    union_range = assignment.union_range
    isempty(union_range) && throw(ArgumentError("hybrid LU-certified CUDA worker range must be nonempty"))
    worker_runtime = _polynomial_runtime_for_gpu_device(runtime, assignment.device)
    active_rank = size(preparation.effective_active_overlap, 1)
    n_state_pair_hessian_pairs = _hessian_pair_count(active_rank)
    state_pair_hessian_response_bytes =
        estimate_array_bytes(T, n_state_pair_hessian_pairs, n_state_pair_hessian_pairs)
    batch_width, available_before = _lu_certified_hybrid_batch_width(
        T,
        n_active_occupied,
        length(union_range),
        state_pair_hessian_response_bytes,
        worker_runtime,
    )

    q2 = n_active_occupied * n_active_occupied
    n_local_hessian_pairs = _hessian_pair_count(n_active_occupied)
    staging = LUCertifiedHybridHostBuffers(T, n_active_occupied, batch_width)
    gpu_workspace = nothing
    state_pair_hessian_response_device = nothing
    h2d_bytes = 0
    d2h_bytes = 0
    batch_count = 0
    inverse_calls = 0
    accepted = 0
    fallbacks = 0
    cpu_lu_wall_seconds = 0.0
    gpu_second_wall_seconds = 0.0

    _activate_cuda_device(worker_runtime)
    try
        gpu_workspace = LUCertifiedHybridGPUWorkspace(T, n_active_occupied, batch_width, worker_runtime)
        state_pair_hessian_response_device = CUDA.CuArray(state_pair_hessian_response)

        union_first = first(union_range)
        final_union = last(union_range)
        while union_first <= final_union
            union_last = min(final_union, union_first + batch_width - 1)
            width = union_last - union_first + 1

            cpu_started_ns = time_ns()
            batch_statistics = _prepare_lu_certified_hybrid_host_batch!(
                staging,
                preparation,
                coefficients,
                state_pair_hessian_response,
                support,
                cofactor_scalar,
                one_body_active,
                two_body_inactive_active,
                same_spin_two_body,
                first_cofactor_values,
                feature_offsets,
                union_first,
                union_last,
                n_active_occupied,
                runtime,
            )
            cpu_lu_wall_seconds += Float64(time_ns() - cpu_started_ns) / 1.0e9
            inverse_calls += batch_statistics.inverse_calls

            gpu_started_ns = time_ns()
            Base.unsafe_copyto!(
                gpu_workspace.scaled_inverse,
                1,
                staging.scaled_inverse,
                1,
                q2 * width,
            )
            Base.unsafe_copyto!(
                gpu_workspace.active_response_or_hessian_pair_indices,
                1,
                staging.active_response_indices,
                1,
                q2 * width,
            )
            Base.unsafe_copyto!(
                gpu_workspace.determinant_scaled,
                1,
                staging.determinant_scaled,
                1,
                width,
            )
            Base.unsafe_copyto!(gpu_workspace.accepted, 1, staging.accepted, 1, width)

            h2d_bytes = Base.Checked.checked_add(
                h2d_bytes,
                Base.Checked.checked_mul(
                    width,
                    Base.Checked.checked_add(
                        estimate_array_bytes(T, q2 + 1),
                        Base.Checked.checked_add(estimate_array_bytes(UInt16, q2), sizeof(UInt8)),
                    ),
                ),
            )

            CUDA.@cuda threads=AHPC_POLYNOMIAL_GPU_THREADS blocks=cld(width, AHPC_POLYNOMIAL_GPU_THREADS) _polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!(
                gpu_workspace.active_response_or_hessian_pair_indices,
                Int32(n_active_occupied),
                Int32(active_rank),
                Int32(width),
            )
            column_pair_indices_device =
                @view gpu_workspace.active_response_or_hessian_pair_indices[1:n_local_hessian_pairs, :]
            row_pair_indices_device =
                @view gpu_workspace.active_response_or_hessian_pair_indices[(n_local_hessian_pairs + 1):(2 * n_local_hessian_pairs), :]

            _launch_lu_certified_hybrid_second_cofactor!(
                gpu_workspace,
                row_pair_indices_device,
                column_pair_indices_device,
                state_pair_hessian_response_device,
                n_active_occupied,
                width,
            )
            CUDA.synchronize()
            Base.unsafe_copyto!(
                staging.contracted_second_scaled,
                1,
                gpu_workspace.contracted_second_scaled,
                1,
                width,
            )
            d2h_bytes = Base.Checked.checked_add(d2h_bytes, estimate_array_bytes(T, width))
            gpu_second_wall_seconds += Float64(time_ns() - gpu_started_ns) / 1.0e9

            post_gpu_fallback_workspace = SpinPairIntermediateWorkspace{T}()
            _resize_spin_pair_intermediate_workspace!(
                post_gpu_fallback_workspace,
                n_active_occupied,
                runtime,
            )
            @inbounds for local_union in 1:width
                union_id = union_first + local_union - 1
                if staging.accepted[local_union] == UInt8(1)
                    exponent = Int(staging.binary_scaling_exponents[local_union])
                    second_value = n_active_occupied < 2 ? zero(T) :
                        ldexp(
                            staging.contracted_second_scaled[local_union],
                            exponent * (n_active_occupied - 2),
                        )
                    if isfinite(second_value)
                        same_spin_two_body[union_id] = second_value
                        accepted += 1
                    else
                        staging.accepted[local_union] = UInt8(0)
                        _recompute_lu_certified_hybrid_polynomial_fallback!(
                            post_gpu_fallback_workspace,
                            preparation,
                            coefficients,
                            state_pair_hessian_response,
                            support.union_support[union_id],
                            cofactor_scalar,
                            one_body_active,
                            two_body_inactive_active,
                            same_spin_two_body,
                            first_cofactor_values,
                            union_id,
                            feature_offsets[union_id],
                            active_rank,
                        )
                        fallbacks += 1
                    end
                else
                    fallbacks += 1
                end
            end

            batch_count += 1
            union_first = union_last + 1
        end
    finally
        state_pair_hessian_response_device === nothing ||
            CUDA.unsafe_free!(state_pair_hessian_response_device)
        gpu_workspace === nothing || _close_lu_certified_hybrid_gpu_workspace!(gpu_workspace)
    end

    return (
        device=assignment.device,
        union_range=union_range,
        batch_count=batch_count,
        batch_width=batch_width,
        h2d_bytes=h2d_bytes,
        d2h_bytes=d2h_bytes,
        available_before=available_before,
        accepted=accepted,
        fallbacks=fallbacks,
        inverse_calls=inverse_calls,
        cpu_lu_wall_seconds=cpu_lu_wall_seconds,
        gpu_second_wall_seconds=gpu_second_wall_seconds,
        worker_wall_seconds=Float64(time_ns() - worker_started_ns) / 1.0e9,
    )
end
