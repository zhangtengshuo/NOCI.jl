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

# File: src/runtime/cuda_resources/lu_certified_spin_pair_cuda.jl
# Purpose: Estimate and manage CUDA resources for LU-certified spin-pair cofactors.
# SPDX-License-Identifier: Apache-2.0

function _lu_certified_spin_pair_gpu_workspace_bytes(
    ::Type{T},
    n_active_occupied::Integer,
    batch_width::Integer,
) where {T<:Union{Float32,Float64}}
    q = Int(n_active_occupied)
    width = Int(batch_width)
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA LU-certified occupied count must lie in 0:32"))
    width >= 0 || throw(ArgumentError("CUDA LU-certified batch width must be nonnegative"))
    q2 = Base.Checked.checked_mul(q, q)
    floating_entries = Base.Checked.checked_mul(width, Base.Checked.checked_add(Base.Checked.checked_mul(3, q2), 2))
    return Base.Checked.checked_add(
        estimate_array_bytes(T, floating_entries),
        estimate_array_bytes(UInt8, width),
    )
end

function _lu_certified_spin_pair_gpu_per_union_device_bytes(
    ::Type{T},
    n_active_occupied::Integer,
) where {T<:Union{Float32,Float64}}
    q = Int(n_active_occupied)
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA LU-certified occupied count must lie in 0:32"))
    q2 = Base.Checked.checked_mul(q, q)
    return Base.Checked.checked_add(
        _lu_certified_spin_pair_gpu_workspace_bytes(T, q, 1),
        Base.Checked.checked_add(
            estimate_array_bytes(T, q2),
            estimate_array_bytes(UInt16, q2),
        ),
    )
end

"""
    build_spin_pair_intermediates_cuda_lu_certified_distributed(...)

Historical all-GPU LU-certified execution retained as an internal performance
comparator. Fixed-occupied-count unions perform LU, inverse construction,
residual/condition certification, first cofactors, and the O(q^4)
contracted-second formula on CUDA. Production `lu_certified + use_gpu=true`
dispatch uses `build_spin_pair_intermediates_cuda_lu_certified_hybrid_distributed`
instead. Rejected unions here still use only CPU polynomial PDDC; no SVD or
G-NOCI fallback is permitted.
"""
function build_spin_pair_intermediates_cuda_lu_certified_distributed(
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    support::SpinPairSupportDomain;
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("CUDA LU-certified construction requires runtime.use_gpu=true"))
    isempty(runtime.gpu_devices) && throw(ArgumentError("CUDA LU-certified construction requires at least one configured GPU"))
    ahpc_threads_per_gpu(runtime)

    n_active_orbitals = _validate_spin_pair_intermediate_inputs(preparation, coefficients)
    maximum_active_occupied, feature_entries = _spin_pair_support_shape(support.union_support, n_active_orbitals)
    maximum_active_occupied <= runtime.max_active_occupied || throw(ArgumentError("union support occupied count exceeds max_active_occupied"))
    maximum_active_occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA LU-certified occupied count exceeds q=32 limit"))
    n_union = length(support.union_support)
    fixed_active_occupied = n_union == 0 || feature_entries == Base.Checked.checked_mul(
        n_union,
        Base.Checked.checked_mul(maximum_active_occupied, maximum_active_occupied),
    )
    fixed_active_occupied || throw(ArgumentError("CUDA LU-certified construction requires one fixed occupied count per union support"))

    if n_union == 0 || maximum_active_occupied == 0
        return build_spin_pair_intermediates(
            preparation,
            coefficients,
            support;
            cofactor_evaluation=LUCertifiedCofactorEvaluation,
            runtime,
            telemetry,
        )
    end

    assignments = _polynomial_spin_pair_gpu_assignments(n_union, runtime.gpu_devices)
    n_state_pair_hessian_pairs = _hessian_pair_count(n_active_orbitals)
    state_pair_hessian_response_bytes = estimate_array_bytes(T, n_state_pair_hessian_pairs, n_state_pair_hessian_pairs)
    global_output_bytes = _polynomial_spin_pair_gpu_output_bytes(T, n_union, feature_entries)

    host_bytes = Base.Checked.checked_add(global_output_bytes, state_pair_hessian_response_bytes)
    peak_device_bytes = 0
    n_state_pair_active_responses = Base.Checked.checked_mul(n_active_orbitals, n_active_orbitals)
    configured_widths = Int[]
    for assignment in assignments
        worker_runtime = _polynomial_runtime_for_gpu_device(runtime, assignment.device)
        width, _ = _lu_certified_spin_pair_gpu_batch_width(
            T,
            maximum_active_occupied,
            length(assignment.union_range),
            n_state_pair_active_responses,
            worker_runtime,
        )
        push!(configured_widths, width)
        host_bytes = Base.Checked.checked_add(
            host_bytes,
            Base.Checked.checked_add(
                _polynomial_spin_pair_gpu_host_staging_bytes(T, maximum_active_occupied, width),
                estimate_array_bytes(UInt8, width),
            ),
        )
        worker_device_bytes = Base.Checked.checked_add(
            state_pair_hessian_response_bytes,
            Base.Checked.checked_mul(
                _lu_certified_spin_pair_gpu_per_union_device_bytes(T, maximum_active_occupied),
                width,
            ),
        )
        peak_device_bytes = max(peak_device_bytes, worker_device_bytes)
    end
    estimate = MemoryEstimate(host_bytes, peak_device_bytes)
    enforce_memory_limits(estimate, runtime)

    phase_metadata = Dict{Symbol,Any}(
        :strategy => length(assignments) == 1 ?
            :cuda_lu_certified_hessian_pair_response :
            :cuda_multidevice_lu_certified_hessian_pair_response,
        :cofactor_evaluation => :lu_certified,
        :lu_certified_cofactor_engine => true,
        :polynomial_fallback => :cpu_pddc,
        :n_union_spin_pair_support => n_union,
        :maximum_active_occupied => maximum_active_occupied,
        :fixed_active_occupied => true,
        :contracted_second_complexity => :O_n_active_occupied_4_certified_O_n_active_occupied_5_fallback,
        :state_pair_hessian_pair_contraction => true,
        :hessian_pair_index_staging => true,
        :gpu_devices => Tuple(assignment.device for assignment in assignments),
        :gpu_count => length(assignments),
        :gpu_batch_width_min_configured => minimum(configured_widths),
        :gpu_batch_width_max_configured => maximum(configured_widths),
        :gpu_intermediate_scratch_bytes_per_device => runtime.gpu_intermediate_scratch_bytes_per_device,
        :lu_certificate_residual_tolerance => AHPC_LU_CERTIFICATE_RESIDUAL_TOLERANCE,
        :lu_certificate_condition_limit => AHPC_LU_CERTIFICATE_CONDITION_LIMIT,
        :n_state_pair_hessian_pairs => n_state_pair_hessian_pairs,
        :state_pair_hessian_response_bytes => state_pair_hessian_response_bytes,
        :rank_classifier => :lu_residual_condition_certificate,
        :svd_fallback => false,
        :direct_global_output => true,
    )

    work = function ()
        q = maximum_active_occupied
        q2 = Base.Checked.checked_mul(q, q)
        cofactor_scalar = Vector{T}(undef, n_union)
        one_body_active = Vector{T}(undef, n_union)
        two_body_inactive_active = Vector{T}(undef, n_union)
        same_spin_two_body = Vector{T}(undef, n_union)
        first_cofactor_values = Vector{T}(undef, feature_entries)
        feature_offsets = Vector{Int}(undef, n_union + 1)
        feature_offsets[1] = 1
        @inbounds for union_id in 1:n_union
            feature_offsets[union_id + 1] = 1 + union_id * q2
        end
        state_pair_response = state_pair_hessian_response(coefficients, n_active_orbitals)

        tasks = map(assignments) do assignment
            Threads.@spawn _build_lu_certified_spin_pair_gpu_worker!(
                cofactor_scalar,
                one_body_active,
                two_body_inactive_active,
                same_spin_two_body,
                first_cofactor_values,
                state_pair_response,
                preparation,
                coefficients,
                support,
                feature_offsets,
                assignment,
                q,
                runtime,
            )
        end
        worker_info = fetch.(tasks)
        phase_metadata[:gpu_batch_count] = sum(worker -> worker.batch_count, worker_info)
        phase_metadata[:gpu_batch_width_min] = minimum(worker -> worker.batch_width, worker_info)
        phase_metadata[:gpu_batch_width_max] = maximum(worker -> worker.batch_width, worker_info)
        phase_metadata[:gpu_h2d_bytes] = sum(worker -> worker.h2d_bytes, worker_info)
        phase_metadata[:gpu_d2h_bytes] = sum(worker -> worker.d2h_bytes, worker_info)
        phase_metadata[:gpu_available_bytes_before_workspace_min] = minimum(worker -> worker.available_before, worker_info)
        phase_metadata[:gpu_worker_wall_seconds_max] = maximum(worker -> worker.worker_wall_seconds, worker_info)
        phase_metadata[:gpu_worker_wall_seconds_mean] = sum(worker -> worker.worker_wall_seconds, worker_info) / length(worker_info)

        return (
            intermediates=ImplicitSpinPairIntermediates(
                n_active_orbitals,
                cofactor_scalar,
                one_body_active,
                two_body_inactive_active,
                same_spin_two_body,
                feature_offsets,
                first_cofactor_values,
                support.union_support,
                Val(:validated),
            ),
            accepted=sum(worker -> worker.accepted, worker_info),
            fallbacks=sum(worker -> worker.fallbacks, worker_info),
            inverse_calls=sum(worker -> worker.inverse_calls, worker_info),
            batch_count=sum(worker -> worker.batch_count, worker_info),
        )
    end

    build_result = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :build_spin_pair_intermediates;
        estimate,
        metadata=phase_metadata,
        collect_allocations=runtime.collect_allocations,
    )

    if !isnothing(telemetry)
        increment_counter!(telemetry, :n_union_spin_pair_intermediates, n_union)
        increment_counter!(telemetry, :n_spin_pair_kernel_evaluations, n_union)
        increment_counter!(telemetry, :n_spin_pair_feature_entries, feature_entries)
        increment_counter!(telemetry, :n_implicit_spin_feature_entries, feature_entries)
        set_counter!(telemetry, :n_complementary_cofactor_transitions, 0)
        set_counter!(telemetry, :n_structural_response_iterations, 0)
        increment_counter!(telemetry, :n_lu_certificate_evaluations, n_union)
        increment_counter!(telemetry, :n_lu_certified_cofactor_kernel_evaluations, build_result.accepted)
        increment_counter!(telemetry, :n_lu_certificate_fallbacks, build_result.fallbacks)
        increment_counter!(telemetry, :n_polynomial_cofactor_kernel_evaluations, build_result.fallbacks)
        increment_counter!(telemetry, :n_inverse_calls, build_result.inverse_calls)
        increment_counter!(telemetry, :n_rank_classifier_calls, n_union)
        increment_counter!(telemetry, :n_svd_calls, 0)
        increment_counter!(telemetry, :n_packed_spin_intermediate_builds)
        set_counter!(telemetry, :n_union_kernel_workers, length(assignments))
        set_counter!(telemetry, :n_cuda_spin_pair_batches, build_result.batch_count)
        set_counter!(telemetry, :n_cuda_spin_pair_devices, length(assignments))
    end
    return build_result.intermediates
end
