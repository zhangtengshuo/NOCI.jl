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

 File:            src/algorithms/ahpc_noci/polynomial_spin_pair_cuda_multigpu_hessian_pair.jl
 Purpose:         Host and per-device storage estimates for multi-device determinant-Hessian polynomial contraction.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _polynomial_spin_pair_multigpu_hessian_pair_memory(T, preparation, support, runtime)

Estimate host and peak per-device storage for the multi-device determinant-
Hessian pair-space polynomial cofactor contraction. One state-pair Hessian-pair
response is retained on the host and copied once to each worker device; worker-
local storage contains only bounded cofactor staging and polynomial workspace.
Inputs are read-only, units are bytes, no CUDA context is initialized, and
checked overflow raises.
"""
function _polynomial_spin_pair_multigpu_hessian_pair_memory(
    ::Type{T},
    preparation::AHPCStatePairPreparation{T},
    support::SpinPairSupportDomain,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    n_active_orbitals = size(preparation.effective_active_overlap, 1)
    maximum_active_occupied, feature_entries = _spin_pair_support_shape(support.union_support, n_active_orbitals)
    n_union = length(support.union_support)
    assignments = _polynomial_spin_pair_gpu_assignments(n_union, runtime.gpu_devices)
    global_output_bytes = _polynomial_spin_pair_gpu_output_bytes(T, n_union, feature_entries)
    isempty(assignments) && return MemoryEstimate(global_output_bytes, 0)

    n_state_pair_hessian_pairs = _hessian_pair_count(n_active_orbitals)
    state_pair_hessian_response_bytes = estimate_array_bytes(
        T,
        n_state_pair_hessian_pairs,
        n_state_pair_hessian_pairs,
    )
    n_state_pair_active_responses = Base.Checked.checked_mul(n_active_orbitals, n_active_orbitals)
    host_bytes = Base.Checked.checked_add(global_output_bytes, state_pair_hessian_response_bytes)
    peak_device_bytes = 0
    for assignment in assignments
        worker_runtime = _polynomial_runtime_for_gpu_device(runtime, assignment.device)
        width = _polynomial_spin_pair_gpu_configured_width(
            T,
            maximum_active_occupied,
            length(assignment.union_range),
            n_state_pair_active_responses,
            worker_runtime,
        )
        host_bytes = Base.Checked.checked_add(
            host_bytes,
            _polynomial_spin_pair_gpu_host_staging_bytes(T, maximum_active_occupied, width),
        )
        worker_device_bytes = Base.Checked.checked_add(
            state_pair_hessian_response_bytes,
            Base.Checked.checked_mul(
                _polynomial_spin_pair_hessian_pair_gpu_per_union_device_bytes(T, maximum_active_occupied),
                width,
            ),
        )
        peak_device_bytes = max(peak_device_bytes, worker_device_bytes)
    end
    return MemoryEstimate(host_bytes, peak_device_bytes)
end

"""
    build_spin_pair_intermediates_cuda_polynomial_multigpu_hessian_pair(...)

Build one fixed-occupied-count AHPC spin-pair intermediate domain on two or more
CUDA devices using the state-pair determinant-Hessian unordered-pair response for
the contracted second complementary cofactor. Device workers own deterministic
contiguous union ranges and write directly into disjoint slices of one host
result. Scientific ordering and floating-point traversal match the validated
single-device Hessian-pair implementation.
"""
function build_spin_pair_intermediates_cuda_polynomial_multigpu_hessian_pair(
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    support::SpinPairSupportDomain;
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("multi-device Hessian-pair construction requires runtime.use_gpu=true"))
    length(runtime.gpu_devices) >= 2 || throw(ArgumentError("multi-device Hessian-pair construction requires at least two devices"))
    ahpc_threads_per_gpu(runtime)
    n_active_orbitals = _validate_spin_pair_intermediate_inputs(preparation, coefficients)
    maximum_active_occupied, feature_entries = _spin_pair_support_shape(support.union_support, n_active_orbitals)
    maximum_active_occupied <= runtime.max_active_occupied || throw(ArgumentError("union support occupied count exceeds max_active_occupied"))
    maximum_active_occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("occupied count exceeds supported q=32 limit"))
    n_union = length(support.union_support)
    fixed_active_occupied = n_union == 0 || feature_entries == Base.Checked.checked_mul(
        n_union,
        Base.Checked.checked_mul(maximum_active_occupied, maximum_active_occupied),
    )
    fixed_active_occupied || throw(ArgumentError("multi-device Hessian-pair construction requires one fixed occupied count"))
    maximum_active_occupied == 0 && return _build_trivial_spin_pair_intermediates_cuda_polynomial(
        preparation,
        support;
        runtime,
        telemetry,
        strategy=:cuda_multidevice_polynomial_hessian_pair_response,
    )

    assignments = _polynomial_spin_pair_gpu_assignments(n_union, runtime.gpu_devices)
    estimate = _polynomial_spin_pair_multigpu_hessian_pair_memory(T, preparation, support, runtime)
    enforce_memory_limits(estimate, runtime)
    n_state_pair_hessian_pairs = _hessian_pair_count(n_active_orbitals)
    state_pair_hessian_response_bytes = estimate_array_bytes(
        T,
        n_state_pair_hessian_pairs,
        n_state_pair_hessian_pairs,
    )
    batch_memory_bound_per_union_bytes = _polynomial_spin_pair_gpu_per_union_device_bytes(T, maximum_active_occupied)
    actual_per_union_device_bytes = _polynomial_spin_pair_hessian_pair_gpu_per_union_device_bytes(T, maximum_active_occupied)
    second_complementary_cofactor_threads_per_union =
        maximum_active_occupied >= 2 ?
        AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_UNION :
        0
    second_complementary_cofactor_threads_per_block =
        maximum_active_occupied >= 2 ?
        AHPC_SECOND_COMPLEMENTARY_COFACTOR_THREADS_PER_BLOCK :
        0
    second_cofactor_stage_groups =
        maximum_active_occupied >= 3 ?
        _second_complementary_cofactor_blocked_stage_groups(
            T,
            maximum_active_occupied,
        ) :
        0
    second_cofactor_shared_bytes =
        maximum_active_occupied >= 3 ?
        _second_complementary_cofactor_blocked_shared_bytes(
            T,
            maximum_active_occupied,
        ) :
        0
    second_cofactor_dense_response_scans =
        second_cofactor_stage_groups == 0 ?
        0 :
        cld(
            (maximum_active_occupied - 2) ÷ 2,
            second_cofactor_stage_groups,
        )
    phase_metadata = Dict{Symbol,Any}(
        :strategy => :cuda_multidevice_polynomial_hessian_pair_response,
        :n_union_spin_pair_support => n_union,
        :maximum_active_occupied => maximum_active_occupied,
        :occupied_rank_within_polynomial_limit =>
            maximum_active_occupied <= AHPC_MAX_OCCUPIED_RANK,
        :recurrence_storage => :union_contiguous,
        :polynomial_cofactor_engine => true,
        :contracted_second_complexity => :O_n_active_occupied_5,
        :state_pair_hessian_pair_contraction => true,
        :hessian_pair_index_staging => true,
        :conservative_batch_memory_bound => true,
        :batch_width_selection_per_union_bytes => batch_memory_bound_per_union_bytes,
        :actual_per_union_device_bytes => actual_per_union_device_bytes,
        :n_state_pair_hessian_pairs => n_state_pair_hessian_pairs,
        :state_pair_hessian_response_bytes => state_pair_hessian_response_bytes,
        :outer_range_partition => true,
        :deterministic_contiguous_ranges => true,
        :direct_global_output => true,
        :worker_subresults => false,
        :host_result_merge => false,
        :gpu_devices => Tuple(assignment.device for assignment in assignments),
        :gpu_count => length(assignments),
        :gpu_intermediate_scratch_bytes_per_device => runtime.gpu_intermediate_scratch_bytes_per_device,
        :julia_threads => Threads.nthreads(),
        :threads_per_gpu => ahpc_threads_per_gpu(runtime),
        :maximum_active_occupied => maximum_active_occupied,
        :first_complementary_cofactor_evaluation =>
            :shared_direct_power_terminal_recurrence,
        :second_complementary_cofactor_evaluation =>
            maximum_active_occupied >= 2 ?
            :blocked_shared_occupied_orbital_pair_space :
            :not_applicable,
        :second_complementary_cofactor_matrix_stage_budget_bytes =>
            AHPC_SECOND_COMPLEMENTARY_COFACTOR_MATRIX_STAGE_BUDGET_BYTES,
        :second_complementary_cofactor_stage_groups =>
            second_cofactor_stage_groups,
        :second_complementary_cofactor_dynamic_shared_bytes =>
            second_cofactor_shared_bytes,
        :second_complementary_cofactor_dense_response_scans =>
            second_cofactor_dense_response_scans,
        :second_complementary_cofactor_threads_per_union =>
            second_complementary_cofactor_threads_per_union,
        :second_complementary_cofactor_threads_per_block =>
            second_complementary_cofactor_threads_per_block,
        :first_complementary_cofactor_threads_per_union => 1,
    )

    work = function ()
        n_active_occupied = maximum_active_occupied
        n_local_active_responses = Base.Checked.checked_mul(n_active_occupied, n_active_occupied)
        cofactor_scalar = Vector{T}(undef, n_union)
        one_body_active = Vector{T}(undef, n_union)
        two_body_inactive_active = Vector{T}(undef, n_union)
        same_spin_two_body = Vector{T}(undef, n_union)
        first_cofactor_values = Vector{T}(undef, feature_entries)
        feature_offsets = Vector{Int}(undef, n_union + 1)
        feature_offsets[1] = 1
        @inbounds for union_id in 1:n_union
            feature_offsets[union_id + 1] = 1 + union_id * n_local_active_responses
        end
        state_pair_response = state_pair_hessian_response(coefficients, n_active_orbitals)

        tasks = map(assignments) do assignment
            Threads.@spawn _build_polynomial_spin_pair_gpu_hessian_pair_worker!(
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
                n_active_occupied,
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

        return ImplicitSpinPairIntermediates(
            n_active_orbitals,
            cofactor_scalar,
            one_body_active,
            two_body_inactive_active,
            same_spin_two_body,
            feature_offsets,
            first_cofactor_values,
            support.union_support,
            Val(:validated),
        )
    end

    result = isnothing(telemetry) ? work() : record_phase!(
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
        increment_counter!(telemetry, :n_polynomial_cofactor_kernel_evaluations, n_union)
        increment_counter!(telemetry, :n_packed_spin_intermediate_builds)
        set_counter!(telemetry, :n_union_kernel_workers, length(assignments))
        set_counter!(telemetry, :spin_pair_response_id_storage_bytes, 0)
        set_counter!(telemetry, :spin_pair_response_id_payload_bytes, 0)
        set_counter!(telemetry, :n_cuda_spin_pair_batches, Int(get(phase_metadata, :gpu_batch_count, 0)))
        set_counter!(telemetry, :n_cuda_spin_pair_devices, length(assignments))
        set_counter!(telemetry, :polynomial_worker_duplicate_output_bytes, 0)
        set_counter!(telemetry, :polynomial_worker_output_merge_copies, 0)
        increment_counter!(telemetry, :n_inverse_calls, 0)
        increment_counter!(telemetry, :n_rank_classifier_calls, 0)
        increment_counter!(telemetry, :n_svd_calls, 0)
    end
    return result
end
