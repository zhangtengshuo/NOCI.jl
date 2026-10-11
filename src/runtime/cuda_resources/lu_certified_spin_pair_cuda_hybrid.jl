#==============================================================================#
#      ___           ___           ___                                         #
#     /\\__\\         /\\  \\         /\\  \\          ___                           #
#    /::|  |       /::\\  \\       /::\\  \\        /\\  \\                          #
#   /:|:|  |      /:/\\:\\  \\     /:/\\:\\  \\       \\:\\  \\         ___        ___  #
#  /:/|:|  |__   /:/  \\:\\  \\   /:/  \\:\\  \\      /::\\__\\       /\\  \\      /\\__\\ #
# /:/ |:| /\\__\\ /:/__/ \\:\\__\\ /:/__/ \\:\\__\\  __/:/\\/__/       \\:\\  \\    /:/  / #
# \\/__|:|/:/  / \\:\\  \\ /:/  / \\:\\  \\  \\/__/ /\\/:/  /      ___ /::\\__\\  /:/  /  #
#     |:/:/  /   \\:\\  /:/  /   \\:\\  \\       \\::/__/      /\\  /:/\\/__/ /:/__/   #
#     |::/  /     \\:\\/:/  /     \\:\\  \\       \\:\\__\\      \\:\\/:/  /    \\:\\  \\   #
#     /:/  /       \\::/  /       \\:\\__\\       \\/__/       \\::/  /      \\:\\  \\  #
#     \\/__/         \\/__/         \\/__/               ██   \\/__/        \\/__/  #
#                                                                              #
#==============================================================================#

#=
 NOCI.jl — Nonorthogonal Configuration Interaction in Julia

 File:            src/algorithms/ahpc_noci/lu_certified_spin_pair_cuda_hybrid.jl
 Purpose:         CPU-LU/CUDA-second-cofactor hybrid for LU-certified AHPC spin-pair intermediates.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

const AHPC_LU_HYBRID_SECOND_THREADS = 128

"""
    LUCertifiedHybridHostBuffers

Own bounded host staging for the LU-certified hybrid path. CPU LAPACK produces
the scaled inverse, determinant, binary exponent, and first complementary
cofactors. Only the scaled inverse, determinant, active-response map, and one
status byte per union are uploaded for the CUDA contracted-second evaluation.
"""
struct LUCertifiedHybridHostBuffers{T<:Union{Float32,Float64}}
    scaled_inverse::Matrix{T}
    active_response_indices::Matrix{UInt16}
    determinant_scaled::Vector{T}
    binary_scaling_exponents::Vector{Int32}
    accepted::Vector{UInt8}
    contracted_second_scaled::Vector{T}
end

function LUCertifiedHybridHostBuffers(
    ::Type{T},
    n_active_occupied::Integer,
    batch_width::Integer,
) where {T<:Union{Float32,Float64}}
    q = Int(n_active_occupied)
    width = Int(batch_width)
    1 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("hybrid LU-certified occupied count must lie in 1:32"))
    width >= 1 || throw(ArgumentError("hybrid LU-certified batch width must be positive"))
    q2 = Base.Checked.checked_mul(q, q)
    return LUCertifiedHybridHostBuffers(
        Matrix{T}(undef, q2, width),
        Matrix{UInt16}(undef, q2, width),
        Vector{T}(undef, width),
        Vector{Int32}(undef, width),
        zeros(UInt8, width),
        Vector{T}(undef, width),
    )
end

function _lu_certified_hybrid_local_pairs(n_active_occupied::Int)
    n_pairs = _hessian_pair_count(n_active_occupied)
    first = Vector{UInt8}(undef, n_pairs)
    second = Vector{UInt8}(undef, n_pairs)
    pair_id = 0
    @inbounds for first_local in 1:(n_active_occupied - 1)
        for second_local in (first_local + 1):n_active_occupied
            pair_id += 1
            first[pair_id] = UInt8(first_local)
            second[pair_id] = UInt8(second_local)
        end
    end
    return first, second
end

function _lu_certified_hybrid_host_staging_bytes(
    ::Type{T},
    n_active_occupied::Integer,
    batch_width::Integer,
) where {T<:Union{Float32,Float64}}
    q = Int(n_active_occupied)
    width = Int(batch_width)
    q2 = Base.Checked.checked_mul(q, q)
    per_union = Base.Checked.checked_add(
        estimate_array_bytes(T, q2 + 2),
        Base.Checked.checked_add(
            estimate_array_bytes(UInt16, q2),
            Base.Checked.checked_add(sizeof(Int32), sizeof(UInt8)),
        ),
    )
    return Base.Checked.checked_mul(per_union, width)
end

function _lu_certified_hybrid_per_union_device_bytes(
    ::Type{T},
    n_active_occupied::Integer,
) where {T<:Union{Float32,Float64}}
    q = Int(n_active_occupied)
    q2 = Base.Checked.checked_mul(q, q)
    return Base.Checked.checked_add(
        estimate_array_bytes(T, q2 + 2),
        Base.Checked.checked_add(estimate_array_bytes(UInt16, q2), sizeof(UInt8)),
    )
end

function _write_lu_certified_hybrid_first_channels!(
    workspace::SpinPairIntermediateWorkspace{T},
    staging::LUCertifiedHybridHostBuffers{T},
    local_union::Int,
    union_id::Int,
    determinant_scaled::T,
    exponent::Int,
    coefficients::ActiveHamiltonianCoefficients{T},
    cofactor_scalar::Vector{T},
    one_body_active::Vector{T},
    two_body_inactive_active::Vector{T},
    same_spin_two_body::Vector{T},
    first_cofactor_values::Vector{T},
    feature_offsets::Vector{Int},
) where {T<:Union{Float32,Float64}}
    q = size(workspace.selected_overlap, 1)
    determinant = ldexp(determinant_scaled, exponent * q)
    isfinite(determinant) || return false
    first_exponent = exponent * max(q - 1, 0)
    @inbounds for column in 1:q, row in 1:q
        first_cofactor = ldexp(
            determinant_scaled * workspace.cofactors.previous[column, row],
            first_exponent,
        )
        isfinite(first_cofactor) || return false
        workspace.cofactors.first_cofactor[row, column] = first_cofactor
    end

    result = ComplementaryCofactorResult(
        determinant,
        workspace.cofactors.first_cofactor,
        zero(T),
        exponent,
    )
    _write_spin_pair_intermediate!(
        cofactor_scalar,
        one_body_active,
        two_body_inactive_active,
        same_spin_two_body,
        first_cofactor_values,
        union_id,
        feature_offsets[union_id],
        coefficients,
        result,
        workspace.response_indices,
    )

    q2 = q * q
    @inbounds for response in 1:q2
        staging.scaled_inverse[response, local_union] = workspace.cofactors.previous[response]
        state_pair_response = workspace.response_indices[response]
        state_pair_response <= typemax(UInt16) || throw(OverflowError("state-pair active response ID exceeds UInt16"))
        staging.active_response_indices[response, local_union] = UInt16(state_pair_response)
    end
    typemin(Int32) <= exponent <= typemax(Int32) || throw(OverflowError("binary scaling exponent exceeds Int32"))
    @inbounds staging.determinant_scaled[local_union] = determinant_scaled
    @inbounds staging.binary_scaling_exponents[local_union] = Int32(exponent)
    @inbounds staging.accepted[local_union] = UInt8(1)
    return true
end

function _recompute_lu_certified_hybrid_polynomial_fallback!(
    workspace::SpinPairIntermediateWorkspace{T},
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    state_pair_hessian_response::Matrix{T},
    pair::CanonicalSpinStringPair,
    cofactor_scalar::Vector{T},
    one_body_active::Vector{T},
    two_body_inactive_active::Vector{T},
    same_spin_two_body::Vector{T},
    first_cofactor_values::Vector{T},
    union_id::Int,
    feature_offset::Int,
    active_rank::Int,
) where {T<:Union{Float32,Float64}}
    _fill_selected_active_overlap_from_bits!(
        workspace.selected_overlap,
        workspace.response_indices,
        preparation.effective_active_overlap,
        pair,
        active_rank,
    )
    _fill_union_hessian_pair_indices!(
        workspace.hessian_row_pair_indices,
        workspace.hessian_column_pair_indices,
        workspace.response_indices,
        active_rank,
    )
    cofactors = _polynomial_state_pair_hessian_contracted_complementary_cofactors!(
        workspace.cofactors,
        workspace.selected_overlap,
        state_pair_hessian_response,
        workspace.hessian_row_pair_indices,
        workspace.hessian_column_pair_indices,
    )
    _write_spin_pair_intermediate!(
        cofactor_scalar,
        one_body_active,
        two_body_inactive_active,
        same_spin_two_body,
        first_cofactor_values,
        union_id,
        feature_offset,
        coefficients,
        cofactors,
        workspace.response_indices,
    )
    return nothing
end

function _prepare_lu_certified_hybrid_host_batch!(
    staging::LUCertifiedHybridHostBuffers{T},
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    state_pair_hessian_response::Matrix{T},
    support::SpinPairSupportDomain,
    cofactor_scalar::Vector{T},
    one_body_active::Vector{T},
    two_body_inactive_active::Vector{T},
    same_spin_two_body::Vector{T},
    first_cofactor_values::Vector{T},
    feature_offsets::Vector{Int},
    union_first::Int,
    union_last::Int,
    n_active_occupied::Int,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    width = union_last - union_first + 1
    fill!(view(staging.accepted, 1:width), UInt8(0))
    active_rank = size(preparation.effective_active_overlap, 1)
    worker_count = min(Threads.nthreads(), width)
    ranges = _balanced_auxiliary_ranges(width, worker_count)

    tasks = map(ranges) do local_range
        Threads.@spawn begin
            workspace = SpinPairIntermediateWorkspace{T}()
            _resize_spin_pair_intermediate_workspace!(workspace, n_active_occupied, runtime)
            inverse_calls = 0
            accepted = 0
            fallbacks = 0

            @inbounds for local_union in local_range
                union_id = union_first + local_union - 1
                pair = support.union_support[union_id]
                _fill_selected_active_overlap_from_bits!(
                    workspace.selected_overlap,
                    workspace.response_indices,
                    preparation.effective_active_overlap,
                    pair,
                    active_rank,
                )
                _fill_union_hessian_pair_indices!(
                    workspace.hessian_row_pair_indices,
                    workspace.hessian_column_pair_indices,
                    workspace.response_indices,
                    active_rank,
                )

                q2 = n_active_occupied * n_active_occupied
                for response in 1:q2
                    state_pair_response = workspace.response_indices[response]
                    state_pair_response <= typemax(UInt16) || throw(OverflowError("state-pair active response ID exceeds UInt16"))
                    staging.active_response_indices[response, local_union] = UInt16(state_pair_response)
                end

                certificate_accepted, inverse_computed, determinant_scaled, exponent, _, _ =
                    _prepare_lu_full_rank_certificate!(workspace.cofactors, workspace.selected_overlap)
                inverse_calls += inverse_computed

                fast_path_written = false
                if certificate_accepted
                    fast_path_written = _write_lu_certified_hybrid_first_channels!(
                        workspace,
                        staging,
                        local_union,
                        union_id,
                        determinant_scaled,
                        exponent,
                        coefficients,
                        cofactor_scalar,
                        one_body_active,
                        two_body_inactive_active,
                        same_spin_two_body,
                        first_cofactor_values,
                        feature_offsets,
                    )
                end

                if fast_path_written
                    accepted += 1
                else
                    staging.accepted[local_union] = UInt8(0)
                    _recompute_lu_certified_hybrid_polynomial_fallback!(
                        workspace,
                        preparation,
                        coefficients,
                        state_pair_hessian_response,
                        pair,
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
            end
            return (inverse_calls=inverse_calls, accepted=accepted, fallbacks=fallbacks)
        end
    end
    statistics = fetch.(tasks)
    return (
        inverse_calls=sum(value -> value.inverse_calls, statistics),
        accepted=sum(value -> value.accepted, statistics),
        fallbacks=sum(value -> value.fallbacks, statistics),
    )
end

"""
    build_spin_pair_intermediates_cuda_lu_certified_hybrid_distributed(...)

Build fixed-occupied-count AHPC spin-pair intermediates with CPU LAPACK
factorization/inverse certification and CUDA block-parallel contracted-second
cofactors. Certified unions upload only their scaled inverse, determinant, and
response map; determinant and first cofactors stay on CPU, and CUDA returns one
scaled second-cofactor scalar per union. Rejected or ambiguous certificates fall
back only to CPU polynomial PDDC. The historical all-GPU LU implementation is
retained separately as an internal performance comparator.
"""
function build_spin_pair_intermediates_cuda_lu_certified_hybrid_distributed(
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    support::SpinPairSupportDomain;
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("hybrid LU-certified construction requires runtime.use_gpu=true"))
    isempty(runtime.gpu_devices) && throw(ArgumentError("hybrid LU-certified construction requires at least one configured GPU"))
    ahpc_threads_per_gpu(runtime)

    active_rank = _validate_spin_pair_intermediate_inputs(preparation, coefficients)
    maximum_active_occupied, feature_entries =
        _spin_pair_support_shape(support.union_support, active_rank)
    maximum_active_occupied <= runtime.max_active_occupied ||
        throw(ArgumentError("union support occupied count exceeds max_active_occupied"))
    maximum_active_occupied <= AHPC_MAX_OCCUPIED_RANK ||
        throw(ArgumentError("hybrid LU-certified occupied count exceeds q=32 limit"))

    n_union = length(support.union_support)
    fixed_active_occupied = n_union == 0 || feature_entries == Base.Checked.checked_mul(
        n_union,
        Base.Checked.checked_mul(maximum_active_occupied, maximum_active_occupied),
    )
    fixed_active_occupied ||
        throw(ArgumentError("hybrid LU-certified construction requires one fixed occupied count per union support"))

    if n_union == 0 || maximum_active_occupied == 0
        return build_spin_pair_intermediates(
            preparation,
            coefficients,
            support;
            cofactor_evaluation=LUCertifiedCofactorEvaluation,
            runtime=RuntimeOptions(
                use_gpu=false,
                scratch_bytes=runtime.scratch_bytes,
                max_active_occupied=runtime.max_active_occupied,
                collect_allocations=runtime.collect_allocations,
            ),
            telemetry,
        )
    end

    assignments = _polynomial_spin_pair_gpu_assignments(n_union, runtime.gpu_devices)
    n_state_pair_hessian_pairs = _hessian_pair_count(active_rank)
    state_pair_hessian_response_bytes =
        estimate_array_bytes(T, n_state_pair_hessian_pairs, n_state_pair_hessian_pairs)
    global_output_bytes = _polynomial_spin_pair_gpu_output_bytes(T, n_union, feature_entries)

    host_bytes = Base.Checked.checked_add(global_output_bytes, state_pair_hessian_response_bytes)
    peak_device_bytes = 0
    configured_widths = Int[]
    for assignment in assignments
        worker_runtime = _polynomial_runtime_for_gpu_device(runtime, assignment.device)
        width, _ = _lu_certified_hybrid_batch_width(
            T,
            maximum_active_occupied,
            length(assignment.union_range),
            state_pair_hessian_response_bytes,
            worker_runtime,
        )
        push!(configured_widths, width)
        host_bytes = Base.Checked.checked_add(
            host_bytes,
            _lu_certified_hybrid_host_staging_bytes(T, maximum_active_occupied, width),
        )
        worker_device_bytes = Base.Checked.checked_add(
            state_pair_hessian_response_bytes,
            Base.Checked.checked_mul(
                _lu_certified_hybrid_per_union_device_bytes(T, maximum_active_occupied),
                width,
            ),
        )
        peak_device_bytes = max(peak_device_bytes, worker_device_bytes)
    end
    estimate = MemoryEstimate(host_bytes, peak_device_bytes)
    enforce_memory_limits(estimate, runtime)

    phase_metadata = Dict{Symbol,Any}(
        :strategy => length(assignments) == 1 ?
            :cpu_lu_cuda_second_cofactor_hybrid :
            :cpu_lu_cuda_second_cofactor_hybrid_multidevice,
        :cofactor_evaluation => :lu_certified,
        :lu_certified_cofactor_engine => true,
        :lu_location => :cpu_lapack,
        :certificate_location => :cpu,
        :first_cofactor_location => :cpu,
        :second_cofactor_location => :cuda,
        :second_cofactor_threads_per_union => AHPC_LU_HYBRID_SECOND_THREADS,
        :polynomial_fallback => :cpu_pddc,
        :all_gpu_lu_comparator_retained => true,
        :n_union_spin_pair_support => n_union,
        :maximum_active_occupied => maximum_active_occupied,
        :fixed_active_occupied => true,
        :contracted_second_complexity => :O_n_active_occupied_4_cuda_certified_O_n_active_occupied_5_fallback,
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
        q2 = q * q
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
        state_pair_response = state_pair_hessian_response(coefficients, active_rank)

        tasks = map(assignments) do assignment
            Threads.@spawn _build_lu_certified_hybrid_gpu_worker!(
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
        phase_metadata[:gpu_available_bytes_before_workspace_min] =
            minimum(worker -> worker.available_before, worker_info)
        phase_metadata[:cpu_lu_certificate_wall_seconds_sum] =
            sum(worker -> worker.cpu_lu_wall_seconds, worker_info)
        phase_metadata[:gpu_second_cofactor_wall_seconds_sum] =
            sum(worker -> worker.gpu_second_wall_seconds, worker_info)
        phase_metadata[:gpu_worker_wall_seconds_max] =
            maximum(worker -> worker.worker_wall_seconds, worker_info)
        phase_metadata[:gpu_worker_wall_seconds_mean] =
            sum(worker -> worker.worker_wall_seconds, worker_info) / length(worker_info)

        return (
            intermediates=ImplicitSpinPairIntermediates(
                active_rank,
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
        set_counter!(telemetry, :n_cuda_lu_factorizations, 0)
        set_counter!(telemetry, :n_cuda_second_cofactor_contractions, build_result.accepted)
    end
    return build_result.intermediates
end
