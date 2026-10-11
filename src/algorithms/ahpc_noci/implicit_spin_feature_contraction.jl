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

 File:            src/algorithms/ahpc_noci/implicit_spin_feature_contraction.jl
 Purpose:         Phase-timed blocked contraction of structural sparse features into raw state-pair matrix elements.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    AHPCContractionPhaseTiming

Accumulate phase-local wall and GPU-event timings for the blocked AHPC sparse contraction pipeline.
Worker-local preparation timings are merged only by the parent task after each batch joins, so telemetry collection introduces no shared floating-point updates or scientific reduction-order changes.
"""
mutable struct AHPCContractionPhaseTiming
    beta_csr_seconds::Float64
    cuda_scatter_seconds::Float64
    cuda_gemm_seconds::Float64
    d2h_seconds::Float64
    prefix_reduction_seconds::Float64
    checkpoint_seconds::Float64
    progress_io_seconds::Float64
    batch_wait_seconds::Float64
end

AHPCContractionPhaseTiming() = AHPCContractionPhaseTiming(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

@inline function _merge_ahpc_contraction_worker_timing!(
    phase_timing::AHPCContractionPhaseTiming,
    worker_timing::AHPCContractionPhaseTiming,
)
    phase_timing.beta_csr_seconds += worker_timing.beta_csr_seconds
    phase_timing.cuda_scatter_seconds += worker_timing.cuda_scatter_seconds
    phase_timing.cuda_gemm_seconds += worker_timing.cuda_gemm_seconds
    phase_timing.d2h_seconds += worker_timing.d2h_seconds
    return phase_timing
end

@inline function _record_ahpc_progress_event!(
    phase_timing::Union{Nothing,AHPCContractionPhaseTiming},
    progress,
    event::Symbol;
    kwargs...,
)
    if isnothing(phase_timing) || isnothing(progress)
        progress_event!(progress, event; kwargs...)
        return nothing
    end
    progress_started_ns = time_ns()
    progress_event!(progress, event; kwargs...)
    phase_timing.progress_io_seconds += Float64(time_ns() - progress_started_ns) / 1.0e9
    return nothing
end

@inline function _write_ahpc_contraction_timing_metadata!(
    metadata::Dict{Symbol,Any},
    phase_timing::AHPCContractionPhaseTiming,
)
    metadata[:beta_csr_seconds] = phase_timing.beta_csr_seconds
    metadata[:cuda_scatter_seconds] = phase_timing.cuda_scatter_seconds
    metadata[:cuda_gemm_seconds] = phase_timing.cuda_gemm_seconds
    metadata[:d2h_seconds] = phase_timing.d2h_seconds
    metadata[:prefix_reduction_seconds] = phase_timing.prefix_reduction_seconds
    metadata[:checkpoint_seconds] = phase_timing.checkpoint_seconds
    metadata[:progress_io_seconds] = phase_timing.progress_io_seconds
    metadata[:batch_wait_seconds] = phase_timing.batch_wait_seconds
    return metadata
end

"""
    _spin_pair_contraction_block_width(T, sparse, intermediates, runtime)

Return the largest per-worker alpha-support block width that fits the configured aggregate host scratch budget.
GPU scratch is divided equally across configured devices, the calculation allocates nothing, and a nonempty contraction that cannot hold one block per active device raises `ResourceLimitError` rather than selecting a slower algorithm.
"""
function _spin_pair_contraction_block_width(
    ::Type{T},
    sparse::SparseSpinPairWeights,
    intermediates::ImplicitSpinPairIntermediates,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    sparse.n_kappa == 0 && return 0
    n_response = Base.Checked.checked_mul(intermediates.active_rank, intermediates.active_rank)
    response_copies = runtime.use_gpu ? 2 : 3
    entries_per_column = Base.Checked.checked_add(4, Base.Checked.checked_mul(response_copies, n_response))
    bytes_per_column = Base.Checked.checked_mul(sizeof(T), entries_per_column)
    worker_count = runtime.use_gpu ? length(runtime.gpu_devices) : 1
    required_bytes = Base.Checked.checked_mul(bytes_per_column, worker_count)
    runtime.contraction_host_scratch_bytes >= required_bytes || throw(ResourceLimitError(:host, required_bytes, runtime.contraction_host_scratch_bytes))
    return min(sparse.n_kappa, runtime.contraction_host_scratch_bytes ÷ required_bytes)
end

"""
    _accumulate_spin_pair_beta_block!(beta_contracted_channels, sparse, support, intermediates, kappa_first, kappa_last, worker_limit)

Accumulate CSR-weighted beta scalar channels and structural first-cofactor features for one alpha row block.
Each Julia worker owns disjoint output columns, sparse rows retain deterministic edge order, inputs remain read-only, and malformed support mappings raise before or during traversal.
"""
function _accumulate_spin_pair_beta_block!(
    beta_contracted_channels::AbstractMatrix{T},
    sparse::SparseSpinPairWeights{T},
    support::SpinPairSupportDomain,
    intermediates::ImplicitSpinPairIntermediates{T},
    kappa_first::Int,
    kappa_last::Int,
    worker_limit::Int=Threads.nthreads(),
) where {T<:AbstractFloat}
    1 <= kappa_first <= kappa_last <= sparse.n_kappa || throw(BoundsError(sparse.kappa_offsets, kappa_first:kappa_last))
    width = kappa_last - kappa_first + 1
    n_response = Base.Checked.checked_mul(intermediates.active_rank, intermediates.active_rank)
    size(beta_contracted_channels) == (4 + n_response, width) || throw(DimensionMismatch("beta contracted channel block has incorrect dimensions"))
    length(support.beta_union_ids) == sparse.n_lambda || throw(DimensionMismatch("beta-to-union map length differs from sparse columns"))
    fill!(beta_contracted_channels, zero(T))
    worker_limit >= 1 || throw(ArgumentError("beta-block worker limit must be positive"))
    worker_count = min(worker_limit, width)
    ranges = _balanced_auxiliary_ranges(width, worker_count)
    tasks = map(ranges) do local_range
        Threads.@spawn begin
            rank = intermediates.active_rank
            @inbounds for local_kappa in local_range
                kappa = kappa_first + local_kappa - 1
                for edge in sparse.kappa_offsets[kappa]:(sparse.kappa_offsets[kappa + 1] - 1)
                    lambda = Int(sparse.lambda_ids[edge])
                    union_lambda = Int(support.beta_union_ids[lambda])
                    weight = sparse.weights[edge]
                    beta_contracted_channels[1, local_kappa] += weight * intermediates.cofactor_scalar[union_lambda]
                    beta_contracted_channels[2, local_kappa] += weight * intermediates.one_body_active[union_lambda]
                    beta_contracted_channels[3, local_kappa] += weight * intermediates.two_body_inactive_active[union_lambda]
                    beta_contracted_channels[4, local_kappa] += weight * intermediates.same_spin_two_body[union_lambda]
                    pair = intermediates.feature_pairs[union_lambda]
                    feature_slot = intermediates.feature_offsets[union_lambda]
                    expected_end = intermediates.feature_offsets[union_lambda + 1]
                    right_bits = pair.right_bits
                    while !iszero(right_bits)
                        right_active = trailing_zeros(right_bits) + 1
                        left_bits = pair.left_bits
                        while !iszero(left_bits)
                            left_active = trailing_zeros(left_bits) + 1
                            response = left_active + (right_active - 1) * rank
                            beta_contracted_channels[4 + response, local_kappa] += weight * intermediates.first_cofactor_values[feature_slot]
                            feature_slot += 1
                            left_bits &= left_bits - UInt32(1)
                        end
                        right_bits &= right_bits - UInt32(1)
                    end
                    feature_slot == expected_end || throw(ArgumentError("beta feature traversal did not consume the expected structural column"))
                end
            end
            nothing
        end
    end
    fetch.(tasks)
    return beta_contracted_channels
end

"""
    _gather_spin_pair_alpha_first_cofactor_response!(features, intermediates, kappa_first, kappa_last, worker_limit)

Materialize one alpha-support structural feature block directly from occupied bit masks into a reusable dense buffer.
The destination is overwritten, input ordering is read-only and deterministic, no extra response-ID array or CUDA work is used, and invalid feature offsets raise explicitly.
"""
function _gather_spin_pair_alpha_first_cofactor_response!(
    features::AbstractMatrix{T},
    intermediates::ImplicitSpinPairIntermediates{T},
    kappa_first::Int,
    kappa_last::Int,
    worker_limit::Int=Threads.nthreads(),
) where {T<:AbstractFloat}
    1 <= kappa_first <= kappa_last <= length(intermediates.feature_pairs) || throw(BoundsError(intermediates.feature_pairs, kappa_first:kappa_last))
    width = kappa_last - kappa_first + 1
    n_response = Base.Checked.checked_mul(intermediates.active_rank, intermediates.active_rank)
    size(features) == (n_response, width) || throw(DimensionMismatch("alpha feature block has incorrect dimensions"))
    fill!(features, zero(T))
    worker_limit >= 1 || throw(ArgumentError("alpha-feature worker limit must be positive"))
    rank = intermediates.active_rank
    ranges = _balanced_auxiliary_ranges(width, min(worker_limit, width))
    tasks = map(ranges) do local_range
        Threads.@spawn begin
            @inbounds for local_kappa in local_range
                union_kappa = kappa_first + local_kappa - 1
                pair = intermediates.feature_pairs[union_kappa]
                feature_slot = intermediates.feature_offsets[union_kappa]
                expected_end = intermediates.feature_offsets[union_kappa + 1]
                right_bits = pair.right_bits
                while !iszero(right_bits)
                    right_active = trailing_zeros(right_bits) + 1
                    left_bits = pair.left_bits
                    while !iszero(left_bits)
                        left_active = trailing_zeros(left_bits) + 1
                        response = left_active + (right_active - 1) * rank
                        features[response, local_kappa] = intermediates.first_cofactor_values[feature_slot]
                        feature_slot += 1
                        left_bits &= left_bits - UInt32(1)
                    end
                    right_bits &= right_bits - UInt32(1)
                end
                feature_slot == expected_end || throw(ArgumentError("alpha feature traversal did not consume the expected structural column"))
            end
            nothing
        end
    end
    fetch.(tasks)
    return features
end

"""
    _project_spin_pair_alpha_first_cofactor_response_gpu!(coulomb_projected_alpha_response, features, coulomb_device, alpha_first_cofactor_response_device, coulomb_projected_alpha_response_device)

Project one complete alpha feature block by the transposed Coulomb-response matrix on the selected CUDA device and copy the result to CPU.
Inputs are read-only, preallocated device buffers are reused, the destination must be a contiguous prefix of its parent host matrix and is overwritten in energy units, and CUDA errors propagate without CPU fallback.
"""
function _project_spin_pair_alpha_first_cofactor_response_gpu!(
    coulomb_projected_alpha_response::AbstractMatrix{T},
    features::AbstractMatrix{T},
    coulomb_device,
    alpha_first_cofactor_response_device_storage,
    coulomb_projected_alpha_response_device_storage,
) where {T<:AbstractFloat}
    size(coulomb_projected_alpha_response) == size(features) || throw(DimensionMismatch("Coulomb-projected and source alpha-response blocks differ"))
    size(alpha_first_cofactor_response_device_storage, 1) == size(features, 1) && size(alpha_first_cofactor_response_device_storage, 2) >= size(features, 2) || throw(DimensionMismatch("CUDA alpha-feature storage is too small"))
    size(coulomb_projected_alpha_response_device_storage) == size(alpha_first_cofactor_response_device_storage) || throw(DimensionMismatch("CUDA Coulomb-projected alpha-response storage differs from source storage"))
    alpha_first_cofactor_response_device = @view alpha_first_cofactor_response_device_storage[:, 1:size(features, 2)]
    coulomb_projected_alpha_response_device = @view coulomb_projected_alpha_response_device_storage[:, 1:size(features, 2)]
    source = parent(features)
    pointer(features) == pointer(source) || throw(ArgumentError("alpha-feature CUDA source must begin at the first element of its parent host matrix"))
    Base.unsafe_copyto!(alpha_first_cofactor_response_device_storage, 1, source, 1, length(features))
    mul!(coulomb_projected_alpha_response_device, transpose(coulomb_device), alpha_first_cofactor_response_device)
    destination = parent(coulomb_projected_alpha_response)
    length(destination) >= length(coulomb_projected_alpha_response_device) || throw(DimensionMismatch("Coulomb-projected alpha-response host storage is smaller than the CUDA block"))
    pointer(coulomb_projected_alpha_response) == pointer(destination) || throw(ArgumentError("Coulomb-projected alpha-response CUDA destination must begin at the first element of its parent host matrix"))
    Base.unsafe_copyto!(destination, 1, coulomb_projected_alpha_response_device, 1, length(coulomb_projected_alpha_response_device))
    return coulomb_projected_alpha_response
end

"""
    AHPCContractionPrefix

Store the three unscaled AHPC contraction accumulators, next alpha-support index, and completed block count at a deterministic recovery boundary.
Values are small CPU scalars, overlap is dimensionless while one-/two-body sums use energy units, and no support, CSR, intermediate, or CUDA allocation is retained.
"""
mutable struct AHPCContractionPrefix{T<:AbstractFloat}
    overlap_sum::T
    one_body_sum::T
    two_body_sum::T
    next_kappa::Int
    completed_blocks::Int
end

"""
    AHPCContractionWorker

Own the reusable host blocks, selected device identifier, optional device Coulomb-response matrix, and three reusable CUDA timing events for one concurrent AHPC contraction worker.
Each instance is confined to one driver task at a time, scientific source arrays are borrowed separately, host storage is dimensionless or follows response energy units, and GPU workers never share device arrays or timing events.
"""
struct AHPCContractionWorker{T<:AbstractFloat,D2,D1,DI,DP,E}
    device::Int
    beta_contracted_channels::Matrix{T}
    alpha_first_cofactor_response::Matrix{T}
    coulomb_projected_alpha_response::Matrix{T}
    coulomb_device::D2
    alpha_first_cofactor_response_device::D2
    coulomb_projected_alpha_response_device::D2
    packed_value_device::D1
    feature_offset_device::DI
    feature_pair_device::DP
    cuda_scatter_start_event::E
    cuda_scatter_stop_event::E
    cuda_gemm_stop_event::E
end

"""
    _prepare_spin_pair_contraction_block!(worker, sparse, support, coefficients, intermediates, kappa_first, kappa_last, runtime, cpu_worker_limit; phase_timing=nothing)

Prepare one independent AHPC alpha-support block through threaded CPU feature assembly and CPU or selected-device Coulomb projection without changing scalar accumulators.
The worker buffers are overwritten, scientific inputs remain read-only, GPU execution is bound to `worker.device` with no fallback, and disjoint blocks may execute concurrently before ordered reduction.
Optional phase timing records worker-local CSR and CUDA projection stages only; the parent merges those timings after task join.
"""
function _prepare_spin_pair_contraction_block!(
    worker::AHPCContractionWorker{T},
    sparse::SparseSpinPairWeights{T},
    support::SpinPairSupportDomain,
    coefficients::ActiveHamiltonianCoefficients{T},
    intermediates::ImplicitSpinPairIntermediates{T},
    kappa_first::Integer,
    kappa_last::Integer,
    runtime::RuntimeOptions,
    cpu_worker_limit::Integer;
    phase_timing::Union{Nothing,AHPCContractionPhaseTiming}=nothing,
) where {T<:AbstractFloat}
    first_kappa = Int(kappa_first)
    last_kappa = Int(kappa_last)
    width = last_kappa - first_kappa + 1
    width >= 1 || throw(ArgumentError("AHPC contraction block must be nonempty"))
    n_response = Base.Checked.checked_mul(intermediates.active_rank, intermediates.active_rank)
    size(worker.beta_contracted_channels, 1) == 4 + n_response && size(worker.beta_contracted_channels, 2) >= width || throw(DimensionMismatch("beta-contracted AHPC channel scratch is too small"))
    size(worker.coulomb_projected_alpha_response, 1) == n_response && size(worker.coulomb_projected_alpha_response, 2) >= width || throw(DimensionMismatch("Coulomb-projected alpha-response AHPC scratch is too small"))
    beta_contracted_channels = @view worker.beta_contracted_channels[:, 1:width]
    coulomb_projected_alpha_response = @view worker.coulomb_projected_alpha_response[:, 1:width]
    beta_csr_started_ns = isnothing(phase_timing) ? UInt64(0) : time_ns()
    _accumulate_spin_pair_beta_block!(beta_contracted_channels, sparse, support, intermediates, first_kappa, last_kappa, Int(cpu_worker_limit))
    if !isnothing(phase_timing)
        phase_timing.beta_csr_seconds += Float64(time_ns() - beta_csr_started_ns) / 1.0e9
    end
    if runtime.use_gpu
        _cuda_module().device!(worker.device)
        _project_spin_pair_alpha_first_cofactor_response_gpu_packed!(
            coulomb_projected_alpha_response,
            intermediates,
            first_kappa,
            last_kappa,
            worker.coulomb_device,
            worker.alpha_first_cofactor_response_device,
            worker.coulomb_projected_alpha_response_device,
            worker.packed_value_device,
            worker.feature_offset_device,
            worker.feature_pair_device;
            phase_timing,
            cuda_scatter_start_event=worker.cuda_scatter_start_event,
            cuda_scatter_stop_event=worker.cuda_scatter_stop_event,
            cuda_gemm_stop_event=worker.cuda_gemm_stop_event,
        )
    else
        size(worker.alpha_first_cofactor_response, 1) == n_response && size(worker.alpha_first_cofactor_response, 2) >= width || throw(DimensionMismatch("alpha AHPC scratch is too small"))
        alpha_first_cofactor_response = @view worker.alpha_first_cofactor_response[:, 1:width]
        _gather_spin_pair_alpha_first_cofactor_response!(alpha_first_cofactor_response, intermediates, first_kappa, last_kappa, Int(cpu_worker_limit))
        mul!(coulomb_projected_alpha_response, transpose(coefficients.coulomb_response), alpha_first_cofactor_response)
    end
    return worker
end

"""
    _accumulate_prepared_spin_pair_contraction_block!(prefix, worker, coefficients, intermediates, kappa_first, kappa_last)

Reduce one prepared AHPC block into its cumulative prefix using the original global kappa/response scalar-addition order.
Only the prefix is mutated, worker and scientific arrays are read-only, no CPU task or CUDA work is launched, and a block not contiguous with the prefix raises `ArgumentError`.
"""
function _accumulate_prepared_spin_pair_contraction_block!(
    prefix::AHPCContractionPrefix{T},
    worker::AHPCContractionWorker{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    intermediates::ImplicitSpinPairIntermediates{T},
    kappa_first::Integer,
    kappa_last::Integer,
) where {T<:AbstractFloat}
    first_kappa = Int(kappa_first)
    last_kappa = Int(kappa_last)
    prefix.next_kappa == first_kappa || throw(ArgumentError("AHPC contraction prefix does not match requested block start"))
    width = last_kappa - first_kappa + 1
    n_response = Base.Checked.checked_mul(intermediates.active_rank, intermediates.active_rank)
    @inbounds for local_kappa in 1:width
        kappa = first_kappa + local_kappa - 1
        alpha_scalar = intermediates.cofactor_scalar[kappa]
        beta_scalar = worker.beta_contracted_channels[1, local_kappa]
        product = alpha_scalar * beta_scalar
        prefix.overlap_sum += product
        prefix.one_body_sum += 2 * coefficients.one_body_inactive * product + intermediates.one_body_active[kappa] * beta_scalar + alpha_scalar * worker.beta_contracted_channels[2, local_kappa]
        prefix.two_body_sum += coefficients.two_body_inactive_inactive * product + intermediates.two_body_inactive_active[kappa] * beta_scalar + alpha_scalar * worker.beta_contracted_channels[3, local_kappa] + T(0.25) * intermediates.same_spin_two_body[kappa] * beta_scalar + T(0.25) * alpha_scalar * worker.beta_contracted_channels[4, local_kappa]
        for response in 1:n_response
            prefix.two_body_sum += worker.coulomb_projected_alpha_response[response, local_kappa] * worker.beta_contracted_channels[4 + response, local_kappa]
        end
    end
    prefix.next_kappa = last_kappa + 1
    prefix.completed_blocks += 1
    return prefix
end

"""
    _finalize_spin_pair_contraction(prefix, preparation)

Apply the squared inactive-overlap determinant to one completed unscaled AHPC prefix and return its matrix element.
Inputs are read-only, overlap and energy units are preserved, no array or CUDA work occurs, and nonfinite values are rejected by `AHPCSpinPairMatrixElement` construction if applicable.
"""
function _finalize_spin_pair_contraction(
    prefix::AHPCContractionPrefix{T},
    preparation::AHPCStatePairPreparation{T},
) where {T<:AbstractFloat}
    scale = preparation.inactive_overlap_determinant^2
    return AHPCSpinPairMatrixElement(scale * prefix.overlap_sum, scale * prefix.one_body_sum, scale * prefix.two_body_sum)
end

"""
    _close_ahpc_contraction_workers!(workers)

Release reusable CUDA arrays owned by AHPC contraction workers and return `nothing` after every device task joins.
CPU workers are ignored, host matrices and scientific inputs remain intact, cleanup is safe only after all block tasks complete, and the first CUDA cleanup failure propagates after every worker has been attempted.
CUDA event objects are Julia-managed and remain bounded at three reusable events per timed GPU worker.
"""
function _close_ahpc_contraction_workers!(workers)
    tasks = map(workers) do worker
        Threads.@spawn begin
            worker.device < 0 && return nothing
            _cuda_module().device!(worker.device)
            _cuda_module().unsafe_free!(worker.coulomb_device)
            _cuda_module().unsafe_free!(worker.alpha_first_cofactor_response_device)
            _cuda_module().unsafe_free!(worker.coulomb_projected_alpha_response_device)
            _cuda_module().unsafe_free!(worker.packed_value_device)
            _cuda_module().unsafe_free!(worker.feature_offset_device)
            _cuda_module().unsafe_free!(worker.feature_pair_device)
            return nothing
        end
    end
    first_error = nothing
    for task in tasks
        try
            fetch(task)
        catch error
            isnothing(first_error) && (first_error = error)
        end
    end
    isnothing(first_error) || throw(first_error)
    return nothing
end

"""
    contract_sparse_spin_pair_weights(sparse, support, preparation, coefficients, intermediates; runtime=RuntimeOptions(), telemetry=nothing, checkpoint=nothing, progress=nothing, left_mebf=0, right_mebf=0)

Contract production AHPC weights by alpha blocks using all configured GPUs concurrently, with an equal integer share of Julia threads driving each device.
Prepared blocks are reduced and checkpointed in original kappa order, aggregate host and peak per-device scratch are preflighted, CPU uses BLAS, GPU uses CUDA without fallback, and asymmetric noninteger CPU/GPU resource ratios are rejected before allocation.
When execution telemetry is present, R5 phase-local timings are collected without changing CSR traversal, scalar accumulation order, checkpoint boundaries, or scientific values.
"""
function contract_sparse_spin_pair_weights(
    sparse::SparseSpinPairWeights{T},
    support::SpinPairSupportDomain,
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    intermediates::ImplicitSpinPairIntermediates{T};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
    checkpoint::Union{Nothing,CheckpointSession}=nothing,
    progress::Union{Nothing,ProgressLogger}=nothing,
    left_mebf::Integer=0,
    right_mebf::Integer=0,
) where {T<:AbstractFloat}
    sparse.n_kappa == length(support.alpha_support) || throw(DimensionMismatch("sparse kappa rows differ from alpha support"))
    sparse.n_lambda == length(support.beta_support) || throw(DimensionMismatch("sparse lambda columns differ from beta support"))
    length(intermediates.cofactor_scalar) == length(support.union_support) || throw(DimensionMismatch("intermediate count differs from union support"))
    intermediates.feature_pairs === support.union_support || throw(ArgumentError("intermediate features must share the production union support"))
    isempty(support.alpha_union_ids) || throw(ArgumentError("production AHPC requires implicit alpha-to-union identity"))
    support.union_support[1:sparse.n_kappa] == support.alpha_support || throw(ArgumentError("implicit alpha-to-union identity does not match alpha-first union ordering"))
    threads_per_gpu = ahpc_threads_per_gpu(runtime)
    contraction_devices = runtime.use_gpu ? runtime.gpu_devices : (-1,)
    contraction_worker_count = length(contraction_devices)
    block_width = _spin_pair_contraction_block_width(T, sparse, intermediates, runtime)
    estimate = sparse_spin_pair_contraction_memory(T, sparse, intermediates, runtime)
    enforce_memory_limits(estimate, runtime)
    phase_timing = isnothing(telemetry) ? nothing : AHPCContractionPhaseTiming()
    phase_metadata = Dict{Symbol,Any}(
        :strategy => runtime.use_gpu ? :cuda_multi_device_packed_alpha_scatter_csr_gemm : :cpu_blocked_alpha_csr_gemm,
        :alpha_block_width => block_width,
        :concurrent_blocks => contraction_worker_count,
        :julia_threads => Threads.nthreads(),
        :gpu_devices => runtime.use_gpu ? runtime.gpu_devices : (),
        :gpu_count => runtime.use_gpu ? contraction_worker_count : 0,
        :threads_per_gpu => runtime.use_gpu ? threads_per_gpu : Threads.nthreads(),
        :beta_csr_seconds => 0.0,
        :cuda_scatter_seconds => 0.0,
        :cuda_gemm_seconds => 0.0,
        :d2h_seconds => 0.0,
        :prefix_reduction_seconds => 0.0,
        :checkpoint_seconds => 0.0,
        :progress_io_seconds => 0.0,
        :batch_wait_seconds => 0.0,
        :phase_timing_instrumented => !isnothing(telemetry),
        :worker_stage_timing_semantics => :sum_across_prepared_blocks,
        :parent_stage_timing_semantics => :host_wall_seconds,
        :cuda_stage_timing_semantics => :device_event_seconds,
    )

    work = function ()
        n_response = Base.Checked.checked_mul(intermediates.active_rank, intermediates.active_rank)
        workers = if runtime.use_gpu
            tasks = map(contraction_devices) do device
                Threads.@spawn begin
                    worker_runtime = _runtime_for_gpu_device(runtime, device)
                    _activate_cuda_device(worker_runtime)
                    coulomb_device = _cuda_module().CuArray(coefficients.coulomb_response)
                    alpha_first_cofactor_response_device = _cuda_module().CuArray{T}(undef, n_response, block_width)
                    coulomb_projected_alpha_response_device = _cuda_module().CuArray{T}(undef, n_response, block_width)
                    packed_value_device = _cuda_module().CuArray{T}(undef, n_response * block_width)
                    feature_offset_device = _cuda_module().CuArray{Int}(undef, block_width + 1)
                    feature_pair_device = _cuda_module().CuArray{CanonicalSpinStringPair}(undef, block_width)
                    cuda_scatter_start_event = isnothing(phase_timing) ? nothing : _cuda_module().CuEvent()
                    cuda_scatter_stop_event = isnothing(phase_timing) ? nothing : _cuda_module().CuEvent()
                    cuda_gemm_stop_event = isnothing(phase_timing) ? nothing : _cuda_module().CuEvent()
                    return AHPCContractionWorker(
                        device,
                        Matrix{T}(undef, 4 + n_response, block_width),
                        Matrix{T}(undef, 0, 0),
                        Matrix{T}(undef, n_response, block_width),
                        coulomb_device,
                        alpha_first_cofactor_response_device,
                        coulomb_projected_alpha_response_device,
                        packed_value_device,
                        feature_offset_device,
                        feature_pair_device,
                        cuda_scatter_start_event,
                        cuda_scatter_stop_event,
                        cuda_gemm_stop_event,
                    )
                end
            end
            collect(fetch.(tasks))
        else
            [AHPCContractionWorker(
                -1,
                Matrix{T}(undef, 4 + n_response, block_width),
                Matrix{T}(undef, n_response, block_width),
                Matrix{T}(undef, n_response, block_width),
                nothing,
                nothing,
                nothing,
                nothing,
                nothing,
                nothing,
                nothing,
                nothing,
                nothing,
            )]
        end
        try
        prefix = AHPCContractionPrefix(zero(T), zero(T), zero(T), 1, 0)
        if !isnothing(checkpoint)
            left_mebf >= 1 && right_mebf >= 1 || throw(ArgumentError("AHPC checkpoint contraction requires positive state-pair indices"))
            checkpoint_started_ns = isnothing(phase_timing) ? UInt64(0) : time_ns()
            inflight = load_inflight_checkpoint(checkpoint, :ahpc, left_mebf, right_mebf)
            if !isnothing(phase_timing)
                phase_timing.checkpoint_seconds += Float64(time_ns() - checkpoint_started_ns) / 1.0e9
            end
            if !isnothing(inflight)
                Int(inflight["total_count"]) == sparse.n_kappa || throw(ArgumentError("AHPC prefix alpha-support count differs from rebuilt support"))
                Int(inflight["block_width"]) == block_width || throw(ArgumentError("AHPC prefix block width differs from current execution"))
                Int(inflight["n_lambda"]) == sparse.n_lambda || throw(ArgumentError("AHPC prefix beta-support count differs from rebuilt support"))
                Int(inflight["n_edges"]) == length(sparse.weights) || throw(ArgumentError("AHPC prefix edge count differs from rebuilt incidence"))
                Int(inflight["n_union"]) == length(support.union_support) || throw(ArgumentError("AHPC prefix union count differs from rebuilt support"))
                prefix.next_kappa = Int(inflight["next_index"])
                prefix.next_kappa == sparse.n_kappa + 1 || (prefix.next_kappa - 1) % block_width == 0 || throw(ArgumentError("AHPC prefix does not end on a contraction block boundary"))
                prefix.overlap_sum = T(inflight["overlap"])
                prefix.one_body_sum = T(inflight["one_body"])
                prefix.two_body_sum = T(inflight["two_body"])
                prefix.completed_blocks = Int(inflight["completed_blocks"])
                _record_ahpc_progress_event!(phase_timing, progress, :ahpc_contraction_prefix_reused; left_mebf=left_mebf, right_mebf=right_mebf, next_kappa=prefix.next_kappa, n_kappa=sparse.n_kappa, completed_blocks=prefix.completed_blocks)
                !isnothing(telemetry) && increment_counter!(telemetry, :n_spin_pair_contraction_blocks_reused, prefix.completed_blocks)
            end
        end
        block_starts = collect(prefix.next_kappa:block_width:sparse.n_kappa)
        for batch_first in 1:contraction_worker_count:length(block_starts)
            batch_last = min(length(block_starts), batch_first + contraction_worker_count - 1)
            batch_indices = batch_first:batch_last
            tasks = map(enumerate(batch_indices)) do (local_worker, block_index)
                kappa_first = block_starts[block_index]
                kappa_last = min(sparse.n_kappa, kappa_first + block_width - 1)
                Threads.@spawn begin
                    worker_timing = isnothing(phase_timing) ? nothing : AHPCContractionPhaseTiming()
                    _prepare_spin_pair_contraction_block!(
                        workers[local_worker],
                        sparse,
                        support,
                        coefficients,
                        intermediates,
                        kappa_first,
                        kappa_last,
                        runtime,
                        threads_per_gpu;
                        phase_timing=worker_timing,
                    )
                    return kappa_first, kappa_last, worker_timing
                end
            end
            batch_wait_started_ns = isnothing(phase_timing) ? UInt64(0) : time_ns()
            prepared_blocks = fetch.(tasks)
            if !isnothing(phase_timing)
                phase_timing.batch_wait_seconds += Float64(time_ns() - batch_wait_started_ns) / 1.0e9
                for (_, _, worker_timing) in prepared_blocks
                    _merge_ahpc_contraction_worker_timing!(phase_timing, worker_timing)
                end
            end
            for (local_worker, (kappa_first, kappa_last, _)) in pairs(prepared_blocks)
                prefix_reduction_started_ns = isnothing(phase_timing) ? UInt64(0) : time_ns()
                _accumulate_prepared_spin_pair_contraction_block!(prefix, workers[local_worker], coefficients, intermediates, kappa_first, kappa_last)
                if !isnothing(phase_timing)
                    phase_timing.prefix_reduction_seconds += Float64(time_ns() - prefix_reduction_started_ns) / 1.0e9
                end
                if !isnothing(checkpoint) && (prefix.completed_blocks % checkpoint.ahpc_checkpoint_blocks == 0 || kappa_last == sparse.n_kappa)
                    next_block_kappa = prefix.next_kappa
                    edge_count = Int(sparse.kappa_offsets[next_block_kappa] - 1)
                    checkpoint_started_ns = isnothing(phase_timing) ? UInt64(0) : time_ns()
                    save_inflight_checkpoint!(
                        checkpoint,
                        :ahpc,
                        left_mebf,
                        right_mebf,
                        next_block_kappa,
                        sparse.n_kappa,
                        prefix.overlap_sum,
                        prefix.one_body_sum,
                        prefix.two_body_sum;
                        metadata=Dict{String,Any}(
                            "block_width" => block_width,
                            "completed_blocks" => prefix.completed_blocks,
                            "n_lambda" => sparse.n_lambda,
                            "n_edges" => length(sparse.weights),
                            "n_union" => length(support.union_support),
                            "processed_edges" => edge_count,
                            "contraction_gpu_count" => runtime.use_gpu ? contraction_worker_count : 0,
                            "threads_per_gpu" => runtime.use_gpu ? threads_per_gpu : 0,
                        ),
                    )
                    if !isnothing(phase_timing)
                        phase_timing.checkpoint_seconds += Float64(time_ns() - checkpoint_started_ns) / 1.0e9
                    end
                    _record_ahpc_progress_event!(phase_timing, progress, :ahpc_contraction_prefix_committed; left_mebf=left_mebf, right_mebf=right_mebf, next_kappa=next_block_kappa, n_kappa=sparse.n_kappa, processed_edges=edge_count, n_edges=length(sparse.weights), completed_blocks=prefix.completed_blocks)
                else
                    _record_ahpc_progress_event!(phase_timing, progress, :ahpc_contraction_block_complete; left_mebf=left_mebf, right_mebf=right_mebf, next_kappa=prefix.next_kappa, n_kappa=sparse.n_kappa, completed_blocks=prefix.completed_blocks)
                end
            end
        end
        !isnothing(phase_timing) && _write_ahpc_contraction_timing_metadata!(phase_metadata, phase_timing)
        return _finalize_spin_pair_contraction(prefix, preparation), prefix.completed_blocks
        finally
            _close_ahpc_contraction_workers!(workers)
        end
    end

    element, block_count = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :contract_sparse_spin_pair_weights;
        estimate,
        metadata=phase_metadata,
        collect_allocations=runtime.collect_allocations,
    )
    if !isnothing(telemetry)
        increment_counter!(telemetry, :n_sparse_spin_pair_weight_evaluations, length(sparse.weights))
        increment_counter!(telemetry, :n_spin_pair_contraction_blocks, block_count)
        set_counter!(telemetry, :n_spin_pair_contraction_workers, contraction_worker_count)
        runtime.use_gpu && increment_counter!(telemetry, :n_cuda_projection_blocks, block_count)
    end
    return element
end
