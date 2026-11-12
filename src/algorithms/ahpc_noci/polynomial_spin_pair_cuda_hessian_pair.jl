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

 File:            src/algorithms/ahpc_noci/polynomial_spin_pair_cuda_hessian_pair.jl
 Purpose:         Per-worker reusable CUDA storage for polynomial determinant-Hessian unordered-pair contraction.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    PolynomialSpinPairHessianPairGPUWorkspace

Own one CUDA worker's reusable polynomial-cofactor storage for contraction of the
second complementary cofactor in the determinant-Hessian unordered-pair basis.
The workspace stores elementary symmetric coefficients, the shared direct-power
backbone, a Newton transformation scratch matrix, first complementary cofactors,
and the zeroth/contracted-second outputs. It owns device memory only and contains no
numerical-rank classifier, inverse, SVD, or determinant-threshold state.
"""
mutable struct PolynomialSpinPairHessianPairGPUWorkspace{T<:Union{Float32,Float64}}
    device::Int
    n_active_occupied::Int
    batch_width::Int
    elementary_symmetric_coefficients::CUDA.CuMatrix{T}
    matrix_powers::CUDA.CuMatrix{T}
    newton_cofactor_matrix::CUDA.CuMatrix{T}
    first_cofactor_output::CUDA.CuMatrix{T}
    cofactor_scalar_and_contracted_second_output::CUDA.CuMatrix{T}
    closed::Bool
end

"""
    PolynomialSpinPairHessianPairGPUWorkspace(T, n_active_occupied, batch_width, runtime)

Allocate the reusable determinant-Hessian pair-space polynomial workspace on the
single CUDA device selected by `runtime`. The runtime is read-only; allocated
arrays are device-owned and must be released with
`_close_polynomial_spin_pair_hessian_pair_gpu_workspace!`.
"""
function PolynomialSpinPairHessianPairGPUWorkspace(
    ::Type{T},
    n_active_occupied::Integer,
    batch_width::Integer,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    occupied = Int(n_active_occupied)
    width = Int(batch_width)
    0 <= occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA occupied count must lie in 0:32"))
    width >= 1 || throw(ArgumentError("CUDA batch width must be positive"))
    runtime.use_gpu || throw(ArgumentError("CUDA cofactor workspace requires runtime.use_gpu=true"))
    length(runtime.gpu_devices) == 1 || throw(ArgumentError("CUDA cofactor workspace requires exactly one GPU"))
    _activate_cuda_device(runtime)
    n_active_responses = occupied * occupied
    n_power_entries = n_active_responses * occupied
    allocated = Any[]
    try
        elementary = CUDA.CuArray{T}(undef, occupied + 1, width); push!(allocated, elementary)
        powers = CUDA.CuArray{T}(undef, n_power_entries, width); push!(allocated, powers)
        newton = CUDA.CuArray{T}(undef, n_active_responses, width); push!(allocated, newton)
        first_cofactor = CUDA.CuArray{T}(undef, n_active_responses, width); push!(allocated, first_cofactor)
        cofactor_outputs = CUDA.CuArray{T}(undef, 2, width); push!(allocated, cofactor_outputs)
        return PolynomialSpinPairHessianPairGPUWorkspace(
            runtime.gpu_device,
            occupied,
            width,
            elementary,
            powers,
            newton,
            first_cofactor,
            cofactor_outputs,
            false,
        )
    catch
        for array in Iterators.reverse(allocated)
            CUDA.unsafe_free!(array)
        end
        rethrow()
    end
end

"""
    _close_polynomial_spin_pair_hessian_pair_gpu_workspace!(workspace)

Release every CUDA allocation owned by one determinant-Hessian pair-space
polynomial workspace. The workspace is mutated only by marking it closed after
all arrays have been freed; repeated calls are no-ops.
"""
function _close_polynomial_spin_pair_hessian_pair_gpu_workspace!(
    workspace::PolynomialSpinPairHessianPairGPUWorkspace,
)
    workspace.closed && return nothing
    CUDA.device!(workspace.device)
    for array in (
        workspace.cofactor_scalar_and_contracted_second_output,
        workspace.first_cofactor_output,
        workspace.newton_cofactor_matrix,
        workspace.matrix_powers,
        workspace.elementary_symmetric_coefficients,
    )
        CUDA.unsafe_free!(array)
    end
    workspace.closed = true
    return nothing
end

"""
    _polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!(...)

Transform each uploaded column-major active-response index grid into canonical
column and row determinant-Hessian pair indices using the same
`n_active_occupied^2` UInt16 device storage. The identity
`2*binomial(n_active_occupied,2)+n_active_occupied=n_active_occupied^2` makes the
transformation storage-neutral. One CUDA thread owns one union and scientific
host staging is not mutated.
"""
function _polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!(
    active_response_or_hessian_pair_indices,
    n_active_occupied_device::Int32,
    n_active_orbitals_device::Int32,
    batch_count::Int32,
)
    union_id = (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) + Int(CUDA.threadIdx().x)
    union_id <= Int(batch_count) || return nothing
    n_active_occupied = Int(n_active_occupied_device)
    n_active_orbitals = Int(n_active_orbitals_device)
    n_hessian_pairs = (n_active_occupied * (n_active_occupied - 1)) ÷ 2
    right_active_scratch_offset = 2 * n_hessian_pairs

    @inbounds for local_column in n_active_occupied:-1:1
        source_slot = 1 + (local_column - 1) * n_active_occupied
        response_id = Int(active_response_or_hessian_pair_indices[source_slot, union_id])
        right_active = div(response_id - 1, n_active_orbitals) + 1
        active_response_or_hessian_pair_indices[right_active_scratch_offset + local_column, union_id] = UInt16(right_active)
    end

    hessian_pair_id = 0
    @inbounds for first_local in 1:(n_active_occupied - 1)
        first_response = Int(active_response_or_hessian_pair_indices[first_local, union_id])
        first_active = (first_response - 1) % n_active_orbitals + 1
        for second_local in (first_local + 1):n_active_occupied
            second_response = Int(active_response_or_hessian_pair_indices[second_local, union_id])
            second_active = (second_response - 1) % n_active_orbitals + 1
            hessian_pair_id += 1
            active_response_or_hessian_pair_indices[n_hessian_pairs + hessian_pair_id, union_id] = UInt16(
                _hessian_pair_index_unchecked(first_active, second_active, n_active_orbitals),
            )
        end
    end

    hessian_pair_id = 0
    @inbounds for first_local in 1:(n_active_occupied - 1)
        first_active = Int(active_response_or_hessian_pair_indices[right_active_scratch_offset + first_local, union_id])
        for second_local in (first_local + 1):n_active_occupied
            second_active = Int(active_response_or_hessian_pair_indices[right_active_scratch_offset + second_local, union_id])
            hessian_pair_id += 1
            active_response_or_hessian_pair_indices[hessian_pair_id, union_id] = UInt16(
                _hessian_pair_index_unchecked(first_active, second_active, n_active_orbitals),
            )
        end
    end
    return nothing
end

"""
    _polynomial_spin_pair_hessian_pair_gpu_kernel!(...)

Evaluate one scaled occupied-overlap determinant/cofactor problem per CUDA thread.
Faddeev determinant arithmetic, matrix powers, Newton transformations, and first
complementary cofactors retain the canonical polynomial order. The contracted
second complementary cofactor is evaluated in the unordered determinant-Hessian
pair basis using the state-pair Hessian response table. No inverse, numerical-
rank classifier, SVD, determinant threshold, or subset branch is used.
"""
function _polynomial_spin_pair_hessian_pair_gpu_kernel!(
    faddeev_previous,
    faddeev_next,
    elementary_symmetric_coefficients,
    matrix_powers,
    newton_cofactor_matrix,
    first_cofactor_output,
    cofactor_scalar_and_contracted_second_output,
    scaled_effective_active_overlap,
    hessian_row_pair_indices,
    hessian_column_pair_indices,
    state_pair_hessian_response_device,
    n_active_occupied_device::Int32,
    batch_count::Int32,
)
    union_id = (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) + Int(CUDA.threadIdx().x)
    union_id <= Int(batch_count) || return nothing
    n_active_occupied = Int(n_active_occupied_device)
    n_active_responses = n_active_occupied * n_active_occupied

    @inbounds for order_index in 1:(n_active_occupied + 1)
        elementary_symmetric_coefficients[order_index, union_id] = zero(eltype(elementary_symmetric_coefficients))
    end
    @inbounds elementary_symmetric_coefficients[1, union_id] = one(eltype(elementary_symmetric_coefficients))

    @inbounds for column in 1:n_active_occupied, row in 1:n_active_occupied
        active_response = row + (column - 1) * n_active_occupied
        faddeev_previous[active_response, union_id] = row == column ? one(eltype(faddeev_previous)) : zero(eltype(faddeev_previous))
    end

    @inbounds for order in 1:n_active_occupied
        if n_active_occupied < 32 && order == 1
            for active_response in 1:n_active_responses
                faddeev_next[active_response, union_id] = scaled_effective_active_overlap[active_response, union_id]
            end
        elseif n_active_occupied < 32 && order == n_active_occupied
            for diagonal in 1:n_active_occupied
                diagonal_product = zero(eltype(faddeev_next))
                for occupied in 1:n_active_occupied
                    diagonal_product += scaled_effective_active_overlap[diagonal + (occupied - 1) * n_active_occupied, union_id] * faddeev_previous[occupied + (diagonal - 1) * n_active_occupied, union_id]
                end
                faddeev_next[diagonal + (diagonal - 1) * n_active_occupied, union_id] = diagonal_product
            end
        else
            for column in 1:n_active_occupied, row in 1:n_active_occupied
                value = zero(eltype(faddeev_next))
                for inner in 1:n_active_occupied
                    left_response = row + (inner - 1) * n_active_occupied
                    right_response = inner + (column - 1) * n_active_occupied
                    value += scaled_effective_active_overlap[left_response, union_id] * faddeev_previous[right_response, union_id]
                end
                faddeev_next[row + (column - 1) * n_active_occupied, union_id] = value
            end
        end
        trace_value = zero(eltype(faddeev_next))
        for diagonal in 1:n_active_occupied
            trace_value += faddeev_next[diagonal + (diagonal - 1) * n_active_occupied, union_id]
        end
        characteristic = -trace_value / order
        elementary_symmetric_coefficients[order + 1, union_id] = isodd(order) ? -characteristic : characteristic
        if order < n_active_occupied
            for diagonal in 1:n_active_occupied
                active_response = diagonal + (diagonal - 1) * n_active_occupied
                faddeev_next[active_response, union_id] += characteristic
            end
            for active_response in 1:n_active_responses
                faddeev_previous[active_response, union_id] = faddeev_next[active_response, union_id]
            end
        end
    end

    if n_active_occupied > 0
        @inbounds for column in 1:n_active_occupied, row in 1:n_active_occupied
            active_response = row + (column - 1) * n_active_occupied
            matrix_powers[active_response, union_id] = row == column ? one(eltype(matrix_powers)) : zero(eltype(matrix_powers))
        end
        @inbounds for exponent in 1:(n_active_occupied - 1)
            previous_power_offset = (exponent - 1) * n_active_responses
            current_power_offset = exponent * n_active_responses
            if n_active_occupied < 32 && exponent == 1
                for active_response in 1:n_active_responses
                    matrix_powers[current_power_offset + active_response, union_id] = scaled_effective_active_overlap[active_response, union_id]
                end
                continue
            end
            for column in 1:n_active_occupied, row in 1:n_active_occupied
                value = zero(eltype(matrix_powers))
                for inner in 1:n_active_occupied
                    left = previous_power_offset + row + (inner - 1) * n_active_occupied
                    right = inner + (column - 1) * n_active_occupied
                    value += matrix_powers[left, union_id] * scaled_effective_active_overlap[right, union_id]
                end
                matrix_powers[current_power_offset + row + (column - 1) * n_active_occupied, union_id] = value
            end
        end

        _polynomial_gpu_fill_newton!(
            newton_cofactor_matrix,
            elementary_symmetric_coefficients,
            matrix_powers,
            n_active_responses,
            union_id,
            n_active_occupied - 1,
        )
        @inbounds for column in 1:n_active_occupied, row in 1:n_active_occupied
            active_response = row + (column - 1) * n_active_occupied
            transposed_response = column + (row - 1) * n_active_occupied
            first_cofactor_output[active_response, union_id] = newton_cofactor_matrix[transposed_response, union_id]
        end
    end

    contracted_second_cofactor = zero(eltype(cofactor_scalar_and_contracted_second_output))
    if n_active_occupied >= 2
        @inbounds for power in 0:(n_active_occupied - 2)
            elementary_order = n_active_occupied - 1 - power
            sign = isodd(power) ? -one(eltype(cofactor_scalar_and_contracted_second_output)) : one(eltype(cofactor_scalar_and_contracted_second_output))
            _polynomial_gpu_fill_newton!(
                newton_cofactor_matrix,
                elementary_symmetric_coefficients,
                matrix_powers,
                n_active_responses,
                union_id,
                elementary_order - 1,
            )
            power_offset = power * n_active_responses
            identity_factor = n_active_occupied < 32 && (power == 0 || power == n_active_occupied - 2)
            hessian_pair_contraction = zero(eltype(cofactor_scalar_and_contracted_second_output))
            column_pair_id = 0
            for first_column in 1:(n_active_occupied - 1)
                for second_column in (first_column + 1):n_active_occupied
                    column_pair_id += 1
                    state_pair_column_pair = Int(hessian_column_pair_indices[column_pair_id, union_id])
                    row_pair_id = 0
                    for first_row in 1:(n_active_occupied - 1)
                        for second_row in _hessian_second_row_indices(first_row, first_column, second_column, n_active_occupied, identity_factor)
                            row_pair_id = _occupied_hessian_pair_index(first_row, second_row, n_active_occupied)
                            left_11 = matrix_powers[power_offset + first_column + (first_row - 1) * n_active_occupied, union_id]
                            left_21 = matrix_powers[power_offset + second_column + (first_row - 1) * n_active_occupied, union_id]
                            left_12 = matrix_powers[power_offset + first_column + (second_row - 1) * n_active_occupied, union_id]
                            left_22 = matrix_powers[power_offset + second_column + (second_row - 1) * n_active_occupied, union_id]
                            right_22 = newton_cofactor_matrix[second_column + (second_row - 1) * n_active_occupied, union_id]
                            right_12 = newton_cofactor_matrix[first_column + (second_row - 1) * n_active_occupied, union_id]
                            right_21 = newton_cofactor_matrix[second_column + (first_row - 1) * n_active_occupied, union_id]
                            right_11 = newton_cofactor_matrix[first_column + (first_row - 1) * n_active_occupied, union_id]
                            wedge_bilinear =
                                left_11 * right_22 -
                                left_21 * right_12 -
                                left_12 * right_21 +
                                left_22 * right_11
                            state_pair_row_pair = Int(hessian_row_pair_indices[row_pair_id, union_id])
                            hessian_pair_contraction += state_pair_hessian_response_device[state_pair_row_pair, state_pair_column_pair] * wedge_bilinear
                        end
                    end
                end
            end
            contracted_second_cofactor += sign * hessian_pair_contraction
        end
    end

    @inbounds cofactor_scalar_and_contracted_second_output[1, union_id] = elementary_symmetric_coefficients[n_active_occupied + 1, union_id]
    @inbounds cofactor_scalar_and_contracted_second_output[2, union_id] = contracted_second_cofactor
    return nothing
end

"""
    build_spin_pair_intermediates_cuda_polynomial_hessian_pair(...)

Build one fixed-occupied-count AHPC spin-pair intermediate domain on exactly one
CUDA device using the state-pair determinant-Hessian unordered-pair response for
the contracted second complementary cofactor. Host active-response staging,
union order, first-cofactor order, binary scaling, batch-width selection, and
floating-point traversal match the validated implementation. Inputs are read-only
and the result is returned in deterministic union-support order.
"""
function build_spin_pair_intermediates_cuda_polynomial_hessian_pair(
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    support::SpinPairSupportDomain;
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    (T === Float32 || T === Float64) || throw(ArgumentError("CUDA polynomial Hessian-pair engine supports Float32 and Float64"))
    runtime.use_gpu || throw(ArgumentError("CUDA polynomial Hessian-pair construction requires runtime.use_gpu=true"))
    length(runtime.gpu_devices) == 1 || throw(ArgumentError("single-device Hessian-pair construction requires exactly one configured GPU"))
    n_active_orbitals = _validate_spin_pair_intermediate_inputs(preparation, coefficients)
    maximum_active_occupied, feature_entries = _spin_pair_support_shape(support.union_support, n_active_orbitals)
    maximum_active_occupied <= runtime.max_active_occupied || throw(ArgumentError("union support occupied count exceeds max_active_occupied"))
    maximum_active_occupied <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA polynomial occupied count exceeds occupied-rank limit 32"))
    n_union = length(support.union_support)
    fixed_active_occupied = n_union == 0 || feature_entries == Base.Checked.checked_mul(
        n_union,
        Base.Checked.checked_mul(maximum_active_occupied, maximum_active_occupied),
    )
    fixed_active_occupied || throw(ArgumentError("single-device Hessian-pair construction requires one fixed occupied count per union support"))
    maximum_active_occupied == 0 && return _build_trivial_spin_pair_intermediates_cuda_polynomial(
        preparation,
        support;
        runtime,
        telemetry,
        strategy=:cuda_polynomial_hessian_pair_response,
    )

    n_state_pair_active_responses = Base.Checked.checked_mul(n_active_orbitals, n_active_orbitals)
    batch_width, available_before = _polynomial_spin_pair_gpu_batch_width(
        T,
        maximum_active_occupied,
        n_union,
        n_state_pair_active_responses,
        runtime,
    )
    estimate = _polynomial_spin_pair_hessian_pair_gpu_memory(
        T,
        preparation,
        n_union,
        feature_entries,
        maximum_active_occupied,
        batch_width,
    )
    enforce_memory_limits(estimate, runtime)
    batch_count = cld(n_union, batch_width)
    batch_memory_bound_per_union_bytes = _polynomial_spin_pair_gpu_per_union_device_bytes(T, maximum_active_occupied)
    actual_per_union_device_bytes = _polynomial_spin_pair_hessian_pair_gpu_per_union_device_bytes(T, maximum_active_occupied)
    n_state_pair_hessian_pairs = _hessian_pair_count(n_active_orbitals)
    state_pair_hessian_response_bytes = estimate_array_bytes(T, n_state_pair_hessian_pairs, n_state_pair_hessian_pairs)
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
        :strategy => :cuda_polynomial_hessian_pair_response,
        :n_union_spin_pair_support => n_union,
        :maximum_active_occupied => maximum_active_occupied,
        :occupied_rank_within_polynomial_limit =>
            maximum_active_occupied <= AHPC_MAX_OCCUPIED_RANK,
        :recurrence_storage => :union_contiguous,
        :fixed_active_occupied => true,
        :polynomial_cofactor_engine => true,
        :contracted_second_complexity => :O_n_active_occupied_5,
        :polynomial_workspace_reuse => true,
        :subset_recurrence_runtime => false,
        :state_pair_hessian_pair_contraction => true,
        :hessian_pair_index_staging => true,
        :conservative_batch_memory_bound => true,
        :batch_width_selection_per_union_bytes => batch_memory_bound_per_union_bytes,
        :actual_per_union_device_bytes => actual_per_union_device_bytes,
        :n_state_pair_hessian_pairs => n_state_pair_hessian_pairs,
        :state_pair_hessian_response_bytes => state_pair_hessian_response_bytes,
        :rank_uniform_arithmetic => true,
        :implicit_response_ids => true,
        :packed_spin_construction => true,
        :structural_feature_kernel => true,
        :gpu_devices => runtime.gpu_devices,
        :gpu_count => 1,
        :gpu_batch_count => batch_count,
        :gpu_batch_width => batch_width,
        :scratch_bytes_per_device => runtime.gpu_intermediate_scratch_bytes_per_device,
        :available_device_bytes_before_workspace => available_before,
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
        n_local_active_responses = n_active_occupied * n_active_occupied
        n_local_hessian_pairs = _hessian_pair_count(n_active_occupied)
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
        feature_offsets[end] == feature_entries + 1 || throw(ArgumentError("CUDA Hessian-pair feature count mismatch"))

        state_pair_response = state_pair_hessian_response(coefficients, n_active_orbitals)
        cofactor_staging = PolynomialSpinPairGPUHostBuffers(T, n_active_occupied, batch_width)
        _activate_cuda_device(runtime)
        scaled_effective_active_overlap_device = nothing
        active_response_or_hessian_pair_indices_device = nothing
        state_pair_hessian_response_device = nothing
        polynomial_workspace = nothing
        h2d_bytes = 0
        d2h_bytes = 0
        try
            scaled_effective_active_overlap_device = CUDA.CuArray{T}(undef, n_local_active_responses, batch_width)
            active_response_or_hessian_pair_indices_device = CUDA.CuArray{UInt16}(undef, n_local_active_responses, batch_width)
            state_pair_hessian_response_device = CUDA.CuArray(state_pair_response)
            polynomial_workspace = PolynomialSpinPairHessianPairGPUWorkspace(
                T,
                n_active_occupied,
                batch_width,
                runtime,
            )

            union_first = 1
            while union_first <= n_union
                union_last = min(n_union, union_first + batch_width - 1)
                width = _fill_polynomial_spin_pair_gpu_host_batch!(
                    cofactor_staging,
                    support,
                    preparation,
                    n_active_orbitals,
                    union_first,
                    union_last,
                    n_active_occupied,
                )
                Base.unsafe_copyto!(
                    scaled_effective_active_overlap_device,
                    1,
                    cofactor_staging.scaled_effective_active_overlap,
                    1,
                    n_local_active_responses * width,
                )
                Base.unsafe_copyto!(
                    active_response_or_hessian_pair_indices_device,
                    1,
                    cofactor_staging.active_response_indices,
                    1,
                    n_local_active_responses * width,
                )
                h2d_bytes = Base.Checked.checked_add(
                    h2d_bytes,
                    Base.Checked.checked_mul(
                        width,
                        Base.Checked.checked_add(
                            estimate_array_bytes(T, n_local_active_responses),
                            estimate_array_bytes(UInt16, n_local_active_responses),
                        ),
                    ),
                )

                CUDA.@cuda threads=AHPC_POLYNOMIAL_GPU_THREADS blocks=cld(width, AHPC_POLYNOMIAL_GPU_THREADS) _polynomial_spin_pair_response_ids_to_hessian_pair_indices_kernel!(
                    active_response_or_hessian_pair_indices_device,
                    Int32(n_active_occupied),
                    Int32(n_active_orbitals),
                    Int32(width),
                )
                column_pair_indices_device = @view active_response_or_hessian_pair_indices_device[1:n_local_hessian_pairs, :]
                row_pair_indices_device = @view active_response_or_hessian_pair_indices_device[(n_local_hessian_pairs + 1):(2 * n_local_hessian_pairs), :]
                _launch_polynomial_spin_pair_hessian_pair_gpu!((
                    polynomial_workspace.elementary_symmetric_coefficients,
                    polynomial_workspace.matrix_powers,
                    polynomial_workspace.newton_cofactor_matrix,
                    polynomial_workspace.first_cofactor_output,
                    polynomial_workspace.cofactor_scalar_and_contracted_second_output,
                    scaled_effective_active_overlap_device,
                    row_pair_indices_device,
                    column_pair_indices_device,
                    state_pair_hessian_response_device),
                    Int32(n_active_occupied),
                    Int32(width),
                )
                CUDA.synchronize()
                Base.unsafe_copyto!(
                    cofactor_staging.first_cofactor_output,
                    1,
                    polynomial_workspace.first_cofactor_output,
                    1,
                    n_local_active_responses * width,
                )
                Base.unsafe_copyto!(
                    cofactor_staging.cofactor_scalar_and_contracted_second_output,
                    1,
                    polynomial_workspace.cofactor_scalar_and_contracted_second_output,
                    1,
                    2 * width,
                )
                d2h_bytes = Base.Checked.checked_add(
                    d2h_bytes,
                    estimate_array_bytes(T, width, n_local_active_responses + 2),
                )

                @inbounds for local_union in 1:width
                    _finish_polynomial_spin_pair_gpu_host_union!(
                        cofactor_scalar,
                        one_body_active,
                        two_body_inactive_active,
                        same_spin_two_body,
                        first_cofactor_values,
                        cofactor_staging,
                        coefficients,
                        feature_offsets,
                        union_first + local_union - 1,
                        local_union,
                        n_active_occupied,
                    )
                end
                union_first = union_last + 1
            end
            phase_metadata[:gpu_h2d_bytes] = h2d_bytes
            phase_metadata[:gpu_d2h_bytes] = d2h_bytes
        finally
            polynomial_workspace === nothing || _close_polynomial_spin_pair_hessian_pair_gpu_workspace!(polynomial_workspace)
            state_pair_hessian_response_device === nothing || CUDA.unsafe_free!(state_pair_hessian_response_device)
            active_response_or_hessian_pair_indices_device === nothing || CUDA.unsafe_free!(active_response_or_hessian_pair_indices_device)
            scaled_effective_active_overlap_device === nothing || CUDA.unsafe_free!(scaled_effective_active_overlap_device)
        end

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

    intermediates = isnothing(telemetry) ? work() : record_phase!(
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
        set_counter!(telemetry, :n_union_kernel_workers, 1)
        set_counter!(telemetry, :spin_pair_response_id_storage_bytes, 0)
        set_counter!(telemetry, :spin_pair_response_id_payload_bytes, 0)
        set_counter!(telemetry, :n_cuda_spin_pair_batches, batch_count)
        set_counter!(telemetry, :n_cuda_spin_pair_devices, 1)
        increment_counter!(telemetry, :n_inverse_calls, 0)
        increment_counter!(telemetry, :n_rank_classifier_calls, 0)
        increment_counter!(telemetry, :n_svd_calls, 0)
    end
    return intermediates
end
