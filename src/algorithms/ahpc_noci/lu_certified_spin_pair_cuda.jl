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

 File:            src/algorithms/ahpc_noci/lu_certified_spin_pair_cuda.jl
 Purpose:         Historical all-GPU LU-certified comparator with polynomial CPU fallback.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    LUCertifiedSpinPairGPUWorkspace

Own one CUDA worker's reusable storage for certificate-guarded full-rank LU
complementary cofactors. Each union retains an in-place LU factorization, the
scaled inverse, first complementary cofactors, two scalar outputs, and one status
byte. Status 0 means LU singular before inverse construction, 1 means an inverse
was computed but the numerical certificate/output checks rejected the fast path,
and 2 means the LU result is certified.
"""
mutable struct LUCertifiedSpinPairGPUWorkspace{T<:Union{Float32,Float64}}
    device::Int
    n_active_occupied::Int
    batch_width::Int
    lu_factors::CUDA.CuMatrix{T}
    inverse::CUDA.CuMatrix{T}
    first_cofactor_output::CUDA.CuMatrix{T}
    cofactor_scalar_and_contracted_second_output::CUDA.CuMatrix{T}
    status::CUDA.CuVector{UInt8}
    closed::Bool
end

function LUCertifiedSpinPairGPUWorkspace(
    ::Type{T},
    n_active_occupied::Integer,
    batch_width::Integer,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    q = Int(n_active_occupied)
    width = Int(batch_width)
    1 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA LU-certified occupied count must lie in 1:32"))
    width >= 1 || throw(ArgumentError("CUDA LU-certified batch width must be positive"))
    runtime.use_gpu || throw(ArgumentError("CUDA LU-certified workspace requires runtime.use_gpu=true"))
    length(runtime.gpu_devices) == 1 || throw(ArgumentError("CUDA LU-certified workspace requires exactly one GPU"))
    _activate_cuda_device(runtime)
    q2 = Base.Checked.checked_mul(q, q)
    allocated = Any[]
    try
        lu_factors = CUDA.CuArray{T}(undef, q2, width); push!(allocated, lu_factors)
        inverse = CUDA.CuArray{T}(undef, q2, width); push!(allocated, inverse)
        first_cofactor = CUDA.CuArray{T}(undef, q2, width); push!(allocated, first_cofactor)
        cofactor_outputs = CUDA.CuArray{T}(undef, 2, width); push!(allocated, cofactor_outputs)
        status = CUDA.CuArray{UInt8}(undef, width); push!(allocated, status)
        return LUCertifiedSpinPairGPUWorkspace(
            runtime.gpu_device,
            q,
            width,
            lu_factors,
            inverse,
            first_cofactor,
            cofactor_outputs,
            status,
            false,
        )
    catch
        for array in Iterators.reverse(allocated)
            CUDA.unsafe_free!(array)
        end
        rethrow()
    end
end

function _close_lu_certified_spin_pair_gpu_workspace!(workspace::LUCertifiedSpinPairGPUWorkspace)
    workspace.closed && return nothing
    CUDA.device!(workspace.device)
    for array in (
        workspace.status,
        workspace.cofactor_scalar_and_contracted_second_output,
        workspace.first_cofactor_output,
        workspace.inverse,
        workspace.lu_factors,
    )
        CUDA.unsafe_free!(array)
    end
    workspace.closed = true
    return nothing
end

function _lu_certified_spin_pair_gpu_batch_width(
    ::Type{T},
    n_active_occupied::Integer,
    n_union::Integer,
    n_state_pair_active_responses::Integer,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    q = Int(n_active_occupied)
    unions = Int(n_union)
    n_active_responses = Int(n_state_pair_active_responses)
    1 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("CUDA LU-certified occupied count must lie in 1:32"))
    unions >= 1 || throw(ArgumentError("CUDA LU-certified batch selection requires at least one union"))
    runtime.use_gpu || throw(ArgumentError("CUDA LU-certified batch selection requires runtime.use_gpu=true"))
    length(runtime.gpu_devices) == 1 || throw(ArgumentError("one CUDA LU-certified worker requires exactly one configured GPU"))
    _activate_cuda_device(runtime)

    per_union = _lu_certified_spin_pair_gpu_per_union_device_bytes(T, q)
    fixed_device = estimate_array_bytes(T, n_active_responses, n_active_responses)
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

"""
    _lu_certified_spin_pair_gpu_kernel!(...)

Evaluate one binary-scaled occupied-overlap problem per CUDA thread by partial-
pivoted LU. Row swaps are applied simultaneously to the identity right-hand side;
forward elimination therefore constructs L⁻¹P and backward substitution produces
the inverse without storing a separate pivot vector. The same infinity-norm
residual and condition limits as the CPU LU certificate are enforced. Certified
unions emit determinant, first complementary cofactors, and the contracted second
cofactor in scaled units; rejected unions emit only a status code and are later
recomputed by the polynomial PDDC fallback.
"""
function _lu_certified_spin_pair_gpu_kernel!(
    lu_factors,
    inverse,
    first_cofactor_output,
    cofactor_scalar_and_contracted_second_output,
    status,
    scaled_effective_active_overlap,
    hessian_row_pair_indices,
    hessian_column_pair_indices,
    state_pair_hessian_response_device,
    n_active_occupied_device::Int32,
    batch_count::Int32,
    residual_tolerance,
    condition_limit,
)
    union_id = (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) + Int(CUDA.threadIdx().x)
    union_id <= Int(batch_count) || return nothing
    q = Int(n_active_occupied_device)
    T = eltype(lu_factors)

    @inbounds status[union_id] = UInt8(0)

    # Copy A_s into the LU workspace and initialize the right-hand side to I.
    @inbounds for column in 1:q, row in 1:q
        response = row + (column - 1) * q
        value = scaled_effective_active_overlap[response, union_id]
        if !isfinite(value)
            return nothing
        end
        lu_factors[response, union_id] = value
        inverse[response, union_id] = row == column ? one(T) : zero(T)
    end

    determinant_sign = one(T)

    # Partial-pivoted LU. Elimination is also applied to the identity RHS, so
    # after this loop inverse[:,union] stores L^{-1} P.
    @inbounds for pivot_column in 1:q
        pivot_row = pivot_column
        pivot_absolute = abs(lu_factors[pivot_column + (pivot_column - 1) * q, union_id])
        for row in (pivot_column + 1):q
            candidate = abs(lu_factors[row + (pivot_column - 1) * q, union_id])
            if candidate > pivot_absolute
                pivot_absolute = candidate
                pivot_row = row
            end
        end
        (!isfinite(pivot_absolute) || iszero(pivot_absolute)) && return nothing

        if pivot_row != pivot_column
            for column in 1:q
                first_index = pivot_column + (column - 1) * q
                second_index = pivot_row + (column - 1) * q
                temporary = lu_factors[first_index, union_id]
                lu_factors[first_index, union_id] = lu_factors[second_index, union_id]
                lu_factors[second_index, union_id] = temporary

                temporary = inverse[first_index, union_id]
                inverse[first_index, union_id] = inverse[second_index, union_id]
                inverse[second_index, union_id] = temporary
            end
            determinant_sign = -determinant_sign
        end

        pivot_value = lu_factors[pivot_column + (pivot_column - 1) * q, union_id]
        (!isfinite(pivot_value) || iszero(pivot_value)) && return nothing

        for row in (pivot_column + 1):q
            lower_index = row + (pivot_column - 1) * q
            multiplier = lu_factors[lower_index, union_id] / pivot_value
            isfinite(multiplier) || return nothing
            lu_factors[lower_index, union_id] = multiplier
            for column in (pivot_column + 1):q
                index = row + (column - 1) * q
                lu_factors[index, union_id] -=
                    multiplier * lu_factors[pivot_column + (column - 1) * q, union_id]
            end
            for rhs_column in 1:q
                index = row + (rhs_column - 1) * q
                inverse[index, union_id] -=
                    multiplier * inverse[pivot_column + (rhs_column - 1) * q, union_id]
            end
        end
    end

    # Back substitution U X = L^{-1} P.
    @inbounds for rhs_column in 1:q
        for row in q:-1:1
            value = inverse[row + (rhs_column - 1) * q, union_id]
            for column in (row + 1):q
                value -= lu_factors[row + (column - 1) * q, union_id] *
                    inverse[column + (rhs_column - 1) * q, union_id]
            end
            diagonal = lu_factors[row + (row - 1) * q, union_id]
            (!isfinite(diagonal) || iszero(diagonal)) && return nothing
            value /= diagonal
            isfinite(value) || return nothing
            inverse[row + (rhs_column - 1) * q, union_id] = value
        end
    end
    @inbounds status[union_id] = UInt8(1)

    determinant_scaled = determinant_sign
    @inbounds for diagonal in 1:q
        determinant_scaled *= lu_factors[diagonal + (diagonal - 1) * q, union_id]
    end
    (!isfinite(determinant_scaled) || iszero(determinant_scaled)) && return nothing

    residual_inf = zero(T)
    scaled_matrix_inf = zero(T)
    inverse_inf = zero(T)
    @inbounds for row in 1:q
        residual_row_sum = zero(T)
        matrix_row_sum = zero(T)
        inverse_row_sum = zero(T)
        for column in 1:q
            product = zero(T)
            for inner in 1:q
                product +=
                    scaled_effective_active_overlap[row + (inner - 1) * q, union_id] *
                    inverse[inner + (column - 1) * q, union_id]
            end
            row == column && (product -= one(T))
            residual_row_sum += abs(product)
            matrix_row_sum += abs(scaled_effective_active_overlap[row + (column - 1) * q, union_id])
            inverse_row_sum += abs(inverse[row + (column - 1) * q, union_id])
        end
        residual_inf = max(residual_inf, residual_row_sum)
        scaled_matrix_inf = max(scaled_matrix_inf, matrix_row_sum)
        inverse_inf = max(inverse_inf, inverse_row_sum)
    end
    condition_inf = scaled_matrix_inf * inverse_inf
    if !isfinite(residual_inf) || !isfinite(condition_inf) ||
       residual_inf > residual_tolerance || condition_inf > condition_limit
        return nothing
    end

    @inbounds for column in 1:q, row in 1:q
        response = row + (column - 1) * q
        first_cofactor = determinant_scaled * inverse[column + (row - 1) * q, union_id]
        isfinite(first_cofactor) || return nothing
        first_cofactor_output[response, union_id] = first_cofactor
    end

    second_scaled = zero(T)
    if q >= 2
        column_pair_id = 0
        @inbounds for first_column in 1:(q - 1)
            for second_column in (first_column + 1):q
                column_pair_id += 1
                state_pair_column_pair = Int(hessian_column_pair_indices[column_pair_id, union_id])
                row_pair_id = 0
                for first_row in 1:(q - 1)
                    for second_row in (first_row + 1):q
                        row_pair_id += 1
                        state_pair_row_pair = Int(hessian_row_pair_indices[row_pair_id, union_id])
                        inverse_wedge =
                            inverse[first_column + (first_row - 1) * q, union_id] *
                            inverse[second_column + (second_row - 1) * q, union_id] -
                            inverse[first_column + (second_row - 1) * q, union_id] *
                            inverse[second_column + (first_row - 1) * q, union_id]
                        second_scaled +=
                            state_pair_hessian_response_device[state_pair_row_pair, state_pair_column_pair] *
                            inverse_wedge
                    end
                end
            end
        end
        second_scaled *= T(2) * determinant_scaled
    end
    isfinite(second_scaled) || return nothing

    @inbounds cofactor_scalar_and_contracted_second_output[1, union_id] = determinant_scaled
    @inbounds cofactor_scalar_and_contracted_second_output[2, union_id] = second_scaled
    @inbounds status[union_id] = UInt8(2)
    return nothing
end

function _launch_lu_certified_spin_pair_gpu!(
    workspace::LUCertifiedSpinPairGPUWorkspace{T},
    scaled_effective_active_overlap_device,
    row_pair_indices_device,
    column_pair_indices_device,
    state_pair_hessian_response_device,
    n_active_occupied::Int,
    width::Int,
) where {T<:Union{Float32,Float64}}
    CUDA.@cuda threads=AHPC_POLYNOMIAL_GPU_THREADS blocks=cld(width, AHPC_POLYNOMIAL_GPU_THREADS) _lu_certified_spin_pair_gpu_kernel!(
        workspace.lu_factors,
        workspace.inverse,
        workspace.first_cofactor_output,
        workspace.cofactor_scalar_and_contracted_second_output,
        workspace.status,
        scaled_effective_active_overlap_device,
        row_pair_indices_device,
        column_pair_indices_device,
        state_pair_hessian_response_device,
        Int32(n_active_occupied),
        Int32(width),
        T(AHPC_LU_CERTIFICATE_RESIDUAL_TOLERANCE),
        T(AHPC_LU_CERTIFICATE_CONDITION_LIMIT),
    )
    return nothing
end

function _build_lu_certified_spin_pair_gpu_worker!(
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
    isempty(union_range) && throw(ArgumentError("LU-certified CUDA worker range must be nonempty"))
    worker_runtime = _polynomial_runtime_for_gpu_device(runtime, assignment.device)
    n_active_orbitals = size(preparation.effective_active_overlap, 1)
    n_state_pair_active_responses = Base.Checked.checked_mul(n_active_orbitals, n_active_orbitals)
    n_worker_unions = length(union_range)
    batch_width, available_before = _lu_certified_spin_pair_gpu_batch_width(
        T,
        n_active_occupied,
        n_worker_unions,
        n_state_pair_active_responses,
        worker_runtime,
    )

    n_local_active_responses = Base.Checked.checked_mul(n_active_occupied, n_active_occupied)
    n_local_hessian_pairs = _hessian_pair_count(n_active_occupied)
    cofactor_staging = PolynomialSpinPairGPUHostBuffers(T, n_active_occupied, batch_width)
    status_host = Vector{UInt8}(undef, batch_width)
    fallback_workspace = SpinPairIntermediateWorkspace{T}()
    _resize_spin_pair_intermediate_workspace!(fallback_workspace, n_active_occupied, runtime)

    _activate_cuda_device(worker_runtime)
    scaled_effective_active_overlap_device = nothing
    active_response_or_hessian_pair_indices_device = nothing
    state_pair_hessian_response_device = nothing
    lu_workspace = nothing
    h2d_bytes = 0
    d2h_bytes = 0
    batch_count = 0
    accepted = 0
    fallbacks = 0
    inverse_calls = 0
    try
        scaled_effective_active_overlap_device = CUDA.CuArray{T}(undef, n_local_active_responses, batch_width)
        active_response_or_hessian_pair_indices_device = CUDA.CuArray{UInt16}(undef, n_local_active_responses, batch_width)
        state_pair_hessian_response_device = CUDA.CuArray(state_pair_hessian_response)
        lu_workspace = LUCertifiedSpinPairGPUWorkspace(T, n_active_occupied, batch_width, worker_runtime)

        union_first = first(union_range)
        final_union = last(union_range)
        while union_first <= final_union
            union_last = min(final_union, union_first + batch_width - 1)
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

            _launch_lu_certified_spin_pair_gpu!(
                lu_workspace,
                scaled_effective_active_overlap_device,
                row_pair_indices_device,
                column_pair_indices_device,
                state_pair_hessian_response_device,
                n_active_occupied,
                width,
            )
            CUDA.synchronize()

            Base.unsafe_copyto!(
                cofactor_staging.first_cofactor_output,
                1,
                lu_workspace.first_cofactor_output,
                1,
                n_local_active_responses * width,
            )
            Base.unsafe_copyto!(
                cofactor_staging.cofactor_scalar_and_contracted_second_output,
                1,
                lu_workspace.cofactor_scalar_and_contracted_second_output,
                1,
                2 * width,
            )
            Base.unsafe_copyto!(status_host, 1, lu_workspace.status, 1, width)
            d2h_bytes = Base.Checked.checked_add(
                d2h_bytes,
                Base.Checked.checked_add(
                    estimate_array_bytes(T, width, n_local_active_responses + 2),
                    estimate_array_bytes(UInt8, width),
                ),
            )

            @inbounds for local_union in 1:width
                union_id = union_first + local_union - 1
                status = status_host[local_union]
                status >= UInt8(1) && (inverse_calls += 1)
                if status == UInt8(2)
                    accepted += 1
                    _finish_polynomial_spin_pair_gpu_host_union!(
                        cofactor_scalar,
                        one_body_active,
                        two_body_inactive_active,
                        same_spin_two_body,
                        first_cofactor_values,
                        cofactor_staging,
                        coefficients,
                        feature_offsets,
                        union_id,
                        local_union,
                        n_active_occupied,
                    )
                else
                    fallbacks += 1
                    _build_spin_pair_intermediate!(
                        fallback_workspace,
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
                        n_active_orbitals,
                        PolynomialCofactorEvaluation,
                    )
                end
            end
            batch_count = Base.Checked.checked_add(batch_count, 1)
            union_first = union_last + 1
        end
    finally
        lu_workspace === nothing || _close_lu_certified_spin_pair_gpu_workspace!(lu_workspace)
        state_pair_hessian_response_device === nothing || CUDA.unsafe_free!(state_pair_hessian_response_device)
        active_response_or_hessian_pair_indices_device === nothing || CUDA.unsafe_free!(active_response_or_hessian_pair_indices_device)
        scaled_effective_active_overlap_device === nothing || CUDA.unsafe_free!(scaled_effective_active_overlap_device)
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
        worker_wall_seconds=Float64(time_ns() - worker_started_ns) / 1.0e9,
    )
end
