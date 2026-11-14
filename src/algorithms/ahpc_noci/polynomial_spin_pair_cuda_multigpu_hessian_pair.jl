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

# File: src/algorithms/ahpc_noci/polynomial_spin_pair_cuda_multigpu_hessian_pair.jl
# Purpose: Distribute spin-pair Hessian-response contraction across CUDA workers.
# SPDX-License-Identifier: Apache-2.0

"""
    _build_polynomial_spin_pair_gpu_hessian_pair_worker!(...)

Evaluate one deterministic contiguous union range on one CUDA device using the
state-pair determinant-Hessian response and write directly into disjoint slices
of the final host arrays. Scientific inputs are read-only, no worker-local final
result is retained, and the configured batch-memory bound is preserved.
"""
function _build_polynomial_spin_pair_gpu_hessian_pair_worker!(
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
    isempty(union_range) && throw(ArgumentError("Hessian-pair worker range must be nonempty"))
    worker_runtime = _polynomial_runtime_for_gpu_device(runtime, assignment.device)
    n_active_orbitals = size(preparation.effective_active_overlap, 1)
    n_state_pair_active_responses = Base.Checked.checked_mul(n_active_orbitals, n_active_orbitals)
    n_worker_unions = length(union_range)

    batch_width, available_before = _polynomial_spin_pair_gpu_batch_width(
        T,
        n_active_occupied,
        n_worker_unions,
        n_state_pair_active_responses,
        worker_runtime,
    )
    n_local_active_responses = Base.Checked.checked_mul(n_active_occupied, n_active_occupied)
    n_local_hessian_pairs = _hessian_pair_count(n_active_occupied)
    cofactor_staging = PolynomialSpinPairGPUHostBuffers(T, n_active_occupied, batch_width)

    _activate_cuda_device(worker_runtime)
    scaled_effective_active_overlap_device = nothing
    active_response_or_hessian_pair_indices_device = nothing
    state_pair_hessian_response_device = nothing
    polynomial_workspace = nothing
    h2d_bytes = 0
    d2h_bytes = 0
    batch_count = 0
    try
        scaled_effective_active_overlap_device = CUDA.CuArray{T}(undef, n_local_active_responses, batch_width)
        active_response_or_hessian_pair_indices_device = CUDA.CuArray{UInt16}(undef, n_local_active_responses, batch_width)
        state_pair_hessian_response_device = CUDA.CuArray(state_pair_hessian_response)
        polynomial_workspace = PolynomialSpinPairHessianPairGPUWorkspace(
            T,
            n_active_occupied,
            batch_width,
            worker_runtime,
        )

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
            second_cofactor_buffers = (
                polynomial_workspace.elementary_symmetric_coefficients,
                polynomial_workspace.matrix_powers,
                polynomial_workspace.newton_cofactor_matrix,
                polynomial_workspace.first_cofactor_output,
                polynomial_workspace.cofactor_scalar_and_contracted_second_output,
                scaled_effective_active_overlap_device,
                row_pair_indices_device,
                column_pair_indices_device,
                state_pair_hessian_response_device)
            _launch_polynomial_spin_pair_hessian_pair_gpu!(
                second_cofactor_buffers,
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
            batch_count = Base.Checked.checked_add(batch_count, 1)
            union_first = union_last + 1
        end
    finally
        polynomial_workspace === nothing || _close_polynomial_spin_pair_hessian_pair_gpu_workspace!(polynomial_workspace)
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
        worker_wall_seconds=Float64(time_ns() - worker_started_ns) / 1.0e9,
    )
end
