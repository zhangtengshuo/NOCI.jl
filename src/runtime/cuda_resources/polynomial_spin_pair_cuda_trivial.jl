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

 File:            src/algorithms/ahpc_noci/polynomial_spin_pair_cuda_trivial.jl
 Purpose:         Exact empty and rank-zero endpoint of the polynomial determinant-cofactor CUDA construction.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _build_trivial_spin_pair_intermediates_cuda_polynomial(preparation, support; runtime, telemetry, strategy)

Return the exact empty/rank-zero endpoint of the polynomial determinant-cofactor
formulation without launching CUDA or selecting another numerical algorithm.
Telemetry uses the normal spin-pair phase/counter schema with zero CUDA workers
and batches.
"""
function _build_trivial_spin_pair_intermediates_cuda_polynomial(
    preparation::AHPCStatePairPreparation{T},
    support::SpinPairSupportDomain;
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry},
    strategy::Symbol,
) where {T<:Union{Float32,Float64}}
    n_active_orbitals = size(preparation.effective_active_overlap, 1)
    n_union = length(support.union_support)
    _, feature_entries = _spin_pair_support_shape(support.union_support, n_active_orbitals)
    estimate = MemoryEstimate(_polynomial_spin_pair_gpu_output_bytes(T, n_union, feature_entries), 0)
    enforce_memory_limits(estimate, runtime)
    metadata = Dict{Symbol,Any}(
        :strategy => strategy,
        :n_union_spin_pair_support => n_union,
        :maximum_active_occupied => 0,
        :fixed_active_occupied => true,
        :polynomial_cofactor_engine => true,
        :contracted_second_complexity => :O_n_active_occupied_5,
        :state_pair_hessian_pair_contraction => true,
        :rank_uniform_arithmetic => true,
        :gpu_devices => runtime.gpu_devices,
        :gpu_count => 0,
        :gpu_batch_count => 0,
        :gpu_batch_width => 0,
    )
    work = () -> _build_trivial_spin_pair_intermediates(T, n_active_orbitals, support)
    result = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :build_spin_pair_intermediates;
        estimate,
        metadata,
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
        n_union > 0 && increment_counter!(telemetry, :n_packed_spin_intermediate_builds)
        set_counter!(telemetry, :n_union_kernel_workers, 0)
        set_counter!(telemetry, :spin_pair_response_id_storage_bytes, 0)
        set_counter!(telemetry, :spin_pair_response_id_payload_bytes, 0)
        set_counter!(telemetry, :n_cuda_spin_pair_batches, 0)
        set_counter!(telemetry, :n_cuda_spin_pair_devices, 0)
        set_counter!(telemetry, :polynomial_worker_duplicate_output_bytes, 0)
        set_counter!(telemetry, :polynomial_worker_output_merge_copies, 0)
        increment_counter!(telemetry, :n_inverse_calls, 0)
        increment_counter!(telemetry, :n_rank_classifier_calls, 0)
        increment_counter!(telemetry, :n_svd_calls, 0)
    end
    return result
end
