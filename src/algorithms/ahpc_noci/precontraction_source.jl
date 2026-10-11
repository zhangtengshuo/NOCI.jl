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

 File:            src/algorithms/ahpc_noci/precontraction_source.jl
 Purpose:         Backend dispatch preparing ERI-factor shards, packed ERIs, or expanded CPU superoperators.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _prepare_ahpc_precontraction_source(integrals; runtime=RuntimeOptions(), telemetry=nothing)

Return metric-whitened ERI factors unchanged on CPU, upload reusable ERI-factor shards once on CUDA, retain packed ERIs for streamed CUDA precontraction, or expand packed ERIs once into reusable CPU superoperators.
Inputs are read-only, CPU expansion and CUDA residency are memory-preflighted and optionally timed, the returned CUDA source owns device resources until matrix assembly cleanup, and resource-limit errors precede large allocation.
"""
function _prepare_ahpc_precontraction_source(
    integrals::Union{PackedERIValues{T},ERIFactorValues{T}};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    if integrals isa ERIFactorValues
        runtime.use_gpu || return integrals
        estimate = _eri_factors_gpu_resident_memory(T, integrals, runtime)
        build_work = () -> _prepare_eri_factors_gpu_precontraction_source(integrals, runtime)
        source = isnothing(telemetry) ? build_work() : record_phase!(
            build_work,
            telemetry,
            :prepare_eri_factor_gpu_source;
            estimate,
            metadata=Dict(
                :strategy => :resident_auxiliary_shards,
                :gpu_devices => runtime.gpu_devices,
                :gpu_count => length(runtime.gpu_devices),
                :n_orbital_pairs => size(integrals.coefficients, 1),
                :n_auxiliary => size(integrals.coefficients, 2),
                :projection_tile_cap => CHOLESKY_GPU_PROJECTION_TILE_CAP,
            ),
            collect_allocations=runtime.collect_allocations,
        )
        if !isnothing(telemetry)
            set_counter!(telemetry, :n_eri_factor_gpu_uploads, length(source.shards))
            set_counter!(telemetry, :n_eri_factor_gpu_resident_shards, length(source.shards))
        end
        return source
    end
    runtime.use_gpu && return integrals
    estimate = packed_eri_superoperator_memory(T, integrals.n_orbitals)
    enforce_memory_limits(estimate, runtime)
    build_work = () -> build_packed_eri_superoperators(integrals; runtime)
    return isnothing(telemetry) ? build_work() : record_phase!(
        build_work,
        telemetry,
        :build_packed_eri_superoperators;
        estimate,
        metadata=Dict(:n_orbitals => integrals.n_orbitals),
        collect_allocations=runtime.collect_allocations,
    )
end
