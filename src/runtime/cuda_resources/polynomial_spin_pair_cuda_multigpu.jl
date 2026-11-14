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

# File: src/runtime/cuda_resources/polynomial_spin_pair_cuda_multigpu.jl
# Purpose: Coordinate multi-GPU spin-pair polynomial contraction.
# SPDX-License-Identifier: Apache-2.0

"""
    _build_spin_pair_intermediates_cuda_polynomial_hessian_pair(...)

Select the determinant-Hessian pair-space CUDA implementation by configured
device count. One device uses the single-device builder; two or more devices use
deterministic contiguous union ranges with direct writes into disjoint host
output slices. Scientific inputs are read-only and no algorithmic selector or
fallback is exposed to callers.
"""
function _build_spin_pair_intermediates_cuda_polynomial_hessian_pair(
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    support::SpinPairSupportDomain;
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("polynomial CUDA construction requires runtime.use_gpu=true"))
    isempty(runtime.gpu_devices) && throw(ArgumentError("polynomial CUDA construction requires at least one configured GPU"))
    if length(runtime.gpu_devices) == 1
        return build_spin_pair_intermediates_cuda_polynomial_hessian_pair(
            preparation,
            coefficients,
            support;
            runtime,
            telemetry,
        )
    end
    return build_spin_pair_intermediates_cuda_polynomial_multigpu_hessian_pair(
        preparation,
        coefficients,
        support;
        runtime,
        telemetry,
    )
end

"""
    build_spin_pair_intermediates_cuda_polynomial_distributed(preparation, coefficients, support; runtime, telemetry=nothing)

Build supported Float64 CUDA AHPC spin-pair intermediates with the rank-uniform
polynomial complementary-cofactor formulation in determinant-Hessian pair space.
Device topology changes scheduling only; scientific ordering and arithmetic are
fixed by the common formulation.
"""
function build_spin_pair_intermediates_cuda_polynomial_distributed(
    preparation::AHPCStatePairPreparation{Float64},
    coefficients::ActiveHamiltonianCoefficients{Float64},
    support::SpinPairSupportDomain;
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
)
    return _build_spin_pair_intermediates_cuda_polynomial_hessian_pair(
        preparation,
        coefficients,
        support;
        runtime,
        telemetry,
    )
end

"""
    build_spin_pair_intermediates_cuda_polynomial_distributed(preparation, coefficients, support; runtime, telemetry=nothing)

Build supported Float32 CUDA AHPC spin-pair intermediates with the rank-uniform
polynomial complementary-cofactor formulation in determinant-Hessian pair space.
Device topology changes scheduling only; scientific ordering and arithmetic are
fixed by the common formulation.
"""
function build_spin_pair_intermediates_cuda_polynomial_distributed(
    preparation::AHPCStatePairPreparation{Float32},
    coefficients::ActiveHamiltonianCoefficients{Float32},
    support::SpinPairSupportDomain;
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
)
    return _build_spin_pair_intermediates_cuda_polynomial_hessian_pair(
        preparation,
        coefficients,
        support;
        runtime,
        telemetry,
    )
end
