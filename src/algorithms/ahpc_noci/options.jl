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

 File:            src/algorithms/ahpc_noci/options.jl
 Purpose:         Typed AHPC acceleration options holding the mutually exclusive support-lookup implementations.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    SupportLookup

Mutually exclusive presized support-index lookup implementations for production AHPC-NOCI.
All values preserve the same compact support and floating-point reduction order.
"""
@enum SupportLookup begin
    DictPresizedLookup
    FlatHashPresizedLookup
    AdaptiveLookup
end

"""
    CofactorEvaluation

Select the occupied-spin complementary-cofactor evaluation used by AHPC union kernels.
`PolynomialCofactorEvaluation` is the rank-uniform PDDC path and remains the production default; `LUCertifiedCofactorEvaluation` attempts a certificate-guarded full-rank LU path on CPU or CUDA and falls back to the polynomial PDDC path whenever the certificate is not satisfied.
"""
@enum CofactorEvaluation begin
    PolynomialCofactorEvaluation
    LUCertifiedCofactorEvaluation
end

"""
    AHPCAccelerationOptions(; support_lookup=AdaptiveLookup, cofactor_evaluation=PolynomialCofactorEvaluation)

Configure production AHPC support lookup and occupied-spin cofactor evaluation.
Construction performs no allocation proportional to the molecular system, the polynomial cofactor path remains the default, and the LU-certified path is available on CPU and CUDA.
"""
Base.@kwdef struct AHPCAccelerationOptions
    support_lookup::SupportLookup = AdaptiveLookup
    cofactor_evaluation::CofactorEvaluation = PolynomialCofactorEvaluation
end

"""
    AHPCSpinStringBranch(; acceleration=AHPCAccelerationOptions())

Select the AHPC production path using canonical spin-string-pair reuse, direct injective sparse weights, and typed complementary-cofactor evaluation.
The acceleration object controls support lookup and selects either rank-uniform polynomial cofactors or a certificate-guarded full-rank LU path with polynomial PDDC fallback on CPU or CUDA; construction performs no molecular-scale allocation.
"""
struct AHPCSpinStringBranch <: AHPCBranch
    acceleration::AHPCAccelerationOptions
end

"""
    AHPCSpinStringBranch(; acceleration=AHPCAccelerationOptions())

Construct the AHPC spin-string path with an explicit or default acceleration configuration.
The supplied immutable configuration is retained without mutation.
"""
function AHPCSpinStringBranch(; acceleration::AHPCAccelerationOptions=AHPCAccelerationOptions())
    return AHPCSpinStringBranch(acceleration)
end

"""
    AHPCNOCI(; branch=AHPCSpinStringBranch())

Select AHPC-NOCI with its production spin-string contraction after active-Hamiltonian precontraction.
Construction retains only immutable algorithm options and performs no molecular-scale allocation.
"""
struct AHPCNOCI{B<:AHPCBranch} <: NOCIAlgorithm
    branch::B
end

"""
    AHPCNOCI(; branch=AHPCSpinStringBranch())

Construct AHPC-NOCI with the production spin-string path.
The branch is immutable and construction does not allocate scientific workspaces.
"""
function AHPCNOCI(; branch::AHPCBranch=AHPCSpinStringBranch())
    return AHPCNOCI(branch)
end

"""
    algorithm_name(algorithm)

Return the stable CLI/report identifier for a public NOCI algorithm.
The result is independent of AHPC acceleration choices.
"""
algorithm_name(::DPNOCI) = "dp-noci"

"""
    algorithm_name(algorithm::AHPCNOCI)

Return `"ahpc-noci"` for AHPC-NOCI without inspecting molecular data.
"""
algorithm_name(::AHPCNOCI) = "ahpc-noci"

"""
    branch_name(branch::AHPCSpinStringBranch)

Return the stable telemetry identifier for the AHPC spin-string production path.
"""
branch_name(::AHPCSpinStringBranch) = "spin-string"
