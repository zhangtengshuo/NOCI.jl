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

# File: src/integrals/factor_provenance.jl
# Purpose: Track the origin, metric, and metadata of electron-repulsion integral factors.
# SPDX-License-Identifier: Apache-2.0

"""
    ERIFactorProvenance(; origin=:unknown, ri_type=0, auxiliary_basis_kind=:unknown, threshold_kind=:unknown, generation_threshold=nothing)

Describe the physical origin of unit-metric Gram ERI factors without changing their numerical representation.
`origin` is `:conventional_cd`, `:ricd`, or `:unknown`; RICD uses RI type 4 and an aCD/acCD auxiliary basis, while conventional CD uses no auxiliary basis.
The optional positive finite threshold is in hartree and its kind distinguishes molecular ERI decomposition from atomic auxiliary-basis generation; unknown legacy origins carry no inferred threshold.
Construction copies no scientific arrays, mutates no input, initializes no device, and rejects inconsistent or nonfinite provenance with `ArgumentError`.
"""
struct ERIFactorProvenance
    origin::Symbol
    ri_type::Int
    auxiliary_basis_kind::Symbol
    threshold_kind::Symbol
    generation_threshold::Union{Nothing,Float64}

    """
        ERIFactorProvenance(; kwargs...)

    Validate factor origin, RI type, auxiliary-basis kind, and threshold semantics and return immutable CPU metadata.
    Counts are dimensionless and thresholds are positive hartree values; no inputs are mutated or device/runtime initialized, and unsupported combinations raise `ArgumentError`.
    """
    function ERIFactorProvenance(;
        origin::Symbol=:unknown,
        ri_type::Integer=0,
        auxiliary_basis_kind::Symbol=:unknown,
        threshold_kind::Symbol=:unknown,
        generation_threshold::Union{Nothing,Real}=nothing,
    )
        origin in (:unknown, :conventional_cd, :ricd) || throw(ArgumentError("unsupported ERI factor origin: $origin"))
        threshold = isnothing(generation_threshold) ? nothing : Float64(generation_threshold)
        isnothing(threshold) || (isfinite(threshold) && threshold > 0) || throw(ArgumentError("factor generation threshold must be finite and positive"))
        if origin == :conventional_cd
            ri_type == 0 && auxiliary_basis_kind == :none && threshold_kind == :molecular_eri_cd || throw(ArgumentError("inconsistent conventional-CD provenance"))
        elseif origin == :ricd
            ri_type == 4 && auxiliary_basis_kind in (:acd, :accd) && threshold_kind == :atomic_auxiliary_basis_cd || throw(ArgumentError("inconsistent RICD provenance"))
        else
            ri_type == 0 && auxiliary_basis_kind == :unknown && threshold_kind == :unknown && isnothing(threshold) || throw(ArgumentError("legacy unknown factor origin cannot carry inferred method metadata"))
        end
        return new(origin, Int(ri_type), auxiliary_basis_kind, threshold_kind, threshold)
    end
end

"""
    eri_factor_provenance_dict(provenance)

Return owned scalar/string metadata suitable for JLD2 and JSON, without scientific arrays or file/device handles.
The immutable input is not changed, the threshold uses hartree, and CPU allocation is limited to the small dictionary.
"""
function eri_factor_provenance_dict(provenance::ERIFactorProvenance)
    return Dict{String,Any}(
        "origin" => String(provenance.origin),
        "ri_type" => provenance.ri_type,
        "auxiliary_basis_kind" => String(provenance.auxiliary_basis_kind),
        "threshold_kind" => String(provenance.threshold_kind),
        "generation_threshold" => provenance.generation_threshold,
        "threshold_units" => "hartree",
    )
end

"""
    _read_eri_factor_provenance(mapping)

Decode and validate explicit factor provenance from one small JLD2-compatible mapping and return immutable metadata.
Inputs are read-only, thresholds must declare hartree units, missing/unknown fields and inconsistent origin/type/threshold values raise before numerical factor allocation, and no device is initialized.
"""
function _read_eri_factor_provenance(mapping)
    required = ("origin", "ri_type", "auxiliary_basis_kind", "threshold_kind", "generation_threshold", "threshold_units")
    mapping isa AbstractDict && all(key -> haskey(mapping, key), required) && length(mapping) == length(required) || throw(ArgumentError("incomplete or unsupported ERI factor provenance"))
    mapping["threshold_units"] == "hartree" || throw(ArgumentError("factor threshold units must be hartree"))
    return ERIFactorProvenance(
        origin=Symbol(mapping["origin"]), ri_type=mapping["ri_type"],
        auxiliary_basis_kind=Symbol(mapping["auxiliary_basis_kind"]), threshold_kind=Symbol(mapping["threshold_kind"]),
        generation_threshold=mapping["generation_threshold"],
    )
end
