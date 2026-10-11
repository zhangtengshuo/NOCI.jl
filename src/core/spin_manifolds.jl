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

 File:            src/core/spin_manifolds.jl
 Purpose:         Complete spin-coupling chain enumeration, manifold dimensions, and explicit-chain validation.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _validate_spin_multiplicities(fragment_spin_multiplicities, total_spin_multiplicity)

Validate positive fragment and total spin multiplicities and return owned integer values.
Multiplicities are dimensionless values 2S+1, inputs are not mutated, no molecular or CUDA data are touched, and empty or nonpositive inputs raise ArgumentError.
"""
function _validate_spin_multiplicities(
    fragment_spin_multiplicities::AbstractVector{<:Integer},
    total_spin_multiplicity::Integer,
)
    multiplicities = Int.(fragment_spin_multiplicities)
    isempty(multiplicities) && throw(ArgumentError("at least one fragment spin multiplicity is required"))
    all(>(0), multiplicities) || throw(ArgumentError("fragment spin multiplicities must be positive"))
    total = Int(total_spin_multiplicity)
    total > 0 || throw(ArgumentError("total spin multiplicity must be positive"))
    return multiplicities, total
end

"""
    _coupled_twice_spin_values(left_twice_spin, right_twice_spin)

Return every allowed doubled spin 2S from coupling two angular momenta under the exact triangle and parity rules.
Arguments and returned values are dimensionless nonnegative integers, no floating-point arithmetic or CUDA work occurs, and negative doubled spins raise ArgumentError.
"""
function _coupled_twice_spin_values(
    left_twice_spin::Integer,
    right_twice_spin::Integer,
)
    left = Int(left_twice_spin)
    right = Int(right_twice_spin)
    left >= 0 && right >= 0 || throw(ArgumentError("doubled spins must be nonnegative"))
    return collect(abs(left - right):2:(left + right))
end

"""
    spin_coupling_chains(fragment_spin_multiplicities, total_spin_multiplicity)

Enumerate the complete deterministic left-to-right spin-coupling basis for the requested total spin.
Each returned vector stores multiplicities 2S+1 after successive fragment couplings and therefore has length n_fragments-1; the final entry equals the requested total multiplicity.
Inputs are read-only and dimensionless, enumeration uses exact integer doubled-spin arithmetic on CPU, no molecular determinant or CUDA storage is created, and an unreachable target returns an empty vector.
"""
function spin_coupling_chains(
    fragment_spin_multiplicities::AbstractVector{<:Integer},
    total_spin_multiplicity::Integer,
)
    multiplicities, total = _validate_spin_multiplicities(
        fragment_spin_multiplicities,
        total_spin_multiplicity,
    )
    fragment_count = length(multiplicities)
    target_twice_spin = total - 1
    if fragment_count == 1
        return multiplicities[1] == total ? [Int[]] : Vector{Vector{Int}}()
    end

    chains = Vector{Vector{Int}}()
    current_chain = Int[]
    local_twice_spins = multiplicities .- 1

    function descend(fragment::Int, previous_twice_spin::Int)
        next_twice_spin_values = _coupled_twice_spin_values(
            previous_twice_spin,
            local_twice_spins[fragment],
        )
        if fragment == fragment_count
            target_twice_spin in next_twice_spin_values || return nothing
            push!(current_chain, total)
            push!(chains, copy(current_chain))
            pop!(current_chain)
            return nothing
        end
        for next_twice_spin in next_twice_spin_values
            push!(current_chain, next_twice_spin + 1)
            descend(fragment + 1, next_twice_spin)
            pop!(current_chain)
        end
        return nothing
    end

    descend(2, local_twice_spins[1])
    return chains
end

"""
    spin_manifold_dimension(fragment_spin_multiplicities, total_spin_multiplicity)

Return the multiplicity of the requested total-spin irreducible representation in the ordered fragment-spin direct product.
The result is a dimensionless integer obtained by an independent dynamic-programming count in doubled-spin space; inputs are read-only, no coupling-chain vectors or molecular data are retained, and integer overflow raises through checked arithmetic.
"""
function spin_manifold_dimension(
    fragment_spin_multiplicities::AbstractVector{<:Integer},
    total_spin_multiplicity::Integer,
)
    multiplicities, total = _validate_spin_multiplicities(
        fragment_spin_multiplicities,
        total_spin_multiplicity,
    )
    counts = Dict{Int,Int}(multiplicities[1] - 1 => 1)
    for fragment in 2:length(multiplicities)
        local_twice_spin = multiplicities[fragment] - 1
        next_counts = Dict{Int,Int}()
        for (previous_twice_spin, previous_count) in counts
            for next_twice_spin in _coupled_twice_spin_values(
                previous_twice_spin,
                local_twice_spin,
            )
                next_counts[next_twice_spin] = Base.Checked.checked_add(
                    get(next_counts, next_twice_spin, 0),
                    previous_count,
                )
            end
        end
        counts = next_counts
    end
    return get(counts, total - 1, 0)
end

"""
    _explicit_spin_coupling_targets(fragment_count, total_spin_multiplicity, coupling_multiplicities)

Normalize one legacy explicit coupling specification to a complete chain including the final total-spin multiplicity.
Inputs and result are dimensionless integer metadata, inputs are not mutated, no molecular or CUDA data are touched, and invalid chain lengths or final multiplicities raise ArgumentError.
"""
function _explicit_spin_coupling_targets(
    fragment_count::Integer,
    total_spin_multiplicity::Integer,
    coupling_multiplicities::AbstractVector{<:Integer},
)
    count = Int(fragment_count)
    count >= 2 || throw(ArgumentError("explicit coupling targets require at least two fragments"))
    total = Int(total_spin_multiplicity)
    total > 0 || throw(ArgumentError("total spin multiplicity must be positive"))
    requested = Int.(coupling_multiplicities)
    all(>(0), requested) || throw(ArgumentError("coupling multiplicities must be positive"))
    if length(requested) == count - 2
        return vcat(requested, total)
    end
    length(requested) == count - 1 || throw(ArgumentError("invalid number of coupling multiplicities"))
    last(requested) == total || throw(ArgumentError("final coupling multiplicity must match the system multiplicity"))
    return requested
end

"""
    _resolved_spin_coupling_chains(fragment_spin_multiplicities, total_spin_multiplicity, coupling_multiplicities)

Return either the complete target-spin coupling manifold for an empty production specification or one validated legacy explicit chain.
All multiplicities are dimensionless 2S+1 values, returned chains are CPU-owned, no molecular determinant or CUDA work occurs, and an unreachable total spin or illegal explicit chain raises ArgumentError.
"""
function _resolved_spin_coupling_chains(
    fragment_spin_multiplicities::AbstractVector{<:Integer},
    total_spin_multiplicity::Integer,
    coupling_multiplicities::AbstractVector{<:Integer},
)
    complete = spin_coupling_chains(
        fragment_spin_multiplicities,
        total_spin_multiplicity,
    )
    isempty(complete) && throw(ArgumentError(
        "fragment spins cannot couple to total multiplicity $(Int(total_spin_multiplicity))",
    ))
    isempty(coupling_multiplicities) && return complete
    targets = _explicit_spin_coupling_targets(
        length(fragment_spin_multiplicities),
        total_spin_multiplicity,
        coupling_multiplicities,
    )
    any(chain -> chain == targets, complete) || throw(ArgumentError(
        "explicit coupling multiplicities $(targets) do not define a legal target-spin basis function",
    ))
    return [targets]
end

"""
    _spin_coupled_basis_label(configuration_label, coupling_multiplicities, manifold_dimension)

Return a deterministic internal MEBF label for one member of a multidimensional complete spin manifold.
The label and integer coupling metadata are read-only, the returned string is CPU-owned, no scientific arrays are touched, and nonpositive manifold dimensions raise ArgumentError.
"""
function _spin_coupled_basis_label(
    configuration_label::AbstractString,
    coupling_multiplicities::AbstractVector{<:Integer},
    manifold_dimension::Integer,
)
    dimension = Int(manifold_dimension)
    dimension > 0 || throw(ArgumentError("spin-manifold dimension must be positive"))
    dimension == 1 && return String(configuration_label)
    return string(
        configuration_label,
        "::spin[",
        join(Int.(coupling_multiplicities), ","),
        "]",
    )
end

"""
    _spin_coupled_basis_descriptors(configuration_label, fragment_spin_multiplicities, total_spin_multiplicity, coupling_multiplicities)

Resolve one electronic configuration into deterministic internal spin-coupled basis descriptors.
An empty coupling specification generates the complete target-spin manifold and labels every member when its dimension exceeds one; a nonempty specification retains the configuration label and selects one validated legacy chain.
Returned metadata are CPU-owned and dimensionless except labels, inputs are not mutated, no molecular determinant or CUDA work occurs, and inconsistent manifold counts raise ArgumentError.
"""
function _spin_coupled_basis_descriptors(
    configuration_label::AbstractString,
    fragment_spin_multiplicities::AbstractVector{<:Integer},
    total_spin_multiplicity::Integer,
    coupling_multiplicities::AbstractVector{<:Integer},
)
    dimension = spin_manifold_dimension(
        fragment_spin_multiplicities,
        total_spin_multiplicity,
    )
    dimension > 0 || throw(ArgumentError(
        "configuration $(configuration_label) cannot reach total multiplicity $(Int(total_spin_multiplicity))",
    ))
    chains = _resolved_spin_coupling_chains(
        fragment_spin_multiplicities,
        total_spin_multiplicity,
        coupling_multiplicities,
    )
    complete_manifold = isempty(coupling_multiplicities)
    if complete_manifold && length(chains) != dimension
        throw(ArgumentError("spin-chain enumeration count differs from the independent manifold dimension"))
    end
    descriptors = NamedTuple[]
    for chain in chains
        label = complete_manifold ?
            _spin_coupled_basis_label(configuration_label, chain, dimension) :
            String(configuration_label)
        push!(
            descriptors,
            (
                label=label,
                configuration_label=String(configuration_label),
                coupling_multiplicities=copy(chain),
                manifold_dimension=dimension,
                complete_manifold=complete_manifold,
            ),
        )
    end
    return descriptors
end
