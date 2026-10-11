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

 File:            src/core/multifragment_count_preflight.jl
 Purpose:         Count-only projection of arbitrary-fragment MEBF construction before determinant materialization.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    MultifragmentMEBFCountPreflight

Store an exact count-only projection of arbitrary-fragment MEBF construction before any molecular determinant product is materialized.
All counts and projected bytes are CPU metadata, candidate/retained counts follow `problem.mebfs` order, determinant-pair block bounds follow production lower-triangular MEBF order, byte/count projections use `UInt128` so physically impossible materializations can still be reported without `Int` overflow, and no molecular orbital matrix, determinant product, pair domain, integral, or CUDA storage is retained.
"""
struct MultifragmentMEBFCountPreflight{T<:AbstractFloat}
    labels::Vector{String}
    fragment_counts::Vector{Int}
    active_orbital_ranks::Vector{Int}
    spin_sector_path_counts::Vector{Int}
    local_microstate_counts::Vector{Vector{Int}}
    naive_microstate_cartesian_counts::Vector{UInt128}
    candidate_determinant_counts::Vector{Int}
    normalized_max_coefficients::Vector{T}
    collection_max_coefficient::T
    collection_cutoff::T
    retained_determinant_counts::Vector{Int}
    candidate_product_storage_bytes::Vector{UInt128}
    retained_expansion_storage_bytes::Vector{UInt128}
    projected_peak_materialization_bytes::UInt128
    full_determinant_pair_block_upper_bounds::Vector{UInt128}
    full_determinant_pair_upper_bound::UInt128
    projected_pair_cutoff_storage_bytes::UInt128
    determinant_index_int32_compatible::Bool
end

"""
    _multifragment_product_magnitude_statistics(prepared, paths, tau_ci)

Traverse all exactly screened determinant tuples for a prepared multifragment MEBF and return `(count, norm_squared, maximum_absolute_coefficient)` before molecular normalization.
Prepared fragment caches and legal sector paths are read-only, coefficient arithmetic and traversal order match `build_multifragment_product64`, no molecular masks/product arrays/CUDA state are created, count addition is checked, and zero-survivor or non-finite statistics raise explicitly.
"""
function _multifragment_product_magnitude_statistics(
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    paths::AbstractVector{MultifragmentSectorPath{T}},
    tau_ci::T,
) where {T<:AbstractFloat}
    count = 0
    norm_squared = zero(T)
    maximum_absolute_coefficient = zero(T)
    for path in paths
        _visit_multifragment_path_products!(prepared, path, tau_ci) do _, local_product
            coefficient = local_product * path.clebsch_product
            magnitude = abs(coefficient)
            count = Base.Checked.checked_add(count, 1)
            norm_squared += abs2(coefficient)
            maximum_absolute_coefficient = max(maximum_absolute_coefficient, magnitude)
        end
    end
    count > 0 || throw(ArgumentError("no molecular determinant products survived multifragment screening"))
    isfinite(norm_squared) && norm_squared > zero(T) || throw(ArgumentError("multifragment count-only norm is non-finite or zero"))
    isfinite(maximum_absolute_coefficient) && maximum_absolute_coefficient > zero(T) || throw(ArgumentError("multifragment count-only maximum coefficient is non-finite or zero"))
    return count, norm_squared, maximum_absolute_coefficient
end

"""
    _multifragment_retained_count(prepared, paths, tau_ci, norm, collection_cutoff)

Count exactly how many candidate molecular determinants survive the established collection-wide normalized-coefficient cutoff without materializing any determinant product.
Inputs are read-only and dimensionless, fragment-product screening still uses the existing pre-Clebsch `tau_ci` rule, the final strict `abs(C)>collection_cutoff` comparison matches `_reduce_multifragment_basis_collection`, no molecular masks/CUDA storage are created, and checked count overflow raises.
"""
function _multifragment_retained_count(
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    paths::AbstractVector{MultifragmentSectorPath{T}},
    tau_ci::T,
    norm::T,
    collection_cutoff::T,
) where {T<:AbstractFloat}
    isfinite(norm) && norm > zero(T) || throw(ArgumentError("multifragment normalization factor must be finite and positive"))
    isfinite(collection_cutoff) && collection_cutoff >= zero(T) || throw(ArgumentError("collection cutoff must be finite and nonnegative"))
    retained = 0
    for path in paths
        _visit_multifragment_path_products!(prepared, path, tau_ci) do _, local_product
            normalized_magnitude = abs(local_product * path.clebsch_product) / norm
            normalized_magnitude > collection_cutoff && (retained = Base.Checked.checked_add(retained, 1))
        end
    end
    retained > 0 || throw(ArgumentError("collection-wide count-only preflight removes every determinant from one MEBF"))
    return retained
end

@inline function _multifragment_projected_packed_bytes(::Type{T}, count::Integer) where {T<:AbstractFloat}
    count >= 0 || throw(ArgumentError("projected determinant count must be nonnegative"))
    return Base.Checked.checked_mul(UInt128(count), UInt128(sizeof(T) + 2 * sizeof(UInt64)))
end

"""
    _multifragment_full_pair_upper_bounds(retained_determinant_counts)

Project the unscreened determinant-pair cardinality and row-cutoff storage implied by retained MEBF determinant counts using the same lower-triangular block convention as `full_determinant_pair_count`.
Counts are read-only and dimensionless, diagonal blocks use `N(N+1)/2`, off-diagonal blocks use `N_left*N_right`, all cardinalities/bytes use checked `UInt128`, no pair domain or coefficient vector is materialized, and the returned Int32-compatibility flag records whether current production determinant indices could represent every retained MEBF.
"""
function _multifragment_full_pair_upper_bounds(retained_determinant_counts::AbstractVector{<:Integer})
    all(>(0), retained_determinant_counts) || throw(ArgumentError("retained determinant counts must be positive"))
    block_counts = UInt128[]
    total = UInt128(0)
    cutoff_storage = UInt128(0)
    for left_mebf in eachindex(retained_determinant_counts), right_mebf in 1:left_mebf
        n_left = UInt128(retained_determinant_counts[left_mebf])
        n_right = UInt128(retained_determinant_counts[right_mebf])
        block_count = if left_mebf == right_mebf
            Base.Checked.checked_mul(n_left, Base.Checked.checked_add(n_left, UInt128(1))) ÷ UInt128(2)
        else
            Base.Checked.checked_mul(n_left, n_right)
        end
        push!(block_counts, block_count)
        total = Base.Checked.checked_add(total, block_count)
        cutoff_storage = Base.Checked.checked_add(
            cutoff_storage,
            Base.Checked.checked_mul(n_left, UInt128(sizeof(Int32))),
        )
    end
    compatible = all(count -> count <= typemax(Int32), retained_determinant_counts)
    return block_counts, total, cutoff_storage, compatible
end

"""
    multifragment_mebf_count_preflight(problem; tau_ci=1e-6)

Perform the exact count-only preflight for every arbitrary-fragment MEBF, including current fragment-product screening, per-MEBF normalization, collection-wide `tau_ci/Cmax` pruning, retained-expansion byte projection, and full determinant-pair upper bounds, without materializing molecular determinants or pair domains.
The loaded fragment wavefunctions are read-only, returned metadata include legal sector paths, local microstate cardinalities, naive Cartesian upper bounds, exact candidate/retained counts, lower-triangular full-pair bounds, Int32 determinant-index compatibility, and `UInt128` byte projections, no ERI/integral/CUDA work occurs, and this function intentionally stops before molecular orbital/product allocation so it can be used on real 7-mer inputs before a full NOCI run is authorized.
"""
function multifragment_mebf_count_preflight(
    problem::NOCIProblemDefinition{T};
    tau_ci::Real=T(1.0e-6),
) where {T<:Union{Float32,Float64}}
    threshold = T(tau_ci)
    isfinite(threshold) && threshold >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    prepared_cache = _prepare_fragment_microstate_cache64(problem)
    n_mebf = length(problem.mebfs)
    n_mebf >= 1 || throw(ArgumentError("count-only MEBF preflight requires at least one MEBF"))

    labels = Vector{String}(undef, n_mebf)
    fragment_counts = Vector{Int}(undef, n_mebf)
    active_orbital_ranks = Vector{Int}(undef, n_mebf)
    spin_sector_path_counts = Vector{Int}(undef, n_mebf)
    local_microstate_counts = Vector{Vector{Int}}(undef, n_mebf)
    naive_microstate_cartesian_counts = Vector{UInt128}(undef, n_mebf)
    candidate_determinant_counts = Vector{Int}(undef, n_mebf)
    norm_squared = Vector{T}(undef, n_mebf)
    normalized_max_coefficients = Vector{T}(undef, n_mebf)
    prepared_by_mebf = Vector{Vector{PreparedPackedFragment64{T}}}(undef, n_mebf)
    paths_by_mebf = Vector{Vector{MultifragmentSectorPath{T}}}(undef, n_mebf)

    for mebf_index in eachindex(problem.mebfs)
        composition = problem.mebfs[mebf_index]
        fragment_count = length(composition.fragment_state_indices)
        fragment_count >= 2 || throw(ArgumentError("count-only multifragment MEBF requires at least two fragment states"))
        labels[mebf_index] = composition.label
        fragment_counts[mebf_index] = fragment_count
        prepared = PreparedPackedFragment64{T}[]
        for state_index in composition.fragment_state_indices
            haskey(prepared_cache, state_index) || throw(ArgumentError("missing packed fragment cache entry $state_index"))
            push!(prepared, prepared_cache[state_index])
        end
        prepared_by_mebf[mebf_index] = prepared
        active_rank = sum(fragment.microstates.n_active_orbitals for fragment in prepared)
        active_rank <= 64 || throw(ArgumentError("count-only multifragment MEBF active rank exceeds 64"))
        active_orbital_ranks[mebf_index] = active_rank
        local_counts = [length(fragment.microstates.coefficients) for fragment in prepared]
        local_microstate_counts[mebf_index] = local_counts
        naive_count = UInt128(1)
        for local_count in local_counts
            naive_count = Base.Checked.checked_mul(naive_count, UInt128(local_count))
        end
        naive_microstate_cartesian_counts[mebf_index] = naive_count

        targets = _coupling_targets(fragment_count, problem.spin_multiplicity, composition.coupling_multiplicities)
        paths = _multifragment_sector_paths(prepared, targets)
        paths_by_mebf[mebf_index] = paths
        spin_sector_path_counts[mebf_index] = length(paths)
        count, sumsq, maximum_raw = _multifragment_product_magnitude_statistics(prepared, paths, threshold)
        candidate_determinant_counts[mebf_index] = count
        norm_squared[mebf_index] = sumsq
        normalized_max_coefficients[mebf_index] = maximum_raw / sqrt(sumsq)
    end

    collection_max_coefficient = maximum(normalized_max_coefficients)
    collection_max_coefficient > zero(T) || throw(ArgumentError("count-only MEBF collection has zero maximum coefficient"))
    collection_cutoff = threshold / collection_max_coefficient
    retained_determinant_counts = Vector{Int}(undef, n_mebf)
    candidate_product_storage_bytes = Vector{UInt128}(undef, n_mebf)
    retained_expansion_storage_bytes = Vector{UInt128}(undef, n_mebf)
    retained_total_bytes = UInt128(0)
    candidate_total_bytes = UInt128(0)
    orbital_total_bytes = UInt128(0)
    peak_product_scratch = UInt128(0)

    for mebf_index in eachindex(problem.mebfs)
        norm = sqrt(norm_squared[mebf_index])
        retained = _multifragment_retained_count(
            prepared_by_mebf[mebf_index],
            paths_by_mebf[mebf_index],
            threshold,
            norm,
            collection_cutoff,
        )
        retained_determinant_counts[mebf_index] = retained
        candidate_bytes = _multifragment_projected_packed_bytes(T, candidate_determinant_counts[mebf_index])
        retained_bytes = _multifragment_projected_packed_bytes(T, retained)
        candidate_product_storage_bytes[mebf_index] = candidate_bytes
        retained_expansion_storage_bytes[mebf_index] = retained_bytes
        candidate_total_bytes = Base.Checked.checked_add(candidate_total_bytes, candidate_bytes)
        retained_total_bytes = Base.Checked.checked_add(retained_total_bytes, retained_bytes)
        sort_bytes = Base.Checked.checked_mul(
            UInt128(candidate_determinant_counts[mebf_index]),
            UInt128(sizeof(Int) + sizeof(T)),
        )
        peak_product_scratch = max(peak_product_scratch, Base.Checked.checked_add(candidate_bytes, sort_bytes))

        composition = problem.mebfs[mebf_index]
        entries = [problem.fragment_states[state_index] for state_index in composition.fragment_state_indices]
        n_inactive = sum(entry.wavefunction.determinants.expansion.inactive_orbitals for entry in entries)
        n_active = active_orbital_ranks[mebf_index]
        orbital_bytes = Base.Checked.checked_mul(
            Base.Checked.checked_mul(UInt128(problem.n_basis), UInt128(n_inactive + n_active)),
            UInt128(sizeof(T)),
        )
        orbital_total_bytes = Base.Checked.checked_add(orbital_total_bytes, orbital_bytes)
    end

    projected_peak_materialization_bytes = Base.Checked.checked_add(
        orbital_total_bytes,
        Base.Checked.checked_add(
            candidate_total_bytes,
            Base.Checked.checked_add(retained_total_bytes, peak_product_scratch),
        ),
    )
    pair_blocks, full_pair_upper_bound, cutoff_storage, index_compatible =
        _multifragment_full_pair_upper_bounds(retained_determinant_counts)

    return MultifragmentMEBFCountPreflight(
        labels,
        fragment_counts,
        active_orbital_ranks,
        spin_sector_path_counts,
        local_microstate_counts,
        naive_microstate_cartesian_counts,
        candidate_determinant_counts,
        normalized_max_coefficients,
        collection_max_coefficient,
        collection_cutoff,
        retained_determinant_counts,
        candidate_product_storage_bytes,
        retained_expansion_storage_bytes,
        projected_peak_materialization_bytes,
        pair_blocks,
        full_pair_upper_bound,
        cutoff_storage,
        index_compatible,
    )
end
