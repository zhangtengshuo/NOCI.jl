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

 File:            src/core/multifragment_mebf_parallel_fill.jl
 Purpose:         Deterministic intra-MEBF fill chunks executed on Julia threads during parallel assembly.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    MultifragmentFillChunk

Describe one deterministic intra-MEBF fill chunk. `root_positions` is a contiguous range in the
first-fragment sector-local stable magnitude order, `output_range` is the exact disjoint molecular
product slice occupied by all survivors from those roots, and `expected_count == length(output_range)`.
The value is CPU metadata only and owns no molecular product or CUDA storage.
"""
struct MultifragmentFillChunk
    path_index::Int
    root_positions::UnitRange{Int}
    output_range::UnitRange{Int}
    expected_count::Int
end

"""
    _count_multifragment_sector_path_roots(prepared, path, tau_ci)

Count one legal path exactly once and return both the total survivor count and survivor counts grouped
by first-fragment root determinant in the established stable sector-local order. The existing production
branch-and-bound visitor supplies every survivor, scientific screening/order is unchanged, all arithmetic
is checked, and no molecular product or CUDA storage is materialized.
"""
function _count_multifragment_sector_path_roots(
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    path::MultifragmentSectorPath{T},
    tau_ci::T,
) where {T<:AbstractFloat}
    length(path.sectors) == length(prepared) || throw(DimensionMismatch(
        "sector path length differs from fragment count",
    ))
    first_sector = path.sectors[1]
    roots = prepared[1].sorted_indices[first_sector]
    root_position_by_determinant = zeros(Int, length(prepared[1].microstates.coefficients))
    @inbounds for (position, determinant) in enumerate(roots)
        root_position_by_determinant[determinant] = position
    end
    root_counts = zeros(Int, length(roots))
    total = Ref(0)
    _visit_multifragment_path_products!(prepared, path, tau_ci) do selected, _
        root_position = @inbounds root_position_by_determinant[selected[1]]
        root_position > 0 || error("internal multifragment root-position mapping failure")
        @inbounds root_counts[root_position] = Base.Checked.checked_add(root_counts[root_position], 1)
        total[] = Base.Checked.checked_add(total[], 1)
    end
    sum_checked = 0
    for count in root_counts
        sum_checked = Base.Checked.checked_add(sum_checked, count)
    end
    sum_checked == total[] || error("internal multifragment root-count reduction mismatch")
    return total[], root_counts
end

"""
    _visit_multifragment_path_root_range!(visitor, prepared, path, tau_ci, root_positions)

Traverse only a contiguous range of first-fragment roots from one legal spin-sector path while preserving
the exact serial DFS order within that range. The branch-and-bound equations and pre-Clebsch `tau_ci`
semantics are identical to `_visit_multifragment_path_products!`; scientific inputs are read-only, each
call owns its selected-index scratch, and invalid root ranges fail before traversal.
"""
function _visit_multifragment_path_root_range!(
    visitor,
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    path::MultifragmentSectorPath{T},
    tau_ci::T,
    root_positions::UnitRange{Int},
) where {T<:AbstractFloat}
    isfinite(tau_ci) && tau_ci >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    fragment_count = length(prepared)
    fragment_count >= 2 || throw(ArgumentError("multifragment root-range traversal requires at least two fragments"))
    length(path.sectors) == fragment_count || throw(DimensionMismatch("sector path length differs from fragment count"))

    selected = zeros(Int, fragment_count)
    determinant_lists = Vector{Vector{Int}}(undef, fragment_count)
    maximum_magnitudes = Vector{T}(undef, fragment_count)
    for fragment in 1:fragment_count
        sector = path.sectors[fragment]
        determinant_lists[fragment] = prepared[fragment].sorted_indices[sector]
        maximum_magnitudes[fragment] = prepared[fragment].maximum_magnitudes[sector]
    end
    roots = determinant_lists[1]
    isempty(root_positions) && return nothing
    first(root_positions) >= 1 && last(root_positions) <= length(roots) || throw(BoundsError(roots, root_positions))

    suffix_bound = ones(T, fragment_count)
    running = one(T)
    for fragment in fragment_count:-1:1
        suffix_bound[fragment] = running
        running *= maximum_magnitudes[fragment]
    end
    tau_ci > zero(T) && running < tau_ci && return nothing

    function descend(fragment::Int, prefix_magnitude::T, prefix_coefficient::T)
        indices = determinant_lists[fragment]
        microstates = prepared[fragment].microstates
        @inbounds for determinant in indices
            coefficient = microstates.coefficients[determinant]
            magnitude = prefix_magnitude * abs(coefficient)
            magnitude * suffix_bound[fragment] >= tau_ci || break
            selected[fragment] = determinant
            product = prefix_coefficient * coefficient
            if fragment == fragment_count
                magnitude >= tau_ci && visitor(selected, product)
            else
                descend(fragment + 1, magnitude, product)
            end
        end
        return nothing
    end

    first_microstates = prepared[1].microstates
    @inbounds for root_position in root_positions
        determinant = roots[root_position]
        coefficient = first_microstates.coefficients[determinant]
        magnitude = abs(coefficient)
        magnitude * suffix_bound[1] >= tau_ci || break
        selected[1] = determinant
        descend(2, magnitude, coefficient)
    end
    return nothing
end

"""
    _partition_multifragment_root_counts(root_counts, maximum_chunks)

Partition ordered first-root survivor counts into at most `maximum_chunks` deterministic contiguous ranges
using cumulative survivor-weight targets. Zero-survivor roots remain in their serial positions, every
returned range has positive total weight, concatenating ranges preserves root order, and no scientific
array is allocated.
"""
function _partition_multifragment_root_counts(
    root_counts::AbstractVector{<:Integer},
    maximum_chunks::Integer,
)
    chunk_limit = Int(maximum_chunks)
    chunk_limit >= 1 || throw(ArgumentError("maximum_chunks must be positive"))
    counts = Int.(root_counts)
    all(>=(0), counts) || throw(ArgumentError("root survivor counts must be nonnegative"))
    isempty(counts) && return UnitRange{Int}[]
    total = 0
    nonzero_roots = 0
    for count in counts
        total = Base.Checked.checked_add(total, count)
        nonzero_roots += count > 0
    end
    total > 0 || return UnitRange{Int}[]
    desired_chunks = min(chunk_limit, nonzero_roots)
    desired_chunks == 1 && return UnitRange{Int}[1:length(counts)]

    ranges = UnitRange{Int}[]
    start_position = 1
    cumulative = 0
    for root_position in eachindex(counts)
        cumulative = Base.Checked.checked_add(cumulative, counts[root_position])
        boundary_number = length(ranges) + 1
        boundary_number >= desired_chunks && continue
        target_numerator = Base.Checked.checked_mul(total, boundary_number)
        target = cld(target_numerator, desired_chunks)
        cumulative >= target || continue
        has_later_survivor = false
        @inbounds for later in (root_position + 1):length(counts)
            if counts[later] > 0
                has_later_survivor = true
                break
            end
        end
        has_later_survivor || continue
        push!(ranges, start_position:root_position)
        start_position = root_position + 1
    end
    push!(ranges, start_position:length(counts))
    filter!(range -> sum(@view counts[range]) > 0, ranges)
    length(ranges) <= desired_chunks || error("internal multifragment root partition exceeded chunk target")
    return ranges
end

"""
    _multifragment_fill_chunks(path_candidate_counts, path_root_candidate_counts, maximum_chunks_per_path)

Build exact disjoint output slices for deterministic parallel fill. Paths retain their established order;
within each path, contiguous first-root ranges are survivor-weight partitioned into bounded chunks and
mapped to output offsets by checked prefix sums. Returned chunks collectively cover `1:sum(path counts)`
exactly with no overlap or gap.
"""
function _multifragment_fill_chunks(
    path_candidate_counts::AbstractVector{<:Integer},
    path_root_candidate_counts::AbstractVector{<:AbstractVector{<:Integer}},
    maximum_chunks_per_path::Integer,
)
    length(path_candidate_counts) == length(path_root_candidate_counts) || throw(DimensionMismatch(
        "path candidate counts and root-count collections differ in length",
    ))
    chunk_limit = Int(maximum_chunks_per_path)
    chunk_limit >= 1 || throw(ArgumentError("maximum_chunks_per_path must be positive"))
    chunks = MultifragmentFillChunk[]
    path_output_start = 1

    for path_index in eachindex(path_candidate_counts)
        path_count = Int(path_candidate_counts[path_index])
        path_count >= 0 || throw(ArgumentError("path candidate counts must be nonnegative"))
        root_counts = Int.(path_root_candidate_counts[path_index])
        root_total = 0
        for count in root_counts
            count >= 0 || throw(ArgumentError("root survivor counts must be nonnegative"))
            root_total = Base.Checked.checked_add(root_total, count)
        end
        root_total == path_count || throw(ArgumentError("root survivor counts do not sum to path candidate count"))
        if path_count > 0
            ranges = _partition_multifragment_root_counts(root_counts, chunk_limit)
            root_prefix = 0
            range_index = 1
            next_range = isempty(ranges) ? nothing : ranges[1]
            for root_position in eachindex(root_counts)
                if !isnothing(next_range) && root_position == first(next_range)
                    expected = 0
                    @inbounds for position in next_range
                        expected = Base.Checked.checked_add(expected, root_counts[position])
                    end
                    output_start = Base.Checked.checked_add(path_output_start, root_prefix)
                    output_stop = Base.Checked.checked_add(output_start, expected - 1)
                    push!(chunks, MultifragmentFillChunk(path_index, next_range, output_start:output_stop, expected))
                    range_index += 1
                    next_range = range_index <= length(ranges) ? ranges[range_index] : nothing
                end
                root_prefix = Base.Checked.checked_add(root_prefix, root_counts[root_position])
            end
            root_prefix == path_count || error("internal multifragment root prefix mismatch")
        end
        path_output_start = Base.Checked.checked_add(path_output_start, path_count)
    end

    expected_output = 1
    total_count = 0
    for chunk in chunks
        chunk.expected_count == length(chunk.output_range) || error("internal multifragment fill chunk length mismatch")
        first(chunk.output_range) == expected_output || error("multifragment fill chunks contain an output gap or overlap")
        expected_output = last(chunk.output_range) + 1
        total_count = Base.Checked.checked_add(total_count, chunk.expected_count)
    end
    planned_total = 0
    for count in path_candidate_counts
        planned_total = Base.Checked.checked_add(planned_total, Int(count))
    end
    total_count == planned_total || error("multifragment fill chunks do not cover the planned output")
    return chunks
end

"""
    _balanced_multifragment_output_ranges(item_count, worker_count)

Partition an ordered output domain into deterministic contiguous balanced ranges for post-sort MEBF
processing. Ranges cover `1:item_count` exactly, differ in length by at most one, and no work is started.
"""
function _balanced_multifragment_output_ranges(item_count::Integer, worker_count::Integer)
    n_items = Int(item_count)
    n_workers = Int(worker_count)
    n_items >= 1 || throw(ArgumentError("multifragment output partition requires at least one item"))
    1 <= n_workers <= n_items || throw(ArgumentError("multifragment output worker count must lie in 1:item_count"))
    quotient, remainder = divrem(n_items, n_workers)
    ranges = Vector{UnitRange{Int}}(undef, n_workers)
    first_index = 1
    for worker in 1:n_workers
        width = quotient + (worker <= remainder ? 1 : 0)
        last_index = first_index + width - 1
        ranges[worker] = first_index:last_index
        first_index = last_index + 1
    end
    first_index == n_items + 1 || error("internal multifragment output partition error")
    return ranges
end

"""
    _phase_reorder_multifragment_range!(ordered_coefficients, ordered_alpha, ordered_beta,
        molecular_coefficients, order, product, output_range)

Apply the established fermionic phase convention and packed alpha/beta reorder for one fixed contiguous
post-sort output range. Inputs and the global `order` are read-only, outputs are disjoint by caller
construction, and occupation/sign scratch arrays are private to this call so concurrent tasks share no
mutable scientific state.
"""
function _phase_reorder_multifragment_range!(
    ordered_coefficients::Vector{T},
    ordered_alpha::Vector{UInt64},
    ordered_beta::Vector{UInt64},
    molecular_coefficients::AbstractVector{T},
    order::AbstractVector{<:Integer},
    product::MultifragmentProduct64{T},
    output_range::UnitRange{Int},
) where {T<:AbstractFloat}
    n_active = product.n_active_orbitals
    occupation = Vector{Int8}(undef, n_active)
    spin_scratch = Vector{Int8}(undef, 2 * n_active)
    occupied_scratch = Vector{Int8}(undef, n_active)

    @inbounds for determinant in output_range
        source_index = order[determinant]
        alpha_bits = product.alpha_bits[source_index]
        beta_bits = product.beta_bits[source_index]
        for orbital in 1:n_active
            bit = UInt64(1) << (orbital - 1)
            alpha = !iszero(alpha_bits & bit)
            beta = !iszero(beta_bits & bit)
            occupation[orbital] = alpha ? (beta ? Int8(2) : Int8(1)) : (beta ? Int8(-1) : Int8(0))
        end
        alpha_beta_sign = _alpha_beta_permutation_sign_with_scratch!(spin_scratch, occupation)
        closed_shell_sign = _closed_shell_permutation_sign_with_scratch!(occupied_scratch, occupation)
        ordered_coefficients[determinant] = molecular_coefficients[source_index] * T(alpha_beta_sign * closed_shell_sign)
        ordered_alpha[determinant] = alpha_bits
        ordered_beta[determinant] = beta_bits
    end
    return nothing
end

"""
    _multifragment_product_to_expansion_parallel(product, inactive_orbitals; worker_capacity=Threads.nthreads())

Convert one exact multifragment packed product to the standard `DeterminantExpansion` while preserving
the screened molecular amplitudes without renormalization, stable magnitude `sortperm`, fermionic phase convention, and
output order. Validation and sorting remain serial; only the independent post-sort phase/reorder loop
is partitioned across deterministic output ranges with worker-private scratch. No floating-point reduction
is parallelized and no CUDA work occurs.
"""
function _multifragment_product_to_expansion_parallel(
    product::MultifragmentProduct64{T},
    inactive_orbitals::Integer;
    worker_capacity::Integer=Threads.nthreads(),
) where {T<:AbstractFloat}
    inactive_orbitals >= 0 || throw(ArgumentError("inactive_orbitals must be nonnegative"))
    capacity = Int(worker_capacity)
    capacity >= 1 || throw(ArgumentError("worker_capacity must be positive"))

    molecular_coefficients = product.coefficients
    isempty(molecular_coefficients) && throw(ArgumentError("multifragment product must contain at least one determinant"))
    all(isfinite, molecular_coefficients) || throw(ArgumentError("multifragment molecular coefficients must be finite"))
    any(coefficient -> !iszero(coefficient), molecular_coefficients) || throw(ArgumentError("multifragment molecular coefficient vector is identically zero"))
    order = sortperm(abs.(molecular_coefficients); rev=true)
    n_determinants = length(order)
    n_determinants >= 1 || throw(ArgumentError("multifragment product must contain at least one determinant"))
    n_active = product.n_active_orbitals
    ordered_coefficients = Vector{T}(undef, n_determinants)
    ordered_alpha = Vector{UInt64}(undef, n_determinants)
    ordered_beta = Vector{UInt64}(undef, n_determinants)
    worker_count = min(capacity, n_determinants)
    ranges = _balanced_multifragment_output_ranges(n_determinants, worker_count)

    if worker_count == 1
        _phase_reorder_multifragment_range!(
            ordered_coefficients,
            ordered_alpha,
            ordered_beta,
            molecular_coefficients,
            order,
            product,
            only(ranges),
        )
    else
        tasks = map(ranges) do output_range
            Threads.@spawn _phase_reorder_multifragment_range!(
                ordered_coefficients,
                ordered_alpha,
                ordered_beta,
                molecular_coefficients,
                order,
                product,
                output_range,
            )
        end
        foreach(fetch, tasks)
    end

    packed = PackedActiveOccupations(n_active, ordered_alpha, ordered_beta; copy_data=false)
    return DeterminantExpansion(ordered_coefficients, packed, inactive_orbitals, Val(:owned))
end
