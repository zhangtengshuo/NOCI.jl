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

 File:            src/core/multifragment_mebf_plan.jl
 Purpose:         Reusable assembly-plan entries for production multifragment MEBF construction.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    MultifragmentMEBFAssemblyEntry

Store one reusable assembly-plan entry for a production multifragment MEBF.
All fields are CPU metadata in `problem.mebfs` order: fragment-state indices, fixed coupling targets,
legal spin-sector paths, exact survivor counts per path and first-fragment root, total candidate count,
active/inactive orbital counts, and retained molecular-product bytes. No molecular determinant product, orbital matrix,
or CUDA storage is owned.
"""
struct MultifragmentMEBFAssemblyEntry{T<:AbstractFloat}
    composition_index::Int
    fragment_state_indices::Vector{Int}
    spin_coupling_targets::Vector{Int}
    sector_paths::Vector{MultifragmentSectorPath{T}}
    path_candidate_determinant_counts::Vector{Int}
    first_fragment_root_candidate_determinant_counts::Vector{Vector{Int}}
    n_candidate_determinants::Int
    n_inactive_orbitals::Int
    n_active_orbitals::Int
    projected_product_storage_bytes::Int
end

"""
    MultifragmentMEBFAssemblyPlan

Store reusable CPU metadata for all production multifragment MEBFs together with the shared packed
fragment cache and conservative collection-level host-memory estimate. Exact counts may be generated
in parallel across MEBF index ranges; entry order is always `problem.mebfs` order and the plan owns no
molecular-scale product or CUDA allocation.
"""
struct MultifragmentMEBFAssemblyPlan{
    T<:AbstractFloat,
    D<:AbstractDict{Int,PreparedPackedFragment64{T}},
}
    prepared_fragments::D
    entries::Vector{MultifragmentMEBFAssemblyEntry{T}}
    estimate::MemoryEstimate
    count_workers::Int
end

"""
    _count_multifragment_sector_path(prepared, path, tau_ci)

Return the exact number of determinant tuples surviving `tau_ci` on one already-enumerated legal
spin-sector path. This total-only helper is retained as a bounded characterization oracle; production
planning uses `_count_multifragment_sector_path_roots` so root counts needed by deterministic parallel
fill are obtained during the same exact traversal.
"""
function _count_multifragment_sector_path(
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    path::MultifragmentSectorPath{T},
    tau_ci::T,
) where {T<:AbstractFloat}
    count = Ref(0)
    _visit_multifragment_path_products!(prepared, path, tau_ci) do _, _
        count[] = Base.Checked.checked_add(count[], 1)
    end
    return count[]
end

"""
    _balanced_mebf_index_ranges(item_count, worker_count)

Partition `1:item_count` into deterministic contiguous balanced nonempty ranges for MEBF planning.
Arguments are dimensionless integers, the returned ranges differ in length by at most one, no task is
started here, and invalid counts raise before return.
"""
function _balanced_mebf_index_ranges(item_count::Integer, worker_count::Integer)
    n_items = Int(item_count)
    n_workers = Int(worker_count)
    n_items >= 1 || throw(ArgumentError("MEBF planning requires at least one item"))
    1 <= n_workers <= n_items || throw(ArgumentError("MEBF planning worker count must lie in 1:item_count"))
    quotient, remainder = divrem(n_items, n_workers)
    ranges = Vector{UnitRange{Int}}(undef, n_workers)
    first_index = 1
    for worker in 1:n_workers
        width = quotient + (worker <= remainder ? 1 : 0)
        last_index = first_index + width - 1
        ranges[worker] = first_index:last_index
        first_index = last_index + 1
    end
    first_index == n_items + 1 || error("internal MEBF planning partition error")
    return ranges
end

"""
    _build_multifragment_mebf_assembly_entry(problem, prepared_fragments, tau_ci, index)

Build one exact-count MEBF assembly-plan entry without materializing its molecular determinant product.
The problem and shared packed cache are read-only, coupling targets and legal paths are generated once,
each legal path is traversed exactly once to obtain both total and first-root survivor counts with
production screening semantics, and checked overflow or invalid topology raises before return.
Single-fragment entries have one highest-projection path with unit coupling and count their complete cached CI before collection pruning.
"""
function _build_multifragment_mebf_assembly_entry(
    problem::NOCIProblemDefinition{T},
    prepared_fragments::AbstractDict{Int,PreparedPackedFragment64{T}},
    tau_ci::T,
    index::Integer,
) where {T<:Union{Float32,Float64}}
    mebf_index = Int(index)
    composition = problem.mebfs[mebf_index]
    length(composition.fragment_state_indices) >= 1 || throw(ArgumentError(
        "production MEBFs require at least one fragment state",
    ))
    fragment_entries = [problem.fragment_states[state_index] for state_index in composition.fragment_state_indices]
    n_inactive_orbitals = sum(entry.wavefunction.determinants.expansion.inactive_orbitals for entry in fragment_entries)
    n_active_orbitals = sum(active_orbital_count(entry.wavefunction.determinants.expansion) for entry in fragment_entries)
    n_active_orbitals <= 64 || throw(ArgumentError("production multifragment MEBF active rank exceeds 64"))

    packed = PreparedPackedFragment64{T}[]
    for state_index in composition.fragment_state_indices
        haskey(prepared_fragments, state_index) || throw(ArgumentError(
            "missing packed fragment-state cache entry $state_index",
        ))
        push!(packed, prepared_fragments[state_index])
    end
    targets = _multifragment_coupling_targets(
        packed,
        problem.spin_multiplicity,
        composition.coupling_multiplicities,
    )
    paths = _multifragment_sector_paths(packed, targets)
    path_candidate_determinant_counts = Vector{Int}(undef, length(paths))
    first_fragment_root_candidate_determinant_counts = Vector{Vector{Int}}(undef, length(paths))
    n_candidate_determinants = 0
    for path_index in eachindex(paths)
        path_count, root_counts = _count_multifragment_sector_path_roots(
            packed,
            paths[path_index],
            tau_ci,
        )
        path_candidate_determinant_counts[path_index] = path_count
        first_fragment_root_candidate_determinant_counts[path_index] = root_counts
        n_candidate_determinants = Base.Checked.checked_add(n_candidate_determinants, path_count)
    end
    n_candidate_determinants > 0 || throw(ArgumentError(
        "MEBF $(composition.label) has no determinant candidate after exact product screening",
    ))

    projected_product_storage_bytes = multifragment_product_memory(T, n_candidate_determinants).host_bytes
    return MultifragmentMEBFAssemblyEntry(
        mebf_index,
        copy(composition.fragment_state_indices),
        copy(targets),
        paths,
        path_candidate_determinant_counts,
        first_fragment_root_candidate_determinant_counts,
        n_candidate_determinants,
        n_inactive_orbitals,
        n_active_orbitals,
        projected_product_storage_bytes,
    )
end

"""
    _multifragment_mebf_plan_memory(problem, entries)

Return the conservative collection-level host-memory estimate from already exact-counted plan entries.
The inputs are read-only, arithmetic is checked, device bytes are zero, and the formula intentionally
matches the established production preflight guard.
"""
function _multifragment_mebf_plan_memory(
    problem::NOCIProblemDefinition{T},
    entries::AbstractVector{MultifragmentMEBFAssemblyEntry{T}},
) where {T<:Union{Float32,Float64}}
    orbital_bytes_total = 0
    determinant_bytes_total = 0
    peak_product_temporary = 0
    for entry in entries
        orbital_bytes_total = Base.Checked.checked_add(
            orbital_bytes_total,
            estimate_array_bytes(T, problem.n_basis, entry.n_inactive_orbitals + entry.n_active_orbitals),
        )
        determinant_bytes_total = Base.Checked.checked_add(determinant_bytes_total, entry.projected_product_storage_bytes)
        sort_and_normalize_bytes = Base.Checked.checked_add(
            estimate_array_bytes(Int, entry.n_candidate_determinants),
            estimate_array_bytes(T, entry.n_candidate_determinants),
        )
        peak_product_temporary = max(
            peak_product_temporary,
            Base.Checked.checked_add(entry.projected_product_storage_bytes, sort_and_normalize_bytes),
        )
    end
    host_bytes = Base.Checked.checked_add(
        orbital_bytes_total,
        Base.Checked.checked_add(
            Base.Checked.checked_mul(2, determinant_bytes_total),
            peak_product_temporary,
        ),
    )
    return MemoryEstimate(host_bytes, 0)
end

"""
    _build_multifragment_mebf_assembly_plan(problem, prepared_fragments, tau_ci)

Build the reusable production MEBF assembly plan with deterministic MEBF-level parallel exact counts.
At most `min(Threads.nthreads(), length(problem.mebfs))` worker tasks process balanced contiguous index
ranges, scientific inputs are read-only, workers return private entry vectors that are merged by index,
all memory totals are reduced serially in deterministic order, and no molecular product or CUDA storage
is allocated.
"""
function _build_multifragment_mebf_assembly_plan(
    problem::NOCIProblemDefinition{T},
    prepared_fragments::D,
    tau_ci::T,
) where {
    T<:Union{Float32,Float64},
    D<:AbstractDict{Int,PreparedPackedFragment64{T}},
}
    isfinite(tau_ci) && tau_ci >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    mebf_count = length(problem.mebfs)
    mebf_count >= 1 || throw(ArgumentError("MEBF assembly planning requires at least one MEBF"))
    worker_count = min(Threads.nthreads(), mebf_count)
    ranges = _balanced_mebf_index_ranges(mebf_count, worker_count)
    entries = Vector{MultifragmentMEBFAssemblyEntry{T}}(undef, mebf_count)

    if worker_count == 1
        for index in only(ranges)
            entries[index] = _build_multifragment_mebf_assembly_entry(
                problem,
                prepared_fragments,
                tau_ci,
                index,
            )
        end
    else
        tasks = map(ranges) do index_range
            Threads.@spawn begin
                local_entries = Vector{MultifragmentMEBFAssemblyEntry{T}}(undef, length(index_range))
                for (local_index, mebf_index) in enumerate(index_range)
                    local_entries[local_index] = _build_multifragment_mebf_assembly_entry(
                        problem,
                        prepared_fragments,
                        tau_ci,
                        mebf_index,
                    )
                end
                return index_range, local_entries
            end
        end
        for task in tasks
            index_range, local_entries = fetch(task)
            for (local_index, mebf_index) in enumerate(index_range)
                entries[mebf_index] = local_entries[local_index]
            end
        end
    end

    return MultifragmentMEBFAssemblyPlan(
        prepared_fragments,
        entries,
        _multifragment_mebf_plan_memory(problem, entries),
        worker_count,
    )
end

"""
    _plan_multifragment_mebf_assembly(problem, prepared_fragments, tau_ci; telemetry=nothing)

Build the reusable MEBF assembly plan and optionally append planning telemetry.
Scientific inputs remain read-only, telemetry is diagnostic only, exact counts are parallelized across
MEBFs when multiple Julia threads are available, first-root survivor counts are retained for deterministic
parallel fill without an additional count traversal, and the returned plan owns no molecular product.
"""
function _plan_multifragment_mebf_assembly(
    problem::NOCIProblemDefinition{T},
    prepared_fragments::D,
    tau_ci::T;
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {
    T<:Union{Float32,Float64},
    D<:AbstractDict{Int,PreparedPackedFragment64{T}},
}
    start_ns = time_ns()
    plan = _build_multifragment_mebf_assembly_plan(problem, prepared_fragments, tau_ci)
    isnothing(telemetry) && return plan

    wall_seconds = Float64(time_ns() - start_ns) / 1.0e9
    path_count = sum(length(entry.sector_paths) for entry in plan.entries)
    root_branch_counts = [
        sum(count(>(0), root_counts) for root_counts in entry.first_fragment_root_candidate_determinant_counts)
        for entry in plan.entries
    ]
    push!(
        telemetry.phases,
        PhaseTelemetry(
            :plan_multifragment_mebf_assembly,
            wall_seconds,
            plan.estimate.host_bytes,
            plan.estimate.device_bytes,
            nothing,
            Dict{Symbol,Any}(
                :n_mebfs => length(plan.entries),
                :count_workers => plan.count_workers,
                :parallel_count_enabled => plan.count_workers > 1,
                :planned_fill_enabled => true,
                :root_partition_metadata_enabled => true,
                :candidate_determinant_counts => [entry.n_candidate_determinants for entry in plan.entries],
                :active_orbital_ranks => [entry.n_active_orbitals for entry in plan.entries],
                :spin_sector_path_counts => [length(entry.sector_paths) for entry in plan.entries],
                :path_candidate_determinant_counts => [copy(entry.path_candidate_determinant_counts) for entry in plan.entries],
                :nonzero_first_root_branch_counts => root_branch_counts,
            ),
        ),
    )
    set_counter!(telemetry, :n_mebf_plan_count_traversals, length(plan.entries))
    set_counter!(telemetry, :n_mebf_plan_path_count_traversals, path_count)
    return plan
end
