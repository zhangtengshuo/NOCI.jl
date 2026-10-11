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

 File:            src/core/multifragment_mebf_production.jl
 Purpose:         Exact candidate counts and conservative memory preflight for the 64-active MEBF path.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    MultifragmentMEBFPreflight

Store exact candidate determinant counts and conservative production assembly memory for the 64-active arbitrary-fragment MEBF path.
Counts and active ranks are dimensionless CPU metadata, the memory estimate is in bytes, no molecular product or CUDA storage is retained, and every vector follows `problem.mebfs` order.
"""
struct MultifragmentMEBFPreflight
    estimate::MemoryEstimate
    candidate_determinant_counts::Vector{Int}
    active_orbital_ranks::Vector{Int}
    spin_sector_path_counts::Vector{Int}
end

"""
    _multifragment_mebf_preflight(problem, prepared_fragments, tau_ci)

Count every screened arbitrary-fragment molecular determinant candidate exactly and return a conservative collection-level host-memory estimate before any molecular product is materialized.
This serial implementation is retained as an independent characterization oracle for the reusable parallel assembly plan; production assembly no longer calls it in steady-state production.
"""
function _multifragment_mebf_preflight(
    problem::NOCIProblemDefinition{T},
    prepared_fragments::AbstractDict{Int,PreparedPackedFragment64{T}},
    tau_ci::T,
) where {T<:Union{Float32,Float64}}
    isfinite(tau_ci) && tau_ci >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    candidate_determinant_counts = Vector{Int}(undef, length(problem.mebfs))
    active_orbital_ranks = Vector{Int}(undef, length(problem.mebfs))
    spin_sector_path_counts = Vector{Int}(undef, length(problem.mebfs))
    orbital_bytes_total = 0
    determinant_bytes_total = 0
    peak_product_temporary = 0

    for index in eachindex(problem.mebfs)
        composition = problem.mebfs[index]
        length(composition.fragment_state_indices) >= 2 || throw(ArgumentError("production multifragment MEBFs require at least two fragment states"))
        entries = [problem.fragment_states[state_index] for state_index in composition.fragment_state_indices]
        n_inactive_orbitals = sum(entry.wavefunction.determinants.expansion.inactive_orbitals for entry in entries)
        n_active_orbitals = sum(active_orbital_count(entry.wavefunction.determinants.expansion) for entry in entries)
        n_active_orbitals <= 64 || throw(ArgumentError("production multifragment MEBF active rank exceeds 64"))
        active_orbital_ranks[index] = n_active_orbitals
        orbital_bytes_total = Base.Checked.checked_add(
            orbital_bytes_total,
            estimate_array_bytes(T, problem.n_basis, n_inactive_orbitals + n_active_orbitals),
        )

        packed = PreparedPackedFragment64{T}[]
        for state_index in composition.fragment_state_indices
            haskey(prepared_fragments, state_index) || throw(ArgumentError("missing packed fragment-state cache entry $state_index"))
            push!(packed, prepared_fragments[state_index])
        end
        targets = _coupling_targets(length(packed), problem.spin_multiplicity, composition.coupling_multiplicities)
        paths = _multifragment_sector_paths(packed, targets)
        spin_sector_path_counts[index] = length(paths)
        n_candidate_determinants = multifragment_product_count(packed, targets, tau_ci)
        n_candidate_determinants > 0 || throw(ArgumentError("MEBF $(composition.label) has no determinant candidate after exact product screening"))
        candidate_determinant_counts[index] = n_candidate_determinants

        projected_product_storage_bytes = multifragment_product_memory(T, n_candidate_determinants).host_bytes
        determinant_bytes_total = Base.Checked.checked_add(determinant_bytes_total, projected_product_storage_bytes)
        sort_and_normalize_bytes = Base.Checked.checked_add(
            estimate_array_bytes(Int, n_candidate_determinants),
            estimate_array_bytes(T, n_candidate_determinants),
        )
        peak_product_temporary = max(
            peak_product_temporary,
            Base.Checked.checked_add(projected_product_storage_bytes, sort_and_normalize_bytes),
        )
    end

    host_bytes = Base.Checked.checked_add(
        orbital_bytes_total,
        Base.Checked.checked_add(
            Base.Checked.checked_mul(2, determinant_bytes_total),
            peak_product_temporary,
        ),
    )
    return MultifragmentMEBFPreflight(
        MemoryEstimate(host_bytes, 0),
        candidate_determinant_counts,
        active_orbital_ranks,
        spin_sector_path_counts,
    )
end

"""
    _fill_multifragment_product_chunk!(coefficients, alpha_bits, beta_bits, prepared, path, tau_ci, orbital_offsets, chunk)

Fill one exact disjoint molecular-product output slice from a planned first-root range.
The output arrays are shared but each call writes only `chunk.output_range`; scientific inputs and orbital
offsets are read-only, traversal order inside the chunk is serial and identical to the corresponding
segment of the original DFS, and the actual survivor count must exactly match the plan.
"""
function _fill_multifragment_product_chunk!(
    coefficients::Vector{T},
    alpha_bits::Vector{UInt64},
    beta_bits::Vector{UInt64},
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    path::MultifragmentSectorPath{T},
    tau_ci::T,
    orbital_offsets::AbstractVector{<:Integer},
    chunk::MultifragmentFillChunk,
) where {T<:Union{Float32,Float64}}
    fragment_count = length(prepared)
    length(orbital_offsets) == fragment_count || throw(DimensionMismatch("orbital offset count differs from fragment count"))
    first(chunk.output_range) >= 1 && last(chunk.output_range) <= length(coefficients) || throw(BoundsError(coefficients, chunk.output_range))
    length(alpha_bits) == length(coefficients) == length(beta_bits) || throw(DimensionMismatch("multifragment product output arrays differ in length"))

    position = first(chunk.output_range) - 1
    _visit_multifragment_path_root_range!(prepared, path, tau_ci, chunk.root_positions) do selected, local_product
        next_position = Base.Checked.checked_add(position, 1)
        next_position <= last(chunk.output_range) || throw(ArgumentError(
            "multifragment parallel fill chunk exceeded its planned output range",
        ))
        position = next_position
        alpha = UInt64(0)
        beta = UInt64(0)
        @inbounds for fragment in 1:fragment_count
            microstates = prepared[fragment].microstates
            determinant = selected[fragment]
            shift = orbital_offsets[fragment]
            alpha |= microstates.alpha_bits[determinant] << shift
            beta |= microstates.beta_bits[determinant] << shift
        end
        @inbounds coefficients[next_position] = local_product * path.clebsch_product
        @inbounds alpha_bits[next_position] = alpha
        @inbounds beta_bits[next_position] = beta
    end
    actual_count = position - first(chunk.output_range) + 1
    actual_count == chunk.expected_count || throw(ArgumentError(
        "multifragment parallel fill chunk count changed between plan and fill",
    ))
    position == last(chunk.output_range) || throw(ArgumentError(
        "multifragment parallel fill chunk did not fill its exact output range",
    ))
    return actual_count
end

"""
    _multifragment_entry_fill_chunks(entry, maximum_chunks_per_path=Threads.nthreads())

Return the deterministic first-root fill chunks for one exact-count assembly-plan entry.
The plan is read-only, output ranges cover the candidate product exactly, and the number of chunks per
legal path is bounded by the requested worker capacity.
"""
function _multifragment_entry_fill_chunks(
    entry::MultifragmentMEBFAssemblyEntry,
    maximum_chunks_per_path::Integer=Threads.nthreads(),
)
    return _multifragment_fill_chunks(
        entry.path_candidate_determinant_counts,
        entry.first_fragment_root_candidate_determinant_counts,
        maximum_chunks_per_path,
    )
end

"""
    build_multifragment_product64_from_plan(prepared, entry, tau_ci; runtime=RuntimeOptions())

Materialize one screened multifragment molecular determinant product directly from an exact-count
assembly-plan entry without a local recount. The implementation partitions each legal path by pre-counted contiguous
first-fragment root ranges, assigns every chunk a fixed disjoint output slice, and executes chunks on
Julia threads when useful. Molecular product order is therefore identical to the serial DFS independent
of task completion order; local/collection memory guards and exact plan/fill assertions remain active.
"""
function build_multifragment_product64_from_plan(
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    entry::MultifragmentMEBFAssemblyEntry{T},
    tau_ci::T;
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:Union{Float32,Float64}}
    fragment_count = length(prepared)
    fragment_count >= 2 || throw(ArgumentError("multifragment product construction requires at least two fragments"))
    length(entry.fragment_state_indices) == fragment_count || throw(DimensionMismatch(
        "planned fragment-state count differs from prepared fragment count",
    ))
    length(entry.sector_paths) == length(entry.path_candidate_determinant_counts) || throw(DimensionMismatch(
        "planned sector-path and path-count lengths differ",
    ))
    length(entry.sector_paths) == length(entry.first_fragment_root_candidate_determinant_counts) || throw(DimensionMismatch(
        "planned sector-path and root-count collections differ in length",
    ))
    entry.n_candidate_determinants > 0 || throw(ArgumentError("planned multifragment candidate count must be positive"))

    planned_sum = 0
    for path_index in eachindex(entry.path_candidate_determinant_counts)
        path_count = entry.path_candidate_determinant_counts[path_index]
        path_count >= 0 || throw(ArgumentError("planned path candidate counts must be nonnegative"))
        root_counts = entry.first_fragment_root_candidate_determinant_counts[path_index]
        root_sum = 0
        for count in root_counts
            count >= 0 || throw(ArgumentError("planned root candidate counts must be nonnegative"))
            root_sum = Base.Checked.checked_add(root_sum, count)
        end
        root_sum == path_count || throw(ArgumentError("planned root counts do not sum to their path candidate count"))
        first_sector = entry.sector_paths[path_index].sectors[1]
        expected_roots = length(prepared[1].sorted_indices[first_sector])
        length(root_counts) == expected_roots || throw(DimensionMismatch(
            "planned first-root count length differs from prepared first-sector determinant count",
        ))
        planned_sum = Base.Checked.checked_add(planned_sum, path_count)
    end
    planned_sum == entry.n_candidate_determinants || throw(ArgumentError(
        "planned path candidate counts do not sum to the planned total",
    ))

    total_active = sum(fragment.microstates.n_active_orbitals for fragment in prepared)
    total_active == entry.n_active_orbitals || throw(DimensionMismatch("planned active rank differs from prepared active rank"))
    total_active <= 64 || throw(ArgumentError("multifragment product exceeds the 64-active-orbital domain"))
    enforce_memory_limits(MemoryEstimate(entry.projected_product_storage_bytes, 0), runtime)

    coefficients = Vector{T}(undef, entry.n_candidate_determinants)
    alpha_bits = Vector{UInt64}(undef, entry.n_candidate_determinants)
    beta_bits = Vector{UInt64}(undef, entry.n_candidate_determinants)
    orbital_offsets = zeros(Int, fragment_count)
    for fragment in 2:fragment_count
        orbital_offsets[fragment] = orbital_offsets[fragment - 1] + prepared[fragment - 1].microstates.n_active_orbitals
    end

    chunks = _multifragment_entry_fill_chunks(entry, max(1, Threads.nthreads()))
    isempty(chunks) && throw(ArgumentError("planned multifragment product has no nonempty fill chunks"))
    if Threads.nthreads() == 1 || length(chunks) == 1
        for chunk in chunks
            _fill_multifragment_product_chunk!(
                coefficients,
                alpha_bits,
                beta_bits,
                prepared,
                entry.sector_paths[chunk.path_index],
                tau_ci,
                orbital_offsets,
                chunk,
            )
        end
    else
        tasks = map(chunks) do chunk
            Threads.@spawn _fill_multifragment_product_chunk!(
                coefficients,
                alpha_bits,
                beta_bits,
                prepared,
                entry.sector_paths[chunk.path_index],
                tau_ci,
                orbital_offsets,
                chunk,
            )
        end
        filled = 0
        for task in tasks
            filled = Base.Checked.checked_add(filled, fetch(task))
        end
        filled == entry.n_candidate_determinants || throw(ArgumentError(
            "parallel multifragment fill did not produce the planned total candidate count",
        ))
    end

    return MultifragmentProduct64(coefficients, alpha_bits, beta_bits, total_active)
end

"""
    _build_multifragment_molecular_basis_function_from_plan(problem, composition, plan, entry, tau_ci; runtime=RuntimeOptions())

Build one production molecular basis function from an already exact-counted reusable assembly-plan entry.
The plan cache, coupling paths, candidate counts, and first-root fill metadata are reused read-only; no
coupling-target regeneration or local exact-count pass occurs, orbital/normalization/sort semantics remain
unchanged, and post-sort fermionic phase/reorder is parallelized over deterministic output ranges with
worker-private scratch.
"""
function _build_multifragment_molecular_basis_function_from_plan(
    problem::NOCIProblemDefinition{T},
    composition::MEBFComposition,
    plan::MultifragmentMEBFAssemblyPlan{T},
    entry::MultifragmentMEBFAssemblyEntry{T},
    tau_ci::T;
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:Union{Float32,Float64}}
    entry.composition_index in eachindex(problem.mebfs) || throw(BoundsError(problem.mebfs, entry.composition_index))
    problem.mebfs[entry.composition_index] === composition || begin
        entry.fragment_state_indices == composition.fragment_state_indices || throw(ArgumentError(
            "assembly-plan entry does not match the requested MEBF composition",
        ))
    end
    entry.fragment_state_indices == composition.fragment_state_indices || throw(ArgumentError(
        "assembly-plan fragment-state indices differ from the requested MEBF composition",
    ))

    fragment_entries = [problem.fragment_states[index] for index in entry.fragment_state_indices]
    prepared = PreparedPackedFragment64{T}[]
    for index in entry.fragment_state_indices
        haskey(plan.prepared_fragments, index) || throw(ArgumentError(
            "missing packed multifragment microstates for fragment-state index $index",
        ))
        push!(prepared, plan.prepared_fragments[index])
    end
    product = build_multifragment_product64_from_plan(prepared, entry, tau_ci; runtime)
    orbital_coefficients, inactive_orbitals = _molecular_orbitals(problem.fragment_states, entry.fragment_state_indices)
    inactive_orbitals == entry.n_inactive_orbitals || throw(DimensionMismatch(
        "planned inactive-orbital count differs from assembled molecular orbitals",
    ))
    expansion = _multifragment_product_to_expansion_parallel(
        product,
        inactive_orbitals;
        worker_capacity=max(1, Threads.nthreads()),
    )
    correlation_energy_shift = sum(fragment.wavefunction.determinants.correlation_energy for fragment in fragment_entries)
    return MolecularBasisFunction(
        composition.label,
        copy(entry.fragment_state_indices),
        orbital_coefficients,
        expansion,
        correlation_energy_shift,
    )
end

"""
    _build_multifragment_basis_collection_planned(problem, plan, screening, runtime)

Build every production MEBF in deterministic composition order from one reusable assembly plan.
Molecular-scale materialization remains strictly serial to bound host peak memory, while the current MEBF
may use deterministic disjoint-slice fill and post-sort phase/reorder tasks internally. No local recount
occurs and collection order and scientific semantics are unchanged.
"""
function _build_multifragment_basis_collection_planned(
    problem::NOCIProblemDefinition{T},
    plan::MultifragmentMEBFAssemblyPlan{T},
    screening::ScreeningOptions{T},
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    length(plan.entries) == length(problem.mebfs) || throw(DimensionMismatch(
        "assembly-plan entry count differs from problem MEBF count",
    ))
    basis_functions = Vector{MolecularBasisFunction{T}}(undef, length(problem.mebfs))
    for index in eachindex(problem.mebfs)
        entry = plan.entries[index]
        entry.composition_index == index || throw(ArgumentError("assembly-plan entry order differs from problem MEBF order"))
        basis_functions[index] = _build_multifragment_molecular_basis_function_from_plan(
            problem,
            problem.mebfs[index],
            plan,
            entry,
            screening.tau_ci;
            runtime,
        )
    end
    return basis_functions
end

"""
    _build_multifragment_basis_collection_prepared(problem, prepared_fragments, screening, runtime)

Build every production MEBF through the independent count/preflight/fill characterization path using one shared packed fragment cache.
This serial helper is retained only as a bounded characterization oracle; production `build_molecular_basis_functions`
uses the reusable plan and does not call this helper in steady-state production.
"""
function _build_multifragment_basis_collection_prepared(
    problem::NOCIProblemDefinition{T},
    prepared_fragments::AbstractDict{Int,PreparedPackedFragment64{T}},
    screening::ScreeningOptions{T},
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    basis_functions = Vector{MolecularBasisFunction{T}}(undef, length(problem.mebfs))
    for index in eachindex(problem.mebfs)
        basis_functions[index] = _build_multifragment_molecular_basis_function(
            problem,
            problem.mebfs[index],
            prepared_fragments,
            screening.tau_ci;
            runtime,
        )
    end
    return basis_functions
end

"""
    mebf_assembly_memory(problem; tau_ci=0)

Return the conservative exact-count production memory preflight for Float32/Float64 arbitrary-fragment MEBF assembly over active domains up to 64 orbitals.
The problem is read-only, the reusable production plan performs exact counts without molecular materialization, no CUDA storage is allocated, and invalid thresholds/topology or checked overflow raises before return.
"""
function mebf_assembly_memory(
    problem::NOCIProblemDefinition{T};
    tau_ci::Real=zero(T),
) where {T<:Union{Float32,Float64}}
    threshold = T(tau_ci)
    isfinite(threshold) && threshold >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    prepared = _prepare_fragment_microstate_cache64(problem)
    return _build_multifragment_mebf_assembly_plan(problem, prepared, threshold).estimate
end

"""
    build_molecular_basis_functions(problem; screening=ScreeningOptions(), runtime=RuntimeOptions(), telemetry=nothing)

Build production Float32/Float64 molecular MEBFs through a reusable exact-count assembly plan and preserve the established collection-wide `tau_ci/Cmax` pre-pruning without renormalization.
The shared fragment cache is prepared once, MEBF exact counts run in parallel across deterministic balanced index ranges, the collection memory guard executes before molecular materialization, each MEBF uses exact-sized disjoint-slice parallel fill without a local recount, and its post-sort phase/reorder loop is also parallelized without changing normalization or sorting. Only one MEBF is materialized at a time.
"""
function build_molecular_basis_functions(
    problem::NOCIProblemDefinition{T};
    screening::ScreeningOptions{T}=ScreeningOptions(T(1.0e-6), T(1.0e-5), T(1.0e-12)),
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:Union{Float32,Float64}}
    prepared = _prepare_fragment_microstate_cache64(problem)
    plan = _plan_multifragment_mebf_assembly(
        problem,
        prepared,
        screening.tau_ci;
        telemetry,
    )
    enforce_memory_limits(plan.estimate, runtime)

    fill_chunks_per_mebf = [length(_multifragment_entry_fill_chunks(entry, max(1, Threads.nthreads()))) for entry in plan.entries]
    maximum_fill_chunks = maximum(fill_chunks_per_mebf)
    work = function ()
        preliminary = _build_multifragment_basis_collection_planned(problem, plan, screening, runtime)
        return _reduce_multifragment_basis_collection(preliminary, screening.tau_ci)
    end
    if isnothing(telemetry)
        return work()
    end

    result = record_phase!(
        work,
        telemetry,
        :build_molecular_basis_functions;
        estimate=plan.estimate,
        metadata=Dict{Symbol,Any}(
            :n_mebfs => length(problem.mebfs),
            :tau_ci => screening.tau_ci,
            :julia_threads => Threads.nthreads(),
            :mebf_tasks => length(problem.mebfs),
            :count_workers => plan.count_workers,
            :parallel_count_enabled => plan.count_workers > 1,
            :planned_fill_enabled => true,
            :local_recount_enabled => false,
            :intra_mebf_parallel_fill_enabled => Threads.nthreads() > 1 && maximum_fill_chunks > 1,
            :fill_worker_capacity => Threads.nthreads(),
            :fill_chunks_per_mebf => fill_chunks_per_mebf,
            :maximum_fill_chunks_per_mebf => maximum_fill_chunks,
            :parallel_phase_reorder_enabled => Threads.nthreads() > 1,
            :phase_reorder_worker_capacity => Threads.nthreads(),
            :prepared_fragment_states => length(prepared),
            :occupation_representation => :packed_alpha_beta_uint64,
            :multifragment_lazy_product => true,
            :exact_count_preflight => true,
            :serial_materialization => true,
            :maximum_fragments_per_mebf => maximum(length(composition.fragment_state_indices) for composition in problem.mebfs),
            :candidate_determinant_counts => [entry.n_candidate_determinants for entry in plan.entries],
            :active_orbital_ranks => [entry.n_active_orbitals for entry in plan.entries],
            :spin_sector_path_counts => [length(entry.sector_paths) for entry in plan.entries],
        ),
        collect_allocations=runtime.collect_allocations,
    )
    path_count = sum(length(entry.sector_paths) for entry in plan.entries)
    fill_chunk_count = sum(fill_chunks_per_mebf)
    set_counter!(telemetry, :n_mebf_fill_traversals, length(plan.entries))
    set_counter!(telemetry, :n_mebf_fill_path_traversals, path_count)
    set_counter!(telemetry, :n_mebf_fill_chunk_traversals, fill_chunk_count)
    set_counter!(telemetry, :n_redundant_mebf_recounts, 0)
    return result
end
