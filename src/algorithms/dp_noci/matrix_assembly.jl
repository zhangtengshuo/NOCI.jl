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

 File:            src/algorithms/dp_noci/matrix_assembly.jl
 Purpose:         DP raw matrix assembly: CI weights, triangular multiplicity, Vnn, normalization, and solve.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
 Algorithm origin: The determinant-pair method originates from the GronOR program (T. P. Straatsma, R. Broer,
                   A. Sanchez-Mansilla, C. Sousa, and C. de Graaf, J. Chem. Theory Comput. 18, 3549-3565 (2022),
                   doi:10.1021/acs.jctc.2c00266; gitlab.com/gronor/gronor, Apache-2.0). NOCI.jl rewrites the method in
                   Julia with extensive single-node performance optimizations, keeping most functionality and settings
                   consistent with GronOR.
=#

"""
    dp_noci_matrix_memory(T, basis_functions)

Estimate conservative peak host storage for DP raw matrices, determinant-pair domain, one full-occupied preparation, and one direct two-electron contraction.
Inputs are read-only, returned units are bytes with zero device storage, no scientific array is allocated, and incompatible MEBF dimensions or checked overflow raise an exception.
"""
function dp_noci_matrix_memory(
    ::Type{T},
    basis_functions::AbstractVector{<:MolecularBasisFunction},
) where {T<:AbstractFloat}
    isempty(basis_functions) && throw(ArgumentError("at least one MEBF is required"))
    n_basis = size(first(basis_functions).orbital_coefficients, 1)
    largest_local = MemoryEstimate()
    for basis_function in basis_functions
        size(basis_function.orbital_coefficients, 1) == n_basis || throw(DimensionMismatch("MEBF AO dimensions differ"))
        n_active_alpha, n_active_beta = _mebf_active_spin_counts(basis_function)
        n_alpha = basis_function.expansion.inactive_orbitals + n_active_alpha
        n_beta = basis_function.expansion.inactive_orbitals + n_active_beta
        preparation = dp_pair_preparation_memory(T, n_basis, n_alpha, n_beta)
        two_electron = _dp_two_electron_dimension_memory(T, n_basis, n_alpha + n_beta)
        local_estimate = combine_memory_estimates(preparation, two_electron)
        local_estimate.host_bytes > largest_local.host_bytes && (largest_local = local_estimate)
    end
    return combine_memory_estimates(
        raw_noci_matrices_memory(T, length(basis_functions)),
        determinant_pair_domain_memory(basis_functions),
        largest_local,
    )
end

"""
    assemble_dp_noci_matrices(basis_functions, one_electron, integrals, screening; runtime=RuntimeOptions(), telemetry=nothing, checkpoint=nothing, progress=nothing)

Assemble symmetric raw Conventional DP-NOCI matrices from screened determinant pairs, physical `.one/.two` values, CI weights, diagonal triangular multiplicity, and nuclear repulsion on the requested CPU or CUDA two-electron backend.
Inputs are read-only, overlap is dimensionless and Hamiltonian components use integral energy units, host/device memory is preflighted, GPU ERIs/workspaces are reused and explicitly released, and optional checkpoint writes preserve the exact pair traversal and accumulator order while only batching cumulative-prefix I/O.
"""
function assemble_dp_noci_matrices(
    basis_functions::AbstractVector{<:MolecularBasisFunction{T}},
    one_electron::OneElectronValues{T},
    integrals::PackedERIValues{T},
    screening::ScreeningOptions{T};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
    checkpoint::Union{Nothing,CheckpointSession}=nothing,
    progress::Union{Nothing,ProgressLogger}=nothing,
) where {T<:AbstractFloat}
    n_basis_functions = length(basis_functions)
    n_basis_functions >= 1 || throw(ArgumentError("at least one MEBF is required"))
    n_basis = size(one_electron.overlap, 1)
    integrals.n_orbitals == n_basis || throw(DimensionMismatch("one-electron and packed ERI dimensions differ"))
    estimate = dp_noci_matrix_memory(T, basis_functions)
    runtime.use_gpu && (estimate = combine_memory_estimates(estimate, dp_two_electron_gpu_workspace_memory(T, integrals)))
    enforce_memory_limits(estimate, runtime)
    domain = build_determinant_pair_domain(basis_functions, screening; runtime, telemetry)
    overlap_matrix = zeros(T, n_basis_functions, n_basis_functions)
    electronic_one_body = zeros(T, n_basis_functions, n_basis_functions)
    two_body = zeros(T, n_basis_functions, n_basis_functions)
    pair_counts = zeros(Int, n_basis_functions, n_basis_functions)
    one_body_operator = core_hamiltonian(one_electron)
    two_electron_workspace = runtime.use_gpu ? DPDirectTwoElectronGPUWorkspace(integrals, one_body_operator, runtime) : DPDirectTwoElectronWorkspace{T}()
    try
        for group in domain.groups
            left_basis = basis_functions[group.left_mebf]
            right_basis = basis_functions[group.right_mebf]
            completed = isnothing(checkpoint) ? nothing : load_state_pair_checkpoint(checkpoint, :dp, group.left_mebf, group.right_mebf)
            if !isnothing(completed)
                overlap_sum = T(completed["overlap"])
                one_body_sum = T(completed["one_body"])
                two_body_sum = T(completed["two_body"])
                pair_count = Int(completed["pair_count"])
                pair_count == length(group.pairs) || throw(ArgumentError("completed DP checkpoint pair count differs from current state-pair domain"))
                _assign_symmetric!(overlap_matrix, group.left_mebf, group.right_mebf, overlap_sum)
                _assign_symmetric!(electronic_one_body, group.left_mebf, group.right_mebf, one_body_sum)
                _assign_symmetric!(two_body, group.left_mebf, group.right_mebf, two_body_sum)
                _assign_symmetric!(pair_counts, group.left_mebf, group.right_mebf, pair_count)
                progress_event!(progress, :dp_state_pair_reused; left_mebf=group.left_mebf, right_mebf=group.right_mebf, pair_count=pair_count)
                continue
            end
            progress_event!(progress, :dp_state_pair_started; left_mebf=group.left_mebf, right_mebf=group.right_mebf, pair_count=length(group.pairs))
            overlap_sum = zero(T)
            one_body_sum = zero(T)
            two_body_sum = zero(T)
            next_pair_index = 1
            if !isnothing(checkpoint)
                inflight = load_inflight_checkpoint(checkpoint, :dp, group.left_mebf, group.right_mebf)
                if !isnothing(inflight)
                    Int(inflight["total_count"]) == length(group.pairs) || throw(ArgumentError("DP cumulative-prefix total differs from current state-pair domain"))
                    next_pair_index = Int(inflight["next_index"])
                    overlap_sum = T(inflight["overlap"])
                    one_body_sum = T(inflight["one_body"])
                    two_body_sum = T(inflight["two_body"])
                    progress_event!(progress, :dp_prefix_reused; left_mebf=group.left_mebf, right_mebf=group.right_mebf, next_pair_index=next_pair_index, pair_count=length(group.pairs))
                end
            end
            processed_pair_index = next_pair_index - 1
            for reference in Iterators.drop(group.pairs, next_pair_index - 1)
                processed_pair_index += 1
                pair = prepare_determinant_pair(left_basis, right_basis, reference, one_electron.overlap)
                preparation = prepare_dp_determinant_pair(pair; tau_singularity=screening.tau_singularity, runtime, telemetry)
                element = contract_dp_determinant_pair!(two_electron_workspace, preparation, one_body_operator, integrals; runtime, telemetry)
                multiplicity = group.left_mebf == group.right_mebf && reference.left != reference.right ? T(2) : one(T)
                ci_weight = multiplicity * left_basis.expansion.coefficients[reference.left] * right_basis.expansion.coefficients[reference.right]
                overlap_sum += ci_weight * element.overlap
                one_body_sum += ci_weight * element.electronic_one_body
                two_body_sum += ci_weight * element.two_body
                if !isnothing(checkpoint) && processed_pair_index % checkpoint.dp_checkpoint_pairs == 0 && processed_pair_index < length(group.pairs)
                    save_inflight_checkpoint!(
                        checkpoint,
                        :dp,
                        group.left_mebf,
                        group.right_mebf,
                        processed_pair_index + 1,
                        length(group.pairs),
                        overlap_sum,
                        one_body_sum,
                        two_body_sum,
                    )
                    progress_event!(progress, :dp_prefix_committed; left_mebf=group.left_mebf, right_mebf=group.right_mebf, processed_pairs=processed_pair_index, pair_count=length(group.pairs))
                end
            end
            _assign_symmetric!(overlap_matrix, group.left_mebf, group.right_mebf, overlap_sum)
            _assign_symmetric!(electronic_one_body, group.left_mebf, group.right_mebf, one_body_sum)
            _assign_symmetric!(two_body, group.left_mebf, group.right_mebf, two_body_sum)
            _assign_symmetric!(pair_counts, group.left_mebf, group.right_mebf, length(group.pairs))
            if !isnothing(checkpoint)
                save_state_pair_checkpoint!(checkpoint, :dp, group.left_mebf, group.right_mebf, overlap_sum, one_body_sum, two_body_sum, length(group.pairs))
            end
            progress_event!(progress, :dp_state_pair_complete; left_mebf=group.left_mebf, right_mebf=group.right_mebf, pair_count=length(group.pairs))
        end
    finally
        runtime.use_gpu && _close_dp_two_electron_gpu_workspace!(two_electron_workspace)
    end
    nuclear_repulsion = one_electron.nuclear_repulsion .* overlap_matrix
    one_body_with_nuclear_repulsion = electronic_one_body + nuclear_repulsion
    hamiltonian = one_body_with_nuclear_repulsion + two_body
    !isnothing(telemetry) && set_counter!(telemetry, :n_dp_determinant_pairs_contracted, domain.screened_count)
    return RawNOCIMatrices([basis.label for basis in basis_functions], overlap_matrix, electronic_one_body, nuclear_repulsion, one_body_with_nuclear_repulsion, two_body, hamiltonian, pair_counts)
end
