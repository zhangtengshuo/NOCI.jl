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

 File:            src/algorithms/ahpc_noci/spin_matrix_assembly.jl
 Purpose:         Row-compressed determinant-pair screening blocks and raw AHPC matrix assembly.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _ahpc_state_pair_pair_block_memory(left_basis)

Estimate retained host storage for one cutoff-only row-compressed determinant-pair screening block whose rows follow `left_basis` determinants.
The estimate covers one `Int32` last-right cutoff per row, allocates no flattened offsets, scientific data, or CUDA state, and uses checked arithmetic.
"""
function _ahpc_state_pair_pair_block_memory(left_basis::MolecularBasisFunction)
    n_left = determinant_count(left_basis.expansion)
    return MemoryEstimate(estimate_array_bytes(Int32, n_left), 0)
end

"""
    _build_ahpc_state_pair_pairs(left_basis, right_basis, screening, left_mebf, right_mebf; runtime, telemetry)

Build exactly one deterministic row-compressed screened determinant-pair block for the requested lower-triangular MEBF pair.
Inputs are read-only, threshold and replay order are identical to `build_determinant_pair_domain`, host storage is preflighted before allocation, optional telemetry records the local build, and no CUDA work occurs.
"""
function _build_ahpc_state_pair_pairs(
    left_basis::MolecularBasisFunction{T},
    right_basis::MolecularBasisFunction{T},
    screening::ScreeningOptions{T},
    left_mebf::Int,
    right_mebf::Int;
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry},
) where {T<:AbstractFloat}
    1 <= right_mebf <= left_mebf || throw(ArgumentError("MEBF-pair indices must identify a lower-triangular block"))
    diagonal = left_mebf == right_mebf
    left_coefficients = left_basis.expansion.coefficients
    right_coefficients = right_basis.expansion.coefficients
    threshold = diagonal ? screening.tau_ci : screening.tau_ci_offdiagonal
    estimate = _ahpc_state_pair_pair_block_memory(left_basis)
    enforce_memory_limits(estimate, runtime)
    work = () -> _build_compressed_determinant_pairs(left_coefficients, right_coefficients, threshold; diagonal)
    return isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :build_state_pair_determinant_pairs;
        estimate,
        metadata=Dict(:left_mebf => left_mebf, :right_mebf => right_mebf, :diagonal => diagonal, :julia_threads => Threads.nthreads(), :cutoff_workers => min(Threads.nthreads(), length(left_coefficients))),
        collect_allocations=runtime.collect_allocations,
    )
end

"""
    assemble_ahpc_spin_string_matrices(basis_functions, one_electron, integrals, screening, acceleration; runtime=RuntimeOptions(), telemetry=nothing, checkpoint=nothing, progress=nothing)

Assemble symmetric raw NOCI matrices with AHPC precontraction, state-pair-local determinant screening, exact-presized canonical spin incidence, selectable complementary-cofactor intermediates, and blocked sparse contraction.
Inputs are read-only, overlap is dimensionless and Hamiltonian components follow integral energy units, every large phase has host/device preflight and telemetry, CPU and CUDA execution may use rank-uniform polynomial PDDC or LU-certified full-rank cofactors with polynomial fallback, optional test-mode progress adds only `.sec`/stdout heartbeat records, and optional two-level restart stores only completed scalars and cumulative alpha-block prefixes.
"""
function assemble_ahpc_spin_string_matrices(
    basis_functions::AbstractVector{<:MolecularBasisFunction{T}},
    one_electron::OneElectronValues{T},
    integrals::Union{PackedERIValues{T},ERIFactorValues{T}},
    screening::ScreeningOptions{T},
    acceleration::AHPCAccelerationOptions=AHPCAccelerationOptions();
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
    checkpoint::Union{Nothing,CheckpointSession}=nothing,
    progress::Union{Nothing,ProgressLogger}=nothing,
) where {T<:AbstractFloat}
    ahpc_threads_per_gpu(runtime)
    n_basis_functions = length(basis_functions)
    n_basis_functions >= 1 || throw(ArgumentError("at least one MEBF is required"))
    n_orbitals = size(one_electron.overlap, 1)
    integrals.n_orbitals == n_orbitals || throw(DimensionMismatch("one-electron and ERI orbital dimensions differ"))
    matrix_estimate = raw_noci_matrices_memory(T, n_basis_functions)
    enforce_memory_limits(matrix_estimate, runtime)
    canonical = build_canonical_mebf_spin_strings.(basis_functions)
    orbital_basis_keys = [Tuple(basis.fragment_state_indices) for basis in basis_functions]
    unique_orbital_basis_keys = Set{Tuple{Vararg{Int}}}(orbital_basis_keys)
    precontracted_oriented_orbital_basis_pair_keys = Set{Tuple{Tuple{Vararg{Int}},Tuple{Vararg{Int}}}}()
    precontraction_call_count = 0
    overlap_matrix = zeros(T, n_basis_functions, n_basis_functions)
    electronic_one_body = zeros(T, n_basis_functions, n_basis_functions)
    two_body = zeros(T, n_basis_functions, n_basis_functions)
    pair_counts = zeros(Int, n_basis_functions, n_basis_functions)
    one_body_operator = core_hamiltonian(one_electron)
    precontraction_source = nothing
    full_pair_count = full_determinant_pair_count(basis_functions)
    screened_pair_count = 0
    state_pair_count = 0

    try
    for left_mebf in eachindex(basis_functions), right_mebf in 1:left_mebf
        completed = isnothing(checkpoint) ? nothing : load_state_pair_checkpoint(checkpoint, :ahpc, left_mebf, right_mebf)
        if !isnothing(completed)
            overlap_value = T(completed["overlap"])
            one_body_value = T(completed["one_body"])
            two_body_value = T(completed["two_body"])
            pair_count = Int(completed["pair_count"])
            _assign_symmetric!(overlap_matrix, left_mebf, right_mebf, overlap_value)
            _assign_symmetric!(electronic_one_body, left_mebf, right_mebf, one_body_value)
            _assign_symmetric!(two_body, left_mebf, right_mebf, two_body_value)
            _assign_symmetric!(pair_counts, left_mebf, right_mebf, pair_count)
            screened_pair_count = Base.Checked.checked_add(screened_pair_count, pair_count)
            state_pair_count = Base.Checked.checked_add(state_pair_count, 1)
            progress_event!(progress, :ahpc_state_pair_reused; left_mebf=left_mebf, right_mebf=right_mebf, pair_count=pair_count)
            continue
        end
        progress_event!(progress, :ahpc_state_pair_started; left_mebf=left_mebf, right_mebf=right_mebf)
        isnothing(precontraction_source) && (precontraction_source = with_test_progress(
            () -> _prepare_ahpc_precontraction_source(integrals; runtime, telemetry),
            progress,
            :ahpc_precontraction_source_progress;
            left_mebf,
            right_mebf,
            gpu_count=runtime.use_gpu ? length(runtime.gpu_devices) : 0,
        ))
        left_basis = basis_functions[left_mebf]
        right_basis = basis_functions[right_mebf]
        left_strings = canonical[left_mebf]
        right_strings = canonical[right_mebf]
        pairs = with_test_progress(
            () -> _build_ahpc_state_pair_pairs(
                left_basis,
                right_basis,
                screening,
                left_mebf,
                right_mebf;
                runtime,
                telemetry,
            ),
            progress,
            :ahpc_pair_block_progress;
            left_mebf,
            right_mebf,
        )
        progress_event!(progress, :ahpc_pair_block_ready; left_mebf=left_mebf, right_mebf=right_mebf, pair_count=length(pairs))
        isempty(pairs) && throw(ArgumentError("AHPC production screening removed every determinant pair for MEBF block ($left_mebf,$right_mebf)"))
        screened_pair_count = Base.Checked.checked_add(screened_pair_count, length(pairs))
        state_pair_count = Base.Checked.checked_add(state_pair_count, 1)
        oriented_orbital_basis_pair_key = (orbital_basis_keys[left_mebf], orbital_basis_keys[right_mebf])
        push!(precontracted_oriented_orbital_basis_pair_keys, oriented_orbital_basis_pair_key)
        precontraction_call_count = Base.Checked.checked_add(precontraction_call_count, 1)
        preparation = with_test_progress(
            () -> prepare_ahpc_state_pair(
                left_basis,
                right_basis,
                one_electron.overlap;
                tau_singularity=screening.tau_singularity,
                runtime,
                telemetry,
            ),
            progress,
            :ahpc_state_pair_preparation_progress;
            left_mebf,
            right_mebf,
        )
        precontraction_estimate = precontraction_memory(T, preparation.response_space, precontraction_source; runtime)
        precontract_work = () -> with_test_progress(
            () -> precontract_active_hamiltonian(one_body_operator, preparation.response_space, precontraction_source; runtime),
            progress,
            :ahpc_precontraction_progress;
            left_mebf,
            right_mebf,
            gpu_count=runtime.use_gpu ? length(runtime.gpu_devices) : 0,
        )
        coefficients = isnothing(telemetry) ? precontract_work() : record_phase!(
            precontract_work,
            telemetry,
            :precontract_active_hamiltonian;
            estimate=precontraction_estimate,
            metadata=Dict(
                :left_mebf => left_mebf,
                :right_mebf => right_mebf,
                :strategy => runtime.use_gpu ? :cuda_active_hamiltonian_precontraction : :cpu_active_hamiltonian_precontraction,
                :gpu_device => runtime.use_gpu ? runtime.gpu_device : -1,
                :gpu_devices => runtime.use_gpu ? runtime.gpu_devices : (),
                :gpu_count => runtime.use_gpu ? length(runtime.gpu_devices) : 0,
            ),
            collect_allocations=runtime.collect_allocations,
        )
        progress_event!(progress, :ahpc_precontraction_complete; left_mebf=left_mebf, right_mebf=right_mebf)
        incidence = with_test_progress(
            () -> build_spin_pair_incidence(
                left_basis,
                right_basis,
                left_strings,
                right_strings,
                pairs;
                support_lookup=acceleration.support_lookup,
                runtime,
                telemetry,
            ),
            progress,
            :ahpc_incidence_progress;
            left_mebf,
            right_mebf,
            pair_count=length(pairs),
        )
        progress_event!(progress, :ahpc_incidence_complete; left_mebf=left_mebf, right_mebf=right_mebf, n_kappa=incidence.weights.n_kappa, n_lambda=incidence.weights.n_lambda, n_edges=length(incidence.weights.weights), n_union=length(incidence.support.union_support))
        support = incidence.support
        sparse = incidence.weights
        n_union = length(support.union_support)
        intermediate_work = () -> if runtime.use_gpu
            acceleration.cofactor_evaluation == LUCertifiedCofactorEvaluation ?
                build_spin_pair_intermediates_cuda_lu_certified_hybrid_distributed(
                    preparation,
                    coefficients,
                    support;
                    runtime,
                    telemetry,
                ) :
                build_spin_pair_intermediates_cuda_polynomial_distributed(
                    preparation,
                    coefficients,
                    support;
                    runtime,
                    telemetry,
                )
        else
            build_spin_pair_intermediates(
                preparation,
                coefficients,
                support;
                cofactor_evaluation=acceleration.cofactor_evaluation,
                runtime,
                telemetry,
            )
        end
        intermediates = with_test_progress(
            intermediate_work,
            progress,
            :ahpc_intermediates_progress;
            left_mebf,
            right_mebf,
            n_union,
            gpu_count=runtime.use_gpu ? length(runtime.gpu_devices) : 0,
            gpu_devices=runtime.use_gpu ? runtime.gpu_devices : (),
        )
        progress_event!(progress, :ahpc_intermediates_complete; left_mebf=left_mebf, right_mebf=right_mebf, n_union)
        contraction_work = () -> contract_sparse_spin_pair_weights(
            sparse,
            support,
            preparation,
            coefficients,
            intermediates;
            runtime,
            telemetry,
            checkpoint,
            progress,
            left_mebf,
            right_mebf,
        )
        element = with_test_progress(
            contraction_work,
            progress,
            :ahpc_sparse_contraction_progress;
            left_mebf,
            right_mebf,
            n_kappa=sparse.n_kappa,
            n_lambda=sparse.n_lambda,
            n_edges=length(sparse.weights),
            n_union,
            gpu_count=runtime.use_gpu ? length(runtime.gpu_devices) : 0,
        )
        _assign_symmetric!(overlap_matrix, left_mebf, right_mebf, element.overlap)
        _assign_symmetric!(electronic_one_body, left_mebf, right_mebf, element.one_body)
        _assign_symmetric!(two_body, left_mebf, right_mebf, element.two_body)
        _assign_symmetric!(pair_counts, left_mebf, right_mebf, length(pairs))
        if !isnothing(checkpoint)
            save_state_pair_checkpoint!(
                checkpoint,
                :ahpc,
                left_mebf,
                right_mebf,
                element.overlap,
                element.one_body,
                element.two_body,
                length(pairs);
                metadata=Dict{String,Any}(
                    "n_kappa" => sparse.n_kappa,
                    "n_lambda" => sparse.n_lambda,
                    "n_edges" => length(sparse.weights),
                    "n_union" => length(support.union_support),
                ),
            )
        end
        progress_event!(progress, :ahpc_state_pair_complete; left_mebf=left_mebf, right_mebf=right_mebf, pair_count=length(pairs))
    end
    finally
        !isnothing(precontraction_source) && _close_ahpc_precontraction_source!(precontraction_source)
    end

    nuclear_repulsion = one_electron.nuclear_repulsion .* overlap_matrix
    one_body_with_nuclear_repulsion = electronic_one_body + nuclear_repulsion
    hamiltonian = one_body_with_nuclear_repulsion + two_body
    if !isnothing(telemetry)
        unique_oriented_basis_pair_count = length(precontracted_oriented_orbital_basis_pair_keys)
        reusable_precontraction_call_count = precontraction_call_count - unique_oriented_basis_pair_count
        set_counter!(telemetry, :n_determinant_pairs, screened_pair_count)
        set_counter!(telemetry, :n_full_determinant_pairs, full_pair_count)
        set_counter!(telemetry, :n_state_pairs_precontracted, state_pair_count)
        set_counter!(telemetry, :n_unique_orbital_basis_keys, length(unique_orbital_basis_keys))
        set_counter!(telemetry, :n_unique_oriented_basis_pair_keys, unique_oriented_basis_pair_count)
        set_counter!(telemetry, :n_reusable_precontraction_calls, reusable_precontraction_call_count)
        set_counter!(telemetry, :n_ahpc_spin_string_determinant_pairs, screened_pair_count)
        set_counter!(telemetry, :n_state_pair_pair_blocks_built, state_pair_count)
        set_counter!(telemetry, :n_global_pair_domains_built, 0)
        increment_counter!(telemetry, :n_inverse_calls, 0)
        increment_counter!(telemetry, :n_rank_classifier_calls, 0)
        increment_counter!(telemetry, :n_svd_calls, 0)
    end
    return RawNOCIMatrices(
        [basis.label for basis in basis_functions],
        overlap_matrix,
        electronic_one_body,
        nuclear_repulsion,
        one_body_with_nuclear_repulsion,
        two_body,
        hamiltonian,
        pair_counts,
    )
end
