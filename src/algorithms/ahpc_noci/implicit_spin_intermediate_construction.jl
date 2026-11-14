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

 File:            src/algorithms/ahpc_noci/implicit_spin_intermediate_construction.jl
 Purpose:         Julia-thread construction of fixed-rank spin-string-pair union intermediates.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _validate_spin_pair_intermediate_inputs(preparation, coefficients)

Validate state-pair overlap and active-Hamiltonian coefficient dimensions and finite values once before union-kernel construction.
Inputs are read-only CPU arrays, no molecular-scale storage or CUDA state is allocated, and malformed or non-finite data raise before worker tasks start.
"""
function _validate_spin_pair_intermediate_inputs(
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
) where {T<:AbstractFloat}
    active_rank = size(preparation.effective_active_overlap, 1)
    size(preparation.effective_active_overlap, 2) == active_rank || throw(DimensionMismatch("effective_active_overlap must be square"))
    0 <= active_rank <= 64 || throw(ArgumentError("production active rank must lie in 0:64"))
    n_response = Base.Checked.checked_mul(active_rank, active_rank)
    length(coefficients.one_body_active) == n_response || throw(DimensionMismatch("one-body active coefficient dimension mismatch"))
    length(coefficients.two_body_inactive_active) == n_response || throw(DimensionMismatch("inactive-active coefficient dimension mismatch"))
    size(coefficients.coulomb_response) == (n_response, n_response) || throw(DimensionMismatch("Coulomb response dimension mismatch"))
    size(coefficients.exchange_response) == (n_response, n_response) || throw(DimensionMismatch("exchange response dimension mismatch"))
    all(isfinite, preparation.effective_active_overlap) || throw(ArgumentError("effective_active_overlap contains a non-finite value"))
    all(isfinite, coefficients.one_body_active) || throw(ArgumentError("one-body active coefficients contain a non-finite value"))
    all(isfinite, coefficients.two_body_inactive_active) || throw(ArgumentError("inactive-active coefficients contain a non-finite value"))
    all(isfinite, coefficients.coulomb_response) || throw(ArgumentError("Coulomb response contains a non-finite value"))
    all(isfinite, coefficients.exchange_response) || throw(ArgumentError("exchange response contains a non-finite value"))
    return active_rank
end

"""
    _fill_selected_active_overlap_from_bits!(selected_overlap, response_indices, effective_active_overlap, pair, active_rank)

Fill one validated canonical occupied-spin overlap block and state-pair active-response map directly from ordered set bits.
Destination CPU buffers are overwritten in column-major response order, inputs are read-only, no allocation or CUDA work occurs, and all dimensions and bit domains must already be validated.
"""
function _fill_selected_active_overlap_from_bits!(
    selected_overlap::Matrix{T},
    response_indices::Vector{Int},
    effective_active_overlap::Matrix{T},
    pair::CanonicalSpinStringPair,
    active_rank::Int,
) where {T<:AbstractFloat}
    feature_slot = 1
    column = 1
    right_bits = pair.right_bits
    @inbounds while !iszero(right_bits)
        right_active = trailing_zeros(right_bits) + 1
        row = 1
        left_bits = pair.left_bits
        while !iszero(left_bits)
            left_active = trailing_zeros(left_bits) + 1
            selected_overlap[row, column] = effective_active_overlap[left_active, right_active]
            response_indices[feature_slot] = left_active + (right_active - 1) * active_rank
            feature_slot += 1
            row += 1
            left_bits &= left_bits - UInt64(1)
        end
        column += 1
        right_bits &= right_bits - UInt64(1)
    end
    return selected_overlap, response_indices
end

"""
    _fill_local_same_spin_kernel!(local_kernel, response_indices, coefficients)

Gather the ordered Coulomb-minus-exchange response kernel `K_ab = J_ab-K_ab` from validated state-pair coefficients.
This helper is retained as an ordered-response reference/oracle for equivalence tests; CPU production contracts the state-pair determinant-Hessian response directly. The destination is overwritten without symmetrization, coefficient matrices and response indices are read-only, and no allocation occurs.
"""
function _fill_local_same_spin_kernel!(
    local_kernel::Matrix{T},
    response_indices::Vector{Int},
    coefficients::ActiveHamiltonianCoefficients{T},
) where {T<:AbstractFloat}
    n_local_response = length(response_indices)
    size(local_kernel) == (n_local_response, n_local_response) || throw(DimensionMismatch("local same-spin kernel dimension mismatch"))
    @inbounds for second_local in 1:n_local_response
        second_global = response_indices[second_local]
        for first_local in 1:n_local_response
            first_global = response_indices[first_local]
            local_kernel[first_local, second_local] =
                coefficients.coulomb_response[first_global, second_global] -
                coefficients.exchange_response[first_global, second_global]
        end
    end
    return local_kernel
end

"""
    _write_spin_pair_intermediate!(outputs, union_id, feature_offset, coefficients, cofactors, response_indices)

Write one union kernel's scalar channels and structural first-cofactor values while contracting both linear coefficient vectors in a single response pass.
Output CPU arrays are mutated at the supplied disjoint union range, inputs are read-only, accumulation follows column-major response order, and any non-finite scientific result raises immediately.
"""
function _write_spin_pair_intermediate!(
    cofactor_scalar::Vector{T},
    one_body_active::Vector{T},
    two_body_inactive_active::Vector{T},
    same_spin_two_body::Vector{T},
    first_cofactor_values::Vector{T},
    union_id::Int,
    feature_offset::Int,
    coefficients::ActiveHamiltonianCoefficients{T},
    cofactors::ComplementaryCofactorResult{T},
    response_indices::Vector{Int},
) where {T<:AbstractFloat}
    isfinite(cofactors.cofactor_scalar) || throw(ArgumentError("complementary-cofactor scalar is non-finite"))
    isfinite(cofactors.contracted_second) || throw(ArgumentError("contracted second complementary cofactor is non-finite"))
    one_body_value = zero(T)
    two_body_value = zero(T)
    @inbounds for local_response in eachindex(response_indices)
        first_cofactor = cofactors.first_cofactor[local_response]
        isfinite(first_cofactor) || throw(ArgumentError("first complementary cofactor is non-finite"))
        state_pair_response = response_indices[local_response]
        one_body_value += coefficients.one_body_active[state_pair_response] * first_cofactor
        two_body_value += coefficients.two_body_inactive_active[state_pair_response] * first_cofactor
        first_cofactor_values[feature_offset + local_response - 1] = first_cofactor
    end
    isfinite(one_body_value) || throw(ArgumentError("one-body active contraction is non-finite"))
    isfinite(two_body_value) || throw(ArgumentError("inactive-active contraction is non-finite"))
    cofactor_scalar[union_id] = cofactors.cofactor_scalar
    one_body_active[union_id] = one_body_value
    two_body_inactive_active[union_id] = two_body_value
    same_spin_two_body[union_id] = cofactors.contracted_second
    return nothing
end

"""
    _build_spin_pair_intermediate!(workspace, preparation, coefficients, state_pair_hessian_response, pair, outputs, union_id, feature_offset, active_rank, cofactor_evaluation)

Evaluate and write one validated CPU union spin-pair intermediate using either the rank-uniform polynomial PDDC engine or the LU-certified full-rank engine with polynomial fallback.
Worker-local overlap/index/cofactor buffers and disjoint output slots are mutated, scientific inputs and the state-pair response are read-only, the return is `(lu_certificate_accepted, inverse_computed)`, and the LU path never falls back to SVD or G-NOCI.
"""
function _build_spin_pair_intermediate!(
    workspace::SpinPairIntermediateWorkspace{T},
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    state_pair_hessian_response::AbstractMatrix{T},
    pair::CanonicalSpinStringPair,
    cofactor_scalar::Vector{T},
    one_body_active::Vector{T},
    two_body_inactive_active::Vector{T},
    same_spin_two_body::Vector{T},
    first_cofactor_values::Vector{T},
    union_id::Int,
    feature_offset::Int,
    active_rank::Int,
    cofactor_evaluation::CofactorEvaluation,
) where {T<:AbstractFloat}
    _fill_selected_active_overlap_from_bits!(
        workspace.selected_overlap,
        workspace.response_indices,
        preparation.effective_active_overlap,
        pair,
        active_rank,
    )
    _fill_union_hessian_pair_indices!(
        workspace.hessian_row_pair_indices,
        workspace.hessian_column_pair_indices,
        workspace.response_indices,
        active_rank,
    )

    lu_certificate_accepted = false
    inverse_computed = false
    cofactors = if cofactor_evaluation == LUCertifiedCofactorEvaluation
        T <: Union{Float32,Float64} || throw(ArgumentError("LU-certified cofactor evaluation requires Float32 or Float64"))
        lu_result, inverse_computed = _lu_certified_state_pair_hessian_contracted_complementary_cofactors!(
            workspace.cofactors,
            workspace.selected_overlap,
            state_pair_hessian_response,
            workspace.hessian_row_pair_indices,
            workspace.hessian_column_pair_indices,
        )
        if isnothing(lu_result)
            _polynomial_state_pair_hessian_contracted_complementary_cofactors!(
                workspace.cofactors,
                workspace.selected_overlap,
                state_pair_hessian_response,
                workspace.hessian_row_pair_indices,
                workspace.hessian_column_pair_indices,
            )
        else
            lu_certificate_accepted = true
            lu_result
        end
    else
        _polynomial_state_pair_hessian_contracted_complementary_cofactors!(
            workspace.cofactors,
            workspace.selected_overlap,
            state_pair_hessian_response,
            workspace.hessian_row_pair_indices,
            workspace.hessian_column_pair_indices,
        )
    end

    _write_spin_pair_intermediate!(
        cofactor_scalar,
        one_body_active,
        two_body_inactive_active,
        same_spin_two_body,
        first_cofactor_values,
        union_id,
        feature_offset,
        coefficients,
        cofactors,
        workspace.response_indices,
    )
    return lu_certificate_accepted, inverse_computed
end

"""
    build_spin_pair_intermediates(preparation, coefficients, support; cofactor_evaluation=PolynomialCofactorEvaluation, runtime=RuntimeOptions(), telemetry=nothing)

Build CPU implicit AHPC spin-pair intermediates with Julia-thread-parallel union kernels, direct bit traversal, one NOCI state-pair determinant-Hessian response, union Hessian-pair index maps, and one reusable cofactor workspace per worker.
The default polynomial PDDC route remains rank uniform with `O(q^5)` contracted-second work; the optional LU-certified route admits only residual/condition-certified full-rank overlaps and evaluates their contracted second cofactors in `O(q^4)`, while every rejected or ambiguous overlap falls back to the same polynomial route; output traversal is deterministic and no SVD, G-NOCI fallback, or CUDA work occurs.
"""
function build_spin_pair_intermediates(
    preparation::AHPCStatePairPreparation{T},
    coefficients::ActiveHamiltonianCoefficients{T},
    support::SpinPairSupportDomain;
    cofactor_evaluation::CofactorEvaluation=PolynomialCofactorEvaluation,
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    active_rank = _validate_spin_pair_intermediate_inputs(preparation, coefficients)
    maximum_occupied, feature_entries = _spin_pair_support_shape(support.union_support, active_rank)
    maximum_occupied <= runtime.max_active_occupied || throw(ArgumentError("union support occupied count exceeds max_active_occupied"))
    cofactor_evaluation == LUCertifiedCofactorEvaluation && !(T <: Union{Float32,Float64}) &&
        throw(ArgumentError("LU-certified cofactor evaluation requires Float32 or Float64"))
    estimate = spin_pair_intermediate_memory(T, preparation, support)
    enforce_memory_limits(estimate, runtime)
    n_union = length(support.union_support)
    fixed_occupied = n_union == 0 || feature_entries == Base.Checked.checked_mul(n_union, Base.Checked.checked_mul(maximum_occupied, maximum_occupied))
    n_union_hessian_pairs = _hessian_pair_count(maximum_occupied)
    n_state_pair_hessian_pairs = _hessian_pair_count(active_rank)
    n_state_pair_hessian_response_entries = n_union == 0 ? 0 : Base.Checked.checked_mul(n_state_pair_hessian_pairs, n_state_pair_hessian_pairs)
    n_union_hessian_pair_indices = Base.Checked.checked_mul(2, n_union_hessian_pairs)
    using_lu_certificate = cofactor_evaluation == LUCertifiedCofactorEvaluation
    phase_metadata = Dict{Symbol,Any}(
        :n_union_spin_pair_support => n_union,
        :maximum_active_occupied => maximum_occupied,
        :fixed_active_occupied => fixed_occupied,
        :response_id_storage_bytes => 0,
        :implicit_response_ids => true,
        :packed_spin_construction => true,
        :structural_feature_kernel => true,
        :raw_same_spin_kernel => false,
        :hessian_pair_response => true,
        :state_pair_hessian_response => true,
        :state_pair_hessian_contraction => true,
        :local_hessian_pair_matrix => false,
        :state_pair_hessian_response_entries => n_state_pair_hessian_response_entries,
        :hessian_pair_response_entries_per_union => 0,
        :hessian_pair_id_entries_per_union => n_union_hessian_pair_indices,
        :ordered_response_second_contraction => false,
        :cofactor_evaluation => using_lu_certificate ? :lu_certified : :polynomial,
        :polynomial_cofactor_engine => !using_lu_certificate,
        :lu_certified_cofactor_engine => using_lu_certificate,
        :contracted_second_complexity => using_lu_certificate ?
            :O_n_active_occupied_4_certified_O_n_active_occupied_5_fallback :
            :O_n_active_occupied_5,
        :polynomial_workspace_reuse => true,
        :subset_recurrence_runtime => false,
        :julia_threads => Threads.nthreads(),
    )

    work = function ()
        cofactor_scalar = Vector{T}(undef, n_union)
        one_body_active = Vector{T}(undef, n_union)
        two_body_inactive_active = Vector{T}(undef, n_union)
        same_spin_two_body = Vector{T}(undef, n_union)
        feature_offsets = Vector{Int}(undef, n_union + 1)
        first_cofactor_values = Vector{T}(undef, feature_entries)
        feature_offsets[1] = 1
        if fixed_occupied
            feature_width = maximum_occupied * maximum_occupied
            @inbounds for union_id in 1:n_union
                feature_offsets[union_id + 1] = 1 + union_id * feature_width
            end
        else
            for union_id in 1:n_union
                n_active_occupied = count_ones(support.union_support[union_id].left_bits)
                feature_offsets[union_id + 1] = Base.Checked.checked_add(
                    feature_offsets[union_id],
                    Base.Checked.checked_mul(n_active_occupied, n_active_occupied),
                )
            end
        end
        feature_offsets[end] == feature_entries + 1 || throw(ArgumentError("implicit structural first-cofactor feature count mismatch"))

        state_pair_response = n_union == 0 ? Matrix{T}(undef, 0, 0) :
            state_pair_hessian_response(coefficients, active_rank)
        worker_count = min(Threads.nthreads(), n_union)
        ranges = n_union == 0 ? UnitRange{Int}[] : _balanced_auxiliary_ranges(n_union, worker_count)
        tasks = map(ranges) do range
            Threads.@spawn begin
                workspace = SpinPairIntermediateWorkspace{T}()
                lu_certificate_accepts = 0
                lu_certificate_fallbacks = 0
                inverse_calls = 0
                polynomial_evaluations = 0
                if fixed_occupied
                    _resize_spin_pair_intermediate_workspace!(workspace, maximum_occupied, runtime)
                    for union_id in range
                        lu_accepted, inverse_computed = _build_spin_pair_intermediate!(
                            workspace,
                            preparation,
                            coefficients,
                            state_pair_response,
                            support.union_support[union_id],
                            cofactor_scalar,
                            one_body_active,
                            two_body_inactive_active,
                            same_spin_two_body,
                            first_cofactor_values,
                            union_id,
                            feature_offsets[union_id],
                            active_rank,
                            cofactor_evaluation,
                        )
                        if using_lu_certificate
                            inverse_calls += inverse_computed
                            if lu_accepted
                                lu_certificate_accepts += 1
                            else
                                lu_certificate_fallbacks += 1
                                polynomial_evaluations += 1
                            end
                        else
                            polynomial_evaluations += 1
                        end
                    end
                else
                    for union_id in range
                        pair = support.union_support[union_id]
                        n_active_occupied = count_ones(pair.left_bits)
                        _resize_spin_pair_intermediate_workspace!(workspace, n_active_occupied, runtime)
                        lu_accepted, inverse_computed = _build_spin_pair_intermediate!(
                            workspace,
                            preparation,
                            coefficients,
                            state_pair_response,
                            pair,
                            cofactor_scalar,
                            one_body_active,
                            two_body_inactive_active,
                            same_spin_two_body,
                            first_cofactor_values,
                            union_id,
                            feature_offsets[union_id],
                            active_rank,
                            cofactor_evaluation,
                        )
                        if using_lu_certificate
                            inverse_calls += inverse_computed
                            if lu_accepted
                                lu_certificate_accepts += 1
                            else
                                lu_certificate_fallbacks += 1
                                polynomial_evaluations += 1
                            end
                        else
                            polynomial_evaluations += 1
                        end
                    end
                end
                return (
                    lu_certificate_accepts=lu_certificate_accepts,
                    lu_certificate_fallbacks=lu_certificate_fallbacks,
                    inverse_calls=inverse_calls,
                    polynomial_evaluations=polynomial_evaluations,
                )
            end
        end
        worker_statistics = fetch.(tasks)
        intermediates = ImplicitSpinPairIntermediates(
            active_rank,
            cofactor_scalar,
            one_body_active,
            two_body_inactive_active,
            same_spin_two_body,
            feature_offsets,
            first_cofactor_values,
            support.union_support,
            Val(:validated),
        )
        return (
            intermediates=intermediates,
            lu_certificate_accepts=sum((statistics.lu_certificate_accepts for statistics in worker_statistics); init=0),
            lu_certificate_fallbacks=sum((statistics.lu_certificate_fallbacks for statistics in worker_statistics); init=0),
            inverse_calls=sum((statistics.inverse_calls for statistics in worker_statistics); init=0),
            polynomial_evaluations=sum((statistics.polynomial_evaluations for statistics in worker_statistics); init=0),
        )
    end

    build_result = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :build_spin_pair_intermediates;
        estimate,
        metadata=phase_metadata,
        collect_allocations=runtime.collect_allocations,
    )
    intermediates = build_result.intermediates
    if !isnothing(telemetry)
        n_kernels = length(intermediates.cofactor_scalar)
        increment_counter!(telemetry, :n_union_spin_pair_intermediates, n_kernels)
        increment_counter!(telemetry, :n_spin_pair_kernel_evaluations, n_kernels)
        increment_counter!(telemetry, :n_spin_pair_feature_entries, length(intermediates.first_cofactor_values))
        increment_counter!(telemetry, :n_implicit_spin_feature_entries, length(intermediates.first_cofactor_values))
        set_counter!(telemetry, :n_complementary_cofactor_transitions, 0)
        set_counter!(telemetry, :n_structural_response_iterations, 0)
        increment_counter!(telemetry, :n_polynomial_cofactor_kernel_evaluations, build_result.polynomial_evaluations)
        increment_counter!(telemetry, :n_lu_certificate_evaluations, using_lu_certificate ? n_kernels : 0)
        increment_counter!(telemetry, :n_lu_certified_cofactor_kernel_evaluations, build_result.lu_certificate_accepts)
        increment_counter!(telemetry, :n_lu_certificate_fallbacks, build_result.lu_certificate_fallbacks)
        increment_counter!(telemetry, :n_packed_spin_intermediate_builds)
        set_counter!(telemetry, :n_union_kernel_workers, min(Threads.nthreads(), length(support.union_support)))
        set_counter!(telemetry, :spin_pair_response_id_storage_bytes, 0)
        set_counter!(telemetry, :spin_pair_response_id_payload_bytes, 0)
        set_counter!(telemetry, :n_hessian_pair_response_entries_per_union, 0)
        set_counter!(telemetry, :n_hessian_pair_id_entries_per_union, n_union_hessian_pair_indices)
        set_counter!(telemetry, :n_state_pair_hessian_response_entries, n_state_pair_hessian_response_entries)
        increment_counter!(telemetry, :n_inverse_calls, build_result.inverse_calls)
        increment_counter!(telemetry, :n_rank_classifier_calls, using_lu_certificate ? n_kernels : 0)
        increment_counter!(telemetry, :n_svd_calls, 0)
    end
    return intermediates
end
