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

 File:            src/algorithms/ahpc_noci/spin_incidence.jl
 Purpose:         Canonical spin support and injective row-compressed CI-weight incidence per state pair.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    PreparedSpinPairIncidence

Own one state-pair's canonical spin support and injective row-compressed CI-weight incidence matrix.
The object contains only production data needed by union-kernel construction and blocked contraction; temporary lookup tables are released after construction and no ERI or CUDA storage is retained.
"""
struct PreparedSpinPairIncidence{T<:AbstractFloat}
    support::SpinPairSupportDomain
    weights::SparseSpinPairWeights{T}
end

"""
    _spin_pair_incidence_memory(T, n_pairs, n_alpha, n_beta, maximum_row_length, selected_lookup)

Estimate peak host storage for the exact-sized support, selected exact 64- or 128-bit lookup, union, CSR, cursor, and largest-row sorting payloads used by production incidence assembly.
Counts are dimensionless, the result is measured in bytes with zero device storage, no scientific arrays are allocated, and invalid or overflowing inputs raise before return.
"""
function _spin_pair_incidence_memory(
    ::Type{T},
    n_pairs::Int,
    n_alpha::Int,
    n_beta::Int,
    maximum_row_length::Int,
    selected_lookup::SupportLookup,
    ::Type{K}=UInt128,
) where {T<:AbstractFloat,K<:Union{UInt64,UInt128}}
    all(count -> count >= 0, (n_pairs, n_alpha, n_beta, maximum_row_length)) || throw(ArgumentError("incidence memory counts must be nonnegative"))
    n_alpha <= typemax(Int32) && n_beta <= typemax(Int32) || throw(OverflowError("spin-pair support exceeds Int32 local-ID storage"))
    selected_lookup in (DictPresizedLookup, FlatHashPresizedLookup) || throw(ArgumentError("incidence memory requires a concrete production lookup"))
    support_entries = Base.Checked.checked_mul(2, Base.Checked.checked_add(n_alpha, n_beta))
    support_bytes = Base.Checked.checked_add(
        estimate_array_bytes(CanonicalSpinStringPair, support_entries),
        estimate_array_bytes(Int32, n_beta),
    )
    alpha_capacity = _spin_pair_flat_capacity(n_alpha)
    beta_capacity = _spin_pair_flat_capacity(n_beta)
    lookup_capacity = Base.Checked.checked_add(alpha_capacity, beta_capacity)
    lookup_bytes = Base.Checked.checked_add(
        Base.Checked.checked_add(estimate_array_bytes(K, lookup_capacity), estimate_array_bytes(Int32, lookup_capacity)),
        cld(lookup_capacity, 8),
    )
    alpha_integer_entries = Base.Checked.checked_add(Base.Checked.checked_mul(3, n_alpha), 1)
    sort_worker_count = maximum_row_length == 0 ? 0 : min(Threads.nthreads(), n_alpha)
    sort_scratch_entries = Base.Checked.checked_mul(maximum_row_length, sort_worker_count)
    csr_bytes = Base.Checked.checked_add(
        Base.Checked.checked_add(estimate_array_bytes(Int, alpha_integer_entries), estimate_array_bytes(Int32, n_pairs)),
        Base.Checked.checked_add(estimate_array_bytes(T, n_pairs), estimate_array_bytes(Tuple{Int32,T}, sort_scratch_entries)),
    )
    return MemoryEstimate(Base.Checked.checked_add(support_bytes, Base.Checked.checked_add(lookup_bytes, csr_bytes)), 0)
end

"""
    spin_pair_incidence_memory(T, left_strings, right_strings, pair_block; support_lookup=AdaptiveLookup)

Estimate retained production incidence memory from exact nested-window support counts without enumerating screened determinant pairs.
Inputs are read-only, temporary planning dictionaries scale with determinant spin strings, output is in bytes with zero device storage, row-sort scratch receives a second exact preflight after pass one, and nonnested domains or unsupported lookup choices raise explicitly.
"""
function spin_pair_incidence_memory(
    ::Type{T},
    left_strings::CanonicalMEBFSpinStrings,
    right_strings::CanonicalMEBFSpinStrings,
    pair_block::RowCompressedDeterminantPairs;
    support_lookup::SupportLookup=AdaptiveLookup,
) where {T<:AbstractFloat}
    alpha_count, beta_count = _support_counts(left_strings, right_strings, pair_block)
    selected = _selected_support_lookup(support_lookup, max(alpha_count, beta_count), length(pair_block))
    numbered = _numbered_spin_pair_domain(left_strings, right_strings)
    key_type = numbered ? UInt64 : _spin_lookup_key_type(left_strings, right_strings)
    numbering_bytes = numbered ? Base.Checked.checked_add(_canonical_spin_number_bytes(left_strings), _canonical_spin_number_bytes(right_strings)) : 0
    return combine_memory_estimates(_spin_pair_incidence_memory(T, length(pair_block), alpha_count, beta_count, 0, selected, key_type), MemoryEstimate(numbering_bytes, 0))
end

"""
    build_spin_pair_incidence(left_basis, right_basis, left_strings, right_strings, pair_block; support_lookup=AdaptiveLookup, runtime=RuntimeOptions(), telemetry=nothing)

Build exact-presized first-encounter spin support and injective CSR weights in two screened-pair passes, combining 64-bit support insertion with row counting in the first pass.
Scientific inputs are read-only, lookup tables are state-pair-local and discarded on return, host memory is preflighted, optional telemetry records both support and incidence statistics, and no duplicate compaction, third pair replay, fallback, or CUDA work occurs.
"""
function build_spin_pair_incidence(
    left_basis::MolecularBasisFunction{T},
    right_basis::MolecularBasisFunction{T},
    left_strings::CanonicalMEBFSpinStrings,
    right_strings::CanonicalMEBFSpinStrings,
    pair_block::RowCompressedDeterminantPairs;
    support_lookup::SupportLookup=AdaptiveLookup,
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    _validate_spin_weight_inputs(left_basis, right_basis, left_strings, right_strings, pair_block)
    alpha_expected, beta_expected = _support_counts(left_strings, right_strings, pair_block)
    selected = _selected_support_lookup(support_lookup, max(alpha_expected, beta_expected), length(pair_block))
    numbered = _numbered_spin_pair_domain(left_strings, right_strings)
    key_type = numbered ? UInt64 : _spin_lookup_key_type(left_strings, right_strings)
    numbering_bytes = numbered ? Base.Checked.checked_add(_canonical_spin_number_bytes(left_strings), _canonical_spin_number_bytes(right_strings)) : 0
    estimate = combine_memory_estimates(_spin_pair_incidence_memory(T, length(pair_block), alpha_expected, beta_expected, 0, selected, key_type), MemoryEstimate(numbering_bytes, 0))
    enforce_memory_limits(estimate, runtime)
    work = function (::Type{K}, left_numbers, right_numbers) where {K<:Union{UInt64,UInt128}}
        alpha_lookup, alpha_selected = _support_lookup(selected, alpha_expected, length(pair_block), K)
        beta_lookup, beta_selected = _support_lookup(selected, beta_expected, length(pair_block), K)
        selected == alpha_selected == beta_selected || throw(ArgumentError("alpha/beta support lookup selections differ"))
        alpha_support = Vector{CanonicalSpinStringPair}(undef, alpha_expected)
        beta_support = Vector{CanonicalSpinStringPair}(undef, beta_expected)
        row_counts = zeros(Int, alpha_expected)
        alpha_count = 0
        beta_count = 0

        @inbounds for left_index in 1:pair_block.n_left
            first_right = pair_block.diagonal ? left_index : 1
            last_right = Int(pair_block.last_right[left_index])
            first_right > last_right && continue
            left_alpha_bits = left_strings.alpha_bits[left_index]
            left_beta_bits = left_strings.beta_bits[left_index]
            for right_index in first_right:last_right
                right_alpha_bits = right_strings.alpha_bits[right_index]
                right_beta_bits = right_strings.beta_bits[right_index]
                alpha_key = _spin_pair_code(K, left_numbers.alpha[left_index], right_numbers.alpha[right_index])
                beta_key = _spin_pair_code(K, left_numbers.beta[left_index], right_numbers.beta[right_index])
                kappa = _lookup_id(alpha_lookup, alpha_key)
                if kappa == 0
                    alpha_count += 1
                    alpha_count <= alpha_expected || throw(ArgumentError("first-pass alpha support exceeds exact projection"))
                    alpha_support[alpha_count] = CanonicalSpinStringPair(left_alpha_bits, right_alpha_bits)
                    kappa = Int32(alpha_count)
                    _insert_lookup!(alpha_lookup, alpha_key, kappa)
                end
                if _lookup_id(beta_lookup, beta_key) == 0
                    beta_count += 1
                    beta_count <= beta_expected || throw(ArgumentError("first-pass beta support exceeds exact projection"))
                    beta_support[beta_count] = CanonicalSpinStringPair(left_beta_bits, right_beta_bits)
                    _insert_lookup!(beta_lookup, beta_key, beta_count)
                end
                row_counts[Int(kappa)] += 1
            end
        end
        alpha_count == alpha_expected || throw(ArgumentError("exact alpha support projection differs from first-pass support"))
        beta_count == beta_expected || throw(ArgumentError("exact beta support projection differs from first-pass support"))
        maximum_row_length = maximum(row_counts; init=0)
        exact_peak_estimate = combine_memory_estimates(_spin_pair_incidence_memory(T, length(pair_block), alpha_expected, beta_expected, maximum_row_length, selected, K), MemoryEstimate(numbering_bytes, 0))
        enforce_memory_limits(exact_peak_estimate, runtime)

        union_capacity = Base.Checked.checked_add(length(alpha_support), length(beta_support))
        union_support = Vector{CanonicalSpinStringPair}(undef, union_capacity)
        copyto!(union_support, 1, alpha_support, 1, length(alpha_support))
        union_count = length(alpha_support)
        beta_union_ids = Vector{Int32}(undef, length(beta_support))
        intersection_count = 0
        @inbounds for index in eachindex(beta_support)
            pair = beta_support[index]
            union_id = _lookup_id(alpha_lookup, _spin_pair_union_code(K, pair, left_numbers.numbers, right_numbers.numbers))
            if union_id == 0
                union_count += 1
                union_count <= union_capacity || throw(ArgumentError("union support exceeds exact alpha-plus-beta capacity"))
                union_support[union_count] = pair
                union_count <= typemax(Int32) || throw(OverflowError("union spin-pair support exceeds Int32 local-ID storage"))
                union_id = Int32(union_count)
            else
                intersection_count += 1
            end
            beta_union_ids[index] = union_id
        end
        resize!(union_support, union_count)
        rehashes = (alpha_lookup isa SpinPairFlatMap ? alpha_lookup.rehashes : 0) + (beta_lookup isa SpinPairFlatMap ? beta_lookup.rehashes : 0)
        support_domain = SpinPairSupportDomain(alpha_support, beta_support, union_support, Int32[], beta_union_ids, intersection_count, support_lookup, selected, rehashes)

        kappa_offsets = Vector{Int}(undef, length(alpha_support) + 1)
        kappa_offsets[1] = 1
        @inbounds for kappa in eachindex(row_counts)
            kappa_offsets[kappa + 1] = Base.Checked.checked_add(kappa_offsets[kappa], row_counts[kappa])
        end
        kappa_offsets[end] == length(pair_block) + 1 || throw(ArgumentError("CSR row counts differ from screened determinant-pair count"))
        lambda_ids = Vector{Int32}(undef, length(pair_block))
        values = Vector{T}(undef, length(pair_block))
        cursor = copy(@view kappa_offsets[1:end-1])
        raw_weight_sum = zero(T)
        left_coefficients = left_basis.expansion.coefficients
        right_coefficients = right_basis.expansion.coefficients
        left_phases = left_strings.determinant_phase
        right_phases = right_strings.determinant_phase
        @inbounds for left_index in 1:pair_block.n_left
            first_right = pair_block.diagonal ? left_index : 1
            last_right = Int(pair_block.last_right[left_index])
            first_right > last_right && continue
            left_alpha_bits = left_strings.alpha_bits[left_index]
            left_beta_bits = left_strings.beta_bits[left_index]
            left_coefficient = left_coefficients[left_index]
            left_phase = left_phases[left_index]
            for right_index in first_right:last_right
                kappa = _lookup_id(alpha_lookup, _spin_pair_code(K, left_numbers.alpha[left_index], right_numbers.alpha[right_index]))
                lambda = _lookup_id(beta_lookup, _spin_pair_code(K, left_numbers.beta[left_index], right_numbers.beta[right_index]))
                kappa > 0 && lambda > 0 || throw(ArgumentError("spin-pair support domain does not cover every screened determinant pair"))
                slot = cursor[Int(kappa)]
                multiplicity = pair_block.diagonal && left_index != right_index ? T(2) : one(T)
                phase = left_phase * right_phases[right_index]
                weight = multiplicity * left_coefficient * right_coefficients[right_index] * T(phase)
                isfinite(weight) || throw(ArgumentError("screened determinant-pair weight is not finite"))
                lambda_ids[slot] = Int32(lambda)
                values[slot] = weight
                cursor[Int(kappa)] = slot + 1
                raw_weight_sum += weight
            end
        end
        observed_maximum_row_length = _sort_injective_spin_pair_rows!(kappa_offsets, lambda_ids, values)
        observed_maximum_row_length == maximum_row_length || throw(ArgumentError("observed alpha CSR row differs from the pass-one exact count"))
        sparse_weights = SparseSpinPairWeights(length(alpha_support), length(beta_support), kappa_offsets, lambda_ids, values, length(pair_block), raw_weight_sum, sum(values; init=zero(T)))
        return PreparedSpinPairIncidence(support_domain, sparse_weights)
    end

    build_numbered_incidence = () -> work(key_type, _canonical_spin_numbers(left_strings, numbered), _canonical_spin_numbers(right_strings, numbered))
    incidence = isnothing(telemetry) ? build_numbered_incidence() : record_phase!(
        build_numbered_incidence,
        telemetry,
        :build_spin_pair_incidence;
        estimate,
        metadata=Dict(:strategy => :exact_presized_two_pass_injective_csr, :n_determinant_pairs => length(pair_block), :support_lookup => string(support_lookup), :selected_support_lookup => string(selected), :projected_alpha_support => alpha_expected, :projected_beta_support => beta_expected, :support_key_bits => 8 * sizeof(key_type), :canonical_spin_numbering => numbered, :canonical_spin_number_bytes => numbering_bytes),
        collect_allocations=runtime.collect_allocations,
    )
    if !isnothing(telemetry)
        support = incidence.support
        weights = incidence.weights
        increment_counter!(telemetry, :n_alpha_spin_pair_support, length(support.alpha_support))
        increment_counter!(telemetry, :n_beta_spin_pair_support, length(support.beta_support))
        increment_counter!(telemetry, :n_union_spin_pair_support, length(support.union_support))
        increment_counter!(telemetry, :n_shared_spin_pair_support, support.intersection_count)
        increment_counter!(telemetry, :n_raw_spin_pair_edges, weights.screened_pair_count)
        increment_counter!(telemetry, :n_sparse_spin_pair_edges, length(weights.weights))
        increment_counter!(telemetry, :n_implicit_alpha_union_maps, 1)
        increment_counter!(telemetry, :n_screened_pair_incidence_passes, 2)
        set_counter!(telemetry, :n_redundant_union_support_lookups_built, 0)
        set_counter!(telemetry, :n_retained_support_lookup_entries, 0)
        set_counter!(telemetry, :n_reduced_spin_pair_edges, 0)
    end
    return incidence
end

