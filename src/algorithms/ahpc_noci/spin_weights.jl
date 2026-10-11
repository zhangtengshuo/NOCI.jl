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

 File:            src/algorithms/ahpc_noci/spin_weights.jl
 Purpose:         Injective sparse spin-pair weight edges and duplicate-mapping diagnostics.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    NonInjectiveSpinPairMapping(kappa, lambda)

Report a duplicate sparse spin-pair edge that violates the production injective mapping contract.
The stored one-based support IDs are dimensionless, and construction allocates no scientific workspace.
"""
struct NonInjectiveSpinPairMapping <: Exception
    kappa::Int
    lambda::Int
end

"""
    showerror(io, error::NonInjectiveSpinPairMapping)

Render the duplicate `kappa`/`lambda` edge that invalidated production sparse assembly and return `nothing`.
The error is read-only, no scientific arrays or CUDA state are touched, and output mutation is limited to `io`.
"""
function Base.showerror(io::IO, error::NonInjectiveSpinPairMapping)
    print(io, "injective spin-pair mapping contains duplicate edge (kappa=", error.kappa, ", lambda=", error.lambda, ")")
    return nothing
end

"""
    SparseSpinPairWeights

Store the production sparse coefficient matrix `W[kappa, lambda]` in deterministic row-compressed form for one MEBF pair.
Rows are alpha support IDs, columns are beta support IDs, every screened determinant pair owns exactly one edge, arrays reside on CPU, and no matrix-element kernels or ERIs are retained.
"""
struct SparseSpinPairWeights{T<:AbstractFloat}
    n_kappa::Int
    n_lambda::Int
    kappa_offsets::Vector{Int}
    lambda_ids::Vector{Int32}
    weights::Vector{T}
    screened_pair_count::Int
    raw_weight_sum::T
    sparse_weight_sum::T

    """
        SparseSpinPairWeights(n_kappa, n_lambda, kappa_offsets, lambda_ids, weights, screened_pair_count, raw_weight_sum, sparse_weight_sum)

    Construct validated production row-compressed spin-pair weights while retaining owned builder arrays without copying them.
    Counts and weights are dimensionless, inputs are not mutated, no CUDA work occurs, and malformed offsets, IDs, non-finite values, or a noninjective edge count raise an exception.
    """
    function SparseSpinPairWeights(
        n_kappa::Integer,
        n_lambda::Integer,
        kappa_offsets::Vector{Int},
        lambda_ids::Vector{Int32},
        weights::Vector{T},
        screened_pair_count::Integer,
        raw_weight_sum::T,
        sparse_weight_sum::T,
    ) where {T<:AbstractFloat}
        kappa_count = Int(n_kappa)
        lambda_count = Int(n_lambda)
        pair_count = Int(screened_pair_count)
        kappa_count >= 0 && lambda_count >= 0 && pair_count >= 0 || throw(ArgumentError("spin-pair counts must be nonnegative"))
        length(kappa_offsets) == kappa_count + 1 || throw(DimensionMismatch("kappa offset length must equal n_kappa+1"))
        length(lambda_ids) == length(weights) || throw(DimensionMismatch("lambda ID and weight lengths differ"))
        first(kappa_offsets) == 1 || throw(ArgumentError("first kappa offset must be one"))
        last(kappa_offsets) == length(weights) + 1 || throw(ArgumentError("final kappa offset does not match sparse edge count"))
        all(index -> kappa_offsets[index] <= kappa_offsets[index + 1], 1:kappa_count) || throw(ArgumentError("kappa offsets must be nondecreasing"))
        lambda_count <= typemax(Int32) || throw(OverflowError("beta spin-pair support exceeds Int32 local-ID storage"))
        all(lambda -> 1 <= lambda <= lambda_count, lambda_ids) || throw(ArgumentError("lambda support ID lies outside 1:n_lambda"))
        pair_count == length(weights) || throw(ArgumentError("production sparse weights require one edge per screened determinant pair"))
        all(isfinite, weights) && isfinite(raw_weight_sum) && isfinite(sparse_weight_sum) || throw(ArgumentError("spin-pair weights contain a non-finite value"))
        return new{T}(kappa_count, lambda_count, kappa_offsets, lambda_ids, weights, pair_count, raw_weight_sum, sparse_weight_sum)
    end
end

"""
    _spin_pair_local_ids(reference, left, right, alpha_lookup, beta_lookup)

Resolve one screened determinant pair to its one-based alpha `kappa` and beta `lambda` support IDs.
Inputs are read-only and dimensionless, no allocation or device work occurs, and incomplete support domains raise `ArgumentError`.
"""
@inline function _spin_pair_local_ids(
    reference::DeterminantPairReference,
    left::CanonicalMEBFSpinStrings,
    right::CanonicalMEBFSpinStrings,
    alpha_lookup,
    beta_lookup,
)
    alpha_key = _spin_pair_code(CanonicalSpinStringPair(left.alpha_bits[reference.left], right.alpha_bits[reference.right]))
    beta_key = _spin_pair_code(CanonicalSpinStringPair(left.beta_bits[reference.left], right.beta_bits[reference.right]))
    kappa = _lookup_id(alpha_lookup, alpha_key)
    lambda = _lookup_id(beta_lookup, beta_key)
    kappa > 0 && lambda > 0 || throw(ArgumentError("spin-pair support domain does not cover every screened determinant pair"))
    return kappa, lambda
end

"""
    _screened_determinant_pair_weight(reference, left_basis, right_basis, left_strings, right_strings, diagonal)

Return one dimensionless CI weight including left/right canonicalization parities and the factor of two for a stored off-diagonal pair in a diagonal MEBF block.
Inputs are read-only, no allocation or CUDA work occurs, and non-finite floating-point overflow raises `ArgumentError`.
"""
@inline function _screened_determinant_pair_weight(
    reference::DeterminantPairReference,
    left_basis::MolecularBasisFunction{T},
    right_basis::MolecularBasisFunction{T},
    left_strings::CanonicalMEBFSpinStrings,
    right_strings::CanonicalMEBFSpinStrings,
    diagonal::Bool,
) where {T<:AbstractFloat}
    multiplicity = diagonal && reference.left != reference.right ? T(2) : one(T)
    phase = T(left_strings.determinant_phase[reference.left]) * T(right_strings.determinant_phase[reference.right])
    weight = multiplicity * left_basis.expansion.coefficients[reference.left] * right_basis.expansion.coefficients[reference.right] * phase
    isfinite(weight) || throw(ArgumentError("screened determinant-pair weight is not finite"))
    return weight
end

"""
    _validate_spin_weight_inputs(left_basis, right_basis, left_strings, right_strings, pair_block)

Validate determinant counts, active ranks, canonical string planes, and diagonal dimensions required by production sparse weight assembly and return `nothing`.
Inputs are read-only, no molecular-scale storage or CUDA state is allocated, and inconsistencies raise `DimensionMismatch` or `ArgumentError` before edge construction.
"""
function _validate_spin_weight_inputs(
    left_basis::MolecularBasisFunction,
    right_basis::MolecularBasisFunction,
    left_strings::CanonicalMEBFSpinStrings,
    right_strings::CanonicalMEBFSpinStrings,
    pair_block::RowCompressedDeterminantPairs,
)
    left_strings.active_rank == right_strings.active_rank || throw(DimensionMismatch("left/right canonical active ranks differ"))
    determinant_count(left_basis.expansion) == pair_block.n_left || throw(DimensionMismatch("left coefficient count differs from pair block"))
    determinant_count(right_basis.expansion) == pair_block.n_right || throw(DimensionMismatch("right coefficient count differs from pair block"))
    length(left_strings.alpha_bits) == pair_block.n_left || throw(DimensionMismatch("left canonical string count differs from pair block"))
    length(right_strings.alpha_bits) == pair_block.n_right || throw(DimensionMismatch("right canonical string count differs from pair block"))
    length(left_strings.beta_bits) == pair_block.n_left || throw(DimensionMismatch("left beta string count differs from pair block"))
    length(right_strings.beta_bits) == pair_block.n_right || throw(DimensionMismatch("right beta string count differs from pair block"))
    length(left_strings.determinant_phase) == pair_block.n_left || throw(DimensionMismatch("left determinant phase count differs from pair block"))
    length(right_strings.determinant_phase) == pair_block.n_right || throw(DimensionMismatch("right determinant phase count differs from pair block"))
    pair_block.diagonal && pair_block.n_left != pair_block.n_right && throw(ArgumentError("diagonal pair block must be square"))
    return nothing
end

"""
    _sort_injective_spin_pair_rows!(kappa_offsets, lambda_ids, weights)

Sort every sparse row by `lambda` ID in place, reject adjacent duplicates as a failed injectivity contract, and return the maximum row length.
Offsets are read-only while disjoint rows are mutated by Julia workers, each worker owns scratch bounded by its largest assigned row, no CUDA work occurs, and duplicates raise `NonInjectiveSpinPairMapping`.
"""
function _sort_injective_spin_pair_rows!(
    kappa_offsets::Vector{Int},
    lambda_ids::Vector{Int32},
    weights::Vector{T},
) where {T<:AbstractFloat}
    n_kappa = length(kappa_offsets) - 1
    n_kappa >= 0 || throw(ArgumentError("kappa offset vector must contain at least one entry"))
    n_kappa == 0 && return 0
    worker_count = min(Threads.nthreads(), n_kappa)
    ranges = _balanced_auxiliary_ranges(n_kappa, worker_count)
    tasks = map(ranges) do kappa_range
        Threads.@spawn begin
            scratch = Tuple{Int32,T}[]
            local_maximum_row_length = 0
            failure = nothing
            try
                @inbounds for kappa in kappa_range
                    first_slot = kappa_offsets[kappa]
                    last_slot = kappa_offsets[kappa + 1] - 1
                    row_length = last_slot - first_slot + 1
                    local_maximum_row_length = max(local_maximum_row_length, row_length)
                    row_length <= 1 && continue
                    resize!(scratch, row_length)
                    for index in 1:row_length
                        slot = first_slot + index - 1
                        scratch[index] = (lambda_ids[slot], weights[slot])
                    end
                    sort!(scratch; by=first)
                    previous_lambda = Int32(0)
                    for index in 1:row_length
                        lambda, weight = scratch[index]
                        index > 1 && lambda == previous_lambda && throw(NonInjectiveSpinPairMapping(kappa, lambda))
                        slot = first_slot + index - 1
                        lambda_ids[slot] = lambda
                        weights[slot] = weight
                        previous_lambda = lambda
                    end
                end
            catch error
                failure = error
            end
            return local_maximum_row_length, failure
        end
    end
    maximum_row_length = 0
    for task in tasks
        local_maximum_row_length, failure = fetch(task)
        isnothing(failure) || throw(failure)
        maximum_row_length = max(maximum_row_length, local_maximum_row_length)
    end
    return maximum_row_length
end
