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

 File:            src/core/pairs.jl
 Purpose:         Screened determinant-pair references and row-cutoff pairlist construction.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    DeterminantPairReference(left, right)

Identify one pair of one-based determinant columns in a left/right MEBF pair.
The immutable indices allocate no scientific storage and construction rejects nonpositive values.
"""
struct DeterminantPairReference
    left::Int32
    right::Int32

    """
        DeterminantPairReference(left, right)

    Construct a validated one-based determinant-pair reference.
    Inputs are scalar and unmodified, no CPU/GPU arrays are allocated, and nonpositive indices raise `ArgumentError`.
    """
    function DeterminantPairReference(left::Integer, right::Integer)
        left >= 1 && right >= 1 || throw(ArgumentError("determinant-pair indices must be positive"))
        left <= typemax(Int32) && right <= typemax(Int32) || throw(OverflowError("determinant-pair index exceeds Int32 production storage"))
        return new(Int32(left), Int32(right))
    end
end

"""
    RowCompressedDeterminantPairs

Store a coefficient-screened determinant-pair block only as the last retained right index for every left row.
The production representation is exact for magnitude-sorted coefficients, uses one `Int32` cutoff per left determinant, supports linear row-major replay without random indexing, and owns CPU integer arrays only.
"""
struct RowCompressedDeterminantPairs
    n_left::Int
    n_right::Int
    diagonal::Bool
    last_right::Vector{Int32}
    total_count::Int

    """
        RowCompressedDeterminantPairs(n_left, n_right, diagonal, last_right, total_count)

    Construct and validate a cutoff-only row-compressed determinant-pair block, copying and narrowing its cutoff array to `Int32`.
    Inputs are not mutated, no flattened offsets, floating-point arrays, or CUDA storage are allocated, and inconsistent dimensions, bounds, overflow, or totals raise an exception.
    """
    function RowCompressedDeterminantPairs(
        n_left::Integer,
        n_right::Integer,
        diagonal::Bool,
        last_right::AbstractVector{<:Integer},
        total_count::Integer,
    )
        left_count = Int(n_left)
        right_count = Int(n_right)
        left_count >= 1 && right_count >= 1 || throw(ArgumentError("pair-block determinant dimensions must be positive"))
        diagonal && left_count != right_count && throw(DimensionMismatch("diagonal pair block must be square"))
        left_count <= typemax(Int32) && right_count <= typemax(Int32) || throw(OverflowError("determinant count exceeds Int32 production storage"))
        final_indices = Int32.(last_right)
        length(final_indices) == left_count || throw(DimensionMismatch("last_right length must equal n_left"))
        total = Int(total_count)
        total >= 0 || throw(ArgumentError("total_count must be nonnegative"))
        counted = 0
        for left in 1:left_count
            first_right = diagonal ? left : 1
            first_right - 1 <= final_indices[left] <= right_count || throw(ArgumentError("last_right[$left] lies outside its valid row interval"))
            row_count = max(0, final_indices[left] - first_right + 1)
            counted = Base.Checked.checked_add(counted, row_count)
        end
        counted == total || throw(ArgumentError("row cutoffs are inconsistent with total_count"))
        return new(left_count, right_count, diagonal, final_indices, total)
    end
end

"""
    length(pairs::RowCompressedDeterminantPairs)

Return the number of screened determinant pairs represented by a row-compressed block.
The query is allocation-free, does not mutate the block, and has no CPU/GPU side effects.
"""
function Base.length(pairs::RowCompressedDeterminantPairs)
    return pairs.total_count
end

"""
    eltype(RowCompressedDeterminantPairs)

Return `DeterminantPairReference` as the concrete element type of cutoff-only pair iteration.
The type query allocates nothing, mutates no input, and has no CPU/GPU side effects.
"""
Base.eltype(::Type{RowCompressedDeterminantPairs}) = DeterminantPairReference

"""
    IteratorEltype(RowCompressedDeterminantPairs)

Declare that cutoff-only pair iteration has a known concrete element type.
The trait query allocates nothing, mutates no input, and has no CPU/GPU side effects.
"""
Base.IteratorEltype(::Type{RowCompressedDeterminantPairs}) = Base.HasEltype()

"""
    IteratorSize(RowCompressedDeterminantPairs)

Declare that cutoff-only pair iteration has the exact finite length returned by `length`.
The trait query allocates nothing, mutates no input, and has no CPU/GPU side effects.
"""
Base.IteratorSize(::Type{RowCompressedDeterminantPairs}) = Base.HasLength()

"""
    iterate(pairs::RowCompressedDeterminantPairs[, state])

Iterate screened determinant references directly from row cutoffs in deterministic left-row then right-column order.
Iteration is read-only, strictly linear without flattened random indexing or binary searches, allocates only returned immutable references, and touches no CUDA state.
"""
function Base.iterate(pairs::RowCompressedDeterminantPairs)
    return Base.iterate(pairs, (1, 1))
end

"""
    iterate(pairs::RowCompressedDeterminantPairs, state)

Continue cutoff-only row-major determinant-pair replay from the `(left,right)` state returned by the preceding iteration.
The block and state are read-only, empty rows are skipped in linear total work, no heap or CUDA storage is allocated beyond the returned reference, and a terminal state returns `nothing`.
"""
function Base.iterate(pairs::RowCompressedDeterminantPairs, state::Tuple{Int,Int})
    left, right = state
    while left <= pairs.n_left
        first_right = pairs.diagonal ? left : 1
        right = max(right, first_right)
        final_right = Int(pairs.last_right[left])
        if right <= final_right
            return DeterminantPairReference(left, right), (left, right + 1)
        end
        left += 1
        right = pairs.diagonal ? left : 1
    end
    return nothing
end

"""
    DeterminantPairGroup(left_mebf, right_mebf, pairs)

Associate one lower-triangular MEBF matrix position with its screened determinant-pair block.
The group retains the row-compressed block without copying and contains no scientific floating-point or device storage.
"""
struct DeterminantPairGroup
    left_mebf::Int
    right_mebf::Int
    pairs::RowCompressedDeterminantPairs

    """
        DeterminantPairGroup(left_mebf, right_mebf, pairs)

    Construct a lower-triangular MEBF-pair group and validate its diagonal flag.
    Scalar inputs and the pair block are retained without scientific copies, no CUDA work occurs, and invalid indices or diagonal metadata raise `ArgumentError`.
    """
    function DeterminantPairGroup(left_mebf::Integer, right_mebf::Integer, pairs::RowCompressedDeterminantPairs)
        1 <= right_mebf <= left_mebf || throw(ArgumentError("MEBF-pair indices must identify a lower-triangular block"))
        pairs.diagonal == (left_mebf == right_mebf) || throw(ArgumentError("pair-block diagonal flag disagrees with MEBF indices"))
        return new(Int(left_mebf), Int(right_mebf), pairs)
    end
end

"""
    DeterminantPairDomain

Store all lower-triangular MEBF-pair groups together with full and screened determinant-pair counts.
Groups are ordered by left MEBF and then right MEBF, arrays reside on CPU, and the object contains no orbital or integral data.
"""
struct DeterminantPairDomain
    groups::Vector{DeterminantPairGroup}
    full_count::Int
    screened_count::Int

    """
        DeterminantPairDomain(groups, full_count, screened_count)

    Construct a validated determinant-pair domain and copy its group vector.
    Inputs are not mutated, pair index arrays are retained, no CUDA storage is allocated, and inconsistent screened or full counts raise `ArgumentError`.
    """
    function DeterminantPairDomain(
        groups::AbstractVector{DeterminantPairGroup},
        full_count::Integer,
        screened_count::Integer,
    )
        full = Int(full_count)
        screened = Int(screened_count)
        full >= 0 && 0 <= screened <= full || throw(ArgumentError("determinant-pair counts must satisfy 0 <= screened <= full"))
        sum(length(group.pairs) for group in groups) == screened || throw(ArgumentError("group pair counts do not sum to screened_count"))
        return new(collect(groups), full, screened)
    end
end

"""
    OccupiedSpinOrbitals

Store occupied alpha/beta molecular orbital matrices and their source MEBF column indices for one determinant.
Matrices and index vectors are CPU-owned, columns follow inactive, closed-active, then open-active order, and no CUDA storage is retained.
"""
struct OccupiedSpinOrbitals{T<:AbstractFloat}
    alpha::Matrix{T}
    beta::Matrix{T}
    alpha_columns::Vector{Int}
    beta_columns::Vector{Int}
    active_occupation::Vector{Int8}
end

"""
    PreparedDeterminantPair

Store left/right occupied spin orbitals and their alpha/beta occupied cross-overlap matrices for one determinant pair.
All arrays are CPU-owned, overlap values are dimensionless, and the object contains no inverse, rank classification, SVD, ERI, or CUDA state.
"""
struct PreparedDeterminantPair{T<:AbstractFloat}
    reference::DeterminantPairReference
    left::OccupiedSpinOrbitals{T}
    right::OccupiedSpinOrbitals{T}
    alpha_cross_overlap::Matrix{T}
    beta_cross_overlap::Matrix{T}
end

"""
    _coefficients_magnitude_sorted(coefficients)

Return whether coefficient magnitudes are finite and monotonically nonincreasing.
The vector is read-only, the query allocates no scientific storage or CUDA state, and NaN values yield `false`.
"""
function _coefficients_magnitude_sorted(coefficients::AbstractVector{<:AbstractFloat})
    isempty(coefficients) && return true
    isfinite(first(coefficients)) || return false
    for index in 2:length(coefficients)
        isfinite(coefficients[index]) || return false
        abs(coefficients[index - 1]) >= abs(coefficients[index]) || return false
    end
    return true
end

"""
    _last_screened_right(coefficients, coefficient_left, threshold, first_right)

Return the final retained right determinant index in a magnitude-sorted row using binary search, or `first_right-1` for an empty row.
Inputs are read-only, coefficients and threshold are dimensionless, the query allocates nothing, and callers must validate ordering and bounds.
"""
function _last_screened_right(
    coefficients::AbstractVector{T},
    coefficient_left::T,
    threshold::T,
    first_right::Integer,
) where {T<:AbstractFloat}
    last_index = length(coefficients)
    first_right > last_index && return Int(first_right) - 1
    coefficient_left * abs(coefficients[first_right]) >= threshold || return Int(first_right) - 1
    coefficient_left * abs(coefficients[last_index]) >= threshold && return last_index
    low = Int(first_right)
    high = last_index
    while low < high
        middle = (low + high + 1) >>> 1
        if coefficient_left * abs(coefficients[middle]) >= threshold
            low = middle
        else
            high = middle - 1
        end
    end
    return low
end

"""
    _fill_pair_cutoff_range!(last_right, left_coefficients, right_coefficients, threshold, diagonal, left_range)

Fill one disjoint left-row range of a compressed determinant-pair block and return its retained pair count.
Coefficient vectors are read-only, each caller owns the destination rows it mutates, row traversal and binary searches are deterministic, no CUDA work occurs, and checked summation rejects integer overflow.
"""
function _fill_pair_cutoff_range!(
    last_right::Vector{Int32},
    left_coefficients::AbstractVector{T},
    right_coefficients::AbstractVector{T},
    threshold::T,
    diagonal::Bool,
    left_range::UnitRange{Int},
) where {T<:AbstractFloat}
    total_count = 0
    @inbounds for left in left_range
        first_right = diagonal ? left : 1
        final_right = _last_screened_right(right_coefficients, abs(left_coefficients[left]), threshold, first_right)
        last_right[left] = Int32(final_right)
        total_count = Base.Checked.checked_add(total_count, max(0, final_right - first_right + 1))
    end
    return total_count
end

"""
    _build_compressed_determinant_pairs(left_coefficients, right_coefficients, threshold; diagonal)

Build an exact row-compressed screening block for magnitude-sorted MEBF coefficients using `abs(C_left*C_right) >= threshold` with disjoint left-row work on Julia threads.
Inputs are read-only, only `O(n_left)` CPU integer storage plus small task metadata are allocated, no CUDA work occurs, and invalid thresholds, dimensions, or coefficient ordering raise an exception.
"""
function _build_compressed_determinant_pairs(
    left_coefficients::AbstractVector{T},
    right_coefficients::AbstractVector{T},
    threshold::T;
    diagonal::Bool,
) where {T<:AbstractFloat}
    isfinite(threshold) && threshold >= zero(T) || throw(ArgumentError("determinant-pair threshold must be finite and nonnegative"))
    isempty(left_coefficients) && throw(ArgumentError("left coefficient vector must not be empty"))
    isempty(right_coefficients) && throw(ArgumentError("right coefficient vector must not be empty"))
    diagonal && length(left_coefficients) != length(right_coefficients) && throw(DimensionMismatch("diagonal pair block coefficient lengths differ"))
    _coefficients_magnitude_sorted(left_coefficients) || throw(ArgumentError("left coefficients must be sorted by nonincreasing magnitude"))
    _coefficients_magnitude_sorted(right_coefficients) || throw(ArgumentError("right coefficients must be sorted by nonincreasing magnitude"))
    n_left = length(left_coefficients)
    n_right = length(right_coefficients)
    n_right <= typemax(Int32) || throw(OverflowError("right determinant count exceeds Int32 cutoff storage"))
    last_right = Vector{Int32}(undef, n_left)
    worker_count = min(Threads.nthreads(), n_left)
    ranges = _balanced_auxiliary_ranges(n_left, worker_count)
    tasks = map(ranges) do left_range
        Threads.@spawn _fill_pair_cutoff_range!(last_right, left_coefficients, right_coefficients, threshold, diagonal, left_range)
    end
    total_count = 0
    for task in tasks
        total_count = Base.Checked.checked_add(total_count, fetch(task))
    end
    return RowCompressedDeterminantPairs(n_left, n_right, diagonal, last_right, total_count)
end

"""
    full_determinant_pair_count(basis_functions)

Return the unscreened determinant-pair count over the lower-triangular MEBF matrix, using triangular counts on diagonal blocks.
Inputs are read-only, checked integer arithmetic prevents silent overflow, no storage or CUDA state is allocated, and an empty collection returns zero.
"""
function full_determinant_pair_count(basis_functions::AbstractVector{<:MolecularBasisFunction})
    total = 0
    for left_mebf in eachindex(basis_functions), right_mebf in 1:left_mebf
        n_left = determinant_count(basis_functions[left_mebf].expansion)
        n_right = determinant_count(basis_functions[right_mebf].expansion)
        block_count = if left_mebf == right_mebf
            div(Base.Checked.checked_mul(n_left, Base.Checked.checked_add(n_left, 1)), 2)
        else
            Base.Checked.checked_mul(n_left, n_right)
        end
        total = Base.Checked.checked_add(total, block_count)
    end
    return total
end

"""
    determinant_pair_domain_memory(basis_functions)

Estimate host bytes for one `Int32` row cutoff per left determinant in every lower-triangular MEBF-pair block; device bytes are zero.
The estimate is linear in left determinant counts, inputs are read-only, no pair references or CUDA arrays are materialized, and checked arithmetic detects overflow.
"""
function determinant_pair_domain_memory(basis_functions::AbstractVector{<:MolecularBasisFunction})
    integer_count = 0
    for left_mebf in eachindex(basis_functions), _ in 1:left_mebf
        n_left = determinant_count(basis_functions[left_mebf].expansion)
        integer_count = Base.Checked.checked_add(integer_count, n_left)
    end
    return MemoryEstimate(estimate_array_bytes(Int32, integer_count), 0)
end

"""
    build_determinant_pair_domain(basis_functions, screening; runtime=RuntimeOptions(), telemetry=nothing)

Build deterministic row-compressed determinant-pair blocks for every lower-triangular MEBF pair using diagonal/off-diagonal CI thresholds.
Basis functions are read-only, host memory is preflighted before index allocation, optional telemetry records timing/allocations and pair counts, CUDA is not initialized, and invalid coefficient ordering raises `ArgumentError`.
"""
function build_determinant_pair_domain(
    basis_functions::AbstractVector{<:MolecularBasisFunction{T}},
    screening::ScreeningOptions{T};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    estimate = determinant_pair_domain_memory(basis_functions)
    enforce_memory_limits(estimate, runtime)
    work = function ()
        groups = DeterminantPairGroup[]
        screened_count = 0
        for left_mebf in eachindex(basis_functions), right_mebf in 1:left_mebf
            left_coefficients = basis_functions[left_mebf].expansion.coefficients
            right_coefficients = basis_functions[right_mebf].expansion.coefficients
            diagonal = left_mebf == right_mebf
            threshold = diagonal ? screening.tau_ci : screening.tau_ci_offdiagonal
            pairs = _build_compressed_determinant_pairs(left_coefficients, right_coefficients, threshold; diagonal)
            push!(groups, DeterminantPairGroup(left_mebf, right_mebf, pairs))
            screened_count = Base.Checked.checked_add(screened_count, length(pairs))
        end
        return DeterminantPairDomain(groups, full_determinant_pair_count(basis_functions), screened_count)
    end
    domain = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :build_determinant_pair_domain;
        estimate,
        metadata=Dict(:n_mebf_pairs => div(length(basis_functions) * (length(basis_functions) + 1), 2)),
        collect_allocations=runtime.collect_allocations,
    )
    if !isnothing(telemetry)
        set_counter!(telemetry, :n_determinant_pairs, domain.screened_count)
        set_counter!(telemetry, :n_full_determinant_pairs, domain.full_count)
    end
    return domain
end

"""
    occupied_orbital_columns(expansion, determinant)

Return occupied alpha and beta orbital-column indices for one molecular determinant, ordering inactive, doubly occupied active, then singly occupied active columns.
The expansion is read-only, returned CPU vectors are owned, no CUDA work occurs, and invalid determinant indices or occupation codes raise an exception.
"""
function occupied_orbital_columns(expansion::DeterminantExpansion, determinant::Integer)
    1 <= determinant <= determinant_count(expansion) || throw(BoundsError(1:determinant_count(expansion), determinant))
    inactive_columns = collect(1:expansion.inactive_orbitals)
    closed_columns = Int[]
    alpha_open_columns = Int[]
    beta_open_columns = Int[]
    for active in 1:active_orbital_count(expansion)
        value = expansion.active_occupations[active, determinant]
        column = expansion.inactive_orbitals + active
        value == 2 && push!(closed_columns, column)
        value == 1 && push!(alpha_open_columns, column)
        value == -1 && push!(beta_open_columns, column)
        value in (-1, 0, 1, 2) || throw(ArgumentError("unsupported active occupation value $value"))
    end
    return vcat(inactive_columns, closed_columns, alpha_open_columns), vcat(inactive_columns, closed_columns, beta_open_columns)
end

"""
    occupied_spin_orbitals(basis_function, determinant)

Materialize occupied alpha/beta orbital matrices and source columns for one determinant of a molecular basis function.
The basis function is read-only, returned CPU arrays are owned, no CUDA state is initialized, and inconsistent orbital dimensions or invalid indices raise an exception.
"""
function occupied_spin_orbitals(basis_function::MolecularBasisFunction{T}, determinant::Integer) where {T<:AbstractFloat}
    alpha_columns, beta_columns = occupied_orbital_columns(basis_function.expansion, determinant)
    maximum_column = max(maximum(alpha_columns; init=0), maximum(beta_columns; init=0))
    maximum_column <= size(basis_function.orbital_coefficients, 2) || throw(DimensionMismatch("MEBF orbital matrix is too small for determinant occupations"))
    occupation = Vector{Int8}(@view basis_function.expansion.active_occupations[:, determinant])
    return OccupiedSpinOrbitals(
        Matrix(@view basis_function.orbital_coefficients[:, alpha_columns]),
        Matrix(@view basis_function.orbital_coefficients[:, beta_columns]),
        alpha_columns,
        beta_columns,
        occupation,
    )
end

"""
    prepare_determinant_pair(left_basis, right_basis, reference, overlap)

Materialize occupied spin orbitals and compute dimensionless alpha/beta occupied cross-overlaps `C_left' * S_AO * C_right` for one determinant pair.
Inputs are read-only, returned arrays reside on CPU, no inverse, rank classifier, SVD, ERI, or CUDA operation is performed, and electron/basis dimension mismatches raise `DimensionMismatch`.
"""
function prepare_determinant_pair(
    left_basis::MolecularBasisFunction{T},
    right_basis::MolecularBasisFunction{T},
    reference::DeterminantPairReference,
    overlap::AbstractMatrix{T},
) where {T<:AbstractFloat}
    n_basis = size(overlap, 1)
    size(overlap, 2) == n_basis || throw(DimensionMismatch("AO overlap matrix must be square"))
    size(left_basis.orbital_coefficients, 1) == n_basis || throw(DimensionMismatch("left MEBF basis dimension differs from AO overlap"))
    size(right_basis.orbital_coefficients, 1) == n_basis || throw(DimensionMismatch("right MEBF basis dimension differs from AO overlap"))
    left = occupied_spin_orbitals(left_basis, reference.left)
    right = occupied_spin_orbitals(right_basis, reference.right)
    size(left.alpha, 2) == size(right.alpha, 2) || throw(DimensionMismatch("left/right alpha electron counts differ"))
    size(left.beta, 2) == size(right.beta, 2) || throw(DimensionMismatch("left/right beta electron counts differ"))
    alpha_cross_overlap = left.alpha' * overlap * right.alpha
    beta_cross_overlap = left.beta' * overlap * right.beta
    return PreparedDeterminantPair(reference, left, right, alpha_cross_overlap, beta_cross_overlap)
end
