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

 File:            src/integrals/values.jl
 Purpose:         PackedERIValues and ERIFactorValues with canonical pair indexing and reconstruction.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    canonical_pair_count(n_orbitals)

Return `n_orbitals * (n_orbitals + 1) / 2` with checked integer arithmetic.
The orbital count must be nonnegative; the function allocates no integral storage.
"""
function canonical_pair_count(n_orbitals::Integer)
    n = Int(n_orbitals)
    n >= 0 || throw(ArgumentError("n_orbitals must be nonnegative"))
    product = Base.Checked.checked_mul(n, Base.Checked.checked_add(n, 1))
    return div(product, 2)
end

"""
    canonical_pair_index(first_orbital, second_orbital)

Return the one-based upper-triangular canonical index for an unordered positive orbital pair.
The function is symmetric in its arguments and allocates no storage.
"""
function canonical_pair_index(first_orbital::Integer, second_orbital::Integer)
    first_orbital >= 1 || throw(ArgumentError("orbital indices must be positive"))
    second_orbital >= 1 || throw(ArgumentError("orbital indices must be positive"))
    lower = min(Int(first_orbital), Int(second_orbital))
    upper = max(Int(first_orbital), Int(second_orbital))
    offset = div(Base.Checked.checked_mul(upper - 1, upper), 2)
    return Base.Checked.checked_add(offset, lower)
end

"""
    _packed_symmetric_index(first_index, second_index)

Return the one-based upper-triangular index for an unordered positive pair of canonical-pair indices.
This allocation-free helper assumes only positive inputs and uses checked arithmetic through `canonical_pair_index`.
"""
@inline function _packed_symmetric_index(first_index::Integer, second_index::Integer)
    return canonical_pair_index(first_index, second_index)
end

"""
    PackedERIValues(n_orbitals, values)

Store physical canonical electron-repulsion integrals `(pq|rs)` in upper-triangular pair-of-pairs order.
Values contain no `.two` file multiplicity weights; construction copies the input and validates its exact length without performing file I/O.
"""
struct PackedERIValues{T<:AbstractFloat}
    n_orbitals::Int
    values::Vector{T}

    """
        PackedERIValues{T}(n_orbitals, values)

    Construct packed ERIs through the non-bypassable validated inner path and copy the input values.
    Invalid size or non-finite values raise before the object is returned.
    """
    function PackedERIValues{T}(n_orbitals::Integer, values::AbstractVector{T}) where {T<:AbstractFloat}
        n = Int(n_orbitals)
        n_pairs = canonical_pair_count(n)
        expected = canonical_pair_count(n_pairs)
        length(values) == expected || throw(DimensionMismatch(
            "packed ERI length $(length(values)) does not match expected $expected for $n orbitals",
        ))
        all(isfinite, values) || throw(ArgumentError("packed ERI values contain a non-finite value"))
        return new{T}(n, Vector{T}(values))
    end
end

"""
    PackedERIValues(n_orbitals, values)

Construct validated physical packed ERI values from an abstract floating-point vector.
The input is not mutated; invalid size or non-finite values raise before the object is returned.
"""
function PackedERIValues(n_orbitals::Integer, values::AbstractVector{T}) where {T<:AbstractFloat}
    return PackedERIValues{T}(n_orbitals, values)
end

"""
    ERIFactorValues(n_orbitals, coefficients)

Store symmetric metric-whitened ERI factors in canonical-orbital-pair rows and auxiliary-vector columns.
Construction copies coefficients, requires at least one auxiliary vector, and performs no CUDA initialization.
"""
struct ERIFactorValues{T<:AbstractFloat}
    n_orbitals::Int
    coefficients::Matrix{T}

    """
        ERIFactorValues{T}(n_orbitals, coefficients)

    Construct metric-whitened ERI factors through the non-bypassable validated inner path and copy the coefficients.
    Invalid dimensions or non-finite values raise before the object is returned.
    """
    function ERIFactorValues{T}(
        n_orbitals::Integer,
        coefficients::AbstractMatrix{T},
    ) where {T<:AbstractFloat}
        n = Int(n_orbitals)
        expected_pairs = canonical_pair_count(n)
        size(coefficients, 1) == expected_pairs || throw(DimensionMismatch(
            "ERI factor row count $(size(coefficients, 1)) does not match expected $expected_pairs",
        ))
        size(coefficients, 2) >= 1 || throw(ArgumentError("metric-whitened ERI factors require at least one auxiliary vector"))
        all(isfinite, coefficients) || throw(ArgumentError("metric-whitened ERI factors contain a non-finite value"))
        return new{T}(n, Matrix{T}(coefficients))
    end

    """
        ERIFactorValues{T}(n_orbitals, coefficients, Val(:validated_owned))

    Transfer a canonical pair-by-auxiliary matrix already dimension-checked and finite-validated by a versioned internal loader.
    The concrete CPU matrix is retained without copying or rescanning, no CUDA state is initialized, and this constructor is unavailable to abstract external storage.
    """
    function ERIFactorValues{T}(
        n_orbitals::Integer,
        coefficients::Matrix{T},
        ::Val{:validated_owned},
    ) where {T<:AbstractFloat}
        return new{T}(Int(n_orbitals), coefficients)
    end
end

"""
    ERIFactorValues(n_orbitals, coefficients)

Construct validated metric-whitened ERI factors with shape `(canonical_pair_count(n_orbitals), n_cholesky)`.
The input is not mutated; dimension and finite-value errors are raised before use.
"""
function ERIFactorValues(n_orbitals::Integer, coefficients::AbstractMatrix{T}) where {T<:AbstractFloat}
    return ERIFactorValues{T}(n_orbitals, coefficients)
end

"""
    _eri_value(integrals, p, q, r, s)

Return the physical `(pq|rs)` value from canonical packed storage.
All orbital indices are one based and checked against `integrals.n_orbitals`; the function allocates no storage.
"""
@inline function _eri_value(
    integrals::PackedERIValues,
    p::Integer,
    q::Integer,
    r::Integer,
    s::Integer,
)
    n = integrals.n_orbitals
    1 <= p <= n || throw(BoundsError(1:n, p))
    1 <= q <= n || throw(BoundsError(1:n, q))
    1 <= r <= n || throw(BoundsError(1:n, r))
    1 <= s <= n || throw(BoundsError(1:n, s))
    first_pair = canonical_pair_index(p, q)
    second_pair = canonical_pair_index(r, s)
    return integrals.values[_packed_symmetric_index(first_pair, second_pair)]
end

"""
    _eri_factors_gram_block_rows(T, n_pairs, n_auxiliary, runtime)

Return a positive row-block size whose CPU or CUDA Gram workspaces fit the configured generic scratch budget.
Counts are dimensionless, no array is allocated, and zero or insufficient scratch raises `ResourceLimitError` before four-index materialization.
"""
function _eri_factors_gram_block_rows(
    ::Type{T},
    n_pairs::Integer,
    n_auxiliary::Integer,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    row_elements = runtime.use_gpu ? Int(n_pairs) + Int(n_auxiliary) : Int(n_pairs)
    row_bytes = Base.Checked.checked_mul(sizeof(T), row_elements)
    runtime.scratch_bytes >= row_bytes || throw(ResourceLimitError(:host, row_bytes, runtime.scratch_bytes))
    return min(Int(n_pairs), max(1, div(runtime.scratch_bytes, row_bytes)))
end

"""
    packed_eri_from_factors_memory(T, n_orbitals, n_auxiliary; runtime=RuntimeOptions())

Estimate the packed four-index output plus one blocked Cholesky Gram workspace for CPU or CUDA execution.
Inputs are dimensions only, units are bytes, factors already resident on the host are excluded, and invalid or overflowing dimensions raise before allocation.
"""
function packed_eri_from_factors_memory(
    ::Type{T},
    n_orbitals::Integer,
    n_auxiliary::Integer;
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    n_orbitals >= 1 || throw(ArgumentError("orbital count must be positive"))
    n_auxiliary >= 1 || throw(ArgumentError("auxiliary count must be positive"))
    n_pairs = canonical_pair_count(n_orbitals)
    block_rows = _eri_factors_gram_block_rows(T, n_pairs, n_auxiliary, runtime)
    output_bytes = estimate_array_bytes(T, canonical_pair_count(n_pairs))
    gram_bytes = estimate_array_bytes(T, block_rows, n_pairs)
    if runtime.use_gpu
        factor_bytes = estimate_array_bytes(T, n_pairs, n_auxiliary)
        row_bytes = estimate_array_bytes(T, block_rows, n_auxiliary)
        return MemoryEstimate(Base.Checked.checked_add(output_bytes, gram_bytes), sum((factor_bytes, row_bytes, gram_bytes)))
    end
    return MemoryEstimate(Base.Checked.checked_add(output_bytes, gram_bytes), 0)
end

"""
    _store_eri_factors_gram_block!(values, gram, first_row)

Copy the upper pair-of-pairs triangle represented by a dense Gram row block into canonical packed storage.
The function mutates `values`, leaves `gram` unchanged, returns the destination, and assumes physical unweighted integral values.
"""
function _store_eri_factors_gram_block!(
    values::Vector{T},
    gram::AbstractMatrix{T},
    first_row::Integer,
) where {T<:AbstractFloat}
    n_pairs = size(gram, 2)
    @inbounds for local_row in axes(gram, 1)
        first_pair = Int(first_row) + local_row - 1
        for second_pair in first_pair:n_pairs
            values[canonical_pair_index(first_pair, second_pair)] = gram[local_row, second_pair]
        end
    end
    return values
end

"""
    _packed_eri_from_factors_cpu(factors, block_rows)

Materialize physical packed four-index ERIs with blocked multithreaded-BLAS Gram products.
Factors are read-only, the returned vector is CPU-owned and unweighted, and CPU parallelism is controlled by the active BLAS thread count.
"""
function _packed_eri_from_factors_cpu(factors::ERIFactorValues{T}, block_rows::Integer) where {T<:AbstractFloat}
    coefficients = factors.coefficients
    n_pairs = size(coefficients, 1)
    values = Vector{T}(undef, canonical_pair_count(n_pairs))
    gram = Matrix{T}(undef, Int(block_rows), n_pairs)
    for first_row in 1:Int(block_rows):n_pairs
        last_row = min(n_pairs, first_row + Int(block_rows) - 1)
        rows = last_row - first_row + 1
        destination = view(gram, 1:rows, :)
        mul!(destination, view(coefficients, first_row:last_row, :), transpose(coefficients))
        _store_eri_factors_gram_block!(values, destination, first_row)
    end
    return PackedERIValues{T}(factors.n_orbitals, values)
end

"""
    _packed_eri_from_factors_gpu(factors, block_rows, gpu_device)

Materialize physical packed four-index ERIs with one resident factor matrix and blocked cuBLAS Gram products.
The returned ERIs are CPU-owned, CUDA scratch is released in `finally`, and unavailable devices or CUDA failures raise without a CPU fallback.
"""
function _packed_eri_from_factors_gpu(
    factors::ERIFactorValues{T},
    block_rows::Integer,
    gpu_device::Integer,
) where {T<:Union{Float32,Float64}}
    _cuda_module().functional() || throw(GPUUnavailableError(Int(gpu_device), "CUDA is not functional for four-index ERI-factor materialization"))
    devices = collect(_cuda_module().devices())
    gpu_device < length(devices) || throw(GPUUnavailableError(Int(gpu_device), "only $(length(devices)) CUDA devices are visible"))
    _cuda_module().device!(Int(gpu_device))
    n_pairs, n_auxiliary = size(factors.coefficients)
    values = Vector{T}(undef, canonical_pair_count(n_pairs))
    host_gram = Matrix{T}(undef, Int(block_rows), n_pairs)
    factor_device = nothing
    row_device = nothing
    gram_device = nothing
    try
        factor_device = _cuda_module().CuArray(factors.coefficients)
        row_device = _cuda_module().zeros(T, Int(block_rows), n_auxiliary)
        gram_device = _cuda_module().zeros(T, Int(block_rows), n_pairs)
        for first_row in 1:Int(block_rows):n_pairs
            last_row = min(n_pairs, first_row + Int(block_rows) - 1)
            rows = last_row - first_row + 1
            copyto!(view(row_device, 1:rows, :), view(factor_device, first_row:last_row, :))
            _cuda_module().CUBLAS.gemm!(
                'N', 'T', one(T), view(row_device, 1:rows, :), factor_device,
                zero(T), view(gram_device, 1:rows, :),
            )
            copyto!(host_gram, gram_device)
            _store_eri_factors_gram_block!(values, view(host_gram, 1:rows, :), first_row)
        end
    finally
        for array in (factor_device, row_device, gram_device)
            array === nothing || try
                _cuda_module().unsafe_free!(array)
            catch
            end
        end
        _cuda_module().reclaim()
    end
    return PackedERIValues{T}(factors.n_orbitals, values)
end

"""
    packed_eri_from_factors(factors; runtime=RuntimeOptions())

Materialize physical packed `(pq|rs)` values from canonical metric-whitened ERI factors using blocked CPU BLAS or CUDA Gram products.
The factors are not mutated, output host memory is quadratic in canonical-pair count, limits are enforced before allocation, and requested CUDA execution never falls back to CPU.
"""
function packed_eri_from_factors(
    factors::ERIFactorValues{T};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    if runtime.use_gpu && isnothing(Base.get_extension(@__MODULE__, :NOCICUDAExt))
        _cuda_module()
        return Base.invokelatest(packed_eri_from_factors, factors; runtime)
    end
    n_pairs, n_auxiliary = size(factors.coefficients)
    estimate = packed_eri_from_factors_memory(T, factors.n_orbitals, n_auxiliary; runtime)
    enforce_memory_limits(estimate, runtime)
    block_rows = _eri_factors_gram_block_rows(T, n_pairs, n_auxiliary, runtime)
    if runtime.use_gpu
        T in (Float32, Float64) || throw(ArgumentError("CUDA four-index materialization supports Float32 and Float64"))
        return _packed_eri_from_factors_gpu(factors, block_rows, runtime.gpu_device)
    end
    return _packed_eri_from_factors_cpu(factors, block_rows)
end
