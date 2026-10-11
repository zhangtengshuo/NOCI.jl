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

 File:            src/core/solver.jl
 Purpose:         Symmetric generalized NOCI eigensolution over overlap-normalized matrices with residuals.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    NormalizedNOCIMatrices

Store NOCI overlap and Hamiltonian component matrices after symmetric diagonal-overlap normalization.
Overlap is dimensionless, Hamiltonian components use the input energy unit, matrices and labels are CPU-owned, and no correlation shift, coupling analysis, or device state is retained.
"""
struct NormalizedNOCIMatrices{T<:AbstractFloat}
    labels::Vector{String}
    overlap::Matrix{T}
    electronic_one_body::Matrix{T}
    nuclear_repulsion::Matrix{T}
    one_body_with_nuclear_repulsion::Matrix{T}
    two_body::Matrix{T}
    hamiltonian::Matrix{T}
end

"""
    NOCISolution

Store ascending generalized NOCI eigenvalues, overlap-normalized eigenvectors by columns, and Euclidean residual norms.
Energies and residuals follow the Hamiltonian energy unit, arrays are CPU-owned, and no coupling or correlation-shift metadata is included.
"""
struct NOCISolution{T<:AbstractFloat}
    energies::Vector{T}
    vectors::Matrix{T}
    residual_norms::Vector{T}
end

"""
    noci_normalization_memory(T, n_basis_functions)

Estimate host numerical payload for six normalized floating-point matrices plus diagonal scaling scratch and return zero device bytes.
The dimension is not mutated, no arrays are allocated, and negative counts or checked overflow raise an exception.
"""
function noci_normalization_memory(::Type{T}, n_basis_functions::Integer) where {T<:AbstractFloat}
    n = Int(n_basis_functions)
    n >= 0 || throw(ArgumentError("n_basis_functions must be nonnegative"))
    matrix_elements = Base.Checked.checked_mul(n, n)
    floating_elements = Base.Checked.checked_add(Base.Checked.checked_mul(6, matrix_elements), n)
    return MemoryEstimate(estimate_array_bytes(T, floating_elements), 0)
end

"""
    _normalize_operator_matrix(operator, scales)

Return `Diagonal(scales) * operator * Diagonal(scales)` using explicit symmetric host assignment.
Inputs are read-only, units follow `operator`, one dense CPU matrix is allocated, and non-square dimensions or scale-length mismatches raise `DimensionMismatch`.
"""
function _normalize_operator_matrix(
    operator::AbstractMatrix{T},
    scales::AbstractVector{T},
) where {T<:AbstractFloat}
    n_basis_functions = length(scales)
    size(operator) == (n_basis_functions, n_basis_functions) || throw(DimensionMismatch("operator dimension differs from normalization scale count"))
    normalized = Matrix{T}(undef, n_basis_functions, n_basis_functions)
    @inbounds for right in 1:n_basis_functions, left in 1:right
        value = scales[left] * operator[left, right] * scales[right]
        normalized[left, right] = value
        normalized[right, left] = value
    end
    return normalized
end

"""
    _normalize_noci_matrices(raw)

Normalize every raw overlap/Hamiltonian component by `1/sqrt(S_ii*S_jj)` and return `NormalizedNOCIMatrices`.
The raw result is read-only, overlap is dimensionless and energy units are preserved, CPU matrices are newly allocated, and nonpositive or non-finite diagonal overlaps raise `ArgumentError`.
"""
function _normalize_noci_matrices(raw::RawNOCIMatrices{T}) where {T<:AbstractFloat}
    n_basis_functions = length(raw.labels)
    scales = Vector{T}(undef, n_basis_functions)
    for basis_function in 1:n_basis_functions
        diagonal_overlap = raw.overlap[basis_function, basis_function]
        isfinite(diagonal_overlap) && diagonal_overlap > zero(T) || throw(ArgumentError("NOCI diagonal overlaps must be finite and positive"))
        scales[basis_function] = one(T) / sqrt(diagonal_overlap)
    end
    overlap = _normalize_operator_matrix(raw.overlap, scales)
    electronic_one_body = _normalize_operator_matrix(raw.electronic_one_body, scales)
    nuclear_repulsion = _normalize_operator_matrix(raw.nuclear_repulsion, scales)
    one_body_with_nuclear_repulsion = _normalize_operator_matrix(raw.one_body_with_nuclear_repulsion, scales)
    two_body = _normalize_operator_matrix(raw.two_body, scales)
    hamiltonian = _normalize_operator_matrix(raw.hamiltonian, scales)
    return NormalizedNOCIMatrices(
        copy(raw.labels),
        overlap,
        electronic_one_body,
        nuclear_repulsion,
        one_body_with_nuclear_repulsion,
        two_body,
        hamiltonian,
    )
end

"""
    normalize_noci_matrices(raw; runtime=RuntimeOptions(), telemetry=nothing)

Apply symmetric diagonal-overlap normalization to raw NOCI overlap and every Hamiltonian component.
The raw matrices are read-only, output units are preserved, host memory is preflighted before allocation, optional telemetry records seconds and allocations, and no eigensolve or CUDA work occurs.
"""
function normalize_noci_matrices(
    raw::RawNOCIMatrices{T};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    estimate = noci_normalization_memory(T, length(raw.labels))
    enforce_memory_limits(estimate, runtime)
    work = () -> _normalize_noci_matrices(raw)
    return isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :normalize_noci_matrices;
        estimate,
        collect_allocations=runtime.collect_allocations,
    )
end

"""
    generalized_noci_solver_memory(T, n_basis_functions)

Estimate conservative host numerical payload for dense symmetric generalized eigensolution, returned eigenvectors/values/residuals, and matrix copies.
The dimension is read-only and dimensionless, device bytes are zero, no LAPACK call occurs, and negative counts or checked overflow raise an exception.
"""
function generalized_noci_solver_memory(::Type{T}, n_basis_functions::Integer) where {T<:AbstractFloat}
    n = Int(n_basis_functions)
    n >= 0 || throw(ArgumentError("n_basis_functions must be nonnegative"))
    matrix_elements = Base.Checked.checked_mul(n, n)
    floating_elements = Base.Checked.checked_add(Base.Checked.checked_mul(4, matrix_elements), Base.Checked.checked_mul(3, n))
    return MemoryEstimate(estimate_array_bytes(T, floating_elements), 0)
end

"""
    _solve_generalized_noci(normalized)

Solve the symmetric generalized eigenproblem `H*C=S*C*E`, return ascending energies, overlap-normalized eigenvectors, and Euclidean residual norms.
Input matrices are read-only and already normalized by diagonal overlap, CPU LAPACK work and owned outputs are allocated, and a non-positive-definite overlap raises `PosDefException`.
"""
function _solve_generalized_noci(normalized::NormalizedNOCIMatrices{T}) where {T<:AbstractFloat}
    hamiltonian = Matrix(normalized.hamiltonian)
    overlap = Matrix(normalized.overlap)
    factorization = eigen(Symmetric(hamiltonian), Symmetric(overlap))
    energies = Vector{T}(factorization.values)
    vectors = Matrix{T}(factorization.vectors)
    residual_norms = Vector{T}(undef, length(energies))
    for state in eachindex(energies)
        residual_norms[state] = norm(hamiltonian * @view(vectors[:, state]) - energies[state] * overlap * @view(vectors[:, state]))
    end
    return NOCISolution(energies, vectors, residual_norms)
end

"""
    solve_generalized_noci(normalized; runtime=RuntimeOptions(), telemetry=nothing)

Solve one normalized symmetric NOCI generalized eigenproblem on CPU and return energies, vectors, and residual norms.
Inputs are read-only, energies use Hamiltonian units, memory is preflighted, optional telemetry records timing/allocations and state count, and no coupling analysis, shift, or CUDA work occurs.
"""
function solve_generalized_noci(
    normalized::NormalizedNOCIMatrices{T};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    estimate = generalized_noci_solver_memory(T, length(normalized.labels))
    enforce_memory_limits(estimate, runtime)
    work = () -> _solve_generalized_noci(normalized)
    solution = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :solve_generalized_noci;
        estimate,
        collect_allocations=runtime.collect_allocations,
    )
    isnothing(telemetry) || set_counter!(telemetry, :n_noci_states, length(solution.energies))
    return solution
end
