# File: src/core/correlation_shifts.jl
# Purpose: Apply Loewdin orthogonalization, Gallup-Norbeck weights, and fragment correlation shifts.
# Algorithm origin: Gallup-Norbeck weighting and the GronOR gronor_correlation_energy method (Straatsma et al., J. Chem. Theory Comput. 18, 3549-3565, 2022; doi:10.1021/acs.jctc.2c00266).

"""
    lowdin_orthogonalization(overlap)

Return the symmetric Löwdin inverse square root of a real positive-definite overlap matrix.
Overlap is dimensionless and read-only, the returned dense matrix is CPU-owned, no CUDA work occurs, and nonsquare, nonfinite, or non-positive-definite inputs raise an exception.
"""
function lowdin_orthogonalization(overlap::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(overlap, 1)
    size(overlap, 2) == n || throw(DimensionMismatch("overlap matrix must be square"))
    all(isfinite, overlap) || throw(ArgumentError("overlap matrix must be finite"))
    factorization = eigen(Symmetric(Matrix{T}(overlap)))
    all(>(zero(T)), factorization.values) || throw(ArgumentError("overlap matrix must be positive definite"))
    inverse_roots = inv.(sqrt.(factorization.values))
    vectors = Matrix{T}(factorization.vectors)
    return vectors * Diagonal(inverse_roots) * transpose(vectors)
end

"""
    _gallup_norbeck_fortran_weights(lowdin)

Return row-normalized squared Löwdin coefficients using the historical Fortran Gallup-Norbeck convention.
Inputs are dimensionless and read-only, the result is a CPU matrix of dimensionless weights, and zero or nonfinite row norms raise `ArgumentError`.
"""
function _gallup_norbeck_fortran_weights(lowdin::AbstractMatrix{T}) where {T<:AbstractFloat}
    n = size(lowdin, 1)
    size(lowdin, 2) == n || throw(DimensionMismatch("Löwdin matrix must be square"))
    weights = Matrix{T}(undef, n, n)
    for row in 1:n
        row_norm = sum(abs2, @view lowdin[row, :])
        isfinite(row_norm) && row_norm > zero(T) || throw(ArgumentError("Gallup-Norbeck row $row has invalid norm"))
        @views weights[row, :] .= abs2.(lowdin[row, :]) ./ row_norm
    end
    return weights
end

"""
    _gallup_norbeck_paper_weights(lowdin, overlap)

Return the inverse-overlap-diagonal-normalized Gallup-Norbeck weights described by the paper convention.
Inputs are dimensionless and read-only, the result is CPU-owned, and dimension, positivity, or finite-value violations raise an exception.
"""
function _gallup_norbeck_paper_weights(
    lowdin::AbstractMatrix{T},
    overlap::AbstractMatrix{T},
) where {T<:AbstractFloat}
    n = size(lowdin, 1)
    size(lowdin, 2) == n || throw(DimensionMismatch("Löwdin matrix must be square"))
    size(overlap) == (n, n) || throw(DimensionMismatch("overlap and Löwdin dimensions differ"))
    inverse_overlap = Matrix(inv(Symmetric(Matrix{T}(overlap))))
    all(value -> isfinite(value) && value > zero(T), diag(inverse_overlap)) || throw(ArgumentError("inverse-overlap diagonal must be finite and positive"))
    weights = Matrix{T}(undef, n, n)
    for row in 1:n
        @views weights[row, :] .= abs2.(lowdin[row, :]) ./ diag(inverse_overlap)
        row_norm = sum(@view weights[row, :])
        isfinite(row_norm) && row_norm > zero(T) || throw(ArgumentError("Gallup-Norbeck paper row $row has invalid norm"))
        @views weights[row, :] ./= row_norm
    end
    return weights
end

"""
    gallup_norbeck_weights(overlap; formula=:fortran)

Construct dimensionless Gallup-Norbeck state weights from a positive-definite overlap matrix using `:fortran` or `:paper` semantics.
The input is read-only, all work is CPU dense linear algebra, and unsupported formulas or invalid overlap matrices raise an exception.
"""
function gallup_norbeck_weights(overlap::AbstractMatrix{T}; formula::Symbol=:fortran) where {T<:AbstractFloat}
    lowdin = lowdin_orthogonalization(overlap)
    formula === :fortran && return _gallup_norbeck_fortran_weights(lowdin)
    formula === :paper && return _gallup_norbeck_paper_weights(lowdin, overlap)
    throw(ArgumentError("Gallup-Norbeck formula must be :fortran or :paper"))
end

"""
    apply_diagonal_shifts(hamiltonian, overlap, shifts; weighting=:identity)

Apply label-preserving scalar energy shifts on the diagonal of the symmetric Löwdin orthogonal basis and transform the Hamiltonian back.
Hamiltonian and shifts share one energy unit while overlap and weights are dimensionless, inputs are read-only and output is CPU-owned, and invalid dimensions, nonfinite values, unsupported weighting, or non-positive overlap raise an exception.
"""
function apply_diagonal_shifts(
    hamiltonian::AbstractMatrix{T},
    overlap::AbstractMatrix{T},
    shifts::AbstractVector{T};
    weighting::Symbol=:identity,
) where {T<:AbstractFloat}
    n = size(hamiltonian, 1)
    size(hamiltonian) == (n, n) || throw(DimensionMismatch("Hamiltonian must be square"))
    size(overlap) == (n, n) || throw(DimensionMismatch("Hamiltonian and overlap dimensions differ"))
    length(shifts) == n || throw(DimensionMismatch("shift count differs from matrix dimension"))
    all(isfinite, hamiltonian) && all(isfinite, shifts) || throw(ArgumentError("Hamiltonian and shifts must be finite"))
    lowdin = lowdin_orthogonalization(overlap)
    applied_shifts = if weighting === :identity
        collect(shifts)
    elseif weighting === :gallup_norbeck_fortran
        _gallup_norbeck_fortran_weights(lowdin) * shifts
    elseif weighting === :gallup_norbeck_paper
        _gallup_norbeck_paper_weights(lowdin, overlap) * shifts
    else
        throw(ArgumentError("weighting must be :identity, :gallup_norbeck_fortran, or :gallup_norbeck_paper"))
    end
    orthogonal_hamiltonian = transpose(lowdin) * hamiltonian * lowdin
    for index in eachindex(applied_shifts)
        orthogonal_hamiltonian[index, index] += applied_shifts[index]
    end
    inverse_lowdin = inv(lowdin)
    shifted = transpose(inverse_lowdin) * orthogonal_hamiltonian * inverse_lowdin
    return Matrix{T}(Symmetric(shifted))
end

"""
    _normalized_with_hamiltonian(normalized, shifted_hamiltonian)

Return normalized matrix components with only the total Hamiltonian replaced for a shifted generalized solve.
All matrices are read-only and share the same energy unit, returned arrays are CPU-owned copies, and label or dimension inconsistencies raise through `NormalizedNOCIMatrices` construction or explicit checks.
"""
function _normalized_with_hamiltonian(
    normalized::NormalizedNOCIMatrices{T},
    shifted_hamiltonian::AbstractMatrix{T},
) where {T<:AbstractFloat}
    size(shifted_hamiltonian) == size(normalized.hamiltonian) || throw(DimensionMismatch("shifted Hamiltonian dimension differs from normalized matrices"))
    return NormalizedNOCIMatrices(
        copy(normalized.labels),
        copy(normalized.overlap),
        copy(normalized.electronic_one_body),
        copy(normalized.nuclear_repulsion),
        copy(normalized.one_body_with_nuclear_repulsion),
        copy(normalized.two_body),
        Matrix{T}(shifted_hamiltonian),
    )
end
