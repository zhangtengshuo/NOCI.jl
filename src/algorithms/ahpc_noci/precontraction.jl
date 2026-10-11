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

 File:            src/algorithms/ahpc_noci/precontraction.jl
 Purpose:         State-pair active-Hamiltonian precontraction with inactive kernels and active-response orbitals.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    ActiveResponseSpace(inactive_density_kernel, left_active_response, right_active_response)

Store the fixed inactive transition-density kernel, its low-rank left/right factors, and equal-size left/right active-response orbital pools for one state pair.
Matrices are copied, `inactive_density_kernel == inactive_density_left_factor * inactive_density_right_factor'`, active-response component `xi=(a,b)` uses column-major ordering, and construction performs no ERI work or device allocation.
"""
struct ActiveResponseSpace{T<:AbstractFloat}
    inactive_density_kernel::Matrix{T}
    inactive_density_left_factor::Matrix{T}
    inactive_density_right_factor::Matrix{T}
    left_active_response::Matrix{T}
    right_active_response::Matrix{T}

    """
        ActiveResponseSpace{T}(inactive_density_kernel, inactive_density_left_factor, inactive_density_right_factor, left_active_response, right_active_response)

    Construct an active-response space through the non-bypassable validated inner path and copy all five matrices.
    Inputs are dimensionless CPU arrays and are not mutated; inconsistent orbital/factor dimensions or non-finite values raise before the object is returned, and no ERI or CUDA work occurs.
    """
    function ActiveResponseSpace{T}(
        inactive_density_kernel::AbstractMatrix{T},
        inactive_density_left_factor::AbstractMatrix{T},
        inactive_density_right_factor::AbstractMatrix{T},
        left_active_response::AbstractMatrix{T},
        right_active_response::AbstractMatrix{T},
    ) where {T<:AbstractFloat}
        n_orbitals = size(inactive_density_kernel, 1)
        size(inactive_density_kernel, 2) == n_orbitals || throw(DimensionMismatch("inactive_density_kernel must be square"))
        size(inactive_density_left_factor, 1) == n_orbitals || throw(DimensionMismatch("inactive-density left-factor orbital dimension mismatch"))
        size(inactive_density_right_factor, 1) == n_orbitals || throw(DimensionMismatch("inactive-density right-factor orbital dimension mismatch"))
        size(inactive_density_left_factor, 2) == size(inactive_density_right_factor, 2) || throw(DimensionMismatch(
            "inactive-density left/right factor ranks differ",
        ))
        size(left_active_response, 1) == n_orbitals || throw(DimensionMismatch("left_active_response orbital dimension mismatch"))
        size(right_active_response, 1) == n_orbitals || throw(DimensionMismatch("right_active_response orbital dimension mismatch"))
        size(left_active_response, 2) == size(right_active_response, 2) || throw(DimensionMismatch(
            "left and right active-response ranks must be equal",
        ))
        all(isfinite, inactive_density_kernel) || throw(ArgumentError("inactive_density_kernel contains a non-finite value"))
        all(isfinite, inactive_density_left_factor) || throw(ArgumentError("inactive_density_left_factor contains a non-finite value"))
        all(isfinite, inactive_density_right_factor) || throw(ArgumentError("inactive_density_right_factor contains a non-finite value"))
        all(isfinite, left_active_response) || throw(ArgumentError("left_active_response contains a non-finite value"))
        all(isfinite, right_active_response) || throw(ArgumentError("right_active_response contains a non-finite value"))
        return new{T}(
            Matrix{T}(inactive_density_kernel),
            Matrix{T}(inactive_density_left_factor),
            Matrix{T}(inactive_density_right_factor),
            Matrix{T}(left_active_response),
            Matrix{T}(right_active_response),
        )
    end
end

"""
    ActiveResponseSpace(inactive_density_kernel, left_active_response, right_active_response)

Construct a validated active-response space whose three matrices share an orbital row dimension and whose left/right active ranks are equal.
Inputs are not mutated; inconsistent dimensions and non-finite entries raise before construction.
"""
function ActiveResponseSpace(
    inactive_density_kernel::AbstractMatrix{T},
    left_active_response::AbstractMatrix{T},
    right_active_response::AbstractMatrix{T},
) where {T<:AbstractFloat}
    n_orbitals = size(inactive_density_kernel, 1)
    size(inactive_density_kernel, 2) == n_orbitals || throw(DimensionMismatch("inactive_density_kernel must be square"))
    inactive_density_left_factor = Matrix{T}(I, n_orbitals, n_orbitals)
    inactive_density_right_factor = Matrix{T}(transpose(inactive_density_kernel))
    return ActiveResponseSpace{T}(
        inactive_density_kernel,
        inactive_density_left_factor,
        inactive_density_right_factor,
        left_active_response,
        right_active_response,
    )
end

"""
    ActiveResponseSpace(inactive_density_kernel, inactive_density_left_factor, inactive_density_right_factor, left_active_response, right_active_response)

Construct a validated active-response space with an explicit low-rank factorization of the fixed inactive density.
Inputs are dimensionless and not mutated, all returned matrices are CPU-owned copies, incompatible dimensions or non-finite values raise before construction, and no ERI or CUDA work occurs.
"""
function ActiveResponseSpace(
    inactive_density_kernel::AbstractMatrix{T},
    inactive_density_left_factor::AbstractMatrix{T},
    inactive_density_right_factor::AbstractMatrix{T},
    left_active_response::AbstractMatrix{T},
    right_active_response::AbstractMatrix{T},
) where {T<:AbstractFloat}
    return ActiveResponseSpace{T}(
        inactive_density_kernel,
        inactive_density_left_factor,
        inactive_density_right_factor,
        left_active_response,
        right_active_response,
    )
end

"""
    active_response_count(space)

Return the number of ordered active-response components `xi=(a,b)` as the square of the active rank.
The query allocates no storage and returns an `Int`.
"""
function active_response_count(space::ActiveResponseSpace)
    rank = size(space.left_active_response, 2)
    return Base.Checked.checked_mul(rank, rank)
end

"""
    active_response_index(left_active, right_active, active_rank)

Map one-based ordered active indices `(a,b)` to the column-major response index `xi=a+(b-1)*active_rank`.
Out-of-range indices raise `BoundsError`; the function allocates no storage.
"""
function active_response_index(left_active::Integer, right_active::Integer, active_rank::Integer)
    rank = Int(active_rank)
    rank >= 0 || throw(ArgumentError("active_rank must be nonnegative"))
    1 <= left_active <= rank || throw(BoundsError(1:rank, left_active))
    1 <= right_active <= rank || throw(BoundsError(1:rank, right_active))
    return Int(left_active) + (Int(right_active) - 1) * rank
end

"""
    ActiveHamiltonianCoefficients

State-pair AHPC coefficients for one- and two-electron operators.
Fields follow the notation `o_I`, `o_xi`, `h_I,I^(2e)`, `h_I,A^(2e)`, `J_A,A`, and `K_A,A`; units follow the supplied operator and ERI values.
"""
struct ActiveHamiltonianCoefficients{T<:AbstractFloat}
    one_body_inactive::T
    one_body_active::Vector{T}
    two_body_inactive_inactive::T
    two_body_inactive_active::Vector{T}
    coulomb_response::Matrix{T}
    exchange_response::Matrix{T}
end

"""
    _response_density_value(space, response, p, q)

Return `(P_I)[p,q]` when `response==0` or `(B_xi)[p,q]` for a positive active-response index.
The helper allocates no basis matrix and assumes orbital indices have already been validated by its caller.
"""
@inline function _response_density_value(
    space::ActiveResponseSpace{T},
    response::Int,
    p::Int,
    q::Int,
) where {T<:AbstractFloat}
    response == 0 && return space.inactive_density_kernel[p, q]
    active_rank = size(space.left_active_response, 2)
    1 <= response <= active_rank * active_rank || throw(BoundsError(1:(active_rank * active_rank), response))
    left_active = (response - 1) % active_rank + 1
    right_active = div(response - 1, active_rank) + 1
    return space.left_active_response[p, left_active] * space.right_active_response[q, right_active]
end

"""
    _precontract_one_body(one_body, space)

Compute `o_I` and `o_xi` from a full one-electron operator and the active-response space.
Inputs are not mutated, output is host-resident, and matrix dimensions and finite values are validated before allocation proportional to `n_response`.
"""
function _precontract_one_body(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
) where {T<:AbstractFloat}
    n_orbitals = size(space.inactive_density_kernel, 1)
    size(one_body) == (n_orbitals, n_orbitals) || throw(DimensionMismatch("one_body orbital dimension mismatch"))
    all(isfinite, one_body) || throw(ArgumentError("one_body contains a non-finite value"))
    n_response = active_response_count(space)
    one_body_inactive = zero(T)
    one_body_active = zeros(T, n_response)
    @inbounds for q in 1:n_orbitals, p in 1:n_orbitals
        operator_value = one_body[p, q]
        one_body_inactive += operator_value * space.inactive_density_kernel[p, q]
        for response in 1:n_response
            one_body_active[response] += operator_value * _response_density_value(space, response, p, q)
        end
    end
    return one_body_inactive, one_body_active
end

"""
    PackedERISuperoperators

Store dense CPU Coulomb and exchange superoperators expanded once from physical packed ERIs for repeated AHPC response-channel projection.
Both matrices use column-major `vec(density)` ordering, retain the input ERI energy unit, own their storage, and contain no CUDA resources.
"""
struct PackedERISuperoperators{T<:AbstractFloat}
    n_orbitals::Int
    coulomb::Matrix{T}
    exchange::Matrix{T}
end

"""
    packed_eri_superoperator_memory(T, n_orbitals)

Estimate dense host payload for Coulomb and exchange AO superoperators with `n_orbitals^2` rows and columns.
The orbital count is dimensionless, returned units are bytes with zero device storage, no array is allocated, and negative counts or checked overflow raise an exception.
"""
function packed_eri_superoperator_memory(::Type{T}, n_orbitals::Integer) where {T<:AbstractFloat}
    n = Int(n_orbitals)
    n >= 0 || throw(ArgumentError("n_orbitals must be nonnegative"))
    n_squared = Base.Checked.checked_mul(n, n)
    return MemoryEstimate(estimate_array_bytes(T, 2, n_squared, n_squared), 0)
end

"""
    build_packed_eri_superoperators(integrals; runtime=RuntimeOptions())

Expand physical packed ERIs into reusable dense Coulomb and exchange operators in column-major AO-density vector ordering.
The integrals are read-only, returned matrices use ERI energy units and CPU memory, allocation is preflighted, no CUDA work occurs, and checked dimension overflow or resource limits raise before allocation.
"""
function build_packed_eri_superoperators(
    integrals::PackedERIValues{T};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    n_orbitals = integrals.n_orbitals
    estimate = packed_eri_superoperator_memory(T, n_orbitals)
    enforce_memory_limits(estimate, runtime)
    n_squared = Base.Checked.checked_mul(n_orbitals, n_orbitals)
    coulomb = Matrix{T}(undef, n_squared, n_squared)
    exchange = Matrix{T}(undef, n_squared, n_squared)
    @inbounds for s in 1:n_orbitals, r in 1:n_orbitals, q in 1:n_orbitals, p in 1:n_orbitals
        row = p + (q - 1) * n_orbitals
        column = r + (s - 1) * n_orbitals
        coulomb[row, column] = _eri_value(integrals, p, q, r, s)
    end
    @inbounds for s in 1:n_orbitals, r in 1:n_orbitals, q in 1:n_orbitals, p in 1:n_orbitals
        row = p + (q - 1) * n_orbitals
        column = r + (s - 1) * n_orbitals
        exchange[row, column] = coulomb[p + (s - 1) * n_orbitals, r + (q - 1) * n_orbitals]
    end
    return PackedERISuperoperators(n_orbitals, coulomb, exchange)
end

"""
    _packed_projected_precontraction_memory(T, n_orbitals, n_response)

Estimate host payload for AHPC outputs, response-density channels, one projected-channel scratch, and two small channel matrices when dense packed-ERI superoperators already exist.
Counts are dimensionless, returned units are bytes with zero device storage, no array is allocated, and negative counts or checked overflow raise an exception.
"""
function _packed_projected_precontraction_memory(
    ::Type{T},
    n_orbitals::Integer,
    n_response::Integer,
) where {T<:AbstractFloat}
    n = Int(n_orbitals)
    response_count = Int(n_response)
    n >= 0 && response_count >= 0 || throw(ArgumentError("packed projected precontraction dimensions must be nonnegative"))
    n_squared = Base.Checked.checked_mul(n, n)
    n_channels = Base.Checked.checked_add(response_count, 1)
    channel_vectors = estimate_array_bytes(T, 2, n_squared, n_channels)
    channel_matrices = estimate_array_bytes(T, 2, n_channels, n_channels)
    return combine_memory_estimates(
        _precontraction_output_memory(T, response_count),
        MemoryEstimate(Base.Checked.checked_add(channel_vectors, channel_matrices), 0),
    )
end

"""
    _fill_ahpc_response_densities!(densities, space)

Fill columns with the inactive density followed by ordered active-response densities in column-major AO vectorization.
The destination is mutated and the response space is read-only, all values are dimensionless CPU data, no allocation or CUDA work occurs, and inconsistent dimensions raise `DimensionMismatch`.
"""
function _fill_ahpc_response_densities!(
    densities::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
) where {T<:AbstractFloat}
    n_orbitals = size(space.inactive_density_kernel, 1)
    n_response = active_response_count(space)
    size(densities) == (n_orbitals * n_orbitals, n_response + 1) || throw(DimensionMismatch("AHPC response-density workspace has incorrect dimensions"))
    @inbounds for q in 1:n_orbitals, p in 1:n_orbitals
        row = p + (q - 1) * n_orbitals
        densities[row, 1] = space.inactive_density_kernel[p, q]
        for response in 1:n_response
            densities[row, response + 1] = _response_density_value(space, response, p, q)
        end
    end
    return densities
end

"""
    _precontract_active_hamiltonian_projected(one_body, space, superoperators; runtime=RuntimeOptions())

Project reusable conventional-ERI Coulomb and exchange superoperators over all inactive/active AHPC response densities with CPU BLAS and return active-Hamiltonian coefficients.
Inputs are read-only, outputs follow one-body and ERI energy units, local host workspace is preflighted, no CUDA work occurs, and orbital-dimension mismatches raise `DimensionMismatch`.
"""
function _precontract_active_hamiltonian_projected(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    superoperators::PackedERISuperoperators{T};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    n_orbitals = size(space.inactive_density_kernel, 1)
    n_orbitals == superoperators.n_orbitals || throw(DimensionMismatch("active-response and packed-ERI superoperator orbital dimensions differ"))
    n_response = active_response_count(space)
    estimate = _packed_projected_precontraction_memory(T, n_orbitals, n_response)
    enforce_memory_limits(estimate, runtime)
    one_body_inactive, one_body_active = _precontract_one_body(one_body, space)
    n_channels = n_response + 1
    densities = Matrix{T}(undef, n_orbitals * n_orbitals, n_channels)
    projected = similar(densities)
    coulomb_channels = Matrix{T}(undef, n_channels, n_channels)
    exchange_channels = Matrix{T}(undef, n_channels, n_channels)
    _fill_ahpc_response_densities!(densities, space)
    mul!(projected, superoperators.coulomb, densities)
    mul!(coulomb_channels, transpose(densities), projected)
    mul!(projected, superoperators.exchange, densities)
    mul!(exchange_channels, transpose(densities), projected)
    coulomb_ia = Vector{T}(@view coulomb_channels[1, 2:end])
    exchange_ia = Vector{T}(@view exchange_channels[1, 2:end])
    coulomb_aa = Matrix{T}(@view coulomb_channels[2:end, 2:end])
    exchange_aa = Matrix{T}(@view exchange_channels[2:end, 2:end])
    return _assemble_active_hamiltonian(
        one_body_inactive,
        one_body_active,
        coulomb_channels[1, 1],
        exchange_channels[1, 1],
        coulomb_ia,
        exchange_ia,
        coulomb_aa,
        exchange_aa,
    )
end

"""
    precontract_active_hamiltonian(one_body, space, superoperators; runtime=RuntimeOptions())

Compute AHPC coefficients by projecting CPU-resident conventional-ERI AO superoperators over all inactive and active response channels.
Inputs are read-only, outputs follow integral energy units and remain host-owned, local host workspace is preflighted, GPU requests are rejected because production CUDA uses streamed packed ERIs, and orbital-dimension mismatches raise `DimensionMismatch`.
"""
function precontract_active_hamiltonian(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    superoperators::PackedERISuperoperators{T};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    runtime.use_gpu && throw(ArgumentError("production CUDA packed-ERI precontraction requires PackedERIValues, not dense CPU superoperators"))
    return _precontract_active_hamiltonian_projected(one_body, space, superoperators; runtime)
end

"""
    _precontraction_output_memory(T, n_response)

Estimate host bytes for one `ActiveHamiltonianCoefficients` numerical payload and return zero device bytes.
The estimate excludes Julia object headers and allocates no coefficient arrays.
"""
function _precontraction_output_memory(::Type{T}, n_response::Integer) where {T<:AbstractFloat}
    n = Int(n_response)
    n >= 0 || throw(ArgumentError("n_response must be nonnegative"))
    scalar_bytes = estimate_array_bytes(T, 2)
    vector_bytes = estimate_array_bytes(T, 2, n)
    matrix_bytes = estimate_array_bytes(T, 2, n, n)
    return MemoryEstimate(Base.Checked.checked_add(Base.Checked.checked_add(scalar_bytes, vector_bytes), matrix_bytes), 0)
end

"""
    precontraction_memory(T, space, integrals)

Estimate peak host/device numerical payload for conventional packed-ERI AHPC precontraction under the requested runtime.
The CPU estimate includes dense reusable superoperators, the CUDA estimate includes exact streamed canonical-pair features, ERI upload, Gram/tile scratch, and output coefficients, inputs and object headers are excluded, and checked overflow or insufficient scratch raises before allocation.
"""
function precontraction_memory(
    ::Type{T},
    space::ActiveResponseSpace,
    integrals::PackedERIValues,
    ;
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    size(space.inactive_density_kernel, 1) == integrals.n_orbitals || throw(DimensionMismatch("active-response and ERI orbital dimensions differ"))
    n_orbitals = integrals.n_orbitals
    n_response = active_response_count(space)
    runtime.use_gpu && return _packed_streamed_gpu_precontraction_memory(T, space, integrals, runtime)
    cpu_estimate = combine_memory_estimates(
        packed_eri_superoperator_memory(T, n_orbitals),
        _packed_projected_precontraction_memory(T, n_orbitals, n_response),
    )
    return cpu_estimate
end

"""
    precontraction_memory(T, space, superoperators)

Estimate local host payload for CPU projected conventional-ERI AHPC precontraction when reusable AO superoperators are already resident.
Inputs are read-only, returned units are bytes, resident host superoperators are excluded, GPU requests are rejected in favor of streamed packed ERIs, and dimension mismatches or checked overflow raise an exception.
"""
function precontraction_memory(
    ::Type{T},
    space::ActiveResponseSpace,
    superoperators::PackedERISuperoperators,
    ;
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    n_orbitals = size(space.inactive_density_kernel, 1)
    n_orbitals == superoperators.n_orbitals || throw(DimensionMismatch("active-response and packed-ERI superoperator orbital dimensions differ"))
    n_response = active_response_count(space)
    runtime.use_gpu && throw(ArgumentError("production CUDA packed-ERI precontraction requires PackedERIValues, not dense CPU superoperators"))
    return _packed_projected_precontraction_memory(T, n_orbitals, n_response)
end

"""
    _eri_factors_cpu_worker_count(factors)

Return the number of independent Julia workers used by CPU Cholesky active-Hamiltonian precontraction.
The count is the smaller of `Threads.nthreads()` and the positive ERI-factor auxiliary count, allocates no storage, initializes no CUDA state, and is an execution-resource choice rather than a scientific algorithm selector.
"""
function _eri_factors_cpu_worker_count(factors::ERIFactorValues)
    return min(Threads.nthreads(), size(factors.coefficients, 2))
end

"""
    precontraction_memory(T, space, factors; runtime=RuntimeOptions())

Estimate peak host/device numerical payload for Cholesky AHPC precontraction under the requested CPU or CUDA runtime.
Input factors and active-response matrices are excluded from CPU storage; the CPU estimate includes deterministic per-auxiliary projection histories plus private scratch for every concurrent Julia worker, no array is allocated, dimension errors and checked overflow propagate, and the CUDA estimate includes factor upload, projection workspaces, and result matrices.
"""
function precontraction_memory(
    ::Type{T},
    space::ActiveResponseSpace,
    factors::ERIFactorValues,
    ;
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    n_orbitals = size(space.inactive_density_kernel, 1)
    n_orbitals == factors.n_orbitals || throw(DimensionMismatch("active-response and ERI-factor orbital dimensions differ"))
    runtime.use_gpu && return _eri_factors_gpu_precontraction_memory(T, space, factors, runtime)
    active_rank = size(space.left_active_response, 2)
    n_response = active_response_count(space)
    n_auxiliary = size(factors.coefficients, 2)
    worker_count = _eri_factors_cpu_worker_count(factors)
    output = _precontraction_output_memory(T, n_response)
    exchange_vector = estimate_array_bytes(T, n_response)
    worker_square_scratch = estimate_array_bytes(T, 3, n_orbitals, n_orbitals, worker_count)
    worker_response_orbitals = estimate_array_bytes(T, 3, n_orbitals, active_rank, worker_count)
    worker_response_matrices = estimate_array_bytes(T, 2, active_rank, active_rank, worker_count)
    response_histories = estimate_array_bytes(T, 2, n_response, n_auxiliary)
    auxiliary_scalar_histories = estimate_array_bytes(T, 2, n_auxiliary)
    scratch_bytes = Base.Checked.checked_add(
        Base.Checked.checked_add(
            Base.Checked.checked_add(
                Base.Checked.checked_add(
                    Base.Checked.checked_add(exchange_vector, worker_square_scratch),
                    worker_response_orbitals,
                ),
                worker_response_matrices,
            ),
            response_histories,
        ),
        auxiliary_scalar_histories,
    )
    return combine_memory_estimates(output, MemoryEstimate(scratch_bytes, 0))
end

"""
    _assemble_active_hamiltonian(one_body_inactive, one_body_active, coulomb_ii, exchange_ii, coulomb_ia, exchange_ia, coulomb_aa, exchange_aa)

Combine sector-resolved contractions into publication-level AHPC coefficients using `h_II=2J_II-K_II` and `h_IA=2J_IA-K_IA`.
The function reuses and mutates `coulomb_ia` as the returned inactive-active two-electron vector; all other arrays become owned result fields without copies.
"""
function _assemble_active_hamiltonian(
    one_body_inactive::T,
    one_body_active::Vector{T},
    coulomb_ii::T,
    exchange_ii::T,
    coulomb_ia::Vector{T},
    exchange_ia::Vector{T},
    coulomb_aa::Matrix{T},
    exchange_aa::Matrix{T},
) where {T<:AbstractFloat}
    @inbounds for response in eachindex(coulomb_ia, exchange_ia)
        coulomb_ia[response] = 2 * coulomb_ia[response] - exchange_ia[response]
    end
    return ActiveHamiltonianCoefficients(
        one_body_inactive,
        one_body_active,
        2 * coulomb_ii - exchange_ii,
        coulomb_ia,
        coulomb_aa,
        exchange_aa,
    )
end

"""
    precontract_active_hamiltonian(one_body, space, integrals::PackedERIValues; runtime=RuntimeOptions())

Compute all AHPC one- and two-electron coefficients from physical conventional packed ERIs using dense reusable CPU superoperators or exact streamed canonical-pair tiles on CUDA.
The function does not mutate inputs, preflights host/device storage before allocation, returns host-owned coefficients in integral energy units, and a requested unavailable GPU raises without CPU fallback.
"""
function precontract_active_hamiltonian(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    integrals::PackedERIValues{T};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    n_orbitals = size(space.inactive_density_kernel, 1)
    n_orbitals == integrals.n_orbitals || throw(DimensionMismatch("active-response and ERI orbital dimensions differ"))
    memory = precontraction_memory(T, space, integrals; runtime)
    enforce_memory_limits(memory, runtime)
    runtime.use_gpu && return _precontract_packed_streamed_active_hamiltonian_gpu(one_body, space, integrals; runtime)
    superoperators = build_packed_eri_superoperators(integrals; runtime)
    return precontract_active_hamiltonian(one_body, space, superoperators; runtime)
end

"""
    _unpack_eri_factor!(factor_matrix, factors, auxiliary)

Expand one canonical symmetric Cholesky-factor column into a full orbital matrix in place.
The destination must have the matching square shape; no temporary matrix is allocated.
"""
function _unpack_eri_factor!(
    factor_matrix::Matrix{T},
    factors::ERIFactorValues{T},
    auxiliary::Int,
) where {T<:AbstractFloat}
    n_orbitals = factors.n_orbitals
    size(factor_matrix) == (n_orbitals, n_orbitals) || throw(DimensionMismatch("factor_matrix orbital dimension mismatch"))
    auxiliary in axes(factors.coefficients, 2) || throw(BoundsError(axes(factors.coefficients, 2), auxiliary))
    @inbounds for upper in 1:n_orbitals, lower in 1:upper
        value = factors.coefficients[canonical_pair_index(lower, upper), auxiliary]
        factor_matrix[lower, upper] = value
        factor_matrix[upper, lower] = value
    end
    return factor_matrix
end

"""
    _precontract_eri_factors_active_hamiltonian_cpu(one_body, space, factors, worker_count)

Compute CPU Cholesky AHPC coefficients with independent auxiliary ranges evaluated concurrently and all cross-auxiliary floating-point reductions replayed in canonical auxiliary order.
Inputs are read-only and outputs are host-owned in integral energy units; `worker_count` is a dimensionless execution resource in `1:min(Threads.nthreads(), n_auxiliary)`, each worker owns its BLAS scratch, and invalid worker counts raise before scientific workspace allocation.
"""
function _precontract_eri_factors_active_hamiltonian_cpu(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    factors::ERIFactorValues{T},
    worker_count::Integer,
) where {T<:AbstractFloat}
    n_orbitals = size(space.inactive_density_kernel, 1)
    n_orbitals == factors.n_orbitals || throw(DimensionMismatch("active-response and ERI-factor orbital dimensions differ"))
    workers = Int(worker_count)
    maximum_workers = _eri_factors_cpu_worker_count(factors)
    1 <= workers <= maximum_workers || throw(ArgumentError("CPU ERI-factor worker_count must lie in 1:$maximum_workers"))
    one_body_inactive, one_body_active = _precontract_one_body(one_body, space)
    active_rank = size(space.left_active_response, 2)
    n_response = active_response_count(space)
    n_auxiliary = size(factors.coefficients, 2)
    coulomb_ii = zero(T)
    exchange_ii = zero(T)
    coulomb_ia = zeros(T, n_response)
    exchange_ia = zeros(T, n_response)
    coulomb_aa = zeros(T, n_response, n_response)
    exchange_aa = zeros(T, n_response, n_response)
    inactive_projection_history = Vector{T}(undef, n_auxiliary)
    exchange_ii_history = Vector{T}(undef, n_auxiliary)
    active_projection_history = Matrix{T}(undef, n_response, n_auxiliary)
    exchange_ia_history = Matrix{T}(undef, n_response, n_auxiliary)
    inactive_density = space.inactive_density_kernel
    left_active = space.left_active_response
    right_active = space.right_active_response

    worker_body = function (worker::Int)
        factor_matrix = Matrix{T}(undef, n_orbitals, n_orbitals)
        matrix_scratch = Matrix{T}(undef, n_orbitals, n_orbitals)
        exchange_matrix = Matrix{T}(undef, n_orbitals, n_orbitals)
        factor_left = Matrix{T}(undef, n_orbitals, active_rank)
        factor_right = Matrix{T}(undef, n_orbitals, active_rank)
        cross_intermediate = Matrix{T}(undef, n_orbitals, active_rank)
        active_projection = Matrix{T}(undef, active_rank, active_rank)
        exchange_cross = Matrix{T}(undef, active_rank, active_rank)
        base_count = div(n_auxiliary, workers)
        extra_count = rem(n_auxiliary, workers)
        first_auxiliary = (worker - 1) * base_count + min(worker - 1, extra_count) + 1
        local_count = base_count + (worker <= extra_count ? 1 : 0)
        last_auxiliary = first_auxiliary + local_count - 1
        for auxiliary in first_auxiliary:last_auxiliary
            _unpack_eri_factor!(factor_matrix, factors, auxiliary)
            inactive_projection = zero(T)
            @inbounds for q in 1:n_orbitals, p in 1:n_orbitals
                inactive_projection += factor_matrix[p, q] * inactive_density[p, q]
            end
            inactive_projection_history[auxiliary] = inactive_projection
            mul!(factor_left, factor_matrix, left_active)
            mul!(factor_right, factor_matrix, right_active)
            mul!(active_projection, transpose(left_active), factor_right)
            n_response > 0 && copyto!(active_projection_history, (auxiliary - 1) * n_response + 1, active_projection, 1, n_response)
            mul!(matrix_scratch, factor_matrix, transpose(inactive_density))
            mul!(exchange_matrix, matrix_scratch, factor_matrix)
            exchange_ii_value = zero(T)
            @inbounds for q in 1:n_orbitals, p in 1:n_orbitals
                exchange_ii_value += inactive_density[p, q] * exchange_matrix[p, q]
            end
            exchange_ii_history[auxiliary] = exchange_ii_value
            mul!(cross_intermediate, transpose(inactive_density), factor_right)
            mul!(exchange_cross, transpose(factor_left), cross_intermediate)
            @inbounds for right_index in 1:active_rank, left_index in 1:active_rank
                response = active_response_index(left_index, right_index, active_rank)
                exchange_ia_history[response, auxiliary] = exchange_cross[left_index, right_index]
            end
        end
        return nothing
    end

    if workers == 1
        worker_body(1)
    else
        @sync begin
            for worker in 1:workers
                Threads.@spawn worker_body(worker)
            end
        end
    end

    @inbounds for auxiliary in 1:n_auxiliary
        inactive_projection = inactive_projection_history[auxiliary]
        coulomb_ii += inactive_projection * inactive_projection
        exchange_ii += exchange_ii_history[auxiliary]
        for response in 1:n_response
            coulomb_ia[response] += inactive_projection * active_projection_history[response, auxiliary]
            exchange_ia[response] += exchange_ia_history[response, auxiliary]
        end
    end
    if n_response > 0
        mul!(coulomb_aa, active_projection_history, transpose(active_projection_history))
        @inbounds for second_right in 1:active_rank, second_left in 1:active_rank
            second_response = active_response_index(second_left, second_right, active_rank)
            for first_right in 1:active_rank, first_left in 1:active_rank
                first_response = active_response_index(first_left, first_right, active_rank)
                exchange_source_row = active_response_index(first_left, second_right, active_rank)
                exchange_source_column = active_response_index(second_left, first_right, active_rank)
                exchange_aa[first_response, second_response] = coulomb_aa[exchange_source_row, exchange_source_column]
            end
        end
    end
    return _assemble_active_hamiltonian(
        one_body_inactive,
        one_body_active,
        coulomb_ii,
        exchange_ii,
        coulomb_ia,
        exchange_ia,
        coulomb_aa,
        exchange_aa,
    )
end

"""
    precontract_active_hamiltonian(one_body, space, factors::ERIFactorValues; runtime=RuntimeOptions())

Compute all AHPC one- and two-electron coefficients directly from symmetric metric-whitened ERI factors without materializing four-index ERIs.
The CPU path parallelizes independent Cholesky-auxiliary projection work across Julia threads with private BLAS scratch, replays II/IA reductions in canonical auxiliary order, and evaluates active-active Coulomb response with one response-space Gram GEMM followed by deterministic exchange-index permutation; the selected CPU or CUDA kernel does not mutate inputs, returns host-owned coefficients in integral energy units, enforces host/device estimates before allocation, and raises `GPUUnavailableError` rather than silently falling back when GPU execution was requested.
"""
function precontract_active_hamiltonian(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    factors::ERIFactorValues{T};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    n_orbitals = size(space.inactive_density_kernel, 1)
    n_orbitals == factors.n_orbitals || throw(DimensionMismatch("active-response and ERI-factor orbital dimensions differ"))
    runtime.use_gpu && return _precontract_eri_factors_active_hamiltonian_gpu(one_body, space, factors; runtime)
    memory = precontraction_memory(T, space, factors; runtime)
    enforce_memory_limits(memory, runtime)
    worker_count = _eri_factors_cpu_worker_count(factors)
    return _precontract_eri_factors_active_hamiltonian_cpu(one_body, space, factors, worker_count)
end
