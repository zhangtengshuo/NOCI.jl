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

 File:            src/algorithms/ahpc_noci/spin_contraction.jl
 Purpose:         CUDA device activation and blocked sparse contraction of spin-pair features into matrix elements.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _activate_cuda_device(runtime)

Validate CUDA functionality, select the requested zero-based device, and return its CUDA device handle when GPU execution is enabled.
The runtime is read-only, no scientific array is allocated, and initialization or selection failures are wrapped in `GPUUnavailableError` without CPU fallback.
"""
function _activate_cuda_device(runtime::RuntimeOptions)
    runtime.use_gpu || return nothing
    try
        Base.invokelatest(_cuda_module().functional) || throw(GPUUnavailableError(runtime.gpu_device, "_cuda_module().functional() returned false"))
        Base.invokelatest(_cuda_module().device!, runtime.gpu_device)
        return Base.invokelatest(_cuda_module().device)
    catch error
        error isa GPUUnavailableError && rethrow()
        throw(GPUUnavailableError(runtime.gpu_device, sprint(showerror, error)))
    end
end

"""
    _contracted_same_spin_second_from_polynomial_basis!(workspace, same_spin_response)

Evaluate the rank-uniform same-spin contracted second determinant derivative from
prepared elementary symmetric coefficients and matrix powers when the response
kernel is antisymmetric under the determinant-Hessian crossed permutation
`K[(i,l),(k,j)] = -K[(i,j),(k,l)]`.

Only the coefficient-derivative half is evaluated and doubled. The response and
workspace are read-only except for the reusable Newton matrix, no determinant
inverse, rank classifier, threshold, SVD, subset fallback, projection buffer, or
CUDA work is used, and singular matrices follow the same polynomial circuit.
"""
function _contracted_same_spin_second_from_polynomial_basis!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    same_spin_response::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = workspace.q
    n_response = q * q
    size(same_spin_response) == (n_response, n_response) ||
        throw(DimensionMismatch("same-spin response must have size (q^2,q^2)"))
    q < 2 && return zero(T)

    coefficient_derivative = zero(T)
    @inbounds for power in 0:(q - 1)
        elementary_order = q - 1 - power
        sign = isodd(power) ? -one(T) : one(T)
        if elementary_order > 0
            _newton_transformation_workspace!(workspace, elementary_order - 1)
            coefficient_derivative += sign * _trace_basis_kernel_bilinear_workspace(
                same_spin_response,
                workspace.powers[power + 1],
                workspace.newton,
            )
        end
    end
    return coefficient_derivative + coefficient_derivative
end

"""
    _polynomial_same_spin_contracted_complementary_cofactors!(workspace, matrix, same_spin_response)

Compute determinant, complete first complementary cofactors, and the contracted
second determinant derivative for a crossed-antisymmetric same-spin response
with one rank-uniform polynomial workspace.

The overlap/response are read-only, exact power-of-two scaling matches the
canonical polynomial path, the second contraction evaluates only one of the two
algebraically identical `O(q^5)` response traversals, and no inverse, rank
classifier, determinant threshold, SVD, subset fallback, or CUDA work occurs.
"""
function _polynomial_same_spin_contracted_complementary_cofactors!(
    workspace::PolynomialContractedCofactorWorkspace{T},
    matrix::AbstractMatrix{T},
    same_spin_response::AbstractMatrix{T},
) where {T<:AbstractFloat}
    q = size(matrix, 1)
    size(matrix, 2) == q || throw(DimensionMismatch("polynomial cofactor input must be square"))
    0 <= q <= AHPC_MAX_OCCUPIED_RANK || throw(ArgumentError("production polynomial cofactor engine supports occupied rank 0:32"))
    size(same_spin_response) == (q * q, q * q) ||
        throw(DimensionMismatch("same-spin response must have size (q^2,q^2)"))
    _resize_polynomial_contracted_workspace!(workspace, q)

    exponent = _scale_polynomial_input!(workspace, matrix)
    _elementary_symmetrics_scaled!(workspace)
    _matrix_powers_scaled!(workspace)

    determinant_scaled = workspace.elementary[q + 1]
    determinant = ldexp(determinant_scaled, exponent * q)
    first_exponent = exponent * max(q - 1, 0)
    if q > 0
        _newton_transformation_workspace!(workspace, q - 1)
        @inbounds for column in 1:q, row in 1:q
            workspace.first_cofactor[row, column] = ldexp(workspace.newton[column, row], first_exponent)
        end
    end

    second_scaled = _contracted_same_spin_second_from_polynomial_basis!(workspace, same_spin_response)
    contracted_second = ldexp(second_scaled, exponent * max(q - 2, 0))
    isfinite(determinant) || throw(ArgumentError("polynomial determinant is non-finite"))
    all(isfinite, workspace.first_cofactor) ||
        throw(ArgumentError("polynomial first cofactors contain a non-finite value"))
    isfinite(contracted_second) ||
        throw(ArgumentError("polynomial contracted same-spin second cofactor is non-finite"))
    return ComplementaryCofactorResult(
        determinant,
        workspace.first_cofactor,
        contracted_second,
        exponent,
    )
end

"""
    _fill_hessian_relevant_same_spin_kernel!(local_kernel, response_indices, coefficients)

Project an arbitrary local Coulomb-minus-exchange response onto the exact
crossed-antisymmetric subspace seen by the determinant Hessian:
`K^- = (K - X*K)/2`, where `X*K[(i,j),(k,l)] = K[(i,l),(k,j)]`.

The destination is overwritten in canonical local-response order. Cross-paired
entries are formed together, equal-right-index entries are set to exact zero,
coefficient matrices and response indices are read-only, and no rank analysis,
determinant division, allocation, or CUDA work occurs. Contracting the returned
kernel with any determinant Hessian is mathematically identical to contracting
the unprojected input kernel for every matrix rank.
"""
function _fill_hessian_relevant_same_spin_kernel!(
    local_kernel::Matrix{T},
    response_indices::Vector{Int},
    coefficients::ActiveHamiltonianCoefficients{T},
) where {T<:AbstractFloat}
    n_local_response = length(response_indices)
    q = isqrt(n_local_response)
    q * q == n_local_response || throw(DimensionMismatch("local response count must be an occupied-rank square"))
    size(local_kernel) == (n_local_response, n_local_response) ||
        throw(DimensionMismatch("local same-spin kernel dimension mismatch"))

    @inbounds for right_active in 1:q
        for second_row in 1:q, first_row in 1:q
            first_local = first_row + (right_active - 1) * q
            second_local = second_row + (right_active - 1) * q
            local_kernel[first_local, second_local] = zero(T)
        end
    end

    half = T(0.5)
    @inbounds for second_right in 2:q
        for first_right in 1:(second_right - 1)
            for second_row in 1:q, first_row in 1:q
                first_local = first_row + (first_right - 1) * q
                second_local = second_row + (second_right - 1) * q
                crossed_first_local = first_row + (second_right - 1) * q
                crossed_second_local = second_row + (first_right - 1) * q

                first_global = response_indices[first_local]
                second_global = response_indices[second_local]
                crossed_first_global = response_indices[crossed_first_local]
                crossed_second_global = response_indices[crossed_second_local]

                raw_response =
                    coefficients.coulomb_response[first_global, second_global] -
                    coefficients.exchange_response[first_global, second_global]
                crossed_response =
                    coefficients.coulomb_response[crossed_first_global, crossed_second_global] -
                    coefficients.exchange_response[crossed_first_global, crossed_second_global]
                projected_response = half * (raw_response - crossed_response)
                isfinite(projected_response) ||
                    throw(ArgumentError("Hessian-relevant same-spin response is non-finite"))
                local_kernel[first_local, second_local] = projected_response
                local_kernel[crossed_first_local, crossed_second_local] = -projected_response
            end
        end
    end
    return local_kernel
end
