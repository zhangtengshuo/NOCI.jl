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

 File:            src/algorithms/dp_noci/two_electron.jl
 Purpose:         Rank-aware two-electron cofactor contraction over packed ERIs with canonical multiplicities.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
 Algorithm origin: The determinant-pair method originates from the GronOR program (T. P. Straatsma, R. Broer,
                   A. Sanchez-Mansilla, C. Sousa, and C. de Graaf, J. Chem. Theory Comput. 18, 3549-3565 (2022),
                   doi:10.1021/acs.jctc.2c00266; gitlab.com/gronor/gronor, Apache-2.0). NOCI.jl rewrites the method in
                   Julia with extensive single-node performance optimizations, keeping most functionality and settings
                   consistent with GronOR.
=#

"""
    DPDeterminantPairMatrixElement

Store one bare conventional DP-NOCI determinant-pair overlap and electronic one- and two-body matrix elements before CI weighting or nuclear repulsion.
Overlap is dimensionless, Hamiltonian components follow integral energy units, and the immutable result owns no arrays or CUDA resources.
"""
struct DPDeterminantPairMatrixElement{T<:AbstractFloat}
    overlap::T
    electronic_one_body::T
    two_body::T
end

"""
    DPDirectTwoElectronWorkspace{T}()

Own reusable CPU buffers for one rank-aware conventional DP-NOCI direct two-electron contraction.
The empty constructor allocates only zero-length arrays, subsequent contractions resize the workspace after memory preflight, stored factors are dimensionless, and the workspace has no CUDA resources.
"""
mutable struct DPDirectTwoElectronWorkspace{T<:AbstractFloat}
    n_basis::Int
    occupied_count::Int
    occupied_factor::Matrix{T}
    alpha_factor::Matrix{T}
    beta_factor::Matrix{T}
    pair_sum::Matrix{T}
    transform_scratch::Matrix{T}
    left_alpha_null::Vector{T}
    left_beta_null::Vector{T}
    right_alpha_null::Vector{T}
    right_beta_null::Vector{T}
    lower_pair_labels::Vector{Int}
    upper_pair_labels::Vector{Int}
end

"""
    DPDirectTwoElectronWorkspace{T}()

Construct an empty reusable direct DP-NOCI two-electron workspace for floating-point type `T`.
No scientific input is accepted or mutated, all returned CPU buffers initially have zero size, no CUDA state is initialized, and unsupported element types are rejected by the type bound.
"""
function DPDirectTwoElectronWorkspace{T}() where {T<:AbstractFloat}
    return DPDirectTwoElectronWorkspace(
        0,
        0,
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        zeros(T, 0),
        zeros(T, 0),
        zeros(T, 0),
        zeros(T, 0),
        Int[],
        Int[],
    )
end

"""
    dp_two_electron_memory(T, preparation)

Estimate conservative peak CPU storage for AO cofactor factors, null modes, pair-sum matrix, orbital-pair labels, and occupied-to-AO transforms in direct DP two-electron contraction.
The preparation is read-only, returned units are bytes with zero device storage, no array is allocated, and checked overflow raises an exception.
"""
function dp_two_electron_memory(
    ::Type{T},
    preparation::DPDeterminantPairPreparation,
) where {T<:AbstractFloat}
    n_basis = size(preparation.alpha_transition_density, 1)
    occupied_count = size(preparation.cofactors.first_cofactor, 1)
    return _dp_two_electron_dimension_memory(T, n_basis, occupied_count)
end

"""
    _dp_two_electron_dimension_memory(T, n_basis, occupied_count)

Estimate direct DP two-electron CPU workspace from dimension counts without constructing a determinant-pair preparation.
Counts are dimensionless, output is host bytes with zero device bytes, no array is allocated, and negative dimensions or checked overflow raise an exception.
"""
function _dp_two_electron_dimension_memory(::Type{T}, n_basis::Integer, occupied_count::Integer) where {T<:AbstractFloat}
    basis_count = Int(n_basis)
    occupied = Int(occupied_count)
    basis_count >= 0 && occupied >= 0 || throw(ArgumentError("DP two-electron dimensions must be nonnegative"))
    n_pairs = canonical_pair_count(basis_count)
    basis_squared = Base.Checked.checked_mul(basis_count, basis_count)
    occupied_squared = Base.Checked.checked_mul(occupied, occupied)
    transform_entries = Base.Checked.checked_mul(basis_count, occupied)
    float_entries = Base.Checked.checked_add(
        Base.Checked.checked_add(Base.Checked.checked_mul(3, basis_squared), occupied_squared),
        Base.Checked.checked_add(Base.Checked.checked_mul(4, basis_count), transform_entries),
    )
    integer_entries = Base.Checked.checked_mul(2, n_pairs)
    return MemoryEstimate(
        Base.Checked.checked_add(estimate_array_bytes(T, float_entries), estimate_array_bytes(Int, integer_entries)),
        0,
    )
end

"""
    _resize_dp_two_electron_workspace!(workspace, n_basis, occupied_count)

Resize all direct DP-NOCI two-electron CPU buffers and rebuild canonical AO pair labels for exact basis and occupied dimensions.
The workspace is mutated and returned, counts are dimensionless, existing buffers are replaced only when dimensions change, no CUDA work occurs, and nonpositive basis or negative occupied counts raise `ArgumentError`.
"""
function _resize_dp_two_electron_workspace!(
    workspace::DPDirectTwoElectronWorkspace{T},
    n_basis::Integer,
    occupied_count::Integer,
) where {T<:AbstractFloat}
    basis_count = Int(n_basis)
    occupied = Int(occupied_count)
    basis_count >= 1 || throw(ArgumentError("n_basis must be positive"))
    occupied >= 0 || throw(ArgumentError("occupied_count must be nonnegative"))
    workspace.n_basis == basis_count && workspace.occupied_count == occupied && return workspace
    lower_pair_labels, upper_pair_labels = _canonical_orbital_pair_labels(basis_count)
    workspace.n_basis = basis_count
    workspace.occupied_count = occupied
    workspace.occupied_factor = zeros(T, occupied, occupied)
    workspace.alpha_factor = zeros(T, basis_count, basis_count)
    workspace.beta_factor = zeros(T, basis_count, basis_count)
    workspace.pair_sum = zeros(T, basis_count, basis_count)
    workspace.transform_scratch = zeros(T, occupied, basis_count)
    workspace.left_alpha_null = zeros(T, basis_count)
    workspace.left_beta_null = zeros(T, basis_count)
    workspace.right_alpha_null = zeros(T, basis_count)
    workspace.right_beta_null = zeros(T, basis_count)
    workspace.lower_pair_labels = lower_pair_labels
    workspace.upper_pair_labels = upper_pair_labels
    return workspace
end

"""
    _fill_dp_occupied_factor!(occupied_factor, cofactors)

Fill the rank-dependent occupied-space factor used by the direct DP-NOCI two-electron formulas.
The destination is overwritten while cofactors remain read-only, values are dimensionless CPU data, no allocation or CUDA work occurs, and a size mismatch raises `DimensionMismatch`.
"""
function _fill_dp_occupied_factor!(
    occupied_factor::AbstractMatrix{T},
    cofactors::DPFullOccupiedCofactors{T},
) where {T<:AbstractFloat}
    occupied_count = size(cofactors.first_cofactor, 1)
    size(occupied_factor) == (occupied_count, occupied_count) || throw(DimensionMismatch("DP occupied-factor workspace has incorrect dimensions"))
    if cofactors.rank_deficiency == 0
        @inbounds for index in eachindex(occupied_factor, cofactors.reduced_inverse_transpose)
            occupied_factor[index] = T(0.5) * cofactors.reduced_inverse_transpose[index]
        end
    elseif cofactors.rank_deficiency == 1
        copyto!(occupied_factor, cofactors.reduced_inverse_transpose)
    elseif cofactors.rank_deficiency == 2
        @inbounds for column in 1:occupied_count, row in 1:occupied_count
            occupied_factor[row, column] = cofactors.left_null_modes[row, 2] * cofactors.right_null_modes[column, 2]
        end
    else
        fill!(occupied_factor, zero(T))
    end
    return occupied_factor
end

"""
    _fill_dp_ao_two_electron_factors!(workspace, preparation)

Fill reusable AO cofactor factors and transformed left/right null modes for the rank-aware direct `.two` formulas.
The workspace is mutated while the preparation is read-only, all values are dimensionless CPU data, no persistent allocation or CUDA work occurs after matching resize, and inconsistent workspace or spin dimensions raise `DimensionMismatch`.
"""
function _fill_dp_ao_two_electron_factors!(
    workspace::DPDirectTwoElectronWorkspace{T},
    preparation::DPDeterminantPairPreparation{T},
) where {T<:AbstractFloat}
    cofactors = preparation.cofactors
    pair = preparation.pair
    n_alpha = cofactors.n_alpha
    n_beta = size(cofactors.first_cofactor, 1) - n_alpha
    n_basis = size(pair.left.alpha, 1)
    occupied_count = n_alpha + n_beta
    workspace.n_basis == n_basis && workspace.occupied_count == occupied_count || throw(DimensionMismatch("DP two-electron workspace dimensions differ from pair preparation"))
    size(pair.left.alpha, 2) == n_alpha && size(pair.right.alpha, 2) == n_alpha || throw(DimensionMismatch("alpha occupied count differs from cofactor partition"))
    size(pair.left.beta, 2) == n_beta && size(pair.right.beta, 2) == n_beta || throw(DimensionMismatch("beta occupied count differs from cofactor partition"))
    occupied_factor = workspace.occupied_factor
    _fill_dp_occupied_factor!(occupied_factor, cofactors)
    alpha_range = 1:n_alpha
    beta_range = (n_alpha + 1):(n_alpha + n_beta)
    alpha_scratch = @view workspace.transform_scratch[alpha_range, 1:n_basis]
    beta_scratch = @view workspace.transform_scratch[beta_range, 1:n_basis]
    mul!(alpha_scratch, @view(occupied_factor[alpha_range, alpha_range]), transpose(pair.right.alpha))
    mul!(workspace.alpha_factor, pair.left.alpha, alpha_scratch)
    mul!(beta_scratch, @view(occupied_factor[beta_range, beta_range]), transpose(pair.right.beta))
    mul!(workspace.beta_factor, pair.left.beta, beta_scratch)
    fill!(workspace.left_alpha_null, zero(T))
    fill!(workspace.left_beta_null, zero(T))
    fill!(workspace.right_alpha_null, zero(T))
    fill!(workspace.right_beta_null, zero(T))
    if cofactors.rank_deficiency in (1, 2)
        mul!(workspace.left_alpha_null, pair.left.alpha, @view(cofactors.left_null_modes[alpha_range, 1]))
        mul!(workspace.left_beta_null, pair.left.beta, @view(cofactors.left_null_modes[beta_range, 1]))
        workspace.left_alpha_null .*= cofactors.nonzero_cofactor_scale
        workspace.left_beta_null .*= cofactors.nonzero_cofactor_scale
        mul!(workspace.right_alpha_null, pair.right.alpha, @view(cofactors.right_null_modes[alpha_range, 1]))
        mul!(workspace.right_beta_null, pair.right.beta, @view(cofactors.right_null_modes[beta_range, 1]))
    end
    @inbounds for column in 1:n_basis, row in 1:n_basis
        workspace.pair_sum[row, column] = workspace.alpha_factor[row, column] + workspace.alpha_factor[column, row] + workspace.beta_factor[row, column] + workspace.beta_factor[column, row]
    end
    return workspace
end

"""
    _dp_regular_two_electron_term(factors, p, q, r, s)

Return the dimensionless full-rank direct-minus-exchange cofactor expression for one canonical AO pair-of-pairs integral.
Inputs are read-only, no allocation or device work occurs, and orbital bounds are checked by array indexing.
"""
@inline function _dp_regular_two_electron_term(factors, p::Int, q::Int, r::Int, s::Int)
    alpha = factors.alpha_factor
    beta = factors.beta_factor
    pair_sum = factors.pair_sum
    return pair_sum[p, q] * pair_sum[r, s] -
           alpha[p, s] * alpha[r, q] - alpha[s, p] * alpha[q, r] -
           beta[p, s] * beta[r, q] - beta[s, p] * beta[q, r] -
           alpha[p, r] * alpha[s, q] - alpha[r, p] * alpha[q, s] -
           beta[p, r] * beta[s, q] - beta[r, p] * beta[q, s]
end

"""
    _dp_singular_two_electron_term(factors, p, q, r, s)

Return the dimensionless nullity-one-or-two direct-minus-exchange cofactor expression for one canonical AO pair-of-pairs integral.
Inputs are read-only, zero spin components naturally remove inactive terms, no allocation or CUDA work occurs, and orbital bounds are checked by array indexing.
"""
@inline function _dp_singular_two_electron_term(factors, p::Int, q::Int, r::Int, s::Int)
    alpha = factors.alpha_factor
    beta = factors.beta_factor
    pair_sum = factors.pair_sum
    left_alpha = factors.left_alpha_null
    left_beta = factors.left_beta_null
    right_alpha = factors.right_alpha_null
    right_beta = factors.right_beta_null
    first_pair_null = left_alpha[p] * right_alpha[q] + left_alpha[q] * right_alpha[p] + left_beta[p] * right_beta[q] + left_beta[q] * right_beta[p]
    return pair_sum[q, p] * (left_alpha[r] * right_alpha[s] + left_alpha[s] * right_alpha[r] + left_beta[r] * right_beta[s] + left_beta[s] * right_beta[r]) +
           first_pair_null * pair_sum[s, r] -
           right_alpha[p] * (alpha[q, s] * left_alpha[r] + alpha[q, r] * left_alpha[s]) -
           right_beta[p] * (beta[q, s] * left_beta[r] + beta[q, r] * left_beta[s]) -
           right_alpha[r] * (alpha[p, s] * left_alpha[q] + alpha[q, s] * left_alpha[p]) -
           right_beta[r] * (beta[p, s] * left_beta[q] + beta[q, s] * left_beta[p]) -
           right_alpha[q] * (alpha[p, s] * left_alpha[r] + alpha[p, r] * left_alpha[s]) -
           right_beta[q] * (beta[p, s] * left_beta[r] + beta[p, r] * left_beta[s]) -
           right_alpha[s] * (alpha[p, r] * left_alpha[q] + alpha[q, r] * left_alpha[p]) -
           right_beta[s] * (beta[p, r] * left_beta[q] + beta[q, r] * left_beta[p])
end

"""
    _canonical_orbital_pair_labels(n_basis)

Return CPU vectors containing lower and upper one-based orbital labels for every canonical symmetric AO pair.
The basis count is dimensionless, outputs are owned integer arrays, no device work occurs, and nonpositive counts raise `ArgumentError`.
"""
function _canonical_orbital_pair_labels(n_basis::Integer)
    n = Int(n_basis)
    n >= 1 || throw(ArgumentError("n_basis must be positive"))
    n_pairs = canonical_pair_count(n)
    lower_labels = Vector{Int}(undef, n_pairs)
    upper_labels = Vector{Int}(undef, n_pairs)
    pair_index = 0
    for upper in 1:n, lower in 1:upper
        pair_index += 1
        lower_labels[pair_index] = lower
        upper_labels[pair_index] = upper
    end
    return lower_labels, upper_labels
end

"""
    contract_dp_two_electron!(workspace, preparation, integrals; runtime=RuntimeOptions(), telemetry=nothing)

Contract physical packed conventional `(pq|rs)` integrals directly with rank-aware full-occupied DP cofactor factors and return the bare electronic two-body matrix element.
The workspace is resized and mutated while scientific inputs are read-only, output follows ERI energy units, host storage is preflighted, physical-to-canonical multiplicities are explicit, nullity three or greater returns zero, and this CPU-workspace overload rejects CUDA requests.
"""
function contract_dp_two_electron!(
    workspace::DPDirectTwoElectronWorkspace{T},
    preparation::DPDeterminantPairPreparation{T},
    integrals::PackedERIValues{T};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    runtime.use_gpu && throw(ArgumentError("a CPU DP two-electron workspace cannot satisfy a CUDA runtime request"))
    n_basis = size(preparation.alpha_transition_density, 1)
    integrals.n_orbitals == n_basis || throw(DimensionMismatch("packed ERI and DP pair AO dimensions differ"))
    estimate = dp_two_electron_memory(T, preparation)
    enforce_memory_limits(estimate, runtime)
    occupied_count = size(preparation.cofactors.first_cofactor, 1)
    needs_resize = workspace.n_basis != n_basis || workspace.occupied_count != occupied_count
    if needs_resize
        resize_work = () -> _resize_dp_two_electron_workspace!(workspace, n_basis, occupied_count)
        isnothing(telemetry) ? resize_work() : record_phase!(
            resize_work,
            telemetry,
            :resize_dp_two_electron_workspace;
            estimate,
            metadata=Dict(:n_basis => n_basis, :occupied_count => occupied_count),
            collect_allocations=runtime.collect_allocations,
        )
        !isnothing(telemetry) && increment_counter!(telemetry, :n_dp_two_electron_workspace_resizes)
    end
    work = function ()
        preparation.cofactors.rank_deficiency >= 3 && return zero(T)
        factors = _fill_dp_ao_two_electron_factors!(workspace, preparation)
        n_pairs = length(workspace.lower_pair_labels)
        accumulation = zero(T)
        @inbounds for first_pair in 1:n_pairs
            p = workspace.lower_pair_labels[first_pair]
            q = workspace.upper_pair_labels[first_pair]
            first_scale = p == q ? T(0.5) : one(T)
            for second_pair in first_pair:n_pairs
                r = workspace.lower_pair_labels[second_pair]
                s = workspace.upper_pair_labels[second_pair]
                second_scale = r == s ? T(0.5) : one(T)
                pair_scale = first_pair == second_pair ? T(0.5) : one(T)
                integral = integrals.values[canonical_pair_index(first_pair, second_pair)] * first_scale * second_scale * pair_scale
                term = preparation.cofactors.rank_deficiency == 0 ?
                       _dp_regular_two_electron_term(factors, p, q, r, s) :
                       _dp_singular_two_electron_term(factors, p, q, r, s)
                accumulation += integral * term
            end
        end
        return preparation.cofactors.rank_deficiency == 0 ? T(4) * preparation.cofactors.overlap * accumulation : accumulation
    end
    value = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :contract_dp_two_electron;
        estimate,
        metadata=Dict(:n_packed_eri_values => length(integrals.values), :rank_deficiency => preparation.cofactors.rank_deficiency, :workspace_reused => !needs_resize),
        collect_allocations=runtime.collect_allocations,
    )
    !isnothing(telemetry) && increment_counter!(telemetry, :n_dp_two_electron_contractions)
    return value
end

"""
    contract_dp_two_electron(preparation, integrals; runtime=RuntimeOptions(), telemetry=nothing)

Allocate one direct DP-NOCI workspace, contract physical packed conventional `(pq|rs)` integrals, and return the bare electronic two-body matrix element.
Scientific inputs are read-only, output follows ERI energy units, all host storage is preflighted, this CPU compatibility wrapper does not retain buffers between calls, and CHO-ERI or CUDA requests are rejected.
"""
function contract_dp_two_electron(
    preparation::DPDeterminantPairPreparation{T},
    integrals::PackedERIValues{T};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    workspace = DPDirectTwoElectronWorkspace{T}()
    return contract_dp_two_electron!(workspace, preparation, integrals; runtime, telemetry)
end

"""
    contract_dp_determinant_pair!(workspace, preparation, one_body_operator, integrals; runtime=RuntimeOptions(), telemetry=nothing)

Return one complete bare conventional DP-NOCI determinant-pair `S/H1/H2` value while reusing the supplied direct two-electron workspace.
The CPU workspace is mutated and scientific inputs are read-only, Hamiltonian components follow integral energy units, host memory is preflighted, no nuclear/CI weighting occurs, CUDA requests are rejected, and dimension errors propagate.
"""
function contract_dp_determinant_pair!(
    workspace::DPDirectTwoElectronWorkspace{T},
    preparation::DPDeterminantPairPreparation{T},
    one_body_operator::AbstractMatrix{T},
    integrals::PackedERIValues{T};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    one_body = contract_dp_one_body(preparation, one_body_operator)
    two_body = contract_dp_two_electron!(workspace, preparation, integrals; runtime, telemetry)
    return DPDeterminantPairMatrixElement(one_body.overlap, one_body.electronic_one_body, two_body)
end

"""
    contract_dp_determinant_pair(preparation, one_body_operator, integrals; runtime=RuntimeOptions(), telemetry=nothing)

Return one complete bare conventional DP-NOCI determinant-pair `S/H1/H2` value from a prepared full-occupied pair, a physical AO one-body operator, and packed `.two` ERIs.
Inputs are read-only, Hamiltonian components follow integral energy units, this compatibility wrapper allocates one preflighted CPU workspace, no nuclear/CI weighting occurs, CUDA requests are rejected, and dimension errors propagate.
"""
function contract_dp_determinant_pair(
    preparation::DPDeterminantPairPreparation{T},
    one_body_operator::AbstractMatrix{T},
    integrals::PackedERIValues{T};
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    workspace = DPDirectTwoElectronWorkspace{T}()
    return contract_dp_determinant_pair!(workspace, preparation, one_body_operator, integrals; runtime, telemetry)
end
