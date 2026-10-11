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

 File:            src/algorithms/ahpc_noci/complementary_cofactors.jl
 Purpose:         Division-free Laplace transitions between adjacent subset-cardinality AHPC cofactor layers.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    ComplementaryCofactorTransition

Store one division-free Laplace transition between adjacent subset-cardinality layers.
Indices and signs are dimensionless, instances are immutable CPU data shared by all worker workspaces, and no numerical matrix values are retained.
"""
struct ComplementaryCofactorTransition
    previous_state::Int32
    next_state::Int32
    previous_gradient_offset::Int32
    next_gradient_map_offset::Int32
    current_gradient::Int32
    current_response::UInt16
    sign::Int8
end

"""
    ComplementaryCofactorLayerPlan

Store the ordered transitions plus structurally nonzero previous-response and next-gradient indices for one processed overlap row.
The plan is read-only, uses one-based local response indices and zero-based transition offsets, owns CPU vectors, and is safe to share across Julia threads.
"""
struct ComplementaryCofactorLayerPlan
    previous_cardinality::Int
    previous_state_count::Int
    next_state_count::Int
    previous_gradient_entries::Int
    next_gradient_entries::Int
    previous_response_indices::Vector{UInt16}
    next_gradient_indices::Vector{Int32}
    transitions::Vector{ComplementaryCofactorTransition}
end

"""
    ComplementaryCofactorPlan

Store the reusable rank-uniform subset circuit for one occupied-spin dimension `q`.
The plan contains only dimensionless CPU metadata, preserves increasing mask and column order, and has no rank classifier, matrix-dependent branch, inverse, SVD, or CUDA state.
"""
struct ComplementaryCofactorPlan
    n_active_occupied::Int
    layers::Vector{ComplementaryCofactorLayerPlan}
    maximum_state_count::Int
    maximum_gradient_entries::Int
    transition_count::Int
    structural_response_iterations::Int
end

const _COMPLEMENTARY_COFACTOR_PLAN_CACHE = Dict{Int,ComplementaryCofactorPlan}()
const _COMPLEMENTARY_COFACTOR_PLAN_LOCK = ReentrantLock()

"""
    _cofactor_workspace_shape(n_active_occupied)

Return maximum cardinality-layer state and structural-gradient entry counts for an occupied-spin dimension.
The result is dimensionless, no arrays are allocated, and dimensions outside `0:20` raise `ArgumentError`.
"""
function _cofactor_workspace_shape(n_active_occupied::Int)
    0 <= n_active_occupied <= 20 || throw(ArgumentError("n_active_occupied must lie in 0:20"))
    maximum_state_count = 1
    maximum_gradient_entries = 0
    for cardinality in 0:n_active_occupied
        state_count = binomial(n_active_occupied, cardinality)
        maximum_state_count = max(maximum_state_count, state_count)
        maximum_gradient_entries = max(
            maximum_gradient_entries,
            Base.Checked.checked_mul(state_count, Base.Checked.checked_mul(cardinality, cardinality)),
        )
    end
    return maximum_state_count, maximum_gradient_entries
end

"""
    _cofactor_plan_memory(n_active_occupied)

Return a conservative host numerical-payload estimate for one immutable compressed recurrence plan and its peak construction scratch.
The estimate counts transition records, response and destination maps, inline layer records, subset-position scratch, and mask scratch in bytes while excluding Julia object headers, reports zero device bytes, allocates no plan, and rejects dimensions outside `0:20`.
"""
function _cofactor_plan_memory(n_active_occupied::Int)
    q = n_active_occupied
    0 <= q <= 20 || throw(ArgumentError("production complementary cofactors support 0:20 occupied active orbitals"))
    n_subsets = 1 << q
    transition_count = q == 0 ? 0 : Base.Checked.checked_mul(q, 1 << (q - 1))
    previous_response_count = 0
    next_gradient_map_count = 0
    maximum_layer_states = 0
    for previous_cardinality in 0:(q - 1)
        previous_state_count = binomial(q, previous_cardinality)
        transitions = Base.Checked.checked_mul(previous_state_count, q - previous_cardinality)
        previous_width = Base.Checked.checked_mul(previous_cardinality, previous_cardinality)
        previous_response_count = Base.Checked.checked_add(
            previous_response_count,
            Base.Checked.checked_mul(previous_state_count, previous_width),
        )
        next_gradient_map_count = Base.Checked.checked_add(
            next_gradient_map_count,
            Base.Checked.checked_mul(transitions, previous_width),
        )
        maximum_layer_states = max(maximum_layer_states, previous_state_count)
    end
    retained_bytes = Base.Checked.checked_add(
        Base.Checked.checked_add(
            Base.Checked.checked_mul(transition_count, sizeof(ComplementaryCofactorTransition)),
            Base.Checked.checked_mul(previous_response_count, sizeof(UInt16)),
        ),
        Base.Checked.checked_add(
            Base.Checked.checked_mul(next_gradient_map_count, sizeof(Int32)),
            Base.Checked.checked_mul(q, sizeof(ComplementaryCofactorLayerPlan)),
        ),
    )
    construction_scratch = Base.Checked.checked_add(
        Base.Checked.checked_mul(n_subsets, sizeof(Int32)),
        Base.Checked.checked_mul(maximum_layer_states, sizeof(Int)),
    )
    return MemoryEstimate(Base.Checked.checked_add(retained_bytes, construction_scratch), 0)
end

"""
    _build_complementary_cofactor_plan(n_active_occupied)

Build the immutable transition circuit and structural response maps for one occupied-spin dimension.
The returned plan is CPU-owned and matrix independent, transition order matches the original row/mask/column recurrence, and invalid or overflowing dimensions raise before numerical work.
"""
function _build_complementary_cofactor_plan(n_active_occupied::Int)
    q = n_active_occupied
    maximum_state_count, maximum_gradient_entries = _cofactor_workspace_shape(q)
    n_subsets = 1 << q
    layers = Vector{ComplementaryCofactorLayerPlan}(undef, q)
    transition_count = 0
    structural_response_iterations = 0

    for row in 1:q
        previous_cardinality = row - 1
        previous_masks = Int[]
        next_state_positions = zeros(Int32, n_subsets)
        next_state_count = 0
        for subset_mask in 0:(n_subsets - 1)
            cardinality = count_ones(subset_mask)
            cardinality == previous_cardinality && push!(previous_masks, subset_mask)
            if cardinality == row
                next_state_count += 1
                next_state_positions[subset_mask + 1] = Int32(next_state_count)
            end
        end

        previous_state_count = length(previous_masks)
        previous_gradient_entries = Base.Checked.checked_mul(previous_state_count, Base.Checked.checked_mul(previous_cardinality, previous_cardinality))
        next_gradient_entries = Base.Checked.checked_mul(next_state_count, Base.Checked.checked_mul(row, row))
        previous_response_indices = Vector{UInt16}(undef, previous_gradient_entries)
        response_slot = 1
        for subset_mask in previous_masks
            remaining_columns = subset_mask
            while !iszero(remaining_columns)
                column = trailing_zeros(remaining_columns) + 1
                for previous_row in 1:previous_cardinality
                    previous_response_indices[response_slot] = UInt16(_response_index(previous_row, column, q))
                    response_slot += 1
                end
                remaining_columns &= remaining_columns - 1
            end
        end
        response_slot == previous_gradient_entries + 1 || throw(ArgumentError("complementary-cofactor structural response plan is incomplete"))

        transitions = Vector{ComplementaryCofactorTransition}()
        next_gradient_indices = Int32[]
        sizehint!(transitions, Base.Checked.checked_mul(previous_state_count, q - previous_cardinality))
        sizehint!(
            next_gradient_indices,
            Base.Checked.checked_mul(
                Base.Checked.checked_mul(previous_state_count, q - previous_cardinality),
                Base.Checked.checked_mul(previous_cardinality, previous_cardinality),
            ),
        )
        for (previous_state, subset_mask) in pairs(previous_masks)
            for column in 1:q
                column_bit = 1 << (column - 1)
                (subset_mask & column_bit) == 0 || continue
                next_mask = subset_mask | column_bit
                next_state = next_state_positions[next_mask + 1]
                next_state > 0 || throw(ArgumentError("complementary-cofactor transition has no next-layer state"))
                insertion_position = count_ones(subset_mask & (column_bit - 1)) + 1
                sign = isodd(row + insertion_position) ? Int8(-1) : Int8(1)
                previous_gradient_offset = (previous_state - 1) * previous_cardinality * previous_cardinality
                next_base = (Int(next_state) - 1) * row * row
                next_gradient_map_offset = length(next_gradient_indices)
                for previous_column_position in 1:previous_cardinality
                    next_column_position = previous_column_position < insertion_position ? previous_column_position : previous_column_position + 1
                    next_column_base = next_base + (next_column_position - 1) * row
                    for previous_row in 1:previous_cardinality
                        push!(next_gradient_indices, Int32(next_column_base + previous_row))
                    end
                end
                current_gradient = next_base + (insertion_position - 1) * row + row
                push!(
                    transitions,
                    ComplementaryCofactorTransition(
                        Int32(previous_state),
                        next_state,
                        Int32(previous_gradient_offset),
                        Int32(next_gradient_map_offset),
                        Int32(current_gradient),
                        UInt16(_response_index(row, column, q)),
                        sign,
                    ),
                )
            end
        end
        transition_count = Base.Checked.checked_add(transition_count, length(transitions))
        structural_response_iterations = Base.Checked.checked_add(
            structural_response_iterations,
            Base.Checked.checked_mul(length(transitions), Base.Checked.checked_mul(previous_cardinality, previous_cardinality)),
        )
        layers[row] = ComplementaryCofactorLayerPlan(
            previous_cardinality,
            previous_state_count,
            next_state_count,
            previous_gradient_entries,
            next_gradient_entries,
            previous_response_indices,
            next_gradient_indices,
            transitions,
        )
    end

    return ComplementaryCofactorPlan(
        q,
        layers,
        maximum_state_count,
        maximum_gradient_entries,
        transition_count,
        structural_response_iterations,
    )
end

"""
    _complementary_cofactor_plan(n_active_occupied)

Return the process-wide immutable recurrence plan for one occupied-spin dimension, building it once on first use.
The cache is protected by a lock, returned plans are read-only and thread safe, and no scientific input or CUDA state is touched.
"""
function _complementary_cofactor_plan(n_active_occupied::Int)
    lock(_COMPLEMENTARY_COFACTOR_PLAN_LOCK) do
        return get!(_COMPLEMENTARY_COFACTOR_PLAN_CACHE, n_active_occupied) do
            _build_complementary_cofactor_plan(n_active_occupied)
        end
    end
end

"""
    ComplementaryCofactorWorkspace{T}()

Reusable host workspace for the AHPC rank-uniform cardinality-layer recurrence with floating-point type `T`.
The workspace owns two scalar layers, two structurally compressed derivative layers, scaled-overlap, symmetric-kernel, and first-cofactor buffers and is not thread safe.
"""
mutable struct ComplementaryCofactorWorkspace{T<:AbstractFloat}
    n_active_occupied::Int
    plan::Union{Nothing,ComplementaryCofactorPlan}
    subset_a::Vector{T}
    subset_b::Vector{T}
    second_a::Vector{T}
    second_b::Vector{T}
    first_a::Vector{T}
    first_b::Vector{T}
    scaled_overlap::Matrix{T}
    symmetric_kernel::Matrix{T}
    first_cofactor::Matrix{T}
end

"""
    ComplementaryCofactorWorkspace{T}()

Construct an empty compressed recurrence workspace without molecular-scale allocation.
The workspace grows on first use, borrows an immutable shared circuit plan, and may then be reused only serially.
"""
function ComplementaryCofactorWorkspace{T}() where {T<:AbstractFloat}
    return ComplementaryCofactorWorkspace{T}(
        -1,
        nothing,
        T[],
        T[],
        T[],
        T[],
        T[],
        T[],
        zeros(T, 0, 0),
        zeros(T, 0, 0),
        zeros(T, 0, 0),
    )
end

"""
    ComplementaryCofactorWorkspace()

Construct an empty `Float64` AHPC compressed recurrence workspace.
This convenience constructor performs no molecular-scale allocation or CUDA initialization.
"""
ComplementaryCofactorWorkspace() = ComplementaryCofactorWorkspace{Float64}()

"""
    ComplementaryCofactorResult

Return value of the AHPC complementary-cofactor recurrence.
`cofactor_scalar` is dimensionless, `first_cofactor` follows the overlap element type, `contracted_second` has the units supplied by the response kernel, and `binary_scaling_exponent` records exact power-of-two normalization.
"""
struct ComplementaryCofactorResult{T<:AbstractFloat,M<:AbstractMatrix{T}}
    cofactor_scalar::T
    first_cofactor::M
    contracted_second::T
    binary_scaling_exponent::Int
end

"""
    cofactor_workspace_memory(T, n_active_occupied)

Return the exact numerical payload estimate for one compressed recurrence workspace as host bytes and zero device bytes.
The estimate includes the symmetric local response kernel but excludes shared immutable plan metadata and Julia object headers, allocates no workspace, and rejects dimensions outside `0:20`.
"""
function cofactor_workspace_memory(::Type{T}, n_active_occupied::Integer) where {T<:AbstractFloat}
    q = Int(n_active_occupied)
    maximum_state_count, maximum_gradient_entries = _cofactor_workspace_shape(q)
    n_response = Base.Checked.checked_mul(q, q)
    scalar_layers = estimate_array_bytes(T, 4, maximum_state_count)
    derivative_layers = estimate_array_bytes(T, 2, maximum_gradient_entries)
    square_buffers = estimate_array_bytes(T, 2, q, q)
    symmetric_kernel = estimate_array_bytes(T, n_response, n_response)
    return MemoryEstimate(
        Base.Checked.checked_add(
            Base.Checked.checked_add(scalar_layers, derivative_layers),
            Base.Checked.checked_add(square_buffers, symmetric_kernel),
        ),
        0,
    )
end

"""
    _response_index(row, column, n_active_occupied)

Map a one-based overlap element to its column-major active-response index.
Arguments must already be in range; this allocation-free helper is used only inside validated recurrence-plan construction and hot loops.
"""
@inline function _response_index(row::Int, column::Int, n_active_occupied::Int)
    return row + (column - 1) * n_active_occupied
end

"""
    _prepare_cofactor_workspace!(workspace, n_active_occupied, runtime)

Validate the occupied-spin dimension, enforce host-memory limits, attach the immutable recurrence plan, and resize worker-local compressed buffers.
The function mutates `workspace`, allocates only after preflight passes, does not clear buffers needed only by a later kernel, and never initializes a GPU.
"""
function _prepare_cofactor_workspace!(
    workspace::ComplementaryCofactorWorkspace{T},
    n_active_occupied::Int,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    0 <= n_active_occupied <= runtime.max_active_occupied || throw(ArgumentError(
        "n_active_occupied=$n_active_occupied exceeds configured max_active_occupied=$(runtime.max_active_occupied)",
    ))
    memory = combine_memory_estimates(
        cofactor_workspace_memory(T, n_active_occupied),
        _cofactor_plan_memory(n_active_occupied),
    )
    enforce_memory_limits(memory, runtime)
    plan = _complementary_cofactor_plan(n_active_occupied)
    q = n_active_occupied
    n_response = q * q
    if workspace.n_active_occupied != q || workspace.plan !== plan
        workspace.n_active_occupied = q
        workspace.plan = plan
        workspace.subset_a = zeros(T, plan.maximum_state_count)
        workspace.subset_b = zeros(T, plan.maximum_state_count)
        workspace.second_a = zeros(T, plan.maximum_state_count)
        workspace.second_b = zeros(T, plan.maximum_state_count)
        workspace.first_a = zeros(T, plan.maximum_gradient_entries)
        workspace.first_b = zeros(T, plan.maximum_gradient_entries)
        workspace.scaled_overlap = zeros(T, q, q)
        workspace.symmetric_kernel = zeros(T, n_response, n_response)
        workspace.first_cofactor = zeros(T, q, q)
    end
    return plan
end

"""
    _fill_symmetric_local_kernel!(destination, response_kernel)

Fill a symmetric local kernel with `response_kernel[first,second] + response_kernel[second,first]` for the contracted-second recurrence.
The destination is overwritten on CPU, the finite source is read-only, addition order follows the upper-triangle orientation, and dimensions must already be validated by the caller.
"""
function _fill_symmetric_local_kernel!(
    destination::Matrix{T},
    response_kernel::AbstractMatrix{<:Real},
) where {T<:AbstractFloat}
    n_response = size(destination, 1)
    @inbounds for second_response in 1:n_response
        for first_response in 1:second_response
            value = T(response_kernel[first_response, second_response]) + T(response_kernel[second_response, first_response])
            destination[first_response, second_response] = value
            destination[second_response, first_response] = value
        end
    end
    return destination
end

"""
    _contracted_complementary_cofactors_symmetric!(workspace, effective_active_overlap, plan)

Evaluate zeroth and first complementary cofactors and the symmetric-kernel-contracted second derivative with the compact cardinality-layer circuit.
Inputs and workspace dimensions must already be validated, `workspace.symmetric_kernel` must already contain the two-direction kernel sum, output aliases `workspace.first_cofactor`, and the CPU routine is division free and rank uniform.
"""
function _contracted_complementary_cofactors_symmetric!(
    workspace::ComplementaryCofactorWorkspace{T},
    effective_active_overlap::AbstractMatrix{<:Real},
    plan::ComplementaryCofactorPlan,
) where {T<:AbstractFloat}
    q = plan.n_active_occupied
    if q == 0
        return ComplementaryCofactorResult(one(T), workspace.first_cofactor, zero(T), 0)
    end

    maximum_absolute_overlap = zero(T)
    @inbounds for column in 1:q, row in 1:q
        value = T(effective_active_overlap[row, column])
        maximum_absolute_overlap = max(maximum_absolute_overlap, abs(value))
    end
    binary_scaling_exponent = maximum_absolute_overlap == zero(T) ? 0 : last(frexp(maximum_absolute_overlap))
    @inbounds for column in 1:q, row in 1:q
        workspace.scaled_overlap[row, column] = ldexp(T(effective_active_overlap[row, column]), -binary_scaling_exponent)
    end

    previous_subset = workspace.subset_a
    next_subset = workspace.subset_b
    previous_second = workspace.second_a
    next_second = workspace.second_b
    previous_first = workspace.first_a
    next_first = workspace.first_b
    previous_subset[1] = one(T)
    previous_second[1] = zero(T)

    @inbounds for row in 1:q
        layer = plan.layers[row]
        next_state_count = layer.next_state_count
        next_gradient_entries = layer.next_gradient_entries
        fill!(view(next_subset, 1:next_state_count), zero(T))
        fill!(view(next_second, 1:next_state_count), zero(T))
        next_gradient_entries > 0 && fill!(view(next_first, 1:next_gradient_entries), zero(T))
        previous_cardinality = layer.previous_cardinality
        previous_state_width = previous_cardinality * previous_cardinality

        for transition in layer.transitions
            previous_state = Int(transition.previous_state)
            next_state = Int(transition.next_state)
            previous_base = Int(transition.previous_gradient_offset)
            next_gradient_map_offset = Int(transition.next_gradient_map_offset)
            current_gradient = Int(transition.current_gradient)
            current_response = Int(transition.current_response)
            sign = T(transition.sign)
            overlap_element = workspace.scaled_overlap[current_response]
            signed_overlap = sign * overlap_element
            previous_delta = previous_subset[previous_state]
            previous_quadratic = previous_second[previous_state]

            next_subset[next_state] += signed_overlap * previous_delta
            cross_term = zero(T)
            for previous_gradient in 1:previous_state_width
                previous_gradient_index = previous_base + previous_gradient
                previous_value = previous_first[previous_gradient_index]
                next_gradient_index = Int(layer.next_gradient_indices[next_gradient_map_offset + previous_gradient])
                next_first[next_gradient_index] += signed_overlap * previous_value
                response = Int(layer.previous_response_indices[previous_gradient_index])
                cross_term += workspace.symmetric_kernel[current_response, response] * previous_value
            end
            next_first[current_gradient] += sign * previous_delta
            next_second[next_state] += sign * (overlap_element * previous_quadratic + cross_term)
        end

        previous_subset, next_subset = next_subset, previous_subset
        previous_second, next_second = next_second, previous_second
        previous_first, next_first = next_first, previous_first
    end

    cofactor_scalar = ldexp(previous_subset[1], binary_scaling_exponent * q)
    first_scaling_exponent = binary_scaling_exponent * (q - 1)
    second_scaling_exponent = binary_scaling_exponent * max(q - 2, 0)
    @inbounds for response in 1:(q * q)
        workspace.first_cofactor[response] = ldexp(previous_first[response], first_scaling_exponent)
    end
    contracted_second = q < 2 ? zero(T) : ldexp(previous_second[1], second_scaling_exponent)
    return ComplementaryCofactorResult(cofactor_scalar, workspace.first_cofactor, contracted_second, binary_scaling_exponent)
end

"""
    contracted_complementary_cofactors!(workspace, effective_active_overlap, coulomb_minus_exchange; runtime=RuntimeOptions())

Validate and evaluate the zeroth and complete first complementary cofactors and the contraction of the second derivative with a local Coulomb-minus-exchange response matrix.
The overlap is square with dimension `q`, the response matrix is `q^2` by `q^2` in column-major response order, inputs are not mutated, output aliases `workspace`, and invalid dimensions, non-finite inputs, outputs, or resource limits raise before returning.
"""
function contracted_complementary_cofactors!(
    workspace::ComplementaryCofactorWorkspace{T},
    effective_active_overlap::AbstractMatrix{<:Real},
    coulomb_minus_exchange::AbstractMatrix{<:Real};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    q = size(effective_active_overlap, 1)
    size(effective_active_overlap, 2) == q || throw(DimensionMismatch("effective_active_overlap must be square"))
    0 <= q <= runtime.max_active_occupied || throw(ArgumentError(
        "n_active_occupied=$q exceeds configured max_active_occupied=$(runtime.max_active_occupied)",
    ))
    n_response = q * q
    size(coulomb_minus_exchange) == (n_response, n_response) || throw(DimensionMismatch(
        "coulomb_minus_exchange must have size (q^2, q^2)",
    ))
    all(isfinite, effective_active_overlap) || throw(ArgumentError("effective_active_overlap contains a non-finite value"))
    all(isfinite, coulomb_minus_exchange) || throw(ArgumentError("coulomb_minus_exchange contains a non-finite value"))

    plan = _prepare_cofactor_workspace!(workspace, q, runtime)
    _fill_symmetric_local_kernel!(workspace.symmetric_kernel, coulomb_minus_exchange)
    result = _contracted_complementary_cofactors_symmetric!(workspace, effective_active_overlap, plan)
    isfinite(result.cofactor_scalar) || throw(ArgumentError("complementary-cofactor scalar is non-finite"))
    all(isfinite, result.first_cofactor) || throw(ArgumentError("first complementary cofactors contain a non-finite value"))
    isfinite(result.contracted_second) || throw(ArgumentError("contracted second complementary cofactor is non-finite"))
    return result
end

"""
    contracted_complementary_cofactors(effective_active_overlap, coulomb_minus_exchange; runtime=RuntimeOptions())

Allocate one compressed recurrence workspace, evaluate rank-uniform complementary cofactors, and return an independently owned first-cofactor matrix.
The wrapper executes on the CPU, does not mutate inputs, and raises dimension, finite-value, or resource-limit errors before returning invalid output.
"""
function contracted_complementary_cofactors(
    effective_active_overlap::AbstractMatrix{T},
    coulomb_minus_exchange::AbstractMatrix{<:Real};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    workspace = ComplementaryCofactorWorkspace{T}()
    result = contracted_complementary_cofactors!(
        workspace,
        effective_active_overlap,
        coulomb_minus_exchange;
        runtime=runtime,
    )
    return ComplementaryCofactorResult(
        result.cofactor_scalar,
        copy(result.first_cofactor),
        result.contracted_second,
        result.binary_scaling_exponent,
    )
end
