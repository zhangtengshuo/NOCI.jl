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

 File:            src/runtime/cuda_resources/precontraction_cuda.jl
 Purpose:         CUDA AHPC precontraction: projected ERI-factor shards and streamed conventional ERI tiles.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _packed_streamed_feature_count(space)

Return the number of canonical-pair feature channels needed for exact streamed conventional-ERI AHPC Coulomb and exchange reconstruction.
The response space is read-only, the result is dimensionless, no host/device array is allocated, and checked overflow raises before execution.
"""
function _packed_streamed_feature_count(space::ActiveResponseSpace)
    inactive_rank = size(space.inactive_density_left_factor, 2)
    active_rank = size(space.left_active_response, 2)
    active_square = Base.Checked.checked_mul(active_rank, active_rank)
    cross = Base.Checked.checked_mul(inactive_rank, active_rank)
    inactive_square = Base.Checked.checked_mul(inactive_rank, inactive_rank)
    return Base.Checked.checked_add(
        Base.Checked.checked_add(1, active_square),
        Base.Checked.checked_add(Base.Checked.checked_mul(2, cross), inactive_square),
    )
end

"""
    _packed_streamed_pair_tile(T, n_pairs, n_features, scratch_bytes)

Choose the largest canonical-pair tile whose square operator tile plus rectangular GEMM scratch fits the typed runtime scratch budget.
Counts are dimensionless and scratch is measured in bytes, no allocation or CUDA initialization occurs, and an insufficient budget raises `ArgumentError`.
"""
function _packed_streamed_pair_tile(
    ::Type{T},
    n_pairs::Integer,
    n_features::Integer,
    scratch_bytes::Integer,
) where {T<:AbstractFloat}
    pair_count = Int(n_pairs)
    feature_count = Int(n_features)
    budget = Int(scratch_bytes)
    pair_count >= 1 && feature_count >= 1 || throw(ArgumentError("streamed packed dimensions must be positive"))
    budget >= 0 || throw(ArgumentError("scratch_bytes must be nonnegative"))
    entry_budget = budget ÷ sizeof(T)
    low = 0
    high = pair_count
    while low < high
        candidate = low + (high - low + 1) ÷ 2
        required = Base.Checked.checked_add(
            Base.Checked.checked_mul(candidate, candidate),
            Base.Checked.checked_mul(candidate, feature_count),
        )
        if required <= entry_budget
            low = candidate
        else
            high = candidate - 1
        end
    end
    low >= 1 || throw(ArgumentError("scratch_bytes must hold at least one streamed ERI row and feature row"))
    return low
end

"""
    _packed_streamed_gpu_precontraction_memory(T, space, integrals, runtime)

Estimate host/device numerical storage for exact canonical-pair-feature AHPC precontraction with streamed conventional-ERI tiles on one GPU.
Inputs are read-only, returned units are bytes, packed ERI input storage is excluded on the host but included for device upload, the tile follows `runtime.scratch_bytes`, and invalid dimensions or checked overflow raise before allocation.
"""
function _packed_streamed_gpu_precontraction_memory(
    ::Type{T},
    space::ActiveResponseSpace,
    integrals::PackedERIValues,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    n_pairs = canonical_pair_count(integrals.n_orbitals)
    n_features = _packed_streamed_feature_count(space)
    tile = _packed_streamed_pair_tile(T, n_pairs, n_features, runtime.scratch_bytes)
    feature_bytes = estimate_array_bytes(T, n_pairs, n_features)
    gram_bytes = estimate_array_bytes(T, n_features, n_features)
    tile_bytes = estimate_array_bytes(T, tile, tile)
    product_bytes = estimate_array_bytes(T, tile, n_features)
    label_bytes = estimate_array_bytes(Int, 2, n_pairs)
    host_bytes = Base.Checked.checked_add(
        _precontraction_output_memory(T, active_response_count(space)).host_bytes,
        Base.Checked.checked_add(label_bytes, Base.Checked.checked_add(feature_bytes, gram_bytes)),
    )
    device_bytes = Base.Checked.checked_add(
        estimate_array_bytes(T, length(integrals.values)),
        Base.Checked.checked_add(
            Base.Checked.checked_add(feature_bytes, gram_bytes),
            Base.Checked.checked_add(tile_bytes, product_bytes),
        ),
    )
    return MemoryEstimate(host_bytes, device_bytes)
end

"""
    _fill_packed_streamed_features!(features, lower_labels, upper_labels, space)

Fill exact canonical-pair density features for inactive, active-active, inactive-active exchange, and inactive-inactive exchange channels.
The host destination is overwritten while labels and response-space factors remain read-only, values are dimensionless, no CUDA operation occurs, and shape inconsistencies raise `DimensionMismatch`.
"""
function _fill_packed_streamed_features!(
    features::AbstractMatrix{T},
    lower_labels::AbstractVector{<:Integer},
    upper_labels::AbstractVector{<:Integer},
    space::ActiveResponseSpace{T},
) where {T<:AbstractFloat}
    n_pairs = length(lower_labels)
    length(upper_labels) == n_pairs || throw(DimensionMismatch("canonical-pair label lengths differ"))
    n_features = _packed_streamed_feature_count(space)
    size(features) == (n_pairs, n_features) || throw(DimensionMismatch("streamed packed feature workspace has incorrect dimensions"))
    inactive_rank = size(space.inactive_density_left_factor, 2)
    active_rank = size(space.left_active_response, 2)
    active_offset = 1
    x_offset = active_offset + active_rank * active_rank
    y_offset = x_offset + inactive_rank * active_rank
    z_offset = y_offset + active_rank * inactive_rank
    @inbounds for pair in 1:n_pairs
        p = Int(lower_labels[pair])
        q = Int(upper_labels[pair])
        offdiagonal = p != q
        features[pair, 1] = space.inactive_density_kernel[p, q] + (offdiagonal ? space.inactive_density_kernel[q, p] : zero(T))
        for b in 1:active_rank, a in 1:active_rank
            channel = active_offset + active_response_index(a, b, active_rank)
            value = space.left_active_response[p, a] * space.right_active_response[q, b]
            offdiagonal && (value += space.left_active_response[q, a] * space.right_active_response[p, b])
            features[pair, channel] = value
        end
        for b in 1:active_rank, i in 1:inactive_rank
            channel = x_offset + i + (b - 1) * inactive_rank
            value = space.inactive_density_left_factor[p, i] * space.right_active_response[q, b]
            offdiagonal && (value += space.inactive_density_left_factor[q, i] * space.right_active_response[p, b])
            features[pair, channel] = value
        end
        for i in 1:inactive_rank, a in 1:active_rank
            channel = y_offset + a + (i - 1) * active_rank
            value = space.left_active_response[p, a] * space.inactive_density_right_factor[q, i]
            offdiagonal && (value += space.left_active_response[q, a] * space.inactive_density_right_factor[p, i])
            features[pair, channel] = value
        end
        for j in 1:inactive_rank, i in 1:inactive_rank
            channel = z_offset + i + (j - 1) * inactive_rank
            value = space.inactive_density_left_factor[p, i] * space.inactive_density_right_factor[q, j]
            offdiagonal && (value += space.inactive_density_left_factor[q, i] * space.inactive_density_right_factor[p, j])
            features[pair, channel] = value
        end
    end
    return features
end

"""
    _assemble_packed_streamed_coefficients(one_body, space, gram)

Reconstruct all AHPC conventional-ERI Coulomb and exchange coefficients from the streamed canonical-pair feature Gram matrix.
Inputs are read-only, returned coefficients are host-owned in integral energy units, only result arrays are allocated, and an inconsistent Gram dimension raises `DimensionMismatch`.
"""
function _assemble_packed_streamed_coefficients(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    gram::AbstractMatrix{T},
) where {T<:AbstractFloat}
    n_features = _packed_streamed_feature_count(space)
    size(gram) == (n_features, n_features) || throw(DimensionMismatch("streamed packed Gram matrix has incorrect dimensions"))
    inactive_rank = size(space.inactive_density_left_factor, 2)
    active_rank = size(space.left_active_response, 2)
    n_response = active_rank * active_rank
    active_offset = 1
    x_offset = active_offset + n_response
    y_offset = x_offset + inactive_rank * active_rank
    z_offset = y_offset + active_rank * inactive_rank
    one_body_inactive, one_body_active = _precontract_one_body(one_body, space)
    coulomb_ia = Vector{T}(@view gram[1, 2:(n_response + 1)])
    coulomb_aa = Matrix{T}(@view gram[2:(n_response + 1), 2:(n_response + 1)])
    exchange_ia = zeros(T, n_response)
    exchange_aa = zeros(T, n_response, n_response)
    exchange_ii = zero(T)
    @inbounds for b in 1:active_rank, a in 1:active_rank
        response = active_response_index(a, b, active_rank)
        for i in 1:inactive_rank
            x = x_offset + i + (b - 1) * inactive_rank
            y = y_offset + a + (i - 1) * active_rank
            exchange_ia[response] += gram[x, y]
        end
        for d in 1:active_rank, c in 1:active_rank
            second = active_response_index(c, d, active_rank)
            ad = active_offset + active_response_index(a, d, active_rank)
            cb = active_offset + active_response_index(c, b, active_rank)
            exchange_aa[response, second] = gram[ad, cb]
        end
    end
    @inbounds for j in 1:inactive_rank, i in 1:inactive_rank
        ij = z_offset + i + (j - 1) * inactive_rank
        ji = z_offset + j + (i - 1) * inactive_rank
        exchange_ii += gram[ij, ji]
    end
    return _assemble_active_hamiltonian(
        one_body_inactive,
        one_body_active,
        gram[1, 1],
        exchange_ii,
        coulomb_ia,
        exchange_ia,
        coulomb_aa,
        exchange_aa,
    )
end

const CHOLESKY_GPU_PROJECTION_TILE_CAP = 256

"""
    _eri_factors_gpu_projection_tile(T, n_orbital_pairs, n_auxiliary, n_projection_rows, runtime)

Return the number of response rows projected per CUDA feature tile under `runtime.scratch_bytes` and the historical 256-row V100 efficiency cap.
Counts are dimensionless and inputs are read-only, no allocation or CUDA initialization occurs, and an insufficient positive scratch budget raises `ArgumentError`.
"""
function _eri_factors_gpu_projection_tile(
    ::Type{T},
    n_orbital_pairs::Integer,
    n_auxiliary::Integer,
    n_projection_rows::Integer,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    n_pairs = Int(n_orbital_pairs)
    n_aux = Int(n_auxiliary)
    n_rows = Int(n_projection_rows)
    n_pairs >= 0 && n_aux >= 0 && n_rows >= 0 || throw(ArgumentError("ERI-factor CUDA projection dimensions must be nonnegative"))
    n_rows == 0 && return 0
    bytes_per_row = estimate_array_bytes(T, Base.Checked.checked_add(n_pairs, n_aux))
    runtime.scratch_bytes >= bytes_per_row || throw(ArgumentError(
        "ERI-factor CUDA precontraction requires scratch_bytes for at least one projection-feature row",
    ))
    scratch_limited = runtime.scratch_bytes ÷ bytes_per_row
    return min(n_rows, CHOLESKY_GPU_PROJECTION_TILE_CAP, scratch_limited)
end

"""
    _eri_factors_single_gpu_precontraction_memory(T, space, factors, runtime, n_auxiliary)

Estimate peak host/device numerical storage for one ERI-factor auxiliary-column shard on one GPU.
Inputs are read-only, returned units are bytes, factor/response/shard dimensions are validated, no CUDA context or array is created, and checked integer overflow or insufficient scratch raises before execution.
"""
function _eri_factors_single_gpu_precontraction_memory(
    ::Type{T},
    space::ActiveResponseSpace,
    factors::ERIFactorValues,
    runtime::RuntimeOptions,
    n_auxiliary::Integer,
) where {T<:AbstractFloat}
    n_orbitals = factors.n_orbitals
    size(space.inactive_density_kernel, 1) == n_orbitals || throw(DimensionMismatch("active-response and ERI-factor orbital dimensions differ"))
    n_pairs, total_auxiliary = size(factors.coefficients)
    n_auxiliary_count = Int(n_auxiliary)
    1 <= n_auxiliary_count <= total_auxiliary || throw(ArgumentError("ERI-factor GPU shard auxiliary count is out of range"))
    n_inactive = size(space.inactive_density_left_factor, 2)
    size(space.inactive_density_right_factor, 2) == n_inactive || throw(DimensionMismatch("inactive-density factor ranks differ"))
    active_rank = size(space.left_active_response, 2)
    n_response = Base.Checked.checked_mul(active_rank, active_rank)
    n_core_response = Base.Checked.checked_mul(n_inactive, n_inactive)
    n_cross_response = Base.Checked.checked_mul(active_rank, n_inactive)
    largest_projection = max(n_core_response, n_cross_response, n_response)
    tile = _eri_factors_gpu_projection_tile(T, n_pairs, n_auxiliary_count, largest_projection, runtime)

    factor_bytes = estimate_array_bytes(T, n_pairs, n_auxiliary_count)
    label_bytes = estimate_array_bytes(Int32, 2, n_pairs)
    basis_columns = Base.Checked.checked_mul(2, Base.Checked.checked_add(n_inactive, active_rank))
    basis_bytes = estimate_array_bytes(T, n_orbitals, basis_columns)
    persistent_bytes = Base.Checked.checked_add(Base.Checked.checked_add(factor_bytes, label_bytes), basis_bytes)
    projection_scratch = tile == 0 ? 0 : estimate_array_bytes(T, Base.Checked.checked_mul(tile, Base.Checked.checked_add(n_pairs, n_auxiliary_count)))

    core_bytes = n_core_response == 0 ? 0 : Base.Checked.checked_add(
        estimate_array_bytes(T, n_core_response, n_auxiliary_count),
        estimate_array_bytes(T, 2, n_auxiliary_count),
    )
    cross_bytes = n_cross_response == 0 ? 0 : estimate_array_bytes(
        T,
        Base.Checked.checked_mul(n_auxiliary_count, Base.Checked.checked_add(Base.Checked.checked_mul(2, n_cross_response), n_response)),
    )
    active_bytes = n_response == 0 ? 0 : Base.Checked.checked_add(
        estimate_array_bytes(T, n_response, n_auxiliary_count),
        Base.Checked.checked_add(estimate_array_bytes(T, n_response, n_response), estimate_array_bytes(T, n_response)),
    )
    stage_bytes = max(core_bytes, cross_bytes, active_bytes)
    device_bytes = Base.Checked.checked_add(persistent_bytes, Base.Checked.checked_add(projection_scratch, stage_bytes))
    host_reduction_bytes = max(
        estimate_array_bytes(T, 2, n_auxiliary_count),
        estimate_array_bytes(T, n_response, n_auxiliary_count),
    )
    host_bytes = Base.Checked.checked_add(_precontraction_output_memory(T, n_response).host_bytes, host_reduction_bytes)
    return MemoryEstimate(host_bytes, device_bytes)
end

"""
    _eri_factors_gpu_precontraction_memory(T, space, factors, runtime)

Estimate aggregate host storage and peak per-device storage for typed multi-GPU Cholesky AHPC precontraction.
Inputs are read-only, returned units are bytes, auxiliary columns are balanced over at most `length(runtime.gpu_devices)` nonempty shards, no CUDA state is initialized, and invalid resource dimensions raise before allocation.
"""
function _eri_factors_gpu_precontraction_memory(
    ::Type{T},
    space::ActiveResponseSpace,
    factors::ERIFactorValues,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    total_auxiliary = size(factors.coefficients, 2)
    worker_count = min(length(runtime.gpu_devices), total_auxiliary)
    worker_count >= 1 || throw(ArgumentError("ERI-factor CUDA precontraction requires at least one GPU device"))
    maximum_shard = cld(total_auxiliary, worker_count)
    per_worker = _eri_factors_single_gpu_precontraction_memory(T, space, factors, runtime, maximum_shard)
    aggregate_host = Base.Checked.checked_mul(per_worker.host_bytes, worker_count)
    return MemoryEstimate(aggregate_host, per_worker.device_bytes)
end

"""
    _eri_factors_canonical_pair_labels(n_orbitals)

Return lower/upper one-based `Int32` orbital labels for each canonical Cholesky-factor row.
The orbital count is dimensionless, outputs are new CPU vectors, no CUDA state is touched, and nonpositive counts or `Int32` overflow raise before allocation.
"""
function _eri_factors_canonical_pair_labels(n_orbitals::Integer)
    n = Int(n_orbitals)
    n >= 1 || throw(ArgumentError("n_orbitals must be positive"))
    n <= typemax(Int32) || throw(OverflowError("orbital labels exceed Int32 capacity"))
    n_pairs = canonical_pair_count(n)
    lower_labels = Vector{Int32}(undef, n_pairs)
    upper_labels = Vector{Int32}(undef, n_pairs)
    pair = 0
    for upper in 1:n, lower in 1:upper
        pair += 1
        lower_labels[pair] = Int32(lower)
        upper_labels[pair] = Int32(upper)
    end
    return lower_labels, upper_labels
end

"""
    _exchange_response_from_coulomb(coulomb_response, active_rank)

Return the AHPC active-active exchange matrix by the exact index permutation `K[(a,b),(c,d)] = J[(a,d),(c,b)]` of the Cholesky Coulomb Gram matrix.
The CPU input is read-only and follows integral energy units, the output owns new host storage, no CUDA work occurs, and inconsistent dimensions raise `DimensionMismatch`.
"""
function _exchange_response_from_coulomb(
    coulomb_response::AbstractMatrix{T},
    active_rank::Integer,
) where {T<:AbstractFloat}
    rank = Int(active_rank)
    n_response = Base.Checked.checked_mul(rank, rank)
    size(coulomb_response) == (n_response, n_response) || throw(DimensionMismatch("Coulomb response dimension differs from active rank"))
    exchange_response = Matrix{T}(undef, n_response, n_response)
    @inbounds for right_second in 1:rank, left_second in 1:rank, right_first in 1:rank, left_first in 1:rank
        first_response = active_response_index(left_first, right_first, rank)
        second_response = active_response_index(left_second, right_second, rank)
        exchange_first = active_response_index(left_first, right_second, rank)
        exchange_second = active_response_index(left_second, right_first, rank)
        exchange_response[first_response, second_response] = coulomb_response[exchange_first, exchange_second]
    end
    return exchange_response
end

"""
    _runtime_for_gpu_device(runtime, gpu_device)

Return a typed runtime copy restricted to one selected CUDA device for an internal multi-GPU worker.
The input is read-only, byte limits and scientific guards are preserved, no CUDA state is initialized, and a negative device index raises `ArgumentError` through `RuntimeOptions`.
"""
function _runtime_for_gpu_device(runtime::RuntimeOptions, gpu_device::Integer)
    return RuntimeOptions(
        use_gpu=true,
        gpu_device=gpu_device,
        gpu_devices=(Int(gpu_device),),
        max_host_bytes=runtime.max_host_bytes,
        max_device_bytes=runtime.max_device_bytes,
        scratch_bytes=runtime.scratch_bytes,
        gpu_intermediate_scratch_bytes_per_device=runtime.gpu_intermediate_scratch_bytes_per_device,
        contraction_host_scratch_bytes=runtime.contraction_host_scratch_bytes,
        max_active_occupied=runtime.max_active_occupied,
        collect_allocations=runtime.collect_allocations,
    )
end

"""
    _balanced_auxiliary_ranges(n_auxiliary, n_workers)

Partition one-based ERI-factor auxiliary columns into ordered nonempty ranges whose sizes differ by at most one.
Counts are dimensionless, returned ranges own only small host metadata, no CUDA work occurs, and nonpositive counts raise `ArgumentError`.
"""
function _balanced_auxiliary_ranges(n_auxiliary::Integer, n_workers::Integer)
    count = Int(n_auxiliary)
    workers = min(Int(n_workers), count)
    count >= 1 && workers >= 1 || throw(ArgumentError("auxiliary and worker counts must be positive"))
    base = count ÷ workers
    remainder = count % workers
    ranges = Vector{UnitRange{Int}}(undef, workers)
    first_column = 1
    for worker in 1:workers
        width = base + (worker <= remainder ? 1 : 0)
        ranges[worker] = first_column:(first_column + width - 1)
        first_column += width
    end
    return ranges
end

"""
    _eri_factors_gpu_resident_memory(T, factors, runtime)

Estimate the peak per-device storage for evaluation-lifetime resident metric-whitened ERI factors and canonical-pair labels.
Inputs are read-only, returned units are bytes, host factor storage is excluded because it is already owned by `factors`, no CUDA state is initialized, and empty device selections or invalid dimensions raise.
"""
function _eri_factors_gpu_resident_memory(
    ::Type{T},
    factors::ERIFactorValues,
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("resident ERI-factor GPU storage requires runtime.use_gpu=true"))
    worker_count = min(length(runtime.gpu_devices), size(factors.coefficients, 2))
    worker_count >= 1 || throw(ArgumentError("resident ERI-factor GPU storage requires at least one device and auxiliary vector"))
    maximum_shard = cld(size(factors.coefficients, 2), worker_count)
    factor_bytes = estimate_array_bytes(T, size(factors.coefficients, 1), maximum_shard)
    label_bytes = estimate_array_bytes(Int32, 2, size(factors.coefficients, 1))
    return MemoryEstimate(0, Base.Checked.checked_add(factor_bytes, label_bytes))
end

"""
    _close_ahpc_precontraction_source!(source)

Release an AHPC precontraction source when it owns evaluation-lifetime resources, otherwise return without action.
The generic method does not mutate ordinary CPU sources or allocate storage, and specialized CUDA cleanup errors propagate.
"""
_close_ahpc_precontraction_source!(source) = nothing


"""
    _reduce_eri_factors_gpu_partials(one_body, space, partials)

Deterministically sum ordered auxiliary-shard two-electron coefficients, evaluate the one-body term once, and form the exchange permutation once after Coulomb reduction.
Inputs are read-only, outputs are new host-owned arrays in integral energy units, no CUDA work occurs, and empty or dimensionally inconsistent partial collections raise.
"""
function _reduce_eri_factors_gpu_partials(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    partials::AbstractVector{<:ActiveHamiltonianCoefficients{T}},
) where {T<:Union{Float32,Float64}}
    isempty(partials) && throw(ArgumentError("at least one ERI-factor GPU partial is required"))
    one_body_inactive, one_body_active = _precontract_one_body(one_body, space)
    n_response = active_response_count(space)
    two_body_inactive_inactive = zero(T)
    two_body_inactive_active = zeros(T, n_response)
    coulomb_response = zeros(T, n_response, n_response)
    for partial in partials
        length(partial.two_body_inactive_active) == n_response || throw(DimensionMismatch("ERI-factor GPU partial active-vector dimension mismatch"))
        size(partial.coulomb_response) == (n_response, n_response) || throw(DimensionMismatch("ERI-factor GPU partial response-matrix dimension mismatch"))
        two_body_inactive_inactive += partial.two_body_inactive_inactive
        two_body_inactive_active .+= partial.two_body_inactive_active
        coulomb_response .+= partial.coulomb_response
    end
    exchange_response = _exchange_response_from_coulomb(coulomb_response, size(space.left_active_response, 2))
    return ActiveHamiltonianCoefficients(
        one_body_inactive,
        one_body_active,
        two_body_inactive_inactive,
        two_body_inactive_active,
        coulomb_response,
        exchange_response,
    )
end

"""
    _precontract_eri_factors_active_hamiltonian_gpu(one_body, space, factors; runtime)

Compute host-owned AHPC coefficients by balancing ERI-factor auxiliary columns over the typed CUDA device list and deterministically summing partial contractions.
Scientific inputs are read-only, outputs follow one-body and ERI energy units, one-body and exchange assembly occur once after ordered reduction, aggregate host and peak per-device memory are preflighted, and unavailable or insufficiently threaded execution raises without CPU fallback.
"""
function _precontract_eri_factors_active_hamiltonian_gpu(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    factors::ERIFactorValues{T};
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("CUDA ERI-factor precontraction requires runtime.use_gpu=true"))
    worker_count = min(length(runtime.gpu_devices), size(factors.coefficients, 2))
    Threads.nthreads() >= worker_count || throw(ArgumentError("multi-GPU ERI-factor precontraction requires at least one Julia thread per GPU"))
    estimate = _eri_factors_gpu_precontraction_memory(T, space, factors, runtime)
    enforce_memory_limits(estimate, runtime)
    ranges = _balanced_auxiliary_ranges(size(factors.coefficients, 2), worker_count)
    tasks = map(eachindex(ranges)) do worker
        device = runtime.gpu_devices[worker]
        worker_runtime = _runtime_for_gpu_device(runtime, device)
        Threads.@spawn _precontract_eri_factors_active_hamiltonian_single_gpu(
            one_body,
            space,
            factors,
            ranges[worker];
            runtime=worker_runtime,
            include_one_body=false,
            include_exchange_response=false,
        )
    end
    partials = fetch.(tasks)
    return _reduce_eri_factors_gpu_partials(one_body, space, partials)
end
