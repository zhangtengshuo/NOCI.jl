
"""
    _unpack_packed_eri_tile_kernel!(tile, integrals, row_first, column_first, n_rows, n_columns)

Expand one rectangular tile of the symmetric canonical-pair ERI operator from physical packed values on the GPU.
The integral vector is read-only, the leading `n_rows` by `n_columns` tile is overwritten in ERI energy units, no allocation occurs, and launch bounds are supplied by the caller.
"""
function _unpack_packed_eri_tile_kernel!(
    tile,
    integrals,
    row_first::Int32,
    column_first::Int32,
    n_rows::Int32,
    n_columns::Int32,
)
    index = (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) + Int(CUDA.threadIdx().x)
    row_count = Int(n_rows)
    column_count = Int(n_columns)
    if index <= row_count * column_count
        local_row = (index - 1) % row_count + 1
        local_column = (index - 1) ÷ row_count + 1
        row = Int(row_first) + local_row - 1
        column = Int(column_first) + local_column - 1
        lower = min(row, column)
        upper = max(row, column)
        packed_index = lower + ((upper - 1) * upper) ÷ 2
        @inbounds tile[local_row, local_column] = integrals[packed_index]
    end
    return nothing
end

"""
    _precontract_packed_streamed_active_hamiltonian_gpu(one_body, space, integrals; runtime)

Compute exact conventional `.two` AHPC coefficients by streaming canonical-pair ERI tiles through single-GPU cuBLAS without materializing dense AO Coulomb/exchange superoperators.
Inputs are read-only, outputs remain host-owned in integral energy units, host/device storage and scratch are preflighted, device allocations are explicitly released, and unavailable CUDA execution raises without CPU fallback.
"""
function _precontract_packed_streamed_active_hamiltonian_gpu(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    integrals::PackedERIValues{T};
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("CUDA streamed packed-ERI precontraction requires runtime.use_gpu=true"))
    n_orbitals = integrals.n_orbitals
    size(space.inactive_density_kernel) == (n_orbitals, n_orbitals) || throw(DimensionMismatch("active-response and packed ERI orbital dimensions differ"))
    size(one_body) == (n_orbitals, n_orbitals) || throw(DimensionMismatch("one_body orbital dimension mismatch"))
    n_pairs = canonical_pair_count(n_orbitals)
    n_features = _packed_streamed_feature_count(space)
    tile_size = _packed_streamed_pair_tile(T, n_pairs, n_features, runtime.scratch_bytes)
    estimate = _packed_streamed_gpu_precontraction_memory(T, space, integrals, runtime)
    enforce_memory_limits(estimate, runtime)
    _activate_cuda_device(runtime)
    lower_labels, upper_labels = _canonical_orbital_pair_labels(n_orbitals)
    features = Matrix{T}(undef, n_pairs, n_features)
    _fill_packed_streamed_features!(features, lower_labels, upper_labels, space)
    gram = Matrix{T}(undef, n_features, n_features)
    integrals_device = nothing
    features_device = nothing
    tile_device = nothing
    product_device = nothing
    gram_device = nothing
    try
        integrals_device = CUDA.CuArray(integrals.values)
        features_device = CUDA.CuArray(features)
        tile_device = CUDA.CuArray{T}(undef, tile_size, tile_size)
        product_device = CUDA.CuArray{T}(undef, tile_size, n_features)
        gram_device = CUDA.zeros(T, n_features, n_features)
        threads = 256
        first_row = 1
        while first_row <= n_pairs
            last_row = min(first_row + tile_size - 1, n_pairs)
            row_count = last_row - first_row + 1
            row_features = @view features_device[first_row:last_row, :]
            first_column = first_row
            while first_column <= n_pairs
                last_column = min(first_column + tile_size - 1, n_pairs)
                column_count = last_column - first_column + 1
                column_features = @view features_device[first_column:last_column, :]
                tile = @view tile_device[1:row_count, 1:column_count]
                CUDA.@cuda threads=threads blocks=cld(row_count * column_count, threads) _unpack_packed_eri_tile_kernel!(
                    tile_device,
                    integrals_device,
                    Int32(first_row),
                    Int32(first_column),
                    Int32(row_count),
                    Int32(column_count),
                )
                row_product = @view product_device[1:row_count, :]
                CUDA.CUBLAS.gemm!('N', 'N', one(T), tile, column_features, zero(T), row_product)
                CUDA.CUBLAS.gemm!('T', 'N', one(T), row_features, row_product, one(T), gram_device)
                if first_column != first_row
                    column_product = @view product_device[1:column_count, :]
                    CUDA.CUBLAS.gemm!('T', 'N', one(T), tile, row_features, zero(T), column_product)
                    CUDA.CUBLAS.gemm!('T', 'N', one(T), column_features, column_product, one(T), gram_device)
                end
                first_column = last_column + 1
            end
            first_row = last_row + 1
        end
        CUDA.synchronize()
        copyto!(gram, gram_device)
    finally
        for array in (gram_device, product_device, tile_device, features_device, integrals_device)
            array === nothing || CUDA.unsafe_free!(array)
        end
    end
    return _assemble_packed_streamed_coefficients(one_body, space, gram)
end

"""
    ERIFactorGPUShard

Own one auxiliary-column factor shard and canonical-pair labels resident on one CUDA device for reuse across AHPC state pairs.
The arrays follow square-root ERI and one-based orbital-label conventions, are mutable only for deterministic cleanup, and must not be used after `closed` becomes true.
"""
mutable struct ERIFactorGPUShard{T<:Union{Float32,Float64}}
    device::Int
    auxiliary_range::UnitRange{Int}
    factors::CUDA.CuArray{T,2}
    lower_labels::CUDA.CuArray{Int32,1}
    upper_labels::CUDA.CuArray{Int32,1}
    closed::Bool
end

"""
    ERIFactorGPUPrecontractionSource

Retain validated host metric-whitened ERI factors together with one resident CUDA shard per selected device for an entire AHPC matrix evaluation.
The source owns its device arrays, preserves deterministic device/range order, performs no scientific reduction itself, and must be released with `_close_eri_factors_gpu_precontraction_source!`.
"""
mutable struct ERIFactorGPUPrecontractionSource{T<:Union{Float32,Float64}}
    factors::ERIFactorValues{T}
    shards::Vector{ERIFactorGPUShard{T}}
    closed::Bool
end

"""
    _fill_eri_factors_projection_features_kernel!(features, lower_labels, upper_labels, left_basis, right_basis, first_response, tile_width, n_left, n_right)

Fill one device feature tile representing symmetric projections `left_basis' * L_P * right_basis` for packed Cholesky rows.
All arrays are device-resident and dimensionless except the inherited factor projection, the destination is mutated, no allocation occurs inside the kernel, and launch dimensions are validated by the host caller.
"""
function _fill_eri_factors_projection_features_kernel!(
    features,
    lower_labels,
    upper_labels,
    left_basis,
    right_basis,
    first_response::Int32,
    tile_width::Int32,
    n_left::Int32,
    n_right::Int32,
)
    index = (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) + Int(CUDA.threadIdx().x)
    n_pairs = length(lower_labels)
    width = Int(tile_width)
    if index <= n_pairs * width
        pair = (index - 1) % n_pairs + 1
        local_response = (index - 1) ÷ n_pairs + 1
        response = Int(first_response) + local_response - 1
        left_count = Int(n_left)
        response_count = left_count * Int(n_right)
        if response <= response_count
            left_index = (response - 1) % left_count + 1
            right_index = (response - 1) ÷ left_count + 1
            lower = Int(@inbounds lower_labels[pair])
            upper = Int(@inbounds upper_labels[pair])
            value = @inbounds left_basis[lower, left_index] * right_basis[upper, right_index]
            lower == upper || (value += @inbounds left_basis[upper, left_index] * right_basis[lower, right_index])
            @inbounds features[pair, local_response] = value
        else
            @inbounds features[pair, local_response] = zero(eltype(features))
        end
    end
    return nothing
end

"""
    _copy_eri_factors_projection_tile_kernel!(projection, tile_projection, first_response, valid_rows)

Copy valid rows from one device projection tile into the full response-by-auxiliary destination.
The destination is mutated, inputs remain device-resident and read-only, no allocation occurs, and the host caller guarantees bounds and matching auxiliary dimensions.
"""
function _copy_eri_factors_projection_tile_kernel!(
    projection,
    tile_projection,
    first_response::Int32,
    valid_rows::Int32,
)
    valid = Int(valid_rows)
    n_auxiliary = size(projection, 2)
    index = (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) + Int(CUDA.threadIdx().x)
    if index <= valid * n_auxiliary
        local_response = (index - 1) % valid + 1
        auxiliary = (index - 1) ÷ valid + 1
        @inbounds projection[Int(first_response) + local_response - 1, auxiliary] = tile_projection[local_response, auxiliary]
    end
    return nothing
end

"""
    _project_eri_factors_gpu(factors_device, lower_labels_device, upper_labels_device, left_basis_device, right_basis_device, tile_width)

Project every packed Cholesky vector into all ordered left/right response products and return a device matrix with shape `(n_left*n_right, n_auxiliary)`.
Inputs are read-only device arrays, output follows square-root ERI units, temporary device storage is explicitly released, CUDA errors propagate, and no host scientific array is materialized.
"""
function _project_eri_factors_gpu(
    factors_device,
    lower_labels_device,
    upper_labels_device,
    left_basis_device,
    right_basis_device,
    tile_width::Int,
)
    n_pairs, n_auxiliary = size(factors_device)
    n_left = size(left_basis_device, 2)
    n_right = size(right_basis_device, 2)
    n_response = Base.Checked.checked_mul(n_left, n_right)
    n_response > 0 || return CUDA.CuArray{eltype(factors_device)}(undef, 0, n_auxiliary)
    width = min(tile_width, n_response)
    projection = CUDA.CuArray{eltype(factors_device)}(undef, n_response, n_auxiliary)
    features = CUDA.CuArray{eltype(factors_device)}(undef, n_pairs, width)
    tile_projection = CUDA.CuArray{eltype(factors_device)}(undef, width, n_auxiliary)
    try
        threads = 256
        for first_response in 1:width:n_response
            valid_rows = min(width, n_response - first_response + 1)
            CUDA.@cuda threads=threads blocks=cld(n_pairs * width, threads) _fill_eri_factors_projection_features_kernel!(
                features,
                lower_labels_device,
                upper_labels_device,
                left_basis_device,
                right_basis_device,
                Int32(first_response),
                Int32(width),
                Int32(n_left),
                Int32(n_right),
            )
            CUDA.CUBLAS.gemm!('T', 'N', one(eltype(features)), features, factors_device, zero(eltype(features)), tile_projection)
            CUDA.@cuda threads=threads blocks=cld(valid_rows * n_auxiliary, threads) _copy_eri_factors_projection_tile_kernel!(
                projection,
                tile_projection,
                Int32(first_response),
                Int32(valid_rows),
            )
        end
        CUDA.synchronize()
        return projection
    catch
        CUDA.unsafe_free!(projection)
        rethrow()
    finally
        CUDA.unsafe_free!(features)
        CUDA.unsafe_free!(tile_projection)
    end
end

"""
    _eri_factors_core_channels_kernel!(inactive_projection, exchange_by_auxiliary, core_projection, n_inactive)

Compute the inactive Coulomb projection and inactive exchange contribution for every ERI-factor auxiliary vector on the GPU.
Inputs are read-only device arrays, both destinations are overwritten in square-root/integral energy units, no allocation occurs, and one CUDA thread handles each auxiliary vector.
"""
function _eri_factors_core_channels_kernel!(
    inactive_projection,
    exchange_by_auxiliary,
    core_projection,
    n_inactive::Int32,
)
    auxiliary = (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) + Int(CUDA.threadIdx().x)
    n_auxiliary = size(core_projection, 2)
    n = Int(n_inactive)
    if auxiliary <= n_auxiliary
        coulomb_value = zero(eltype(core_projection))
        exchange_value = zero(eltype(core_projection))
        @inbounds for index in 1:n
            coulomb_value += core_projection[index + (index - 1) * n, auxiliary]
        end
        @inbounds for column in 1:n, row in 1:n
            exchange_value += core_projection[row + (column - 1) * n, auxiliary] * core_projection[column + (row - 1) * n, auxiliary]
        end
        @inbounds inactive_projection[auxiliary] = coulomb_value
        @inbounds exchange_by_auxiliary[auxiliary] = exchange_value
    end
    return nothing
end

"""
    _eri_factors_inactive_active_exchange_kernel!(exchange_by_auxiliary, left_cross_projection, right_cross_projection, active_rank, n_inactive)

Compute each auxiliary-vector contribution to the inactive-active exchange response `left_cross * right_cross` on the GPU.
Device inputs are read-only, the response-by-auxiliary destination is overwritten in integral energy units, no allocation occurs, and the host caller validates projection dimensions.
"""
function _eri_factors_inactive_active_exchange_kernel!(
    exchange_by_auxiliary,
    left_cross_projection,
    right_cross_projection,
    active_rank::Int32,
    n_inactive::Int32,
)
    rank = Int(active_rank)
    inactive = Int(n_inactive)
    n_response = rank * rank
    n_auxiliary = size(exchange_by_auxiliary, 2)
    index = (Int(CUDA.blockIdx().x) - 1) * Int(CUDA.blockDim().x) + Int(CUDA.threadIdx().x)
    if index <= n_response * n_auxiliary
        response = (index - 1) % n_response + 1
        auxiliary = (index - 1) ÷ n_response + 1
        left_active = (response - 1) % rank + 1
        right_active = (response - 1) ÷ rank + 1
        value = zero(eltype(exchange_by_auxiliary))
        @inbounds for inactive_index in 1:inactive
            left_slot = left_active + (inactive_index - 1) * rank
            right_slot = inactive_index + (right_active - 1) * inactive
            value += left_cross_projection[left_slot, auxiliary] * right_cross_projection[right_slot, auxiliary]
        end
        @inbounds exchange_by_auxiliary[response, auxiliary] = value
    end
    return nothing
end

"""
    _upload_eri_factors_gpu_shard(factors, lower_labels, upper_labels, auxiliary_range, runtime)

Upload one contiguous auxiliary-column shard and the shared canonical labels to the selected CUDA device.
Host inputs are read-only, returned arrays are device-owned and remain live until explicit cleanup, transfer failures release partial allocations, and device/range inconsistencies raise without fallback.
"""
function _upload_eri_factors_gpu_shard(
    factors::ERIFactorValues{T},
    lower_labels::Vector{Int32},
    upper_labels::Vector{Int32},
    auxiliary_range::UnitRange{Int},
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("ERI-factor shard upload requires runtime.use_gpu=true"))
    length(runtime.gpu_devices) == 1 || throw(ArgumentError("one ERI-factor shard upload must select exactly one CUDA device"))
    first(auxiliary_range) >= 1 && last(auxiliary_range) <= size(factors.coefficients, 2) || throw(BoundsError(axes(factors.coefficients, 2), auxiliary_range))
    length(lower_labels) == size(factors.coefficients, 1) || throw(DimensionMismatch("lower ERI-factor label count differs from factor rows"))
    length(upper_labels) == size(factors.coefficients, 1) || throw(DimensionMismatch("upper ERI-factor label count differs from factor rows"))
    _activate_cuda_device(runtime)
    factors_device = nothing
    lower_labels_device = nothing
    upper_labels_device = nothing
    try
        factors_device = CUDA.CuArray(@view factors.coefficients[:, auxiliary_range])
        lower_labels_device = CUDA.CuArray(lower_labels)
        upper_labels_device = CUDA.CuArray(upper_labels)
        CUDA.synchronize()
        return ERIFactorGPUShard(
            first(runtime.gpu_devices),
            auxiliary_range,
            factors_device,
            lower_labels_device,
            upper_labels_device,
            false,
        )
    catch
        for array in (upper_labels_device, lower_labels_device, factors_device)
            array === nothing || CUDA.unsafe_free!(array)
        end
        rethrow()
    end
end

"""
    _close_eri_factors_gpu_shard!(shard)

Release all CUDA allocations owned by one resident ERI-factor shard and mark it closed.
The operation is idempotent, mutates only the shard lifecycle, performs no host scientific allocation, and CUDA cleanup errors propagate.
"""
function _close_eri_factors_gpu_shard!(shard::ERIFactorGPUShard)
    shard.closed && return nothing
    CUDA.device!(shard.device)
    for array in (shard.upper_labels, shard.lower_labels, shard.factors)
        CUDA.unsafe_free!(array)
    end
    shard.closed = true
    CUDA.reclaim()
    return nothing
end

"""
    _close_eri_factors_gpu_precontraction_source!(source)

Release every device shard owned by an evaluation-lifetime ERI-factor GPU source and mark the source closed.
The operation is idempotent, preserves host factors, mutates only device-resource lifecycle fields, and attempts every shard cleanup before propagating the first CUDA error.
"""
function _close_eri_factors_gpu_precontraction_source!(source::ERIFactorGPUPrecontractionSource)
    source.closed && return nothing
    first_error = nothing
    for shard in source.shards
        try
            _close_eri_factors_gpu_shard!(shard)
        catch error
            isnothing(first_error) && (first_error = error)
        end
    end
    source.closed = true
    isnothing(first_error) || throw(first_error)
    return nothing
end

"""
    _close_ahpc_precontraction_source!(source::ERIFactorGPUPrecontractionSource)

Release an evaluation-lifetime resident ERI-factor GPU source through its idempotent lifecycle method.
The source is mutated only by closing device resources, host factors remain valid, no scientific result is changed, and cleanup errors propagate.
"""
_close_ahpc_precontraction_source!(source::ERIFactorGPUPrecontractionSource) =
    _close_eri_factors_gpu_precontraction_source!(source)

"""
    _prepare_eri_factors_gpu_precontraction_source(factors, runtime)

Upload balanced ERI-factor auxiliary shards and canonical labels once for reuse across every AHPC state pair in one matrix evaluation.
Host factors are read-only, device allocation is preflighted and parallelized over the typed device order, partial uploads are released on failure, and the returned source requires explicit cleanup.
"""
function _prepare_eri_factors_gpu_precontraction_source(
    factors::ERIFactorValues{T},
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("resident ERI-factor GPU source requires runtime.use_gpu=true"))
    worker_count = min(length(runtime.gpu_devices), size(factors.coefficients, 2))
    Threads.nthreads() >= worker_count || throw(ArgumentError("multi-GPU ERI-factor source preparation requires at least one Julia thread per GPU"))
    estimate = _eri_factors_gpu_resident_memory(T, factors, runtime)
    enforce_memory_limits(estimate, runtime)
    ranges = _balanced_auxiliary_ranges(size(factors.coefficients, 2), worker_count)
    lower_labels, upper_labels = _eri_factors_canonical_pair_labels(factors.n_orbitals)
    tasks = map(eachindex(ranges)) do worker
        worker_runtime = _runtime_for_gpu_device(runtime, runtime.gpu_devices[worker])
        Threads.@spawn _upload_eri_factors_gpu_shard(
            factors,
            lower_labels,
            upper_labels,
            ranges[worker],
            worker_runtime,
        )
    end
    shards = Vector{Union{Nothing,ERIFactorGPUShard{T}}}(nothing, length(tasks))
    first_error = nothing
    for worker in eachindex(tasks)
        try
            shards[worker] = fetch(tasks[worker])
        catch error
            isnothing(first_error) && (first_error = error)
        end
    end
    if !isnothing(first_error)
        for shard in shards
            isnothing(shard) || _close_eri_factors_gpu_shard!(shard)
        end
        throw(first_error)
    end
    return ERIFactorGPUPrecontractionSource(factors, ERIFactorGPUShard{T}[shard::ERIFactorGPUShard{T} for shard in shards], false)
end

"""
    _precontract_eri_factors_active_hamiltonian_single_gpu(one_body, space, factors, auxiliary_range; runtime, resident_shard=nothing, include_one_body=true, include_exchange_response=true)

Compute one host-owned partial AHPC coefficient set from a contiguous ERI-factor auxiliary-column shard using one selected GPU.
Scientific inputs are read-only, outputs follow one-body and ERI energy units, optional resident factors/labels are borrowed without release, per-device storage is preflighted, and unavailable CUDA execution raises without CPU fallback.
"""
function _precontract_eri_factors_active_hamiltonian_single_gpu(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    factors::ERIFactorValues{T},
    auxiliary_range::UnitRange{Int};
    runtime::RuntimeOptions,
    resident_shard::Union{Nothing,ERIFactorGPUShard{T}}=nothing,
    include_one_body::Bool=true,
    include_exchange_response::Bool=true,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("CUDA ERI-factor precontraction requires runtime.use_gpu=true"))
    n_orbitals = factors.n_orbitals
    size(space.inactive_density_kernel) == (n_orbitals, n_orbitals) || throw(DimensionMismatch("active-response and ERI-factor orbital dimensions differ"))
    size(one_body) == (n_orbitals, n_orbitals) || throw(DimensionMismatch("one_body orbital dimension mismatch"))
    first(auxiliary_range) >= 1 && last(auxiliary_range) <= size(factors.coefficients, 2) || throw(BoundsError(axes(factors.coefficients, 2), auxiliary_range))
    estimate = _eri_factors_single_gpu_precontraction_memory(T, space, factors, runtime, length(auxiliary_range))
    enforce_memory_limits(estimate, runtime)
    _activate_cuda_device(runtime)
    if !isnothing(resident_shard)
        resident_shard.closed && throw(ArgumentError("resident ERI-factor GPU shard is closed"))
        resident_shard.device == first(runtime.gpu_devices) || throw(ArgumentError("resident ERI-factor GPU shard device differs from runtime device"))
        resident_shard.auxiliary_range == auxiliary_range || throw(DimensionMismatch("resident ERI-factor GPU shard auxiliary range differs from requested range"))
    end
    one_body_inactive, one_body_active = include_one_body ? _precontract_one_body(one_body, space) : (zero(T), zeros(T, active_response_count(space)))
    factor_coefficients = @view factors.coefficients[:, auxiliary_range]
    n_pairs, n_auxiliary = size(factor_coefficients)
    n_inactive = size(space.inactive_density_left_factor, 2)
    active_rank = size(space.left_active_response, 2)
    n_response = active_response_count(space)
    largest_projection = max(n_inactive * n_inactive, active_rank * n_inactive, n_response)
    tile_width = _eri_factors_gpu_projection_tile(T, n_pairs, n_auxiliary, largest_projection, runtime)
    lower_labels, upper_labels = isnothing(resident_shard) ? _eri_factors_canonical_pair_labels(n_orbitals) : (Int32[], Int32[])

    owns_factor_storage = isnothing(resident_shard)
    factors_device = isnothing(resident_shard) ? nothing : resident_shard.factors
    lower_labels_device = isnothing(resident_shard) ? nothing : resident_shard.lower_labels
    upper_labels_device = isnothing(resident_shard) ? nothing : resident_shard.upper_labels
    inactive_left_device = nothing
    inactive_right_device = nothing
    active_left_device = nothing
    active_right_device = nothing
    inactive_projection_device = nothing
    try
        if owns_factor_storage
            factors_device = CUDA.CuArray(factor_coefficients)
            lower_labels_device = CUDA.CuArray(lower_labels)
            upper_labels_device = CUDA.CuArray(upper_labels)
        end
        inactive_left_device = CUDA.CuArray(space.inactive_density_left_factor)
        inactive_right_device = CUDA.CuArray(space.inactive_density_right_factor)
        active_left_device = CUDA.CuArray(space.left_active_response)
        active_right_device = CUDA.CuArray(space.right_active_response)

        coulomb_ii = zero(T)
        exchange_ii = zero(T)
        if n_inactive > 0
            core_projection = _project_eri_factors_gpu(
                factors_device,
                lower_labels_device,
                upper_labels_device,
                inactive_left_device,
                inactive_right_device,
                tile_width,
            )
            inactive_projection_device = CUDA.CuArray{T}(undef, n_auxiliary)
            exchange_core_device = CUDA.CuArray{T}(undef, n_auxiliary)
            try
                threads = 256
                CUDA.@cuda threads=threads blocks=cld(n_auxiliary, threads) _eri_factors_core_channels_kernel!(
                    inactive_projection_device,
                    exchange_core_device,
                    core_projection,
                    Int32(n_inactive),
                )
                CUDA.synchronize()
                inactive_projection = Array(inactive_projection_device)
                coulomb_ii = sum(abs2, inactive_projection)
                exchange_ii = sum(Array(exchange_core_device))
            finally
                CUDA.unsafe_free!(core_projection)
                CUDA.unsafe_free!(exchange_core_device)
            end
        else
            inactive_projection_device = CUDA.zeros(T, n_auxiliary)
        end

        exchange_ia = zeros(T, n_response)
        if n_inactive > 0 && n_response > 0
            left_cross_projection = _project_eri_factors_gpu(
                factors_device,
                lower_labels_device,
                upper_labels_device,
                active_left_device,
                inactive_right_device,
                tile_width,
            )
            right_cross_projection = _project_eri_factors_gpu(
                factors_device,
                lower_labels_device,
                upper_labels_device,
                inactive_left_device,
                active_right_device,
                tile_width,
            )
            exchange_by_auxiliary = CUDA.CuArray{T}(undef, n_response, n_auxiliary)
            try
                threads = 256
                CUDA.@cuda threads=threads blocks=cld(n_response * n_auxiliary, threads) _eri_factors_inactive_active_exchange_kernel!(
                    exchange_by_auxiliary,
                    left_cross_projection,
                    right_cross_projection,
                    Int32(active_rank),
                    Int32(n_inactive),
                )
                CUDA.synchronize()
                exchange_ia .= vec(sum(Array(exchange_by_auxiliary); dims=2))
            finally
                CUDA.unsafe_free!(left_cross_projection)
                CUDA.unsafe_free!(right_cross_projection)
                CUDA.unsafe_free!(exchange_by_auxiliary)
            end
        end

        coulomb_ia = zeros(T, n_response)
        coulomb_response = zeros(T, n_response, n_response)
        if n_response > 0
            active_projection = _project_eri_factors_gpu(
                factors_device,
                lower_labels_device,
                upper_labels_device,
                active_left_device,
                active_right_device,
                tile_width,
            )
            coulomb_ia_device = CUDA.CuArray{T}(undef, n_response)
            coulomb_response_device = CUDA.CuArray{T}(undef, n_response, n_response)
            try
                CUDA.CUBLAS.gemv!('N', one(T), active_projection, inactive_projection_device, zero(T), coulomb_ia_device)
                CUDA.CUBLAS.gemm!('N', 'T', one(T), active_projection, active_projection, zero(T), coulomb_response_device)
                CUDA.synchronize()
                copyto!(coulomb_ia, coulomb_ia_device)
                copyto!(coulomb_response, coulomb_response_device)
            finally
                CUDA.unsafe_free!(active_projection)
                CUDA.unsafe_free!(coulomb_ia_device)
                CUDA.unsafe_free!(coulomb_response_device)
            end
        end
        exchange_response = include_exchange_response ? _exchange_response_from_coulomb(coulomb_response, active_rank) : zeros(T, n_response, n_response)
        return _assemble_active_hamiltonian(
            one_body_inactive,
            one_body_active,
            coulomb_ii,
            exchange_ii,
            coulomb_ia,
            exchange_ia,
            coulomb_response,
            exchange_response,
        )
    finally
        for array in (inactive_projection_device, active_right_device, active_left_device, inactive_right_device, inactive_left_device)
            array === nothing || CUDA.unsafe_free!(array)
        end
        if owns_factor_storage
            for array in (upper_labels_device, lower_labels_device, factors_device)
                array === nothing || CUDA.unsafe_free!(array)
            end
        end
    end
end

"""
    precontraction_memory(T, space, source::ERIFactorGPUPrecontractionSource; runtime=RuntimeOptions())

Estimate peak host/device storage for one AHPC state-pair contraction while evaluation-lifetime ERI-factor shards are resident on their selected GPUs.
Inputs are read-only, the estimate includes resident factors plus transient projections to enforce the true live device limit, no CUDA operation occurs, and closed or mismatched sources raise.
"""
function precontraction_memory(
    ::Type{T},
    space::ActiveResponseSpace,
    source::ERIFactorGPUPrecontractionSource,
    ;
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:Union{Float32,Float64}}
    source.closed && throw(ArgumentError("resident ERI-factor GPU source is closed"))
    runtime.use_gpu || throw(ArgumentError("resident ERI-factor GPU source requires runtime.use_gpu=true"))
    Tuple(shard.device for shard in source.shards) == runtime.gpu_devices || throw(ArgumentError("resident ERI-factor GPU source devices differ from runtime devices"))
    return _eri_factors_gpu_precontraction_memory(T, space, source.factors, runtime)
end

"""
    precontract_active_hamiltonian(one_body, space, source::ERIFactorGPUPrecontractionSource; runtime=RuntimeOptions())

Compute AHPC coefficients with evaluation-lifetime resident ERI factor shards, reusing device factors and labels across state pairs.
Inputs are read-only, outputs are host-owned in integral energy units, one task runs per resident GPU in deterministic order, one-body/exchange assembly occurs once, and closed, mismatched, or insufficiently threaded execution raises without fallback.
"""
function precontract_active_hamiltonian(
    one_body::AbstractMatrix{T},
    space::ActiveResponseSpace{T},
    source::ERIFactorGPUPrecontractionSource{T};
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:Union{Float32,Float64}}
    source.closed && throw(ArgumentError("resident ERI-factor GPU source is closed"))
    runtime.use_gpu || throw(ArgumentError("resident ERI-factor GPU source requires runtime.use_gpu=true"))
    worker_count = length(source.shards)
    worker_count == length(runtime.gpu_devices) || throw(ArgumentError("resident ERI-factor GPU source device count differs from runtime"))
    Threads.nthreads() >= worker_count || throw(ArgumentError("multi-GPU resident ERI-factor precontraction requires at least one Julia thread per GPU"))
    estimate = precontraction_memory(T, space, source; runtime)
    enforce_memory_limits(estimate, runtime)
    tasks = map(eachindex(source.shards)) do worker
        shard = source.shards[worker]
        worker_runtime = _runtime_for_gpu_device(runtime, shard.device)
        Threads.@spawn _precontract_eri_factors_active_hamiltonian_single_gpu(
            one_body,
            space,
            source.factors,
            shard.auxiliary_range;
            runtime=worker_runtime,
            resident_shard=shard,
            include_one_body=false,
            include_exchange_response=false,
        )
    end
    partials = fetch.(tasks)
    return _reduce_eri_factors_gpu_partials(one_body, space, partials)
end
