# File: src/algorithms/gronor_noci/two_electron_cuda.jl
# Purpose: Implement GPU workspaces and direct two-electron contractions for G-NOCI.
# Algorithm origin: GronOR NOCI-F (Straatsma et al., JCTC 2022, doi:10.1021/acs.jctc.2c00266); rank-aware Julia CPU/CUDA reimplementation.


"""
    GronorNOCIDirectTwoElectronGPUWorkspace

Own one reusable CPU factor workspace plus device-resident conventional ERIs, one-body operator, selected-orbital transform buffers, AO factors, null vectors, and reduction partials for G-NOCI.
All device arrays belong to one selected CUDA device, values follow input ERI or dimensionless cofactor units, and callers must release the workspace with `_close_gronor_noci_two_electron_gpu_workspace!`.
"""
mutable struct GronorNOCIDirectTwoElectronGPUWorkspace{T<:Union{Float32,Float64}}
    n_basis::Int
    n_orbital_pairs::Int
    gpu_device::Int
    host::GronorNOCIDirectTwoElectronWorkspace{T}
    integrals_device::CUDA.CuVector{T}
    lower_labels_device::CUDA.CuVector{Int32}
    upper_labels_device::CUDA.CuVector{Int32}
    one_body_operator_device::CUDA.CuMatrix{T}
    left_orbitals_device::CUDA.CuMatrix{T}
    right_orbitals_device::CUDA.CuMatrix{T}
    occupied_factor_device::CUDA.CuMatrix{T}
    transform_scratch_device::CUDA.CuMatrix{T}
    alpha_factor_device::CUDA.CuMatrix{T}
    beta_factor_device::CUDA.CuMatrix{T}
    pair_sum_device::CUDA.CuMatrix{T}
    null_mode_device::CUDA.CuVector{T}
    left_alpha_null_device::CUDA.CuVector{T}
    left_beta_null_device::CUDA.CuVector{T}
    right_alpha_null_device::CUDA.CuVector{T}
    right_beta_null_device::CUDA.CuVector{T}
    partials_device::CUDA.CuVector{T}
    closed::Bool
end

"""
    GronorNOCIDirectTwoElectronGPUWorkspace(integrals, one_body_operator, runtime)

Construct a reusable conventional G-NOCI CUDA workspace on `runtime.gpu_device` and upload the packed physical ERIs and one-body operator once.
Scientific inputs are not mutated, device memory is preflighted before allocation, unavailable CUDA execution raises without CPU fallback, and partial allocations are explicitly released on construction failure.
"""
function GronorNOCIDirectTwoElectronGPUWorkspace(
    integrals::PackedERIValues{T},
    one_body_operator::AbstractMatrix{T},
    runtime::RuntimeOptions,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("G-NOCI CUDA workspace requires runtime.use_gpu=true"))
    estimate = gronor_noci_two_electron_gpu_workspace_memory(T, integrals)
    enforce_memory_limits(estimate, runtime)
    _activate_cuda_device(runtime)
    n_basis = integrals.n_orbitals
    size(one_body_operator) == (n_basis, n_basis) || throw(DimensionMismatch("one-body operator and packed ERI dimensions differ"))
    n_pairs = canonical_pair_count(n_basis)
    lower_labels, upper_labels = _gronor_noci_gpu_pair_labels(n_basis)
    allocated = Any[]
    try
        integrals_device = CUDA.CuArray(integrals.values); push!(allocated, integrals_device)
        lower_labels_device = CUDA.CuArray(lower_labels); push!(allocated, lower_labels_device)
        upper_labels_device = CUDA.CuArray(upper_labels); push!(allocated, upper_labels_device)
        one_body_operator_device = CUDA.CuArray(one_body_operator); push!(allocated, one_body_operator_device)
        left_orbitals_device = CUDA.CuArray{T}(undef, n_basis, n_basis); push!(allocated, left_orbitals_device)
        right_orbitals_device = CUDA.CuArray{T}(undef, n_basis, n_basis); push!(allocated, right_orbitals_device)
        occupied_factor_device = CUDA.CuArray{T}(undef, n_basis, n_basis); push!(allocated, occupied_factor_device)
        transform_scratch_device = CUDA.CuArray{T}(undef, n_basis, n_basis); push!(allocated, transform_scratch_device)
        alpha_factor_device = CUDA.CuArray{T}(undef, n_basis, n_basis); push!(allocated, alpha_factor_device)
        beta_factor_device = CUDA.CuArray{T}(undef, n_basis, n_basis); push!(allocated, beta_factor_device)
        pair_sum_device = CUDA.CuArray{T}(undef, n_basis, n_basis); push!(allocated, pair_sum_device)
        null_mode_device = CUDA.CuArray{T}(undef, n_basis); push!(allocated, null_mode_device)
        left_alpha_null_device = CUDA.CuArray{T}(undef, n_basis); push!(allocated, left_alpha_null_device)
        left_beta_null_device = CUDA.CuArray{T}(undef, n_basis); push!(allocated, left_beta_null_device)
        right_alpha_null_device = CUDA.CuArray{T}(undef, n_basis); push!(allocated, right_alpha_null_device)
        right_beta_null_device = CUDA.CuArray{T}(undef, n_basis); push!(allocated, right_beta_null_device)
        partials_device = CUDA.CuArray{T}(undef, n_pairs); push!(allocated, partials_device)
        return GronorNOCIDirectTwoElectronGPUWorkspace(
            n_basis,
            n_pairs,
            runtime.gpu_device,
            GronorNOCIDirectTwoElectronWorkspace{T}(),
            integrals_device,
            lower_labels_device,
            upper_labels_device,
            one_body_operator_device,
            left_orbitals_device,
            right_orbitals_device,
            occupied_factor_device,
            transform_scratch_device,
            alpha_factor_device,
            beta_factor_device,
            pair_sum_device,
            null_mode_device,
            left_alpha_null_device,
            left_beta_null_device,
            right_alpha_null_device,
            right_beta_null_device,
            partials_device,
            false,
        )
    catch
        for array in Iterators.reverse(allocated)
            CUDA.unsafe_free!(array)
        end
        rethrow()
    end
end

"""
    _close_gronor_noci_two_electron_gpu_workspace!(workspace)

Release every device allocation owned by a G-NOCI CUDA workspace and return `nothing`.
The workspace is mutated to an idempotent closed state, scientific host inputs are unaffected, no result is retained on device, and subsequent contraction attempts raise `ArgumentError`.
"""
function _close_gronor_noci_two_electron_gpu_workspace!(workspace::GronorNOCIDirectTwoElectronGPUWorkspace)
    workspace.closed && return nothing
    CUDA.device!(workspace.gpu_device)
    for array in (
        workspace.partials_device,
        workspace.right_beta_null_device,
        workspace.right_alpha_null_device,
        workspace.left_beta_null_device,
        workspace.left_alpha_null_device,
        workspace.null_mode_device,
        workspace.pair_sum_device,
        workspace.beta_factor_device,
        workspace.alpha_factor_device,
        workspace.transform_scratch_device,
        workspace.occupied_factor_device,
        workspace.right_orbitals_device,
        workspace.left_orbitals_device,
        workspace.one_body_operator_device,
        workspace.upper_labels_device,
        workspace.lower_labels_device,
        workspace.integrals_device,
    )
        CUDA.unsafe_free!(array)
    end
    workspace.closed = true
    return nothing
end

"""
    _contract_gronor_noci_two_electron_pair_rows_kernel!(partials, integrals, lower_labels, upper_labels, alpha_factor, beta_factor, pair_sum, left_alpha, left_beta, right_alpha, right_beta, singular)

Contract one canonical first-orbital-pair row per CUDA block and write its reduced physical packed-ERI contribution.
Device inputs are read-only, `partials` is overwritten in ERI energy units, canonical diagonal multiplicities are applied exactly, no allocation occurs, and the launch uses `GRONOR_NOCI_TWO_ELECTRON_GPU_THREADS` threads per block.
"""
function _contract_gronor_noci_two_electron_pair_rows_kernel!(
    partials,
    integrals,
    lower_labels,
    upper_labels,
    alpha_factor,
    beta_factor,
    pair_sum,
    left_alpha,
    left_beta,
    right_alpha,
    right_beta,
    singular::Bool,
)
    first_pair = Int(CUDA.blockIdx().x)
    lane = Int(CUDA.threadIdx().x)
    n_pairs = length(lower_labels)
    p = Int(@inbounds lower_labels[first_pair])
    q = Int(@inbounds upper_labels[first_pair])
    first_scale = p == q ? eltype(partials)(0.5) : one(eltype(partials))
    accumulation = zero(eltype(partials))
    second_pair = first_pair + lane - 1
    while second_pair <= n_pairs
        r = Int(@inbounds lower_labels[second_pair])
        s = Int(@inbounds upper_labels[second_pair])
        second_scale = r == s ? eltype(partials)(0.5) : one(eltype(partials))
        pair_scale = first_pair == second_pair ? eltype(partials)(0.5) : one(eltype(partials))
        packed_index = first_pair + ((second_pair - 1) * second_pair) ÷ 2
        integral = @inbounds integrals[packed_index] * first_scale * second_scale * pair_scale
        term = singular ? _gronor_noci_singular_two_electron_term_gpu(
            alpha_factor,
            beta_factor,
            pair_sum,
            left_alpha,
            left_beta,
            right_alpha,
            right_beta,
            p,
            q,
            r,
            s,
        ) : _gronor_noci_regular_two_electron_term_gpu(alpha_factor, beta_factor, pair_sum, p, q, r, s)
        accumulation += integral * term
        second_pair += GRONOR_NOCI_TWO_ELECTRON_GPU_THREADS
    end
    shared = CUDA.CuStaticSharedArray(eltype(partials), GRONOR_NOCI_TWO_ELECTRON_GPU_THREADS)
    @inbounds shared[lane] = accumulation
    CUDA.sync_threads()
    offset = GRONOR_NOCI_TWO_ELECTRON_GPU_THREADS >>> 1
    while offset >= 1
        if lane <= offset
            @inbounds shared[lane] += shared[lane + offset]
        end
        CUDA.sync_threads()
        offset >>>= 1
    end
    lane == 1 && (@inbounds partials[first_pair] = shared[1])
    return nothing
end

"""
    _fill_gronor_noci_ao_two_electron_factors_gpu!(workspace, preparation)

Build rank-aware AO cofactor matrices and null vectors with reusable single-GPU cuBLAS workspaces for one prepared G-NOCI pair.
The CUDA workspace and its nested host occupied-factor buffer are mutated while the preparation remains read-only, values are dimensionless, no result is copied back to the host, and dimension/device inconsistencies raise before contraction.
"""
function _fill_gronor_noci_ao_two_electron_factors_gpu!(
    workspace::GronorNOCIDirectTwoElectronGPUWorkspace{T},
    preparation::GronorNOCIDeterminantPairPreparation{T},
) where {T<:Union{Float32,Float64}}
    cofactors = preparation.cofactors
    pair = preparation.pair
    n_alpha = cofactors.n_alpha
    occupied_count = size(cofactors.first_cofactor, 1)
    n_beta = occupied_count - n_alpha
    n_basis = workspace.n_basis
    workspace.host.n_basis == n_basis && workspace.host.occupied_count == occupied_count || throw(DimensionMismatch("G-NOCI CUDA host workspace dimensions differ from pair preparation"))
    size(pair.left.alpha) == (n_basis, n_alpha) && size(pair.right.alpha) == (n_basis, n_alpha) || throw(DimensionMismatch("alpha occupied orbitals differ from cofactor partition"))
    size(pair.left.beta) == (n_basis, n_beta) && size(pair.right.beta) == (n_basis, n_beta) || throw(DimensionMismatch("beta occupied orbitals differ from cofactor partition"))
    _fill_gronor_noci_occupied_factor!(workspace.host.occupied_factor, cofactors)
    transform_spin! = function (left_host, right_host, factor_host, output_device, n_spin)
        if n_spin == 0
            fill!(output_device, zero(T))
            return nothing
        end
        left_device = @view workspace.left_orbitals_device[:, 1:n_spin]
        right_device = @view workspace.right_orbitals_device[:, 1:n_spin]
        factor_device = @view workspace.occupied_factor_device[1:n_spin, 1:n_spin]
        scratch_device = @view workspace.transform_scratch_device[1:n_spin, :]
        copyto!(workspace.left_orbitals_device, 1, left_host, 1, length(left_host))
        copyto!(workspace.right_orbitals_device, 1, right_host, 1, length(right_host))
        factor_stage = workspace.host.alpha_factor
        @views factor_stage[1:n_spin, 1:n_spin] .= factor_host
        copyto!(workspace.occupied_factor_device, 1, factor_stage, 1, n_basis * n_spin)
        CUDA.CUBLAS.gemm!('N', 'T', one(T), factor_device, right_device, zero(T), scratch_device)
        CUDA.CUBLAS.gemm!('N', 'N', one(T), left_device, scratch_device, zero(T), output_device)
        return nothing
    end
    alpha_range = 1:n_alpha
    beta_range = (n_alpha + 1):(n_alpha + n_beta)
    transform_spin!(pair.left.alpha, pair.right.alpha, @view(workspace.host.occupied_factor[alpha_range, alpha_range]), workspace.alpha_factor_device, n_alpha)
    transform_spin!(pair.left.beta, pair.right.beta, @view(workspace.host.occupied_factor[beta_range, beta_range]), workspace.beta_factor_device, n_beta)
    for vector in (
        workspace.left_alpha_null_device,
        workspace.left_beta_null_device,
        workspace.right_alpha_null_device,
        workspace.right_beta_null_device,
    )
        fill!(vector, zero(T))
    end
    if cofactors.rank_deficiency in (1, 2)
        transform_null! = function (orbitals_host, mode_host, output_device, n_spin, scale)
            n_spin == 0 && return nothing
            orbitals_device = @view workspace.left_orbitals_device[:, 1:n_spin]
            mode_device = @view workspace.null_mode_device[1:n_spin]
            copyto!(workspace.left_orbitals_device, 1, orbitals_host, 1, length(orbitals_host))
            mode_stage = workspace.host.left_alpha_null
            copyto!(mode_stage, 1, mode_host, 1, n_spin)
            copyto!(workspace.null_mode_device, 1, mode_stage, 1, n_spin)
            CUDA.CUBLAS.gemv!('N', one(T), orbitals_device, mode_device, zero(T), output_device)
            scale == one(T) || (output_device .*= scale)
            return nothing
        end
        transform_null!(pair.left.alpha, @view(cofactors.left_null_modes[alpha_range, 1]), workspace.left_alpha_null_device, n_alpha, cofactors.nonzero_cofactor_scale)
        transform_null!(pair.left.beta, @view(cofactors.left_null_modes[beta_range, 1]), workspace.left_beta_null_device, n_beta, cofactors.nonzero_cofactor_scale)
        transform_null!(pair.right.alpha, @view(cofactors.right_null_modes[alpha_range, 1]), workspace.right_alpha_null_device, n_alpha, one(T))
        transform_null!(pair.right.beta, @view(cofactors.right_null_modes[beta_range, 1]), workspace.right_beta_null_device, n_beta, one(T))
    end
    workspace.pair_sum_device .= workspace.alpha_factor_device .+ transpose(workspace.alpha_factor_device) .+ workspace.beta_factor_device .+ transpose(workspace.beta_factor_device)
    return workspace
end

"""
    _contract_gronor_noci_one_body_gpu!(workspace, preparation; runtime, telemetry=nothing)

Contract one prepared G-NOCI transition density with the device-resident one-body operator and return its bare overlap/electronic contribution.
The reusable CUDA density buffers are overwritten while scientific inputs remain read-only, output follows one-body energy units, device memory is preflighted, and no CPU contraction fallback is permitted.
"""
function _contract_gronor_noci_one_body_gpu!(
    workspace::GronorNOCIDirectTwoElectronGPUWorkspace{T},
    preparation::GronorNOCIDeterminantPairPreparation{T};
    runtime::RuntimeOptions,
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("G-NOCI CUDA one-body contraction requires runtime.use_gpu=true"))
    workspace.closed && throw(ArgumentError("G-NOCI CUDA workspace is closed"))
    runtime.gpu_device == workspace.gpu_device || throw(DimensionMismatch("runtime GPU device differs from the G-NOCI CUDA workspace"))
    estimate = MemoryEstimate(0, estimate_array_bytes(T, 4, workspace.n_basis, workspace.n_basis))
    enforce_memory_limits(estimate, runtime)
    _activate_cuda_device(runtime)
    work = function ()
        copyto!(workspace.alpha_factor_device, preparation.alpha_transition_density)
        copyto!(workspace.beta_factor_device, preparation.beta_transition_density)
        workspace.pair_sum_device .= workspace.alpha_factor_device .+ workspace.beta_factor_device
        CUDA.synchronize()
        value = dot(workspace.one_body_operator_device, workspace.pair_sum_device)
        return GronorNOCIDeterminantPairOneBodyElement(preparation.cofactors.overlap, value)
    end
    value = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :contract_gronor_noci_one_body;
        estimate,
        metadata=Dict(:strategy => :cuda_density_dot, :gpu_device => runtime.gpu_device),
        collect_allocations=runtime.collect_allocations,
    )
    !isnothing(telemetry) && increment_counter!(telemetry, :n_gronor_noci_cuda_one_body_contractions)
    return value
end

"""
    contract_gronor_noci_two_electron!(workspace::GronorNOCIDirectTwoElectronGPUWorkspace, preparation, integrals; runtime, telemetry=nothing)

Contract one prepared conventional G-NOCI determinant pair with reusable device-resident ERIs and AO-factor workspaces.
The workspace and its nested CPU buffers are mutated while scientific inputs remain read-only, output follows ERI energy units, device selection and memory limits are enforced, nullity three or greater returns zero, and no CPU contraction fallback is permitted.
"""
function contract_gronor_noci_two_electron!(
    workspace::GronorNOCIDirectTwoElectronGPUWorkspace{T},
    preparation::GronorNOCIDeterminantPairPreparation{T},
    integrals::PackedERIValues{T};
    runtime::RuntimeOptions=RuntimeOptions(use_gpu=true),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:Union{Float32,Float64}}
    runtime.use_gpu || throw(ArgumentError("G-NOCI CUDA contraction requires runtime.use_gpu=true"))
    workspace.closed && throw(ArgumentError("G-NOCI CUDA workspace is closed"))
    runtime.gpu_device == workspace.gpu_device || throw(DimensionMismatch("runtime GPU device differs from the G-NOCI CUDA workspace"))
    integrals.n_orbitals == workspace.n_basis || throw(DimensionMismatch("packed ERI and G-NOCI CUDA workspace dimensions differ"))
    _activate_cuda_device(runtime)
    host_estimate = gronor_noci_two_electron_memory(T, preparation)
    device_estimate = gronor_noci_two_electron_gpu_workspace_memory(T, integrals)
    estimate = combine_memory_estimates(host_estimate, device_estimate)
    enforce_memory_limits(estimate, runtime)
    occupied_count = size(preparation.cofactors.first_cofactor, 1)
    needs_resize = workspace.host.n_basis != workspace.n_basis || workspace.host.occupied_count != occupied_count
    needs_resize && _resize_gronor_noci_two_electron_workspace!(workspace.host, workspace.n_basis, occupied_count)
    work = function ()
        preparation.cofactors.rank_deficiency >= 3 && return zero(T)
        _fill_gronor_noci_ao_two_electron_factors_gpu!(workspace, preparation)
        CUDA.@cuda threads=GRONOR_NOCI_TWO_ELECTRON_GPU_THREADS blocks=workspace.n_orbital_pairs _contract_gronor_noci_two_electron_pair_rows_kernel!(
            workspace.partials_device,
            workspace.integrals_device,
            workspace.lower_labels_device,
            workspace.upper_labels_device,
            workspace.alpha_factor_device,
            workspace.beta_factor_device,
            workspace.pair_sum_device,
            workspace.left_alpha_null_device,
            workspace.left_beta_null_device,
            workspace.right_alpha_null_device,
            workspace.right_beta_null_device,
            preparation.cofactors.rank_deficiency != 0,
        )
        CUDA.synchronize()
        accumulation = sum(Array(workspace.partials_device))
        return preparation.cofactors.rank_deficiency == 0 ? T(4) * preparation.cofactors.overlap * accumulation : accumulation
    end
    value = isnothing(telemetry) ? work() : record_phase!(
        work,
        telemetry,
        :contract_gronor_noci_two_electron;
        estimate,
        metadata=Dict(
            :n_packed_eri_values => length(integrals.values),
            :rank_deficiency => preparation.cofactors.rank_deficiency,
            :workspace_reused => !needs_resize,
            :strategy => :cuda_packed_pair_rows,
            :ao_factor_strategy => :cuda_cublas,
            :gpu_device => runtime.gpu_device,
        ),
        collect_allocations=runtime.collect_allocations,
    )
    if !isnothing(telemetry)
        needs_resize && increment_counter!(telemetry, :n_gronor_noci_two_electron_workspace_resizes)
        increment_counter!(telemetry, :n_gronor_noci_two_electron_contractions)
        increment_counter!(telemetry, :n_gronor_noci_cuda_two_electron_contractions)
    end
    return value
end

"""
    contract_gronor_noci_determinant_pair!(workspace::GronorNOCIDirectTwoElectronGPUWorkspace, preparation, one_body_operator, integrals; runtime, telemetry=nothing)

Return one complete bare conventional G-NOCI determinant-pair `S/H1/H2` value while reusing the supplied CUDA two-electron workspace.
The workspace is mutated and scientific inputs are read-only, Hamiltonian components follow integral energy units, no nuclear/CI weighting occurs, and unavailable or mismatched GPU execution raises without CPU fallback.
"""
function contract_gronor_noci_determinant_pair!(
    workspace::GronorNOCIDirectTwoElectronGPUWorkspace{T},
    preparation::GronorNOCIDeterminantPairPreparation{T},
    one_body_operator::AbstractMatrix{T},
    integrals::PackedERIValues{T};
    runtime::RuntimeOptions=RuntimeOptions(use_gpu=true),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:Union{Float32,Float64}}
    size(one_body_operator) == (workspace.n_basis, workspace.n_basis) || throw(DimensionMismatch("one-body operator and G-NOCI CUDA workspace dimensions differ"))
    one_body = _contract_gronor_noci_one_body_gpu!(workspace, preparation; runtime, telemetry)
    two_body = contract_gronor_noci_two_electron!(workspace, preparation, integrals; runtime, telemetry)
    return GronorNOCIDeterminantPairMatrixElement(one_body.overlap, one_body.electronic_one_body, two_body)
end
