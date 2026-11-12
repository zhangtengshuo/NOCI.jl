
"""
    JuliaMOTRAGPUWorkspace(orbitals, n_ao, n_orbitals, batch_size, gpu_device)

Construct a CUDA Julia-MOTRA workspace and upload the retained orbital matrix once.
The input matrix is read-only, returned arrays live on `gpu_device`, and CUDA failure is reported without a CPU fallback.
"""
function JuliaMOTRAGPUWorkspace(
    orbitals::Matrix{Float64},
    n_ao::Int,
    n_orbitals::Int,
    batch_size::Int,
    gpu_device::Int,
)
    CUDA.functional() || throw(GPUUnavailableError(gpu_device, "CUDA is not functional for Julia-MOTRA"))
    devices = collect(CUDA.devices())
    gpu_device < length(devices) || throw(GPUUnavailableError(gpu_device, "only $(length(devices)) CUDA devices are visible"))
    size(orbitals) == (n_orbitals, n_ao) || throw(DimensionMismatch("retained orbital matrix shape mismatch"))
    CUDA.device!(gpu_device)
    device_orbitals = CUDA.CuArray(orbitals)
    device_transpose = CUDA.CuArray(Matrix(transpose(orbitals)))
    return JuliaMOTRAGPUWorkspace(
        reshape(device_orbitals, n_orbitals, n_ao, 1),
        reshape(device_transpose, n_ao, n_orbitals, 1),
        CUDA.zeros(Float64, n_ao, n_ao, batch_size),
        CUDA.zeros(Float64, n_orbitals, n_ao, batch_size),
        CUDA.zeros(Float64, n_orbitals, n_orbitals, batch_size),
    )
end

"""
    release_julia_motra_gpu_workspace!(workspace)

Release every CUDA array owned by a Julia-MOTRA workspace and reclaim cached device memory.
The workspace becomes unusable, no host scientific input is modified, and release errors from already-freed arrays are ignored during cleanup.
"""
function release_julia_motra_gpu_workspace!(workspace::JuliaMOTRAGPUWorkspace)
    for array in (workspace.orbitals, workspace.orbitals_transpose, workspace.ao, workspace.half, workspace.mo)
        try
            CUDA.unsafe_free!(array)
        catch
        end
    end
    CUDA.reclaim()
    return nothing
end

"""
    transform_ao_eri_factors_gpu!(host_mo, workspace, host_ao, n_vectors)

Upload one AO-factor batch, perform two strided-batched cuBLAS GEMMs, and copy retained-MO matrices back to CPU storage.
The function mutates `host_mo` and device scratch, leaves `host_ao` unchanged, reports CUDA errors directly, and performs no CPU fallback.
"""
function transform_ao_eri_factors_gpu!(
    host_mo::Array{Float64,3},
    workspace::JuliaMOTRAGPUWorkspace,
    host_ao::Array{Float64,3},
    n_vectors::Integer,
)
    n = Int(n_vectors)
    copyto!(workspace.ao, host_ao)
    CUDA.CUBLAS.gemm_strided_batched!(
        'N', 'N', 1.0, workspace.orbitals, view(workspace.ao, :, :, 1:n),
        0.0, view(workspace.half, :, :, 1:n),
    )
    CUDA.CUBLAS.gemm_strided_batched!(
        'N', 'N', 1.0, view(workspace.half, :, :, 1:n), workspace.orbitals_transpose,
        0.0, view(workspace.mo, :, :, 1:n),
    )
    copyto!(host_mo, workspace.mo)
    return host_mo
end
