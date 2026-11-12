const _cuda_load_lock = ReentrantLock()

"""
    _cuda_module()

Load the explicitly requested optional CUDA backend and return its module.
No scientific input is read or mutated, no package is installed, and missing optional support raises `GPUUnavailableError` before device allocation.
CPU execution never calls this function; installation must have selected the locked GPU profile beforehand.
"""
function _cuda_module()
    noci_gpu_supported() || throw(GPUUnavailableError(0, "this NOCI.jl release supports CPU execution only"))
    extension = Base.get_extension(@__MODULE__, :NOCICUDAExt)
    if isnothing(extension)
        lock(_cuda_load_lock) do
            if isnothing(Base.get_extension(@__MODULE__, :NOCICUDAExt))
                try
                    package = Base.identify_package("CUDA")
                    isnothing(package) && error("CUDA is not installed in the selected profile")
                    Base.require(package)
                catch error
                    throw(GPUUnavailableError(0, "optional CUDA support is unavailable; install this release with --withgpu and use its GPU environment"))
                end
            end
        end
        extension = Base.get_extension(@__MODULE__, :NOCICUDAExt)
    end
    isnothing(extension) && throw(GPUUnavailableError(0, "the optional NOCICUDAExt extension did not load"))
    return getfield(extension, :CUDA)
end

"""
    _cuda_dispatch(name, args...; kwargs...)

Invoke a named optional GPU implementation after loading the installed CUDA extension.
Argument units, mutation, return values, and memory limits belong to the concrete implementation, package installation is never performed, and missing support raises `GPUUnavailableError`.
The latest world is used so an explicit first GPU request can load its implementation safely within an existing CPU call stack.
"""
function _cuda_dispatch(name::Symbol, args...; kwargs...)
    _cuda_module()
    extension = Base.get_extension(@__MODULE__, :NOCICUDAExt)
    return Base.invokelatest(getfield(extension, name), args...; kwargs...)
end
