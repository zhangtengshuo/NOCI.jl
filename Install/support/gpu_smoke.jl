using NOCI
using YAML
using CUDA
using TOML

"""
    check_gpu_installation(workdir)

Verify the optional CUDA extension, a tiny Float64 device matrix product and visible devices.
Only a caller-owned TOML receipt is written, inputs and metadata are dimensionless, device memory is bounded by small matrices, no scientific input is loaded, and unavailable CUDA, missing extension or incorrect arithmetic raises before activation.
"""
function check_gpu_installation(workdir::String)
    NOCI.noci_gpu_supported() || error("product does not declare GPU support")
    CUDA.functional(true) || error("CUDA is not functional")
    NOCI._cuda_module() === CUDA || error("NOCI CUDA extension is not bound to this environment")
    matrix = [1.0 2.0; 3.0 4.0]
    device = CUDA.CuArray(matrix)
    maximum(abs, Array(device * device) - matrix * matrix) <= 1e-12 || error("CUDA Float64 matrix check failed")
    CUDA.synchronize()
    CUDA.versioninfo()
    receipt = Dict("cuda_package_version"=>string(Base.pkgversion(CUDA)),"runtime_version"=>string(CUDA.runtime_version()),"driver_version"=>string(CUDA.driver_version()),"devices"=>[CUDA.name(device) for device in CUDA.devices()],"extension_loaded"=>true,"float64_matrix_check"=>"passed")
    mkpath(workdir)
    open(joinpath(workdir,"gpu-check.toml"),"w") do io
        TOML.print(io,receipt;sorted=true)
    end
    return nothing
end

length(ARGS)==1 || error("usage: gpu_smoke.jl WORK_DIRECTORY")
check_gpu_installation(abspath(ARGS[1]))
