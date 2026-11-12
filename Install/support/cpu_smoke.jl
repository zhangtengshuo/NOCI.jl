using NOCI
using YAML
using TOML

"""
    check_cpu_installation(workdir)

Verify package loading without CUDA in the CPU project and write a small receipt.
Only caller-owned metadata are written, values have no physical units, no scientific input or GPU is initialized, and a CUDA dependency, loaded CUDA module or active extension raises before activation.
"""
function check_cpu_installation(workdir::String)
    project = dirname(dirname(pathof(NOCI)))
    manifest = TOML.parsefile(joinpath(project, "Manifest.toml"))
    forbidden = ("CUDA", "CUDA_Runtime_jll", "CUDA_Driver_jll", "GPUCompiler")
    any(name -> haskey(manifest["deps"], name), forbidden) && error("CPU lock contains CUDA")
    any(id -> id.name in forbidden, keys(Base.loaded_modules)) && error("CPU loading initialized CUDA")
    isnothing(Base.get_extension(NOCI, :NOCICUDAExt)) || error("CUDA extension loaded in the CPU project")
    mkpath(workdir)
    open(joinpath(workdir, "cpu-check.toml"), "w") do io
        TOML.print(io, Dict("version"=>string(NOCI.noci_version()), "cuda_in_cpu_lock"=>false, "cuda_module_loaded"=>false, "cuda_extension_loaded"=>false); sorted=true)
    end
    println("CPU package check passed; CUDA was neither installed by the CPU lock nor loaded.")
    return nothing
end

length(ARGS)==1 || error("usage: cpu_smoke.jl WORK_DIRECTORY")
check_cpu_installation(abspath(ARGS[1]))
