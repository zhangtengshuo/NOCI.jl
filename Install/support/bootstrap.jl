using Pkg
using SHA

"""
    instantiate_locked(project)

Instantiate and precompile a complete locked Julia profile without resolution or updates.
The host project and Manifest must retain their SHA-256 values, downloads populate only the configured depot, no scientific inputs are read or mutated, and package/network/lock failures propagate.
The CPU profile contains no GPU or Conda management Julia dependency.
"""
function instantiate_locked(project::String)
    paths = [joinpath(project, name) for name in ("Project.toml", "Manifest.toml")]
    all(isfile, paths) || error("locked project requires Project.toml and Manifest.toml")
    original = [sha256(read(path)) for path in paths]
    Pkg.activate(project)
    Pkg.instantiate(; julia_version_strict=true, allow_autoprecomp=false)
    Pkg.precompile()
    original == [sha256(read(path)) for path in paths] || error("Project/Manifest changed during instantiation")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 && ARGS[1] == "instantiate" || error("usage: bootstrap.jl instantiate PROJECT")
    instantiate_locked(abspath(ARGS[2]))
end
