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

# File: src/runtime/openmolcas4noci.jl
# Purpose: Discover and validate the external OpenMolcas4NOCI installation and library.
# SPDX-License-Identifier: Apache-2.0

# OpenMolcas discovery never installs packages or resolves an environment.

"""
    openmolcas4noci_executable()

Return the installed absolute `pymolcas4noci` launcher from the installation context or PATH.
`NOCI_OPENMOLCAS4NOCI` has priority, inputs and the process environment are not mutated, no package manager or GPU is loaded, and a missing or non-executable launcher raises `ArgumentError`.
"""
function openmolcas4noci_executable()
    override = strip(get(ENV, "NOCI_OPENMOLCAS4NOCI", ""))
    selected = isempty(override) ? Sys.which("pymolcas4noci") : abspath(expanduser(override))
    isnothing(selected) && throw(ArgumentError("OpenMolcas4NOCI is not installed or configured; use the NOCI installer or set NOCI_OPENMOLCAS4NOCI"))
    path = abspath(String(selected))
    isfile(path) && (uperm(path) & 0x01 != 0) || throw(ArgumentError("OpenMolcas4NOCI launcher is not an executable file: $path"))
    return path
end

"""
    openmolcas4noci_identity(; required_capabilities=String[], molcas_root=nothing)

Read and validate the actual launcher's built-in release identity and optional managed installation records.
Version, build, source revision, upstream commit and declared capabilities must match the package compatibility policy; launcher/root/library/dependency bindings must agree.
Inputs are read-only paths and dimensionless capability names, only small local metadata are read, no binary hashes, downloads, package manager, scientific calculation or GPU initialization occurs, and unknown versions or inconsistent bindings raise `ArgumentError` before backend use.
"""
function openmolcas4noci_identity(; required_capabilities=String[], molcas_root=nothing)
    executable = realpath(openmolcas4noci_executable())
    prefix = dirname(dirname(executable))
    expected_root = realpath(joinpath(prefix, "libexec", "openmolcas4noci"))
    selected_root = isnothing(molcas_root) ? get(ENV, "NOCI_OPENMOLCAS_ROOT", expected_root) : molcas_root
    realpath(expanduser(String(selected_root))) == expected_root || throw(ArgumentError("OpenMolcas launcher and installation root belong to different programs"))
    library = joinpath(expected_root, "lib", "libmolcas.so")
    isfile(library) || throw(ArgumentError("OpenMolcas4NOCI shared library is missing: $library"))
    if haskey(ENV, "MOLCAS_LIBMOLCAS_SO") && !isempty(ENV["MOLCAS_LIBMOLCAS_SO"])
        realpath(expanduser(ENV["MOLCAS_LIBMOLCAS_SO"])) == realpath(library) || throw(ArgumentError("OpenMolcas launcher and configured shared library disagree"))
    end
    metadata_path = joinpath(prefix, "share", "openmolcas4noci", "release-manifest.json")
    isfile(metadata_path) || throw(ArgumentError("OpenMolcas4NOCI release identity is missing: $metadata_path"))
    metadata = YAML.load_file(metadata_path)
    metadata["package"] == "openmolcas4noci" || throw(ArgumentError("not an OpenMolcas4NOCI program identity"))
    policy = NOCI_BACKEND_COMPATIBILITY
    version = String(metadata["version"])
    version in policy["openmolcas4noci"]["accepted_versions"] || throw(ArgumentError("NOCI.jl $(noci_version()) has no declared OpenMolcas4NOCI compatibility for $version; accepted: $(join(policy["openmolcas4noci"]["accepted_versions"], ", "))"))
    expected = policy["versions"][version]
    for key in ("upstream_commit", "source_revision", "build_number")
        metadata[key] == expected[key] || throw(ArgumentError("OpenMolcas4NOCI $key disagrees with the accepted $version build"))
    end
    missing = setdiff(String.(required_capabilities), expected["capabilities"])
    isempty(missing) || throw(ArgumentError("OpenMolcas4NOCI $version lacks declared capabilities: $(join(missing, ", "))"))
    dependencies_identity = "unmanaged"
    program_record = joinpath(prefix, "noci-program.toml")
    if isfile(program_record)
        record = TOML.parsefile(program_record)
        for key in ("version", "source_revision")
            record[key] == metadata[key] || throw(ArgumentError("OpenMolcas installed record disagrees with built-in $key"))
        end
        record["build"] == expected["build"] || throw(ArgumentError("OpenMolcas installed build is incompatible"))
        dependencies_identity = String(record["dependencies_identity"])
        dependencies = strip(get(ENV, "NOCI_OPENMOLCAS_DEPENDENCIES", ""))
        isempty(dependencies) && throw(ArgumentError("managed OpenMolcas dependency prefix is not configured"))
        dependencies = realpath(expanduser(dependencies))
        dependency_record = TOML.parsefile(joinpath(dependencies, "noci-dependencies.toml"))
        dependency_record["identity"] == dependencies_identity || throw(ArgumentError("OpenMolcas dependency identity changed"))
        realpath(joinpath(prefix, "lib")) == realpath(joinpath(dependencies, "lib")) || throw(ArgumentError("OpenMolcas program and dependency libraries are bound inconsistently"))
    end
    return Dict{String,Any}("version" => version, "upstream_version" => expected["upstream_version"],
        "upstream_commit" => metadata["upstream_commit"], "source_revision" => metadata["source_revision"],
        "build" => expected["build"], "dependencies_identity" => dependencies_identity,
        "capabilities" => expected["capabilities"], "launcher" => executable, "root" => expected_root,
        "library" => realpath(library))
end

"""
    validate_openmolcas4noci_library(path)

Validate a shared-library path against the same compatible program identity used by the launcher and return its absolute real path.
Only small local metadata are read, paths have no physical units, no library is loaded and no inputs are mutated; an unknown backend or mixed installation raises before `dlopen`.
"""
function validate_openmolcas4noci_library(path::AbstractString)
    selected = realpath(path)
    identity = openmolcas4noci_identity(; required_capabilities=["library"], molcas_root=dirname(dirname(abspath(path))))
    selected == identity["library"] || throw(ArgumentError("configured shared library does not belong to the compatible OpenMolcas4NOCI program"))
    return selected
end

"""
    with_openmolcas4noci(work)

Run a callback with the installed OpenMolcas program and shared dependency prefix exposed temporarily.
The executable and optional `NOCI_OPENMOLCAS_ROOT`/`NOCI_OPENMOLCAS_DEPENDENCIES` are installation paths, the caller's environment is restored afterward, no resolution, downloads or GPU work occur, and missing files or callback failures propagate.
"""
function with_openmolcas4noci(work::F) where {F<:Function}
    openmolcas4noci_identity(; required_capabilities=["library"])
    executable = openmolcas4noci_executable()
    prefix = dirname(dirname(executable))
    default_root = joinpath(prefix, "libexec", "openmolcas4noci")
    molcas_root = abspath(expanduser(get(ENV, "NOCI_OPENMOLCAS_ROOT", default_root)))
    isdir(molcas_root) || throw(ArgumentError("OpenMolcas4NOCI installation root not found: $molcas_root"))
    libmolcas = joinpath(molcas_root, "lib", "libmolcas.so")
    isfile(libmolcas) || throw(ArgumentError("OpenMolcas4NOCI shared library not found: $libmolcas"))
    dependency_prefix = strip(get(ENV, "NOCI_OPENMOLCAS_DEPENDENCIES", ""))
    paths = [joinpath(molcas_root, "bin"), dirname(executable)]
    if !isempty(dependency_prefix)
        dependency_prefix = abspath(expanduser(dependency_prefix))
        isdir(joinpath(dependency_prefix, "lib")) || throw(ArgumentError("OpenMolcas dependency prefix not found: $dependency_prefix"))
        push!(paths, joinpath(dependency_prefix, "bin"))
    end
    push!(paths, get(ENV, "PATH", ""))
    return withenv(
        "MOLCAS" => molcas_root,
        "OPENMOLCAS" => molcas_root,
        "MOLCAS_LIBMOLCAS_SO" => libmolcas,
        "NOCI_OPENMOLCAS4NOCI_ACTIVE" => executable,
        "PATH" => join(paths, ':'),
    ) do
        work()
    end
end

"""
    run_openmolcas4noci(input; scratch=nothing, output=nothing)

Run one OpenMolcas input through the managed `openmolcas4noci` launcher and return the completed `Process`.
The input and optional scratch/output paths are resolved before execution, the calculation runs in the input directory, `MOLCAS_WORKDIR` is set only when requested, standard output is redirected only when `output` is supplied, and process failures propagate without modifying the input.
"""
function run_openmolcas4noci(
    input::AbstractString;
    scratch::Union{Nothing,AbstractString}=nothing,
    output::Union{Nothing,AbstractString}=nothing,
)
    input_path = abspath(input)
    isfile(input_path) || throw(ArgumentError("OpenMolcas input not found: $input_path"))
    scratch_path = isnothing(scratch) ? nothing : abspath(scratch)
    isnothing(scratch_path) || mkpath(scratch_path)
    output_path = isnothing(output) ? nothing : abspath(output)
    isnothing(output_path) || mkpath(dirname(output_path))
    return with_openmolcas4noci() do
        executable = ENV["NOCI_OPENMOLCAS4NOCI_ACTIVE"]
        command = Cmd(Cmd([executable, basename(input_path)]); dir=dirname(input_path))
        if !isnothing(scratch_path)
            command = addenv(command, "WorkDir" => scratch_path, "MOLCAS_WORKDIR" => scratch_path)
        end
        if isnothing(output_path)
            return run(command)
        end
        return open(output_path, "w") do io
            run(pipeline(command; stdout=io))
        end
    end
end
