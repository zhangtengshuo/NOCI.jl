using TOML
using SHA
using Tar
using Dates
include("bootstrap.jl")

const SOURCE_PAYLOAD = ("LICENSE", "Manifest.toml", "Project.toml", "OpenMolcasCompatibility.toml", "README.md", "README.zh.md", "bin", "src", "tools/import_seward_hdf5.jl", "tools/openmolcas4noci.jl", "tools/seward_to_factor_bundle.jl", "scripts/preparation", "docs/tutorials", "examples/ahpc_noci_standard")

"""
    copy_cpu_product(source, target)

Copy only the CPU product closure to a new version installation directory.
Paths have no physical units, source files remain read-only, GPU implementations and management files are excluded, no dependency operation or scientific work occurs, and missing files, symlinks or copy failures raise without activation.
"""
function copy_cpu_product(source::String, target::String)
    for name in SOURCE_PAYLOAD
        original = joinpath(source, name)
        ispath(original) || error("missing product payload: $name")
        islink(original) && error("product symlink is forbidden: $name")
        if isfile(original)
            destination = joinpath(target, name)
            mkpath(dirname(destination))
            cp(original, destination)
            continue
        end
        for (directory, subdirs, files) in walkdir(original; follow_symlinks=false)
            any(child -> islink(joinpath(directory, child)), subdirs) && error("product directory contains a symlink")
            filter!(child -> !startswith(child, "."), subdirs)
            for file in files
                file == "AGENTS.md" && continue
                startswith(file, ".") && continue
                path = joinpath(directory, file)
                relative = relpath(path, source)
                startswith(relative, joinpath("src", "algorithms")) && occursin("cuda", lowercase(file)) && continue
                islink(path) && error("product file is a symlink: $relative")
                destination = joinpath(target, relative)
                mkpath(dirname(destination))
                cp(path, destination)
            end
        end
    end
    return nothing
end


"""
    write_record(path, record)

Atomically write a TOML installation record at a caller-owned path.
Values contain paths, versions and integrity metadata without scientific units, the destination is replaced after a complete temporary write, and I/O failures propagate without changing other installation records.
"""
function write_record(path::String, record::AbstractDict)
    mkpath(dirname(path))
    temporary, io = mktemp(dirname(path))
    try
        TOML.print(io, record; sorted=true)
        close(io)
        mv(temporary, path; force=true)
    finally
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary)
    end
    return nothing
end

"""
    fetch_verified(asset, cache)

Download or reuse an HTTPS asset verified against its pinned SHA-256 and return its host path.
The cache is the only mutated directory, byte content is never executed before verification, no package resolution occurs, and malformed pins, checksum or network failures propagate.
"""
function fetch_verified(asset::AbstractDict, cache::String)
    digest = asset["sha256"]
    occursin(r"^[a-f0-9]{64}$", digest) || error("unfinalized asset checksum")
    startswith(asset["url"], "https://") || error("asset URL must use HTTPS")
    mkpath(cache)
    path = joinpath(cache, digest)
    isfile(path) && bytes2hex(open(sha256, path)) == digest && return path
    temporary = path * ".partial"
    run(`curl --fail --location --proto =https --proto-redir =https --retry 3 --connect-timeout 20 --max-time 1800 $(asset["url"]) -o $temporary`)
    bytes2hex(open(sha256, temporary)) == digest || error("asset checksum mismatch: $(asset["url"])")
    mv(temporary, path; force=true)
    return path
end

"""
    unpack_spec(archive, destination, top_directory)

Validate and extract the small runtime specification archive, returning its root directory.
Archive paths must stay under the expected root and contain only files/directories, only the task-owned destination changes, and unsafe headers or extraction failures propagate before package installation.
"""
function unpack_spec(archive::String, destination::String, top_directory::String)
    headers = open(`gzip -dc $archive`) do io
        Tar.list(io)
    end
    for header in headers
        parts = split(header.path, '/')
        startswith(header.path, '/') && error("absolute archive path")
        ".." in parts && error("archive traversal")
        first(parts) == top_directory || error("unexpected archive root")
        header.type in (:file, :directory) || error("runtime specification contains a link or special file")
    end
    mkpath(destination)
    run(`tar -xzf $archive -C $destination --no-same-owner`)
    return joinpath(destination, top_directory)
end

"""
    audit_cpu_lock(source)

Reject mandatory GPU or Conda Julia dependencies in a CPU installation profile.
The source Project/Manifest are read-only, package names are dimensionless metadata, no scientific arrays or GPU context are created, and forbidden dependencies raise before instantiation.
"""
function audit_cpu_lock(source::String)
    project = TOML.parsefile(joinpath(source, "Project.toml"))
    manifest = TOML.parsefile(joinpath(source, "Manifest.toml"))
    for name in ("CUDA", "CUDA_Runtime_jll", "CUDA_Driver_jll", "CUDA_Compiler_jll", "GPUCompiler", "CondaPkg", "MicroMamba", "micromamba_jll")
        haskey(project["deps"], name) && error("mandatory optional dependency: $name")
        haskey(manifest["deps"], name) && error("CPU Manifest contains $name")
    end
    return nothing
end

"""
    verify_packages(prefix, expected, yaml)

Compare the complete installed native package set with the release lock metadata.
The prefix and expected lock are read-only, names/version/build/subdir/SHA-256 identify exact package bytes, no solver or scientific work occurs, and missing, extra or changed packages raise.
"""
function verify_packages(prefix::String, expected::Vector, yaml::Module)
    directory = joinpath(prefix, "conda-meta")
    isdir(directory) || error("missing native package metadata")
    actual = Dict{String,Any}()
    for path in readdir(directory; join=true)
        endswith(path, ".json") || continue
        package = Base.invokelatest(yaml.load_file, path)
        haskey(actual, package["name"]) && error("duplicate native package")
        actual[package["name"]] = package
    end
    Set(keys(actual)) == Set(p["name"] for p in expected) || error("native package closure differs from release lock")
    for package in expected, key in ("version", "build", "subdir", "sha256")
        get(actual[package["name"]], key, nothing) == package[key] || error("native package identity changed: $(package["name"]) $key")
    end
    return nothing
end

"""
    verify_program_package(prefix, asset, yaml)

Verify the installed OpenMolcas program package identity for a local-archive install.
Name/version/build are compared with the pinned release asset, while archive byte identity is already guaranteed by the verified SHA-256 download; local-archive installs record an empty conda-meta subdir and no comparable archive digest, so those two keys are intentionally not checked here.
The prefix and release asset are read-only, no solver or network access occurs, and any mismatch propagates before the runtime context is recorded.
"""
function verify_program_package(prefix::String, asset::AbstractDict, yaml::Module)
    directory = joinpath(prefix, "conda-meta")
    isdir(directory) || error("missing program package metadata")
    records = [f for f in readdir(directory; join=true) if endswith(f, ".json")]
    length(records) == 1 || error("program prefix must contain exactly one package")
    actual = Base.invokelatest(yaml.load_file, only(records))
    actual["name"] == asset["name"] || error("program package name mismatch")
    actual["version"] == asset["version"] || error("program package version mismatch")
    actual["build"] == asset["build"] || error("program package build mismatch")
    return nothing
end

"""
shell_quote(value)

Quote a host string as one literal Bash argument without evaluating its content.
The input is read-only, the output has no physical units, no subprocess or GPU operation occurs, and allocation failures propagate.
"""
function shell_quote(value::String)
    return "'" * replace(value, "'" => "'\"'\"'") * "'"
end

"""
    install_release(root, source, config_path, features, activate, support)

Install a copied CPU NOCI source profile and explicitly selected OpenMolcas component with immutable release locks.
Host paths, versions and package digests are recorded, only the chosen installation root/depot are mutated, completed native prefixes are verified and reused, CPU is the mandatory base, and missing assets or integrity failures leave activation unchanged.
No CondaPkg, MicroMamba.jl, dependency resolution, package updates or scientific production runs occur; the small optional OpenMolcas acceptance probe must pass before activation.
"""
function install_release(root::String, source::String, config_path::String, features::Set{String}, activate::Bool, support::String)
    VERSION == v"1.12.6" || error("locked release requires Julia 1.12.6")
    config = TOML.parsefile(config_path)
    config["schema"] == 3 && config["platform"] == "linux-x86_64" || error("unsupported release configuration")
    features ⊆ Set(["cpu", "openmolcas"]) && "cpu" in features || error("this release supports CPU and OpenMolcas components only")
    version = config["version"]
    project = TOML.parsefile(joinpath(source, "Project.toml"))
    project["version"] == version || error("installer and source versions disagree")
    policy = TOML.parsefile(joinpath(source, "OpenMolcasCompatibility.toml"))
    policy["release"]["gpu"] == false || error("installer requires the CPU release profile")
    config["program"]["version"] == policy["openmolcas4noci"]["default_version"] || error("backend installation pin disagrees with compatibility policy")
    occursin(r"^[0-9]+\.[0-9]+\.[0-9]+$", version) || error("invalid version")
    audit_cpu_lock(source)
    target = joinpath(root, "v" * version)
    ispath(target) && error("version directory already exists; preserve it and use an isolated acceptance prefix")
    all(name -> ispath(joinpath(source, name)), SOURCE_PAYLOAD) || error("incomplete NOCI source payload")
    mkpath(target)
    copy_cpu_product(source, target)
    instantiate_locked(target)
    scientific_project = target
    runtime_context = Dict{String,String}()
    cache = joinpath(root, "installation", "downloads")
    if "openmolcas" in features
        asset = config["runtime"]
        spec_archive = fetch_verified(asset, cache)
        work = mktempdir(joinpath(root, "installation"); prefix="runtime-spec-")
        try
            spec = unpack_spec(spec_archive, work, asset["top_directory"])
            # YAML's supported JSON syntax avoids adding a separate JSON package.
            Pkg.activate(target)
            yaml = Base.require(Main, :YAML)
            packages = Base.invokelatest(yaml.load_file, joinpath(spec, "packages.json"))["packages"]
            dependencies = [p for p in packages if p["name"] != "openmolcas4noci"]
            canonical = "{\"julia\":\"1.12.6\",\"packages\":[" * join(("{" * join(("\"" * key * "\":\"" * package[key] * "\"" for key in ("build", "name", "sha256", "subdir", "version")), ",") * "}" for package in sort(dependencies; by=p->p["name"])), ",") * "],\"platform\":\"linux-x86_64\"}"
            identity = bytes2hex(sha256(canonical))
            identity == asset["accepted_dependencies_identity"] || error("runtime dependency identity mismatch")
            dependency_prefix = joinpath(root, "OpenMolcas4NOCI", "dependencies", identity)
            program_asset = config["program"]
            program_identity = bytes2hex(sha256(program_asset["sha256"] * "\n" * identity))
            program_prefix = joinpath(root, "OpenMolcas4NOCI", "v" * program_asset["version"] * "-" * program_identity[1:16])
            binary = fetch_verified(config["micromamba"], cache)
            micromamba = joinpath(root, "installation", "micromamba-" * config["micromamba"]["version"], "micromamba")
            mkpath(dirname(micromamba)); cp(binary, micromamba; force=true); chmod(micromamba, 0o755)
            mamba_env = Dict("MAMBA_ROOT_PREFIX" => joinpath(root, "installation", "mamba-cache"))
            if !isdir(dependency_prefix)
                run(addenv(`$micromamba --no-rc create -y -p $dependency_prefix -f $(joinpath(spec,"dependencies.explicit.txt"))`, mamba_env))
            end
            Base.invokelatest(verify_packages, dependency_prefix, dependencies, yaml)
            write_record(joinpath(dependency_prefix, "noci-dependencies.toml"), Dict("identity"=>identity, "count"=>length(dependencies)))
            archive_download = fetch_verified(program_asset, cache)
            archive = joinpath(work, basename(program_asset["url"]))
            cp(archive_download, archive)
            if !isdir(program_prefix)
                run(addenv(`$micromamba --no-rc create -y --no-deps -p $program_prefix $archive`, mamba_env))
                ispath(joinpath(program_prefix, "lib")) && error("program prefix unexpectedly contains lib")
                symlink(relpath(joinpath(dependency_prefix,"lib"), program_prefix), joinpath(program_prefix,"lib"))
            end
            Base.invokelatest(verify_program_package, program_prefix, program_asset, yaml)
            realpath(joinpath(program_prefix,"lib")) == realpath(joinpath(dependency_prefix,"lib")) || error("program/dependency binding changed")
            runtime_context = Dict("NOCI_OPENMOLCAS4NOCI"=>joinpath(program_prefix,"bin","pymolcas4noci"), "NOCI_OPENMOLCAS_ROOT"=>joinpath(program_prefix,"libexec","openmolcas4noci"), "NOCI_OPENMOLCAS_DEPENDENCIES"=>dependency_prefix, "MOLCAS_LIBMOLCAS_SO"=>joinpath(program_prefix,"libexec","openmolcas4noci","lib","libmolcas.so"))
            write_record(joinpath(program_prefix,"noci-program.toml"), Dict("identity"=>program_identity,"dependencies_identity"=>identity,"source_revision"=>asset["source_revision"],"version"=>program_asset["version"],"build"=>program_asset["build"],"sha256"=>program_asset["sha256"]))
            smoke = mktempdir(joinpath(root,"installation"); prefix="smoke-")
            for name in ("h2_chh5.in", "h2_ricd_chh5.in")
                cp(joinpath(support, "support", name), joinpath(smoke, name))
            end
            run(addenv(`$(Base.julia_cmd()) --startup-file=no --project=$target $(joinpath(support,"support","smoke.jl")) $smoke`, runtime_context))
        finally
            rm(work; recursive=true, force=true)
        end
    end
    julia = joinpath(Sys.BINDIR, "julia")
    context = Dict("JULIA_BIN"=>julia,"NOCI_ROOT"=>target,"NOCI_JULIA_PROJECT"=>scientific_project,"JULIA_PROJECT"=>scientific_project,"JULIA_DEPOT_PATH"=>joinpath(root,"Julia","depot"))
    merge!(context, runtime_context)
    environment = joinpath(target,"bin","noci-environment.sh")
    open(environment,"w") do io
        println(io,"# Generated installation context; this version owns its preparation commands.")
        for key in sort!(collect(keys(context))); println(io,"export ",key,"=",shell_quote(context[key])); end
        println(io,"export PATH=",shell_quote(joinpath(target,"bin")),":",shell_quote(Sys.BINDIR),":\"\$PATH\"")
        println(io,"export JULIA_NUM_THREADS=\"\${JULIA_NUM_THREADS:-1}\"")
        println(io,"export OPENBLAS_NUM_THREADS=\"\${OPENBLAS_NUM_THREADS:-1}\"")
    end
    records = joinpath(root,"installation","records")
    record = Dict{String,Any}("schema"=>3,"version"=>version,"source"=>target,"components"=>sort!(collect(features)),"completed_at"=>string(now()),"julia"=>julia,"julia_version"=>string(VERSION),"scientific_project"=>scientific_project,"project_sha256"=>bytes2hex(open(sha256,joinpath(target,"Project.toml"))),"manifest_sha256"=>bytes2hex(open(sha256,joinpath(target,"Manifest.toml"))),"openmolcas"=>runtime_context)
    if "openmolcas" in features
        record["openmolcas_version"]=config["program"]["version"]
        record["openmolcas_build"]=config["program"]["build"]
        record["openmolcas_sha256"]=config["program"]["sha256"]
        record["dependencies_identity"]=config["runtime"]["accepted_dependencies_identity"]
    end
    write_record(joinpath(records,"v"*version*".toml"),record)
    mkpath(joinpath(root,"bin")); cp(joinpath(support,"support","noci.sh"),joinpath(root,"bin","noci");force=true);chmod(joinpath(root,"bin","noci"),0o755)
    if activate
        selected = joinpath(root,"noci")
        (ispath(selected) && !islink(selected)) && error("refusing to replace a real noci file/directory")
        temporary = mktempdir(root; prefix=".noci-selection-")
        try
            link = joinpath(temporary,"noci")
            symlink("v"*version,link)
            run(`mv -Tf $link $selected`)
        finally
            rm(temporary;recursive=true,force=true)
        end
    end
    println("Installed NOCI ",version," at ",target,"; components: ",join(sort!(collect(features)),','))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 6 || error("installer requires root, source, TOML config, components, activation, support")
    install_release(abspath(expanduser(ARGS[1])),abspath(ARGS[2]),abspath(ARGS[3]),Set{String}(split(ARGS[4],',')),ARGS[5]=="true",abspath(ARGS[6]))
end
