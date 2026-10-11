# case_driver.jl — manifest-driven driver for the standard AHPC-NOCI case chain.
#
# Usage: julia --project <NOCI> scripts/preparation/case_driver.jl STAGE MANIFEST.yaml [MODE]
# Stages: validate | states | placement | commonorb | integrals | noci | run
# Modes (GV800 two-phase convention): prepare generates task directories, inputs
# and tasks_to_submit.txt without ever running OpenMolcas (submit those inputs
# under Slurm or another queue when the cluster owns the compute); compute
# executes the generated tasks directly (workstation mode); finalize validates
# the real outputs, extracts data and publishes the case products; all is the
# default workstation sequence prepare -> compute -> finalize.
# Every stage validates the manifest first (validation lives with manifest
# selection in 00), no stage ever computes a seed orbital: initial RasOrb files
# are user-prepared and referenced per fragment state. Scientific products
# leave the OpenMolcas scratch only through `>>> COPY` commands inside the
# OpenMolcas input, so the scratch may live on another node under a queue
# system: NOCI_MOLCAS_SCRATCH_ROOT relocates it and every successful run
# empties it again; the driver never reads the scratch afterwards.

using YAML
using SHA
using TOML
using Dates
using Printf
using NOCI

include(joinpath(@__DIR__, "validate_case.jl"))
include(joinpath(@__DIR__, "standard_manifest.jl"))
include(joinpath(@__DIR__, "molcas_state_energies.jl"))

const STATE_TEMPLATE_VERSION = "native_c1_rasscf_v2"
const STAGE_MODES = ("prepare", "compute", "finalize", "all")

"""
    case_root()

Return the NOCI installation root that owns this driver (two levels above `scripts/preparation`).
The location is read-only information used to reach the maintained preparation libraries and tools.
"""
case_root() = abspath(joinpath(@__DIR__, "..", ".."))

"""
    load_case(manifest_path)

Validate and normalize a schema-4 registry or an existing legacy manifest and return `(manifest, base, root, fragment_states)`.
Paths identify the manifest directory and its parent case root; files are read-only, allocation is small CPU metadata, and missing inputs/schema failures propagate before scientific work.
"""
function load_case(manifest_path::String)
    if get(YAML.load_file(manifest_path), "schema_version", nothing) == 4
        manifest = standard_read(manifest_path)
        base = dirname(abspath(manifest_path))
        return manifest, base, dirname(base), manifest["electronic_structure"]["states"]
    end
    summary = validate_case(manifest_path)
    manifest = YAML.load_file(manifest_path)
    base = dirname(abspath(manifest_path))
    return manifest, base, dirname(base), summary["fragment_states"]
end

"""
    resolve(base, path)

Resolve one manifest path against the manifest directory, rejecting empty values.
Absolute paths pass through unchanged; the returned path is not required to exist.
"""
function resolve(base::String, path)
    text = string(path)
    isempty(text) && error("empty path in manifest")
    return isabspath(text) ? text : abspath(joinpath(base, text))
end

"""
    file_digest(path)

Return the hex SHA-256 of a file, or an empty string when the file is missing.
Missing files are reported by callers so each stage can name the exact expected product.
"""
file_digest(path::String) = isfile(path) ? bytes2hex(open(sha256, path)) : ""

"""
    write_atomic(path, content)

Write text to `path` through a same-directory temporary file and an atomic rename.
A partial write never replaces the destination, and I/O failures propagate.
"""
function write_atomic(path::String, content::String)
    mkpath(dirname(path))
    temporary = path * ".partial"
    open(temporary, "w") do io
        write(io, content)
    end
    mv(temporary, path; force=true)
    return path
end

"""
    write_record(path, record)

Atomically write a TOML completion/identity record at `path` through a temporary file.
"""
function write_record(path::String, record::AbstractDict)
    mkpath(dirname(path))
    temporary = path * ".partial"
    open(temporary, "w") do io
        TOML.print(io, record; sorted=true)
    end
    mv(temporary, path; force=true)
    return path
end

"""
    write_tasks_list(stage_directory, inputs)

Write the queue-oriented task list of one stage as one absolute input path per line.
The list is regenerated atomically on every prepare call and is empty when every task already holds verified products; a queue submission layer consumes it instead of running compute locally.
"""
function write_tasks_list(stage_directory::String, inputs::AbstractVector)
    content = join([abspath(String(path)) for path in inputs], "\n")
    isempty(content) || (content *= "\n")
    return write_atomic(joinpath(stage_directory, "tasks_to_submit.txt"), content)
end

"""
    openmolcas_identity()

Return the installed OpenMolcas4NOCI program identity used inside task identities.
Identity comes from validated built-in release metadata and managed dependency records, without hashing binaries at startup; installation paths and directory names are not version authorities.
"""
function openmolcas_identity()
    identity = NOCI.openmolcas4noci_identity()
    fields = ("version", "source_revision", "upstream_commit", "build", "dependencies_identity")
    buffer = IOBuffer()
    TOML.print(buffer, Dict(key => identity[key] for key in fields); sorted=true)
    return String(take!(buffer))
end

"""
    molcas_scratch_directory(input, default_scratch)

Resolve the task-owned OpenMolcas scratch directory for one input.
`NOCI_MOLCAS_SCRATCH_ROOT` relocates every scratch directory under a queue-provided node-local root as `<last two input-directory segments>__<input stem>`; without it the caller default beside the input is used. The relocated name is deterministic and built only from readable path segments.
"""
function molcas_scratch_directory(input::String, default_scratch::String)
    root = strip(get(ENV, "NOCI_MOLCAS_SCRATCH_ROOT", ""))
    isempty(root) && return default_scratch
    segments = splitpath(dirname(abspath(input)))
    tail = join(segments[max(1, length(segments) - 1):end], "_")
    return joinpath(abspath(root), tail * "__" * splitext(basename(input))[1])
end

"""
    run_molcas(input, default_scratch)

Run one OpenMolcas input with a task-owned scratch directory and return the output path.
The output is written beside the input as `<stem>.out`; scientific products must leave the scratch through `>>> COPY` commands inside the input because the driver never reads the scratch afterwards, the scratch is emptied after a successful run, and a failed run retains it with a reported path for inspection.
"""
function run_molcas(input::String, default_scratch::String)
    scratch = molcas_scratch_directory(input, default_scratch)
    mkpath(scratch)
    output = joinpath(dirname(input), string(splitext(basename(input))[1], ".out"))
    NOCI.run_openmolcas4noci(input; scratch, output)
    if molcas_succeeded(output)
        try
            rm(scratch; recursive=true, force=true)
        catch cleanup_failure
            println(stderr, "WARNING: scratch cleanup failed for $scratch: ", sprint(showerror, cleanup_failure))
        end
    else
        println(stderr, "WARNING: OpenMolcas failed; scratch retained for inspection: $scratch")
    end
    return output
end

"""
    molcas_succeeded(output)

Return true when the OpenMolcas output records a normal shutdown.
"""
molcas_succeeded(output::String) = isfile(output) && occursin("Happy landing", read(output, String))

"""
    rasscf_energies(output)

Extract the finite total energies printed for the optimized RASSCF roots.
Returns the parsed vector; a malformed or absent energy block raises.
"""
function rasscf_energies(output::String)
    text = read(output, String)
    values = Float64[]
    for match in eachmatch(r"Total energy:\s+(-?[0-9]+\.[0-9]+)", text)
        push!(values, parse(Float64, match.captures[1]))
    end
    isempty(values) && error("no RASSCF total energies found in $output")
    all(isfinite, values) || error("non-finite RASSCF energy in $output")
    return values
end

"""
    fortran_real(value)

Format one positive real threshold in OpenMolcas Fortran notation such as `1.0d-4`.
Scientific-notation exponents use the letter `d`, plain values keep their decimal form, and the input is coerced through `Float64`.
"""
fortran_real(value::Real) = replace(string(Float64(value)), "e" => "d")

"""
    gateway_header(geometry, basis; ricd=true, ricd_threshold=nothing)

Render the C1 GATEWAY block used by every standard-chain OpenMolcas input.
Plain `ricd=true` renders a bare `RICD` decomposition; stage 04 additionally passes `ricd_threshold` so the RICD template renders `RICD` plus `CDThreshold` from `integrals.generation_threshold`, while the conventional-Cholesky SEWARD task passes `ricd=false` because the manual marks the CD accuracy keywords as exclusive with `RICD`.
"""
function gateway_header(geometry::String, basis::String; ricd::Bool=true, ricd_threshold=nothing)
    ricd_line = ""
    if ricd
        ricd_line = isnothing(ricd_threshold) ? "  RICD\n" : "  RICD\n  CDThreshold\n  $(fortran_real(ricd_threshold))\n"
    end
    return """
    &GATEWAY
      coord = $geometry
      basis = $basis
      group = nosym
$(ricd_line)End of input
    """
end

"""
    iterfragment(manifest)

Return the ordered fragment labels of the `mebf_table` rows.
"""
iterfragment(manifest) = [String(row[1]) for row in manifest["mebf_table"]]

"""
    fragment_state_labels(fragment_states, label)

Return the sorted state labels defined for one fragment.
"""
fragment_state_labels(fragment_states, label) = sort(collect(string.(keys(fragment_states[label]))))

"""
    state_task_identity(state, elements, basis, guess_digest)

Build the stage-01 monomer task identity digest for one fragment-state pair.
The identity covers the intrinsic geometry content and atom order, every state science keyword, the basis, the template version, the user initial-orbital digest and the installed OpenMolcas program identity; case labels, fragment letters, placements and output locations are excluded so identical tasks deduplicate across fragments and YAMLs.
"""
function state_task_identity(state, elements, basis::String, guess_digest::String)
    canonical = join(["case-task-v2", join(elements, ","), basis,
        string(state["charge"]), string(state["multiplicity"]), string(state["nactel"]), string(state["inactive"]),
        string(state["ras2"]), string(state["symmetry"]), string(state["ciroot"]), string(state["rlxroot"]),
        STATE_TEMPLATE_VERSION, guess_digest, openmolcas_identity()], "|")
    return bytes2hex(sha256(canonical))
end

"""
    state_task_name(state_label, geometry)

Return the human-readable stage-01 task directory name for one fragment state.
The name follows the GV800 content convention `MolX_<STATE>@<geometry tag>`: `MolX` denotes the intrinsic monomer before placement rotates it into the aggregate as A/B/C, and the geometry tag is the fragment XYZ file stem, so identical molecules deduplicate readably while different monomer geometries coexist. The identity digest itself stays inside the task completion record.
"""
state_task_name(state_label::String, geometry::String) = "MolX_$(state_label)@$(splitext(basename(geometry))[1])"

"""
    acquire_lock(directory)

Create an exclusive stage-lock directory and return its path; the task-owned directory records the PID and wall-clock timestamp.
Only lock metadata is written, no backend/device is initialized, and an existing lock raises for manual owner inspection rather than being bypassed automatically.
"""
function acquire_lock(directory::String)
    mkpath(directory)
    lock_path = joinpath(directory, ".lock")
    ispath(lock_path) && error("stage is locked: $lock_path; inspect the owning process before removing a stale lock")
    mkdir(lock_path)
    open(joinpath(lock_path, "owner"), "w") do io
        println(io, getpid(), " ", now())
    end
    return lock_path
end

"""
    verify_products(record, directory)

Return whether every product still matches its recorded size and identity; schema-4 records use modification-time metadata and legacy records use their existing SHA-256 field.
Files and records are read-only, sizes use bytes and times nanoseconds, no device is initialized, and missing or changed products return false so incomplete tasks are not reused.
"""
function verify_products(record::AbstractDict, directory::String)
    products = get(record, "products", nothing)
    products isa AbstractDict || return false
    for (name, meta) in products
        path = joinpath(directory, name)
        meta isa AbstractDict || return false
        isfile(path) || return false
        stat(path).size == meta["bytes"] || return false
        if haskey(meta, "sha256")
            file_digest(path) == meta["sha256"] || return false
        else
            round(Int64, stat(path).mtime * 1.0e9) == meta["mtime_ns"] || return false
        end
    end
    return true
end

"""
    plan_state_tasks(manifest, base, root, fragment_states)

Plan the stage-01 monomer science tasks and return the ordered task records.
Each record carries the fragment/state labels, task identity, directory, inputs and reusable flag; schema 4 uses readable species/geometry/state tasks and explicit metadata comparison, while legacy manifests retain their digest records.
Intrinsic coordinates use Angstrom, inputs are read-only, only small CPU planning data are allocated, and conflicting definitions or missing seeds raise before calculation.
"""
function plan_state_tasks(manifest, base::String, root::String, fragment_states)
    basis = manifest["electronic_structure"]["basis"]
    initial = manifest["electronic_structure"]["initial_orbitals"]
    cache = workflow_stage_directory(manifest, root, "states")
    tasks = Vector{NamedTuple}()
    for label in iterfragment(manifest)
        geometry = resolve(base, manifest["geometry"]["fragments"][label])
        _, elements, coordinates = load_xyz(geometry)
        for state_label in fragment_state_labels(fragment_states, label)
            state = fragment_states[label][state_label]
            guess = resolve(base, initial[label][state_label])
            isfile(guess) || error("initial orbital file for fragment $label state $state_label not found: $guess")
            if standard_case(manifest)
                definition = standard_task_definition(manifest, label, state_label)
                identity = standard_identity(definition)
                binding = manifest["_bindings"][label]
                task_name = joinpath(binding["molecule"], binding["geometry"], state_label)
            else
                # Include coordinates as well as element order in legacy task identities.
                identity = state_task_identity(state, vcat(elements, string.(coordinates)), basis, file_digest(guess))
                task_name = state_task_name(state_label, geometry)
            end
            task_directory = joinpath(cache, task_name)
            record_path = joinpath(task_directory, "completion.toml")
            reusable = false
            definition_path = joinpath(task_directory, "definition.toml")
            if standard_case(manifest) && isfile(definition_path)
                TOML.parsefile(definition_path) == definition || error("changed shared task definition: $task_directory; use a new geometry identifier or explicitly clear the affected task and dependent case stages")
            end
            if isfile(record_path)
                record = TOML.parsefile(record_path)
                record["identity"] == identity ||
                    error("task name collision in the shared state cache: $task_name holds a different science identity (basis, state keywords or initial orbitals differ); use a separate case root, or review and remove the stale task directory $task_directory first")
                reusable = verify_products(record, task_directory) &&
                           molcas_succeeded(joinpath(task_directory, state_label * ".out"))
            end
            push!(tasks, (label=label, state=state_label, identity=identity, directory=task_directory,
                input=joinpath(task_directory, state_label * ".in"), geometry=geometry, guess=guess,
                state_data=state, reusable=reusable))
        end
    end
    return tasks
end

"""
    stage_states(manifest, base, root, fragment_states, mode)

Plan, reuse or compute the stage-01 monomer science tasks into the shared single-fragment cache (01; modes: prepare, compute, finalize, all).
Every instance/state pair consumes the user seed directly; schema-4 tasks are shared as `f/GS0/S1` and per-case bindings publish short instance names, while existing legacy cases retain their task layout.
Prepare writes input decks, compute runs pending CAS tasks and finalize checks normal termination, finite energies and nonmissing products; user files are read-only, energies use hartree and failures propagate before publishing completion.
"""
function stage_states(manifest, base::String, root::String, fragment_states, mode::String="all")
    directory = workflow_stage_directory(manifest, root, "states")
    lock_path = acquire_lock(directory)
    try
        tasks = plan_state_tasks(manifest, base, root, fragment_states)
        verified = count(task -> task.reusable, tasks)
        pending = let seen = Set{String}(), selected = Vector{NamedTuple}()
            for task in tasks
                task.reusable && continue
                task.directory in seen && continue
                push!(seen, task.directory)
                push!(selected, task)
            end
            selected
        end
        reused = verified
        if mode in ("prepare", "all")
            for task in pending
                mkpath(task.directory)
                standard_case(manifest) && write_record(joinpath(task.directory, "definition.toml"), standard_task_definition(manifest, task.label, task.state))
                if standard_case(manifest)
                    write_atomic(task.input, standard_cas_input(manifest, task))
                    continue
                end
                rlx = task.state_data["rlxroot"]
                seward_keywords = standard_case(manifest) && manifest["_standard"]["cas"]["integral_method"] == "cd" ? "  Cholesky\n  ThrCholesky\n  $(fortran_real(manifest["_standard"]["cas"]["generation_threshold"]))\n" : ""
                write_atomic(task.input, workflow_gateway_header(manifest, task.geometry, manifest["electronic_structure"]["basis"]) * """
                &SEWARD
$(seward_keywords)End of input

                &RASSCF
                  charge = $(task.state_data["charge"])
                  nactel = $(task.state_data["nactel"]) 0 0
                  Inactive = $(task.state_data["inactive"])
                  Ras2 = $(task.state_data["ras2"])
                  symmetry = $(task.state_data["symmetry"])
                  spin = $(task.state_data["multiplicity"])
                  fileorb = $(task.guess)
                  ciroot = $(task.state_data["ciroot"])
                  rlxroot = $rlx
                  prwf = 1.0e-5
                  prsd
                End of input

                >>> COPY \$Project.RasOrb.$rlx \$CurrDir/$(task.state).RasOrb
                >>> COPY \$Project.VecDet.$rlx \$CurrDir/$(task.state).VecDet
                """)
            end
            write_tasks_list(directory, [task.input for task in pending])
            println("  stage $(workflow_stage_code(manifest, "states")) prepare: $(length(pending)) task input(s) generated, $reused verified task(s) reused")
        end
        if mode in ("compute", "all")
            for task in pending
                output = joinpath(task.directory, task.state * ".out")
                if molcas_succeeded(output) &&
                   isfile(joinpath(task.directory, task.state * ".RasOrb")) &&
                   isfile(joinpath(task.directory, task.state * ".VecDet"))
                    continue
                end
                output = run_molcas(task.input, joinpath(task.directory, "scratch"))
                molcas_succeeded(output) || error("state $(task.state) RASSCF failed for fragment $(task.label); see $output")
            end
        end
        if mode in ("finalize", "all")
            for task in pending
                output = joinpath(task.directory, task.state * ".out")
                molcas_succeeded(output) || error("state $(task.state) output for fragment $(task.label) is not complete; see $output")
                energy_record = Dict{String,Any}()
                if standard_case(manifest)
                    casscf, caspt2 = read_molcas_state_energies(output, task.state_data["rlxroot"]; require_caspt2=manifest["_standard"]["caspt2"]["enabled"])
                    energies = [casscf]
                    energy_record = Dict("casscf_energy" => casscf, "caspt2_energy" => caspt2, "root" => task.state_data["rlxroot"])
                else
                    energies = rasscf_energies(output)
                end
                products = Dict{String,Any}()
                for name in (task.state * ".RasOrb", task.state * ".VecDet", task.state * ".out")
                    path = joinpath(task.directory, name)
                    isfile(path) || error("state $(task.state) finished without $name")
                    products[name] = standard_case(manifest) ? Dict("bytes" => stat(path).size, "mtime_ns" => round(Int64, stat(path).mtime * 1.0e9)) : Dict("bytes" => stat(path).size, "sha256" => file_digest(path))
                end
                write_record(joinpath(task.directory, "completion.toml"),
                    merge(Dict("identity" => task.identity, "state" => task.state, "energies" => energies,
                        "products" => products, "openmolcas" => openmolcas_identity(),
                        "template" => standard_case(manifest) ? "standard_user_templates_1" : STATE_TEMPLATE_VERSION, "completed_at" => string(now())), energy_record))
            end
            project = manifest["project"]
            binding_directory = state_product_directory(manifest, root)
            mkpath(binding_directory)
            binding_path = joinpath(binding_directory, "binding.toml")
            bound = isfile(binding_path) ? TOML.parsefile(binding_path) : Dict{String,Any}()
            recorded = get(bound, "identities", Dict{String,Any}())
            names = Dict("$(task.label)_$(task.state)" => relpath(task.directory, directory) for task in tasks)
            sources = standard_case(manifest) ? Dict("$(task.label)_$(task.state)" => Dict(suffix => standard_source(joinpath(task.directory, "$(task.state).$suffix")) for suffix in ("RasOrb", "VecDet", "out")) for task in tasks) : Dict()
            rebound = get(bound, "project", "") != project ||
                      recorded != names || (standard_case(manifest) && get(bound, "sources", Dict()) != sources) ||
                      any(!isfile(joinpath(binding_directory, "$(task.label)_$(task.state).$suffix")) for task in tasks for suffix in ("RasOrb", "VecDet", "out"))
            if rebound
                for task in tasks
                    for suffix in ("RasOrb", "VecDet", "out")
                        cp(joinpath(task.directory, "$(task.state).$(suffix)"),
                            joinpath(binding_directory, "$(task.label)_$(task.state).$(suffix)"); force=true)
                    end
                end
                write_record(binding_path, Dict("project" => project, "identities" => names, "sources" => sources))
                println("  fragment-state bindings published for $project")
            end
            println("  stage $(workflow_stage_code(manifest, "states")) summary: fragment-state pairs=$(length(tasks)) pending=$(length(pending)) reused=$reused; cache=$directory")
        end
    finally
        rm(lock_path; recursive=true, force=true)
    end
    return directory
end

"""
    stage_placement(manifest, base, root, fragment_states, mode)

Derive placement matrices by Kabsch alignment, run one C1 reference per placed fragment, and rotate every state orbital into its placed frame (02; modes: prepare, compute, finalize, all).
Schema 4 reuses maintained Kabsch primitives with explicit atom correspondence, rotates intrinsic XYZ and orbitals consistently and publishes instance-labelled matrices, aggregate geometry and short orbital names; legacy cases retain their previous alignment interface.
Coordinates use Angstrom, original inputs are read-only, prepare allocates small CPU geometry data, compute runs C1 references and finalize performs orbital rotation with OpenMolcas basis metadata; missing references or incompatible orbital dimensions raise.
"""
function stage_placement(manifest, base::String, root::String, fragment_states, mode::String="all")
    directory = workflow_stage_directory(manifest, root, "placement")
    lock_path = acquire_lock(directory)
    try
        basis = manifest["electronic_structure"]["basis"]
        fragments = iterfragment(manifest)
        aggregate = resolve(base, manifest["geometry"]["aggregate"])
        _, aggregate_elements, aggregate_coordinates = load_xyz(aggregate)
        aligned = joinpath(directory, "aligned")
        rotated = joinpath(directory, "rotated")
        mkpath(aligned)
        mkpath(rotated)
        fragment_geometries = [resolve(base, manifest["geometry"]["fragments"][label]) for label in fragments]
        julia = joinpath(Sys.BINDIR, "julia")
        pending_inputs = String[]
        if mode in ("prepare", "all")
            standard_case(manifest) && standard_placement(manifest, root)
            all_matrices = [isfile(joinpath(aligned, "matrix_$(label).txt")) for label in fragments]
            if !all(all_matrices)
                run(`$julia --startup-file=no --project=$(case_root()) $(joinpath(case_root(), "scripts", "preparation", "lib", "AutoRotate.jl")) $aggregate $fragment_geometries --out-dir $aligned`)
            end
            offset = 0
            for (index, label) in enumerate(fragments)
                _, elements, _ = load_xyz(fragment_geometries[index])
                matrix = joinpath(aligned, "matrix_$(label).txt")
                isfile(matrix) || error("alignment produced no matrix_$(label).txt")
                placed = joinpath(directory, "placed_$(label).xyz")
                if !standard_case(manifest)
                    buffer = IOBuffer()
                    println(buffer, length(elements))
                    println(buffer, "placed fragment geometry (aggregate block), Angstrom")
                    for element in elements
                        coordinate = aggregate_coordinates[offset+1]
                        @printf(buffer, "%-2s %20.10f %20.10f %20.10f\n", element, coordinate...)
                        offset += 1
                    end
                    write_atomic(placed, String(take!(buffer)))
                end
                reference = joinpath(directory, "$(label)_c1.RunFile")
                oneint = joinpath(directory, "$(label)_c1.OneInt")
                if !isfile(reference) || !isfile(oneint)
                    input = joinpath(directory, "$(label)_c1.in")
                    if standard_case(manifest)
                        parameters = standard_openmolcas_parameters(manifest, placed)
                        parameters["FRAGMENT"] = label
                        write_atomic(input, standard_render_template(manifest, "fragment_reference", parameters))
                    else
                        write_atomic(input, workflow_gateway_header(manifest, placed, basis) * """
                    &SEWARD
                      oneonly
                    End of input

                    >>> COPY \$Project.RunFile \$CurrDir/$(label)_c1.RunFile
                    >>> COPY \$Project.OneInt \$CurrDir/$(label)_c1.OneInt
                    """)
                    end
                    push!(pending_inputs, input)
                end
            end
            write_tasks_list(directory, pending_inputs)
            println("  stage $(workflow_stage_code(manifest, "placement")) prepare: placement matrices verified, $(length(pending_inputs)) C1 reference input(s) generated")
        end
        if mode in ("compute", "all")
            for label in fragments
                reference = joinpath(directory, "$(label)_c1.RunFile")
                oneint = joinpath(directory, "$(label)_c1.OneInt")
                (isfile(reference) && isfile(oneint)) && continue
                input = joinpath(directory, "$(label)_c1.in")
                isfile(input) || error("missing C1 reference input for fragment $label; run the placement stage prepare first: $input")
                output = run_molcas(input, joinpath(directory, "scratch_$(label)_c1"))
                molcas_succeeded(output) || error("fragment $label C1 reference failed; see $output")
            end
        end
        if mode in ("finalize", "all")
            for (index, label) in enumerate(fragments)
                reference = joinpath(directory, "$(label)_c1.RunFile")
                oneint = joinpath(directory, "$(label)_c1.OneInt")
                isfile(reference) || error("missing fragment C1 RunFile for placement finalize: $reference (run the placement stage compute first)")
                isfile(oneint) || error("missing fragment C1 OneInt for placement finalize: $oneint (run the placement stage compute first)")
                # GV800-proven protocol (B1_finalize): libmolcas expects the active runfile
                # to be a file literally named RUNFIL in the process working directory; expose
                # the authoritative RunFile as a literal RUNFIL copy in a task-owned rotation
                # directory and pass neutrally named orbitals so RotateOrb's prefix
                # stripping yields Mol<label>_<state>.orb outputs.
                matrix = joinpath(aligned, "matrix_$(label).txt")
                isfile(matrix) || error("missing placement matrix for fragment $label; run the placement stage prepare first: $matrix")
                rotation = joinpath(directory, "rotate_$(label)")
                mkpath(rotation)
                runfile_link = joinpath(rotation, "RUNFIL")
                rm(runfile_link; force=true)
                cp(reference, runfile_link; force=true)
                neutral = String[]
                for state in fragment_state_labels(fragment_states, label)
                    source = joinpath(state_product_directory(manifest, root), "$(label)_$(state).RasOrb")
                    isfile(source) || error("missing stage-01 orbital for placement: $source (run the states stage first)")
                    staged = joinpath(rotation, "$(state).orb")
                    cp(source, staged; force=true)
                    push!(neutral, staged)
                end
                run(`$julia --startup-file=no --project=$(case_root()) $(joinpath(case_root(), "scripts", "preparation", "lib", "RotateOrb.jl")) $matrix $neutral --runfile $runfile_link --out-dir $rotation`)
                for state in fragment_state_labels(fragment_states, label)
                    produced = joinpath(rotation, "Mol$(label)_$(state).orb")
                    isfile(produced) || error("rotation for $label $state produced no Mol$(label)_$(state).orb")
                    destination = standard_case(manifest) ? "$(label)_$(state).orb" : "Mol$(label)_$(state).orb"
                    mv(produced, joinpath(rotated, destination); force=true)
                end
                println("  fragment $label aligned, C1 reference validated, $(length(fragment_state_labels(fragment_states, label))) orbitals rotated")
            end
        end
    finally
        rm(lock_path; recursive=true, force=true)
    end
    return directory
end

"""
    frozen_specification(manifest, base, fragment_states, label)

Return the explicit frozen-orbital specification of one fragment as `0` (none) or an ascending range list such as `1-8` or `1-3,6-7`.
Schema 4 expands the user's boolean into the automatically counted all-electron core prefix; legacy manifests retain their existing `freeze_core` derivation.
Indices are dimensionless and start at one, inputs are read-only, memory is small CPU metadata, and unresolved core counts or unsupported elements raise before CommonOrb processing.
"""
function frozen_specification(manifest, base::String, fragment_states, label)
    if standard_case(manifest)
        binding = manifest["_bindings"][label]
        binding["frozen_orbitals"] || return "0"
        n_core = binding["frozen_count"]
        n_core >= 0 || error("core count was deferred by structure-only validation; perform full input validation before processing $label")
        return n_core == 0 ? "0" : "1-$(n_core)"
    end
    manifest["common_orbitals"]["freeze_core"] || return "0"
    _, elements, _ = load_xyz(resolve(base, manifest["geometry"]["fragments"][label]))
    n_core = core_orbital_count(elements)
    return n_core == 0 ? "0" : "1-$(n_core)"
end

"""
    stage_commonorb(manifest, base, root, fragment_states, mode)

Generate the CommonOrb control and file map from the manifest, run the maintained `CommonOrbBase.jl`, and record the deleted-orbital count for stage 04 (stage 03; modes: prepare, compute, finalize, all).
Prepare stages the aggregate C1 reference input and the control files; compute runs the aggregate reference OpenMolcas task; finalize runs CommonOrb over the validated references and writes `commonorb_summary.toml` with the explicit per-fragment frozen specifications.
"""
function stage_commonorb(manifest, base::String, root::String, fragment_states, mode::String="all")
    directory = workflow_stage_directory(manifest, root, "commonorb")
    run_directory = joinpath(directory, "gcommon_run")
    lock_path = acquire_lock(directory)
    try
        basis = manifest["electronic_structure"]["basis"]
        aggregate = workflow_aggregate(manifest, base, root)
        project = manifest["project"]
        fragments = iterfragment(manifest)
        frozen_specs = [frozen_specification(manifest, base, fragment_states, label) for label in fragments]
        mkpath(run_directory)
        summary_path = joinpath(directory, "commonorb_summary.toml")
        if mode != "finalize" && isfile(summary_path) && isfile(joinpath(run_directory, "COMMONORB"))
            println("  CommonOrb already complete; rerun skipped: $summary_path")
            return run_directory
        end
        pending_inputs = String[]
        if mode in ("prepare", "all")
            runfile_link = joinpath(run_directory, "RUNFIL")
            if !isfile(runfile_link)
                input = joinpath(run_directory, "aggregate_reference.in")
                if standard_case(manifest)
                    write_atomic(input, standard_render_template(manifest, "aggregate_reference", standard_openmolcas_parameters(manifest, aggregate)))
                else
                    write_atomic(input, workflow_gateway_header(manifest, aggregate, basis) * """
                &SEWARD
                  oneonly
                End of input

                >>> COPY \$Project.RunFile \$CurrDir/RUNFIL
                >>> COPY \$Project.OneInt \$CurrDir/ONEINT
                """)
                end
                push!(pending_inputs, input)
            end
            control = IOBuffer()
            println(control, "Project")
            println(control, project)
            println(control)
            println(control, "Fragments")
            println(control, length(fragments))
            println(control, join([string(length(fragment_state_labels(fragment_states, label))) for label in fragments], " "))
            println(control)
            println(control, "Threshold")
            println(control, manifest["common_orbitals"]["threshold"])
            println(control)
            if any(spec != "0" for spec in frozen_specs)
                println(control, "Frozen")
                println(control, join(frozen_specs, " "))
                println(control)
            end
            println(control, "Labels")
            for label in fragments
                println(control, label, " ", join(fragment_state_labels(fragment_states, label), " "))
            end
            write_atomic(joinpath(directory, "commonorb.input"), String(take!(control)))
            mapping = IOBuffer()
            println(mapping, "# logical\torb_in\tdet_in\tmolcas_out\tvec_out\tdet_out\troot\tcaspt2")
            for label in fragments
                for state in fragment_state_labels(fragment_states, label)
                    logical = fragment_product_label(manifest, label, state)
                    internal = "$(project)$(label)_$(state)"
                    orbital = standard_case(manifest) ? "$(label)_$(state).orb" : "Mol$(label)_$(state).orb"
                    caspt2_flag = standard_case(manifest) ? "\t$(manifest["_standard"]["caspt2"]["enabled"])" : ""
                    println(mapping, internal, "\t", joinpath(workflow_stage_directory(manifest, root, "placement"), "rotated", orbital), "\t",
                        joinpath(state_product_directory(manifest, root), "$(label)_$(state).VecDet"), "\t", joinpath(state_product_directory(manifest, root), "$(label)_$(state).out"), "\t",
                        logical, ".vec\t", logical, ".det\t", fragment_states[label][state]["rlxroot"], caspt2_flag)
                end
            end
            write_atomic(joinpath(directory, "file_map.tsv"), String(take!(mapping)))
            write_tasks_list(directory, pending_inputs)
            println("  stage $(workflow_stage_code(manifest, "commonorb")) prepare: control and file map written, $(length(pending_inputs)) aggregate reference input(s) generated")
        end
        if mode in ("compute", "all")
            runfile_link = joinpath(run_directory, "RUNFIL")
            if !isfile(runfile_link)
                input = joinpath(run_directory, "aggregate_reference.in")
                isfile(input) || error("missing aggregate reference input; run the commonorb stage prepare first: $input")
                output = run_molcas(input, joinpath(directory, "scratch_aggregate"))
                molcas_succeeded(output) || error("aggregate C1 reference failed; see $output")
            end
        end
        if mode in ("finalize", "all")
            isfile(joinpath(run_directory, "RUNFIL")) || error("missing aggregate RUNFIL for CommonOrb; run the commonorb stage compute first")
            isfile(joinpath(run_directory, "ONEINT")) || error("missing aggregate ONEINT for CommonOrb; run the commonorb stage compute first")
            # CommonOrbBase reads per-fragment references as literal RUNFIL<label>/ONEINT<label>
            # in its working directory (namerun("RUNFIL"*fraglabel)); expose the stage-02
            # fragment C1 references under those exact names.
            for label in fragments
                source_run = joinpath(workflow_stage_directory(manifest, root, "placement"), "$(label)_c1.RunFile")
                source_one = joinpath(workflow_stage_directory(manifest, root, "placement"), "$(label)_c1.OneInt")
                isfile(source_run) || error("missing fragment C1 RunFile for CommonOrb: $source_run (run the placement stage first)")
                isfile(source_one) || error("missing fragment C1 OneInt for CommonOrb: $source_one (run the placement stage first)")
                cp(source_run, joinpath(run_directory, "RUNFIL$(label)"); force=true)
                cp(source_one, joinpath(run_directory, "ONEINT$(label)"); force=true)
            end
            julia = joinpath(Sys.BINDIR, "julia")
            control_path = joinpath(directory, "commonorb.input")
            map_path = joinpath(directory, "file_map.tsv")
            isfile(control_path) || error("missing CommonOrb control; run the commonorb stage prepare first: $control_path")
            isfile(map_path) || error("missing CommonOrb file map; run the commonorb stage prepare first: $map_path")
            cd(run_directory) do
                run(pipeline(`$julia --startup-file=no --project=$(case_root()) $(joinpath(case_root(), "scripts", "preparation", "lib", "CommonOrbBase.jl")) --file-map $(abspath(map_path))`, stdin = control_path, stdout = "commonorb.out"))
            end
            isfile(joinpath(run_directory, "COMMONORB")) || error("CommonOrb produced no COMMONORB")
            for label in fragments, state in fragment_state_labels(fragment_states, label)
                logical = fragment_product_label(manifest, label, state)
                for suffix in ("vec", "det")
                    isfile(joinpath(run_directory, logical * "." * suffix)) || error("CommonOrb produced no $logical.$suffix")
                end
            end
            report = read(joinpath(run_directory, "commonorb.out"), String)
            deleted = match(r"Deleted orbitals in MOTRA\D*(\d+)", report)
            deleted === nothing && error("CommonOrb did not report a deleted-orbital count; see commonorb.out")
            deleted_count = parse(Int, deleted.captures[1])
            write_record(summary_path,
                Dict("deleted" => deleted_count, "frozen" => frozen_specs,
                    "freeze_core" => manifest["common_orbitals"]["freeze_core"],
                    "threshold" => string(manifest["common_orbitals"]["threshold"])))
            println("  CommonOrb complete: threshold=$(manifest["common_orbitals"]["threshold"]), frozen=[$(join(frozen_specs, ", "))], deleted=$deleted_count")
        end
    finally
        rm(lock_path; recursive=true, force=true)
    end
    return run_directory
end

"""
    stage_integrals(manifest, base, root, fragment_states, mode)

Run the aggregate C1 CHH5 SEWARD export and the Julia conversion into `.one`/`.sys` and the portable version-2 factor bundle (`factors.jld2` in schema 4, `<project>.two.jld2` for legacy manifests) (04; modes: prepare, compute, finalize, all).
The SEWARD template follows `integrals.method`: `ricd` renders GATEWAY `RICD` plus `CDThreshold` and a SEWARD section with only `CHH5` (requires OpenMolcas4NOCI v26.06.2), while `conventional_cd` renders the SEWARD `Cholesky`/`ThrCholesky` form; both thresholds come from `integrals.generation_threshold`, and the frozen/deleted counts come from the stage-03 summary records.
"""
function stage_integrals(manifest, base::String, root::String, fragment_states, mode::String="all")
    directory = workflow_stage_directory(manifest, root, "integrals")
    lock_path = acquire_lock(directory)
    try
        basis = manifest["electronic_structure"]["basis"]
        aggregate = workflow_aggregate(manifest, base, root)
        prefix = integral_prefix(manifest)
        export_directory = joinpath(directory, "seward_chh5.NOCI_SEWARD_H5")
        factor_bundle = joinpath(directory, standard_case(manifest) ? "factors.jld2" : prefix * ".two.jld2")
        one_electron = joinpath(directory, prefix * ".one")
        pending_inputs = String[]
        if mode in ("prepare", "all")
        if !molcas_succeeded(joinpath(directory, "seward_chh5.out")) || !isdir(export_directory)
            method = manifest["integrals"]["method"]
            threshold = manifest["integrals"]["generation_threshold"]
            input = joinpath(directory, "seward_chh5.in")
            if standard_case(manifest)
                parameters = standard_openmolcas_parameters(manifest, aggregate; integral_stage=true)
                write_atomic(input, standard_render_template(manifest, "integrals", parameters))
            elseif method == "ricd"
                write_atomic(input, gateway_header(aggregate, basis; ricd=true, ricd_threshold=threshold) * """
                &SEWARD
                  CHH5
                End of input
                """)
            else
                write_atomic(input, gateway_header(aggregate, basis; ricd=false) * """
                &SEWARD
                  Cholesky
                  ThrCholesky
                  $(fortran_real(threshold))
                  CHH5
                End of input
                """)
            end
            push!(pending_inputs, input)
        end
            write_tasks_list(directory, pending_inputs)
            println("  stage $(workflow_stage_code(manifest, "integrals")) prepare: $(length(pending_inputs)) SEWARD CHH5 input(s) generated")
        end
        if mode in ("compute", "all")
            if !molcas_succeeded(joinpath(directory, "seward_chh5.out")) || !isdir(export_directory)
                input = joinpath(directory, "seward_chh5.in")
                isfile(input) || error("missing SEWARD CHH5 input; run the integrals stage prepare first: $input")
                output = run_molcas(input, joinpath(directory, "scratch_seward"))
                molcas_succeeded(output) || error("SEWARD CHH5 failed; see $output")
                isdir(export_directory) || error("SEWARD finished without $export_directory")
            end
        end
        if mode in ("finalize", "all")
            isdir(export_directory) || error("missing CHH5 export directory; run the integrals stage compute first: $export_directory")
            summary = TOML.parsefile(joinpath(workflow_stage_directory(manifest, root, "commonorb"), "commonorb_summary.toml"))
            frozen = sum(length(parse_frozen_orbital_spec(String(spec))) for spec in summary["frozen"])
            deleted = Int(summary["deleted"])
            isfile(joinpath(directory, "COMMONORB")) ||
                cp(joinpath(workflow_stage_directory(manifest, root, "commonorb"), "gcommon_run", "COMMONORB"), joinpath(directory, "COMMONORB"); force=true)
            if !isfile(one_electron) || !isfile(factor_bundle) || !isfile(joinpath(directory, prefix * ".sys"))
                result = NOCI.import_seward_hdf5(export_directory, joinpath(directory, "COMMONORB"); n_frozen=frozen, n_deleted=deleted)
                NOCI.write_seward_hdf5_products(result;
                    one_path=one_electron, system_path=joinpath(directory, prefix * ".sys"), bundle_path=factor_bundle)

            end
            isfile(one_electron) || error("conversion produced no $one_electron")
            isfile(joinpath(directory, prefix * ".sys")) || error("conversion produced no .sys")
            isfile(factor_bundle) || error("conversion produced no $factor_bundle")
            println("  integrals complete: $one_electron + factor bundle $factor_bundle (frozen=$frozen, deleted=$deleted)")
        end
    finally
        rm(lock_path; recursive=true, force=true)
    end
    return directory
end

"""
    stage_noci(manifest, base, root, fragment_states)

Generate the traceable production input `<project>.yaml` for the AHPC-NOCI run (stage 05, generation only).
Schema 4 references short instance/state files and `integrals.one`/`factors.jld2`, while legacy inputs retain previous paths; settings come from the manifest and the generated YAML is structurally parsed without loading scientific data.
Original files are read-only, thresholds retain declared units, allocation is small CPU text/metadata, and malformed production input raises before execution.
"""
function stage_noci(manifest, base::String, root::String, fragment_states)
    directory = workflow_stage_directory(manifest, root, "run")
    mkpath(directory)
    project = manifest["project"]
    prefix = replace(project, r"[^A-Za-z0-9_.-]" => "_")
    fragments = iterfragment(manifest)
    entries = Dict{String,Any}[]
    for label in fragments, state in fragment_state_labels(fragment_states, label)
        logical = fragment_product_label(manifest, label, state)
        common = joinpath(workflow_stage_directory(manifest, root, "commonorb"), "gcommon_run")
        push!(entries, Dict("label" => logical, "fragment" => label, "state" => state,
            "determinants" => joinpath(common, logical * ".det"), "orbitals" => joinpath(common, logical * ".vec")))
    end
    table = manifest["mebf_table"]
    mebfs = Dict{String,Any}[]
    for column in 2:length(table[1])
        composition = [fragment_product_label(manifest, String(row[1]), String(row[column])) for row in table]
        label = join([String(row[column]) for row in table], standard_case(manifest) ? "_" : "")
        push!(mebfs, Dict("label" => label, "states" => composition))
    end
    integral_directory = workflow_stage_directory(manifest, root, "integrals")
    input = Dict("algorithm" => Dict("name" => "ahpc-noci"),
        "system" => Dict("label" => project, "spin_multiplicity" => manifest["electronic_structure"]["total_spin_multiplicity"],
            "fragment_states" => entries, "mebfs" => mebfs),
        "integrals" => Dict("backend" => "factorized", "one_electron" => joinpath(integral_directory, integral_prefix(manifest) * ".one"),
            "factor_bundle" => joinpath(integral_directory, standard_case(manifest) ? "factors.jld2" : integral_prefix(manifest) * ".two.jld2")),
        "screening" => Dict("tau_ci" => manifest["screening"]["tau_ci"], "tau_ci_offdiagonal" => manifest["screening"]["tau_ci_offdiag"], "tau_singularity" => 1.0e-12),
        "runtime" => Dict("use_gpu" => manifest["runtime"]["use_gpu"], "scratch_bytes" => manifest["runtime"]["scratch_mb"] * 1024^2))
    generated = joinpath(directory, prefix * ".yaml")
    write_atomic(generated, "# Generated from 00; edit the authoritative manifest, not this file.\n" * YAML.write(input))
    NOCI.read_noci_input(generated) # Parse structure only; no scientific input loading.
    println("  generated $generated")
    return generated
end

"""
    stage_run(manifest, base, root, fragment_states, mode)

Generate the NOCI input and execute the production AHPC-NOCI run (05; modes: prepare, compute, finalize, all).
Prepare writes and parses `<project>.yaml`; compute calls the shared NOCI production API through a fresh Julia worker with explicit Julia/BLAS threads and checkpoint settings, and finalize checks completion JSON/progress.
Inputs are read-only, energies use hartree, scientific memory/preflight remains in NOCI, and missing inputs or failed production completion raise.
"""
function stage_run(manifest, base::String, root::String, fragment_states, mode::String="all")
    directory = workflow_stage_directory(manifest, root, "run")
    project = manifest["project"]
    prefix = replace(project, r"[^A-Za-z0-9_.-]" => "_")
    result = joinpath(directory, prefix * ".json")
    progress = joinpath(directory, prefix * ".sec")
    if mode in ("prepare", "all")
        stage_noci(manifest, base, root, fragment_states)
    end
    if mode in ("compute", "all")
        if isfile(result) && isfile(progress) && occursin("production_run_complete", read(progress, String))
            println("  production output already complete; compute skipped: $result")
        else
            generated = joinpath(directory, prefix * ".yaml")
            isfile(generated) || error("missing generated NOCI input; run the run stage prepare first: $generated")
            aggregate = workflow_aggregate(manifest, base, root)
            julia = joinpath(Sys.BINDIR, "julia")
            hint = "--heap-size-hint=$(manifest["runtime"]["heap_size_hint_mb"])M"
            command = `$julia --threads=$(manifest["runtime"]["threads"]) $hint --startup-file=no --project=$(case_root()) $(joinpath(case_root(), "scripts", "preparation", "prepare.jl")) execute $generated $directory $prefix $aggregate $(manifest["runtime"]["blas_threads"]) $(manifest["output"]["checkpoint"])`
            run(command)
        end
    end
    if mode in ("finalize", "all")
        isfile(result) || error("production run is not complete; missing $result")
        isfile(progress) || error("production run is not complete; missing $progress")
        sec_text = read(progress, String)
        occursin("production_run_complete", sec_text) || error("production run did not reach production_run_complete; see $progress")
        ground = collect(eachmatch(r"production_run_complete[^\n]*?ground_energy=(-?[0-9.eE+-]+)", sec_text))
        println("  production complete: $result")
        println("  matrix snapshot: ", joinpath(directory, prefix * ".matrix_snapshot.jld2"))
        isempty(ground) || println("  ground energy from completion event: ", ground[end].captures[1], " Eh")
    end
    return directory
end

const PREPARATION_STAGE_ADAPTERS = Dict(
    ("openmolcas", "openmolcas", method) => Dict(
        "states" => stage_states, "placement" => stage_placement, "commonorb" => stage_commonorb,
        "integrals" => stage_integrals, "run" => stage_run, "noci" => stage_noci)
    for method in ("ricd", "cd")
)

"""
    main(arguments)

Dispatch one driver stage with an optional mode and return nothing; unknown stages or modes raise with the supported lists.
"""
function main(arguments::Vector{String})
    length(arguments) in (2, 3) || error("usage: case_driver.jl STAGE MANIFEST.yaml [MODE] (stages: validate states placement commonorb integrals noci run; modes: $(join(STAGE_MODES, " ")))")
    stage, manifest_path = arguments[1:2]
    if stage in ("check", "plan")
        structure_only = length(arguments) == 3 && arguments[3] == "--structure-only"
        length(arguments) == 2 || structure_only || error("check/plan accepts only --structure-only")
        standard_plan(manifest_path; structure_only)
        return nothing
    end
    if get(YAML.load_file(manifest_path), "schema_version", nothing) == 4
        manifest, base, root, fragment_states = load_case(abspath(manifest_path))
        mode = length(arguments) == 3 ? arguments[3] : "all"
        mode in STAGE_MODES || error("unsupported mode: $mode")
        standard_backend_ready(manifest)
        if stage in ("states", "placement", "commonorb", "integrals")
            NOCI.openmolcas4noci_identity(; required_capabilities=["cas", "commonorb", "rotateorb", "chh5", String(manifest["_standard"]["integrals"]["method"]), String(manifest["_standard"]["cas"]["integral_method"])])
        end
        if stage == "validate"
            standard_plan(manifest_path)
            return nothing
        end
        raw = manifest["_standard"]
        key = (String(raw["cas"]["backend"]), String(raw["integrals"]["backend"]), String(raw["integrals"]["method"]))
        handlers = PREPARATION_STAGE_ADAPTERS[key]
        stage in keys(handlers) || error("unknown stage: $stage")
        prerequisite = Dict("placement" => "states", "commonorb" => "placement", "integrals" => "commonorb", "noci" => "integrals", "run" => "integrals")
        haskey(prerequisite, stage) && standard_require_products(manifest, root, prerequisite[stage])
        standard_guard(manifest, root, stage)
        if stage == "noci"
            stage_noci(manifest, base, root, fragment_states)
        else
            handlers[stage](manifest, base, root, fragment_states, mode)
        end
        mode in ("all", "finalize") && standard_require_products(manifest, root, stage)
        return nothing
    end
    mode = length(arguments) == 3 ? arguments[3] : "all"
    mode in STAGE_MODES || error("unknown mode $mode; supported: $(join(STAGE_MODES, " "))")
    summary = Dict("validate" => () -> (validate_case(abspath(manifest_path)); nothing),
        "states" => () -> stage_states(load_case(abspath(manifest_path))..., mode),
        "placement" => () -> stage_placement(load_case(abspath(manifest_path))..., mode),
        "commonorb" => () -> stage_commonorb(load_case(abspath(manifest_path))..., mode),
        "integrals" => () -> stage_integrals(load_case(abspath(manifest_path))..., mode),
        "noci" => () -> stage_noci(load_case(abspath(manifest_path))...),
        "run" => () -> stage_run(load_case(abspath(manifest_path))..., mode))
    haskey(summary, stage) || error("unknown stage $stage; supported: validate states placement commonorb integrals noci run")
    summary[stage]()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main(ARGS)
    catch failure
        println(stderr, "STAGE FAILED: ", sprint(showerror, failure))
        exit(1)
    end
end
