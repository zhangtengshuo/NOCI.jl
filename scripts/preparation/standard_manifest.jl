# Declarative species/geometry registry for the recommended C1 workflow.
# Numerical preparation remains in case_driver.jl and its shared libraries.

module StandardPlacementGeometry
include("lib/AutoRotate.jl")
end

const STANDARD_BACKENDS = Dict(
    ("openmolcas", "openmolcas", "ricd") => :implemented,
    ("openmolcas", "openmolcas", "cd") => :implemented,
    ("pyscf", "pyscf", "dfcd") => :reserved,
)

"""
    standard_case(manifest)

Return whether the read-only mapping uses the recommended workflow registry.
The dimensionless flag allocates no scientific arrays and initializes no backend.
"""
standard_case(manifest) = haskey(manifest, "_standard")

"""
    standard_positive(value, location; integer=false)

Validate a finite positive dimensionless count or threshold and return its value.
No input is mutated; malformed values raise before backend initialization on CPU.
"""
function standard_positive(value, location; integer=false)
    value isa Real && !(value isa Bool) && isfinite(value) && value > 0 || error("$location must be finite and positive")
    integer && !(value isa Integer) && error("$location must be an integer")
    return value
end

"""
    standard_token(value, location; pattern=ASCII_IDENTIFIER_PATTERN)

Return a safe, nonempty ASCII identifier, rejecting path components and malformed labels.
Inputs are read-only, identifiers are dimensionless, and no backend is initialized.
"""
function standard_token(value, location; pattern=r"^[A-Za-z0-9_.+-]+$")
    value isa AbstractString && occursin(pattern, value) && !(value in (".", "..")) || error("invalid identifier at $location: $value")
    return String(value)
end

"""
    standard_source(path)

Return lightweight absolute-path, byte-size and modification-time identity of one source file.
No content hash is calculated, no input is mutated, and a missing/empty file raises; bytes are counts and modification time is Unix nanoseconds using NOCI's filesystem-stat convention.
"""
function standard_source(path::String)
    isfile(path) && filesize(path) > 0 || error("missing or empty input: $path")
    info = stat(path)
    return Dict("path" => realpath(path), "bytes" => info.size, "mtime_ns" => round(Int64, info.mtime * 1.0e9))
end

"""
    standard_core_orbital_count(elements)

Return the number of doubly occupied all-electron chemical-core spatial orbitals for XYZ element symbols, following the PySCF `chemcore_atm` convention without importing PySCF.
Counts are dimensionless and summed over atoms; input symbols are read-only, storage is small CPU metadata and unknown/ghost symbols raise rather than guessing.
This is an atomic core-count policy, not an orbital-character or active-space analysis; ECP core subtraction is outside the current all-electron contract.
"""
function standard_core_orbital_count(elements)
    symbols = split("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og")
    boundaries = (4, 12, 30, 38, 48, 70, 80, 102, 112, 118)
    counts = (0, 1, 5, 9, 14, 18, 23, 34, 50, 55)
    total = 0
    for element in elements
        z = findfirst(s -> uppercase(s) == uppercase(element), symbols)
        z === nothing && error("unsupported all-electron core element: $element")
        total += counts[findfirst(limit -> z <= limit, boundaries)]
    end
    return total
end

"""
    standard_initial_orbital(carrier, state, location)

Resolve one user seed path from a carrier's state override, falling back to its `default` entry, and return the nonempty path string.
Inputs and user files are read-only, paths have no physical units, and missing or malformed paths raise before task planning; no backend or scientific arrays are initialized.
"""
function standard_initial_orbital(carrier, state, location)
    seeds = carrier["initial_orbitals"]
    path = get(seeds, state, get(seeds, "default", nothing))
    path isa AbstractString && !isempty(strip(path)) || error("missing or invalid user seed $location/$state; supply initial_orbitals.default or a state override")
    return String(path)
end

"""
    standard_template_path(manifest, role)

Resolve a user input template, preferring its explicit YAML path, then the workflow's `templates/<backend>/` file, then the installed default.
Return an absolute existing nonempty path; files are read-only, no shell evaluation or scientific allocation occurs, and an explicit missing override raises instead of silently falling back.
Role names and paths have no physical units; PySCF files are preparation contracts while that adapter is reserved.
"""
function standard_template_path(manifest, role)
    raw = manifest["_standard"]
    base = dirname(manifest["_standard_source"])
    backend = role == "integrals" ? raw["integrals"]["backend"] : raw["cas"]["backend"]
    extension = backend == "pyscf" ? ".py.in" : ".in"
    filename = role == "integrals" ? "integrals_$(raw["integrals"]["method"])$extension" : "$role$extension"
    overrides = get(raw, "templates", Dict())
    if haskey(overrides, role)
        selected = resolve(base, overrides[role])
    else
        relative = joinpath("templates", backend, filename)
        local_path = joinpath(base, relative)
        selected = isfile(local_path) ? local_path : joinpath(@__DIR__, relative)
    end
    isfile(selected) && filesize(selected) > 0 || error("missing or empty $role input template: $selected")
    return abspath(selected)
end

"""
    standard_template_definition(manifest, role)

Return lightweight path/byte/mtime provenance plus the small template text for one input role.
The files and manifest are read-only, no content hash or backend is used, and missing files raise before reuse; memory scales only with input-template length.
"""
function standard_template_definition(manifest, role)
    path = standard_template_path(manifest, role)
    return Dict("source" => standard_source(path), "text" => read(path, String))
end

"""
    standard_render_template(manifest, role, parameters)

Substitute `{{UPPER_CASE_KEY}}` fields in a selected user/default template and return input text, preserving literal OpenMolcas dollar variables.
Values retain the caller's units and quoting; the manifest and template are read-only, substitution does not execute code or allocate scientific arrays, and unknown/malformed placeholders raise.
"""
function standard_render_template(manifest, role, parameters)
    source = read(standard_template_path(manifest, role), String)
    rendered = replace(source, r"\{\{[A-Z][A-Z0-9_]*\}\}" => token -> begin
        key = String(token[3:end-2])
        haskey(parameters, key) || error("unknown placeholder {{$key}} in $role template")
        string(parameters[key])
    end)
    occursin("{{", rendered) && error("unresolved or malformed placeholder in $role template")
    return rendered
end

"""
    standard_openmolcas_parameters(manifest, geometry; label=nothing, integral_stage=false)

Return the small placeholder mapping for C1 OpenMolcas inputs, separating RICD gateway options from molecular-CD SEWARD options.
XYZ paths refer to Angstrom geometry and decomposition thresholds retain their declared semantics; no files are modified and no backend/scientific arrays are initialized.
An optional fragment selects its molecule's point-group declaration, while aggregate/reference work remains C1; missing settings raise during input preparation.
"""
function standard_openmolcas_parameters(manifest, geometry; label=nothing, integral_stage=false)
    raw = manifest["_standard"]
    settings = integral_stage ? raw["integrals"] : raw["cas"]
    method = integral_stage ? settings["method"] : settings["integral_method"]
    threshold = fortran_real(settings["generation_threshold"])
    point_group = label === nothing ? "C1" : raw["molecules"][manifest["_bindings"][label]["molecule"]]["symmetry"]
    return Dict{String,Any}("XYZ" => geometry, "BASIS" => raw["cas"]["basis"], "POINT_GROUP" => point_group == "C1" ? "nosym" : point_group,
        "GATEWAY_INTEGRAL_OPTIONS" => method == "ricd" ? "  RICD\n  CDThreshold\n  $threshold\n" : "",
        "SEWARD_INTEGRAL_OPTIONS" => method == "cd" ? "  Cholesky\n  ThrCholesky\n  $threshold\n" : "",
        "GENERATION_THRESHOLD" => threshold)
end

"""
    standard_cas_input(manifest, task)

Render one user/default OpenMolcas CAS input and its optional single-root CASPT2 block from the normalized task.
Energies/shifts use hartree and orbitals remain user-defined; no input files are mutated, only small CPU strings are allocated and unresolved template fields raise before a calculation.
CASPT2 copies never replace the unperturbed RasOrb/VecDet outputs, and the selected root is retained explicitly.
"""
function standard_cas_input(manifest, task)
    parameters = standard_openmolcas_parameters(manifest, task.geometry; label=task.label)
    state = task.state_data
    for (placeholder, key) in (("CHARGE", "charge"), ("NACTEL", "nactel"), ("INACTIVE", "inactive"), ("RAS2", "ras2"),
            ("STATE_SYMMETRY", "symmetry"), ("MULTIPLICITY", "multiplicity"), ("CIROOT", "ciroot"), ("RLXROOT", "rlxroot"))
        parameters[placeholder] = state[key]
    end
    parameters["INITIAL_ORBITALS"] = task.guess
    parameters["STATE"] = task.state
    binding = manifest["_bindings"][task.label]
    parameters["MOLECULE"] = binding["molecule"]
    parameters["CARRIER_GEOMETRY"] = binding["geometry"]
    parameters["CASPT2_BLOCK"] = ""
    pt2 = manifest["_standard"]["caspt2"]
    if pt2["enabled"]
        parameters["IPEA_SHIFT"] = fortran_real(pt2["ipea_shift"])
        parameters["IMAGINARY_SHIFT"] = fortran_real(pt2["imaginary_shift"])
        parameters["MAX_ITERATIONS"] = pt2["max_iterations"]
        parameters["CASPT2_BLOCK"] = standard_render_template(manifest, "caspt2", parameters)
    end
    return standard_render_template(manifest, "cas", parameters)
end

"""
    standard_read(path; require_files=true)

Validate schema 4 and normalize the species/geometry registry for the shared C1 preparation implementation.
XYZ coordinates use Angstrom and thresholds retain their declared physical meanings; original mappings/files are not mutated and only small CPU planning data are allocated.
Unknown keys, malformed labels, incompatible backend pairs and missing seeds raise; active-orbital character and scientific state correctness are never inferred.
`require_files=false` permits schema review of future PySCF templates without reading XYZ or seed files.
"""
function standard_read(path::String; require_files::Bool=true)
    raw = YAML.load_file(path)
    raw isa AbstractDict || error("manifest must be a mapping")
    schema = Dict{String,Any}(
        "schema_version" => nothing, "project" => nothing, "method" => nothing,
        "geometry" => Dict{String,Any}("aggregate" => nothing),
        "molecules" => nothing, "fragments" => nothing, "mebf_table" => nothing,
        "cas" => Dict{String,Any}("backend" => nothing, "basis" => nothing, "integral_method" => nothing, "generation_threshold" => nothing, "auxiliary_basis" => nothing),
        "caspt2" => Dict{String,Any}("enabled" => nothing, "ipea_shift" => nothing, "imaginary_shift" => nothing, "max_iterations" => nothing),
        "templates" => Dict{String,Any}(k => nothing for k in ("cas", "caspt2", "fragment_reference", "aggregate_reference", "integrals")),
        "common_orbitals" => Dict{String,Any}("threshold" => nothing),
        "integrals" => Dict{String,Any}("backend" => nothing, "method" => nothing, "generation_threshold" => nothing,
            "auxiliary_basis" => nothing, "metric_lindep" => nothing),
        "system" => Dict{String,Any}("charge" => nothing, "spin_multiplicity" => nothing),
        "screening" => Dict{String,Any}("tau_ci" => nothing, "tau_ci_offdiagonal" => nothing),
        "runtime" => Dict{String,Any}("threads" => nothing, "blas_threads" => nothing, "scratch_mb" => nothing, "heap_size_hint_mb" => nothing),
        "output" => Dict{String,Any}("checkpoint" => nothing))
    check_allowed_keys(raw, schema, "manifest")
    raw["schema_version"] == 4 || error("recommended workflow requires schema_version: 4")
    project = standard_token(raw["project"], "project")
    raw["method"] == "ahpc_noci" || error("method must be ahpc_noci")
    cas = Dict{String,Any}(raw["cas"])
    integrals = Dict{String,Any}(raw["integrals"])
    pair = (String(cas["backend"]), String(integrals["backend"]), String(integrals["method"]))
    haskey(STANDARD_BACKENDS, pair) || error("unsupported backend combination $pair; mixed CAS/integral backends are not implemented in this edition")
    cas["basis"] isa AbstractString && !isempty(strip(cas["basis"])) || error("cas.basis must be an explicit basis")
    if pair[1] == "openmolcas"
        cas["integral_method"] in ("ricd", "cd") || error("cas.integral_method must be ricd or cd")
        standard_positive(cas["generation_threshold"], "cas.generation_threshold")
        haskey(cas, "auxiliary_basis") && error("cas.auxiliary_basis belongs to the PySCF DF adapter; OpenMolcas RICD uses generated atomic auxiliary functions")
    else
        any(haskey(cas, k) for k in ("integral_method", "generation_threshold")) && error("PySCF CAS template does not use OpenMolcas integral settings")
    end
    if pair[3] == "dfcd"
        auxiliary = get(integrals, "auxiliary_basis", get(cas, "auxiliary_basis", nothing))
        auxiliary isa AbstractString && !isempty(strip(auxiliary)) || error("PySCF DF requires an explicit auxiliary_basis in cas or integrals")
        integrals["auxiliary_basis"] = auxiliary
        cas["auxiliary_basis"] = get(cas, "auxiliary_basis", auxiliary)
        cas["auxiliary_basis"] isa AbstractString && !isempty(strip(cas["auxiliary_basis"])) || error("cas.auxiliary_basis must be nonempty")
        integrals["auxiliary_basis"] isa AbstractString && !isempty(strip(integrals["auxiliary_basis"])) || error("DFCD requires integrals.auxiliary_basis")
        standard_positive(integrals["metric_lindep"], "integrals.metric_lindep")
        haskey(integrals, "generation_threshold") && error("DFCD uses metric_lindep, not a molecular/atomic CD threshold")
    else
        standard_positive(integrals["generation_threshold"], "integrals.generation_threshold")
        any(haskey(integrals, key) for key in ("auxiliary_basis", "metric_lindep")) && error("specified auxiliary basis and metric_lindep belong to PySCF DFCD only")
    end
    raw["cas"] = cas
    raw["integrals"] = integrals
    pt2 = merge(Dict{String,Any}("enabled" => true, "ipea_shift" => 0.0, "imaginary_shift" => 0.2, "max_iterations" => 80), get(raw, "caspt2", Dict()))
    pt2["enabled"] isa Bool || error("caspt2.enabled must be boolean")
    for key in ("ipea_shift", "imaginary_shift")
        value = pt2[key]
        value isa Real && !(value isa Bool) && isfinite(value) && value >= 0 || error("caspt2.$key must be finite and nonnegative (hartree)")
    end
    standard_positive(pt2["max_iterations"], "caspt2.max_iterations"; integer=true)
    raw["caspt2"] = pt2
    for (role, template) in get(raw, "templates", Dict())
        template isa AbstractString && !isempty(strip(template)) || error("templates.$role must be a nonempty path")
    end
    common_threshold = get(get(raw, "common_orbitals", Dict()), "threshold", 1.0e-5)
    standard_positive(common_threshold, "common_orbitals.threshold")
    for key in ("threads", "blas_threads", "scratch_mb", "heap_size_hint_mb")
        standard_positive(raw["runtime"][key], "runtime.$key"; integer=true)
    end
    for key in ("tau_ci", "tau_ci_offdiagonal")
        v = raw["screening"][key]
        v isa Real && !(v isa Bool) && isfinite(v) && 0 <= v < 1 || error("screening.$key must be in [0,1)")
    end
    raw["output"]["checkpoint"] isa Bool || error("output.checkpoint must be boolean")
    standard_positive(raw["system"]["spin_multiplicity"], "system.spin_multiplicity"; integer=true)
    raw["system"]["charge"] isa Integer && !(raw["system"]["charge"] isa Bool) || error("system.charge must be integer")
    table = raw["mebf_table"]
    table isa AbstractVector && !isempty(table) && all(row -> row isa AbstractVector && length(row) >= 2, table) || error("mebf_table must have nonempty rows")
    widths = unique(length.(table))
    length(widths) == 1 || error("mebf_table row lengths differ")
    labels = [standard_token(row[1], "mebf_table fragment"; pattern=r"^[A-Za-z0-9]+_[A-Za-z0-9]+$") for row in table]
    length(unique(labels)) == length(labels) || error("duplicate fragment label")
    fragments = raw["fragments"]
    Set(keys(fragments)) == Set(labels) || error("fragments and mebf_table must declare the same instances")
    base = dirname(abspath(path))
    resolve_input = x -> isabspath(String(x)) ? String(x) : abspath(joinpath(base, String(x)))
    geometries = Dict{String,Any}()
    states = Dict{String,Any}()
    initials = Dict{String,Any}()
    bindings = Dict{String,Any}()
    molecules = raw["molecules"]
    molecules isa AbstractDict && !isempty(molecules) || error("molecules must be a registry")
    for (species, molecule) in molecules
        standard_token(species, "molecules species"; pattern=r"^[A-Za-z0-9]+$")
        check_allowed_keys(molecule, Dict{String,Any}("symmetry" => nothing, "geometries" => nothing, "states" => nothing), "molecules.$species")
        standard_token(molecule["symmetry"], "molecules.$species.symmetry"; pattern=r"^[A-Za-z][A-Za-z0-9]*$")
        require_files && molecule["symmetry"] != "C1" && error("molecules.$species.symmetry=$(molecule["symmetry"]) is a future interface; this edition executes only C1")
        for (geometry, carrier) in molecule["geometries"]
            standard_token(geometry, "molecules.$species.geometries")
            check_allowed_keys(carrier, Dict{String,Any}("xyz" => nothing, "initial_orbitals" => nothing), "molecules.$species.geometries.$geometry")
            carrier["initial_orbitals"] isa AbstractDict || error("initial_orbitals must be a default/state mapping at $species/$geometry")
            check_allowed_keys(carrier["initial_orbitals"], Dict{String,Any}(k => nothing for k in union(["default"], String.(collect(keys(molecule["states"]))))), "molecules.$species.geometries.$geometry.initial_orbitals")
            for seed in values(carrier["initial_orbitals"])
                seed isa AbstractString && !isempty(strip(seed)) || error("initial_orbitals paths must be nonempty strings at $species/$geometry")
            end
        end
    end
    for (i, label) in enumerate(labels)
        binding = fragments[label]
        check_allowed_keys(binding, Dict{String,Any}("molecule" => nothing, "geometry" => nothing, "frozen_orbitals" => nothing), "fragments.$label")
        species = String(binding["molecule"])
        geometry = String(binding["geometry"])
        haskey(molecules, species) || error("undefined molecule $species")
        molecule = molecules[species]
        haskey(molecule["geometries"], geometry) || error("undefined geometry $species/$geometry")
        carrier = molecule["geometries"][geometry]
        geo_path = resolve_input(carrier["xyz"])
        geometries[label] = geo_path
        wanted = unique(String.(table[i][2:end]))
        selected = Dict{String,Any}()
        seeds = Dict{String,Any}()
        for state in wanted
            standard_token(state, "state label")
            haskey(molecule["states"], state) || error("undefined state $species/$state")
            data = molecule["states"][state]
            if pair[1] == "openmolcas"
                allowed = Dict{String,Any}(k => nothing for k in ("charge", "multiplicity", "nactel", "inactive", "ras2", "symmetry", "ciroot", "rlxroot"))
                check_allowed_keys(data, allowed, "molecules.$species.states.$state")
                for key in ("charge", "multiplicity", "nactel", "inactive", "ras2", "symmetry", "rlxroot")
                    data[key] isa Integer && !(data[key] isa Bool) || error("$species/$state.$key must be integer")
                end
                molecule["symmetry"] == "C1" && data["symmetry"] != 1 && error("C1 state symmetry must be 1")
                data["inactive"] >= 0 && data["nactel"] >= 0 || error("negative orbital/electron count")
                standard_positive(data["ras2"], "ras2"; integer=true)
                standard_positive(data["multiplicity"], "multiplicity"; integer=true)
                standard_positive(data["rlxroot"], "rlxroot"; integer=true)
                data["ciroot"] isa AbstractString && !isempty(strip(data["ciroot"])) || error("ciroot must be an explicit string")
                selected[state] = data
            else
                allowed = Dict{String,Any}(k => nothing for k in ("charge", "spin_2S", "nelecas", "ncore", "ncas", "root"))
                check_allowed_keys(data, allowed, "molecules.$species.states.$state")
                for key in ("charge", "spin_2S", "ncore", "ncas", "root")
                    data[key] isa Integer && !(data[key] isa Bool) || error("$species/$state.$key must be integer")
                end
                data["nelecas"] isa AbstractVector && length(data["nelecas"]) == 2 && all(x -> x isa Integer && x >= 0, data["nelecas"]) || error("nelecas must be [n_alpha,n_beta]")
                data["spin_2S"] >= 0 && data["ncore"] >= 0 || error("negative spin/core count")
                standard_positive(data["ncas"], "ncas"; integer=true)
                standard_positive(data["root"], "root"; integer=true)
                selected[state] = Dict("charge" => data["charge"], "multiplicity" => data["spin_2S"] + 1,
                    "nactel" => sum(data["nelecas"]), "inactive" => data["ncore"], "ras2" => data["ncas"], "symmetry" => 1,
                    "ciroot" => "reserved", "rlxroot" => data["root"], "pyscf" => data)
            end
            seeds[state] = resolve_input(standard_initial_orbital(carrier, state, "$species/$geometry"))
            require_files && standard_source(seeds[state])
        end
        frozen = get(binding, "frozen_orbitals", false)
        frozen isa Bool || error("fragments.$label.frozen_orbitals must be true or false")
        n_core = frozen ? (require_files ? standard_core_orbital_count(load_xyz(geo_path)[2]) : -1) : 0
        states[label] = selected
        initials[label] = seeds
        bindings[label] = Dict("molecule" => species, "geometry" => geometry, "frozen_orbitals" => frozen, "frozen_count" => n_core)
    end
    for column in 2:first(widths)
        composition = [String(row[column]) for row in table]
        sum(states[label][composition[i]]["charge"] for (i, label) in enumerate(labels)) == raw["system"]["charge"] || error("MEBF column $column charge differs from system.charge")
        spins_can_couple([states[label][composition[i]]["multiplicity"] for (i,label) in enumerate(labels)], Int(raw["system"]["spin_multiplicity"])) || error("MEBF column $column cannot couple to system.spin_multiplicity")
    end
    compositions = [Tuple(row[column] for row in table) for column in 2:first(widths)]
    length(unique(compositions)) == length(compositions) || error("duplicate MEBF columns")
    aggregate = resolve_input(raw["geometry"]["aggregate"])
    if require_files
        _, aggregate_elements, aggregate_coordinates = load_xyz(aggregate)
        all(x -> all(isfinite, x), aggregate_coordinates) || error("nonfinite aggregate coordinates")
        sequence = String[]
        for label in labels
            _, elements, coordinates = load_xyz(geometries[label])
            all(x -> all(isfinite, x), coordinates) || error("nonfinite intrinsic geometry")
            append!(sequence, elements)
        end
        sequence == aggregate_elements || error("aggregate XYZ must contain contiguous fragment blocks in mebf_table order, preserving each intrinsic XYZ atom order")
    end
    manifest = Dict{String,Any}("project" => project, "_standard" => raw, "_standard_source" => abspath(path), "_bindings" => bindings,
        "geometry" => Dict("aggregate" => aggregate, "fragments" => geometries),
        "electronic_structure" => Dict("basis" => cas["basis"], "group" => "nosym", "total_spin_multiplicity" => raw["system"]["spin_multiplicity"], "states" => states, "initial_orbitals" => initials),
        "mebf_table" => table, "common_orbitals" => Dict("threshold" => common_threshold, "freeze_core" => any(b["frozen_orbitals"] for b in values(bindings))),
        "integrals" => Dict("method" => pair[3] == "cd" ? "conventional_cd" : pair[3], "generation_threshold" => get(integrals, "generation_threshold", 0.0), "export" => "chh5"),
        "screening" => Dict("tau_ci" => raw["screening"]["tau_ci"], "tau_ci_offdiag" => raw["screening"]["tau_ci_offdiagonal"]),
        "runtime" => merge(Dict("use_gpu" => false), raw["runtime"]), "output" => raw["output"])
    roles = pair[1] == "openmolcas" ? ["cas", "fragment_reference", "aggregate_reference", "integrals"] : ["cas", "integrals"]
    pair[1] == "openmolcas" && pt2["enabled"] && push!(roles, "caspt2")
    for role in roles
        standard_template_path(manifest, role)
    end
    return manifest
end

"""
    standard_backend_ready(manifest)

Reject a reserved backend before writing task directories or launching scientific jobs.
The read-only registry is dimensionless and allocates no scientific arrays or devices.
"""
function standard_backend_ready(manifest)
    raw = manifest["_standard"]
    all(m["symmetry"] == "C1" for m in values(raw["molecules"])) || error("only per-molecule C1 symmetry is implemented")
    raw["caspt2"]["enabled"] && raw["cas"]["backend"] != "openmolcas" && error("CASPT2 execution requires the OpenMolcas CAS backend; the PySCF CASPT2 adapter is not implemented")
    key = (String(raw["cas"]["backend"]), String(raw["integrals"]["backend"]), String(raw["integrals"]["method"]))
    STANDARD_BACKENDS[key] == :implemented || error("PySCF DFCD is a reserved adapter contract, not an executable backend in this edition; use check/plan --structure-only for its template")
    return nothing
end

"""
    workflow_stage_folder(manifest, stage)

Return the standard or legacy folder name for one preparation responsibility.
Identifiers have no physical units, inputs are read-only, no backend/storage is initialized and unknown stages raise KeyError.
"""
function workflow_stage_folder(manifest, stage)
    folders = Dict("states" => "01_fragment_states", "placement" => "02_fragment_placement", "commonorb" => "03_common_orbitals", "integrals" => "04_integrals", "noci" => "05_noci", "run" => "05_noci")
    return folders[stage]
end

"""
    workflow_stage_directory(manifest, root, stage)

Return the stage directory without creating it; stage 01 holds shared intrinsic tasks and other stages are isolated by project.
Paths have no physical units, inputs are read-only, no scientific storage is allocated and unknown stages raise before backend work.
"""
function workflow_stage_directory(manifest, root, stage)
    folder = workflow_stage_folder(manifest, stage)
    return stage == "states" ? joinpath(root, folder) : joinpath(root, folder, manifest["project"])
end

"""
    workflow_stage_code(manifest, stage)

Return the stage prefix for progress and errors, matching the selected standard or legacy directory layout.
The dimensionless identifier changes no input, initializes no backend and allocates only small CPU text.
"""
workflow_stage_code(manifest, stage) = first(split(workflow_stage_folder(manifest, stage), '_'))

"""
    state_product_directory(manifest, root)

Return the per-case binding directory for recommended manifests and the legacy flat directory otherwise.
Paths have no units, inputs are read-only, and this helper performs no I/O or backend work.
"""
state_product_directory(manifest, root) = standard_case(manifest) ? joinpath(workflow_stage_directory(manifest, root, "states"), "bindings", manifest["project"]) : workflow_stage_directory(manifest, root, "states")

"""
    fragment_product_label(manifest, label, state)

Return short instance/state output names for schema 4, retaining old filenames for existing cases.
Inputs are not mutated and no scientific storage or backend is initialized.
"""
fragment_product_label(manifest, label, state) = standard_case(manifest) ? "$(label)_$(state)" : "$(manifest["project"])$(label)_$(state)"

"""
    integral_prefix(manifest)

Return the stable integral output prefix, independent of backend and case in schema 4.
This dimensionless naming operation changes no files or scientific input.
"""
integral_prefix(manifest) = standard_case(manifest) ? "integrals" : replace(manifest["project"], r"[^A-Za-z0-9_.-]" => "_")

"""
    workflow_gateway_header(manifest, geometry, basis)

Render monomer/C1-reference gateways with explicit CAS RICD/CD origin and threshold semantics.
Geometry uses Angstrom and CD thresholds hartree; the read-only inputs are not interpreted as active-space recommendations and only a small CPU string is allocated.
"""
function workflow_gateway_header(manifest, geometry, basis)
    standard_case(manifest) || return gateway_header(geometry, basis)
    cas = manifest["_standard"]["cas"]
    ricd = cas["integral_method"] == "ricd"
    return gateway_header(geometry, basis; ricd, ricd_threshold=ricd ? cas["generation_threshold"] : nothing)
end

"""
    workflow_aggregate(manifest, base, root)

Return the assembled rotated intrinsic-geometry XYZ for schema 4 or the legacy aggregate source.
No file is changed; a missing placed aggregate raises before CommonOrb/integral preparation.
"""
function workflow_aggregate(manifest, base, root)
    standard_case(manifest) || return resolve(base, manifest["geometry"]["aggregate"])
    path = joinpath(workflow_stage_directory(manifest, root, "placement"), "aggregate.xyz")
    isfile(path) || error("missing placed aggregate $path; complete stage placement first")
    return path
end

"""
    standard_task_definition(manifest, label, state)

Return explicit monomer task identity: species, carrier, parsed XYZ, state/basis/backend and user seed metadata.
Coordinates use Angstrom, source sizes use bytes and times Unix nanoseconds; no hashes, active-space interpretation, mutations or device allocations occur.
Missing source data raises before a cached calculation can be reused.
"""
function standard_task_definition(manifest, label, state)
    binding = manifest["_bindings"][label]
    _, elements, coordinates = load_xyz(manifest["geometry"]["fragments"][label])
    templates = Dict("cas" => standard_template_definition(manifest, "cas"))
    pt2 = manifest["_standard"]["caspt2"]
    pt2["enabled"] && manifest["_standard"]["cas"]["backend"] == "openmolcas" && (templates["caspt2"] = standard_template_definition(manifest, "caspt2"))
    return Dict{String,Any}("molecule" => binding["molecule"], "geometry" => binding["geometry"], "state" => state,
        "symmetry" => manifest["_standard"]["molecules"][binding["molecule"]]["symmetry"],
        "elements" => elements, "coordinates_angstrom" => [collect(c) for c in coordinates],
        "cas" => manifest["_standard"]["cas"], "state_definition" => manifest["electronic_structure"]["states"][label][state],
        "seed" => standard_source(manifest["electronic_structure"]["initial_orbitals"][label][state]),
        "caspt2" => pt2["enabled"] ? pt2 : Dict("enabled" => false), "input_templates" => templates, "template" => "standard_user_templates_1", "openmolcas" => openmolcas_identity())
end

"""
    standard_identity(definition)

Return a deterministic readable TOML serialization for equality comparison of small task definitions.
No hash is produced, inputs are read-only, and storage scales with metadata rather than orbital/factor payloads.
"""
function standard_identity(definition)
    buffer = IOBuffer()
    TOML.print(buffer, definition; sorted=true)
    return String(take!(buffer))
end

"""
    standard_placement(manifest, root)

Fit corresponding atom positions in declared order and rigidly rotate each intrinsic XYZ into its aggregate pose.
CPU Kabsch uses Angstrom coordinates and a proper rotation; no orbitals, active spaces or input files are changed.
The task directory receives instance-labelled matrices/XYZ and a merged aggregate in the same order; malformed correspondences raise and memory is linear in atom count.
"""
function standard_placement(manifest, root)
    directory = workflow_stage_directory(manifest, root, "placement")
    aligned = joinpath(directory, "aligned")
    mkpath(aligned)
    labels = iterfragment(manifest)
    cluster = StandardPlacementGeometry.read_xyz(manifest["geometry"]["aggregate"])
    monomers = [StandardPlacementGeometry.read_xyz(manifest["geometry"]["fragments"][label]) for label in labels]
    blocks = StandardPlacementGeometry.split_cluster(cluster, monomers)
    merged = StandardPlacementGeometry.Atom[]
    for (i, label) in enumerate(labels)
        # Preserve the user's AO-centre order; no implicit atom permutation.
        P = permutedims(reduce(hcat, [a.coord for a in monomers[i]]))
        Q = permutedims(reduce(hcat, [a.coord for a in blocks[i]]))
        R, t = StandardPlacementGeometry.kabsch(P, Q)
        placed = [StandardPlacementGeometry.transform_atom(a, R, t) for a in monomers[i]]
        StandardPlacementGeometry.write_matrix(joinpath(aligned, "matrix_$(label).txt"), R, t, label)
        StandardPlacementGeometry.write_xyz(joinpath(directory, "placed_$(label).xyz"), placed, "rotated intrinsic geometry $label; Angstrom")
        append!(merged, placed)
        println("  $label pose-fit RMSD (Angstrom): ", StandardPlacementGeometry.rmsd_heavy(P, Q, R, t))
    end
    StandardPlacementGeometry.write_xyz(joinpath(directory, "aggregate.xyz"), merged, "placed intrinsic geometries; mebf_table order; Angstrom")
    return nothing
end

"""
    standard_stage_definition(manifest, stage)

Return small explicit input metadata for safe case-stage reuse without content hashing.
XYZ uses Angstrom, files use path/bytes/mtime, and the read-only definition excludes MEBF combinations for shared placement/CommonOrb/integral data.
Production input records include screening/runtime/output choices and MEBF columns so stale completed NOCI results cannot be skipped.
"""
function standard_stage_definition(manifest, stage)
    labels = iterfragment(manifest)
    _, elements, coordinates = load_xyz(manifest["geometry"]["aggregate"])
    definition = Dict{String,Any}("schema_version" => 4, "implementation" => "c1_independent_projects_1", "cas" => manifest["_standard"]["cas"],
        "aggregate_elements" => elements, "aggregate_coordinates_angstrom" => [collect(c) for c in coordinates],
        "fragments" => [Dict("label" => label, "binding" => manifest["_bindings"][label],
            "states" => Dict(state => standard_task_definition(manifest, label, state) for state in fragment_state_labels(manifest["electronic_structure"]["states"], label))) for label in labels])
    if stage in ("commonorb", "integrals", "run", "noci")
        definition["common_threshold"] = manifest["common_orbitals"]["threshold"]
    end
    role = stage == "placement" ? "fragment_reference" : stage == "commonorb" ? "aggregate_reference" : stage == "integrals" ? "integrals" : nothing
    role === nothing || (definition["input_template"] = standard_template_definition(manifest, role))
    if stage in ("integrals", "run", "noci")
        definition["integrals"] = manifest["_standard"]["integrals"]
        root = dirname(dirname(manifest["_standard_source"]))
        common_record = joinpath(workflow_stage_directory(manifest, root, "commonorb"), "definition.toml")
        isfile(common_record) || error("complete CommonOrb before preparing $stage: $common_record")
        definition["common_basis"] = TOML.parsefile(common_record)
    end
    if stage in ("run", "noci")
        for key in ("mebf_table", "screening", "runtime", "output")
            definition[key] = manifest[key]
        end
        definition["spin_multiplicity"] = manifest["electronic_structure"]["total_spin_multiplicity"]
    end
    return definition
end

"""
    standard_guard(manifest, root, stage)

Check or atomically record the current project's stage definition before reusing its own products.
Projects share only intrinsic tasks in stage 01; no donor project is read or copied.
Inputs are read-only, XYZ uses Angstrom and source identities path/bytes/nanosecond mtime, no hashes or scientific allocations occur, and changed or unrecorded stage definitions raise.
Return nothing after recording or verifying small CPU metadata.
"""
function standard_guard(manifest, root, stage)
    stage in ("states", "validate") && return nothing
    directory = workflow_stage_directory(manifest, root, stage)
    record = joinpath(directory, "definition.toml")
    requested = standard_stage_definition(manifest, stage)
    if isfile(record)
        TOML.parsefile(record) == requested || error("inputs changed for $directory; choose a new project label or explicitly remove this task-owned stage directory and its dependent stages")
    else
        isdir(directory) && !isempty(readdir(directory)) && error("unrecorded stage directory exists: $directory")
        write_record(record, requested)
    end
    return nothing
end

"""
    standard_require_products(manifest, root, stage)

Check the minimal completed output closure of one recommended stage without interpreting orbital/active-space character or hashing files.
Inputs are read-only, lengths use bytes, no numerical backend/device is initialized, and missing/empty products raise before downstream use.
"""
function standard_require_products(manifest, root, stage)
    project = manifest["project"]
    states = manifest["electronic_structure"]["states"]
    files = String[]
    if stage == "states"
        for label in iterfragment(manifest), state in fragment_state_labels(states, label), suffix in ("RasOrb", "VecDet", "out")
            push!(files, joinpath(state_product_directory(manifest, root), "$(label)_$(state).$suffix"))
        end
    elseif stage == "placement"
        directory = workflow_stage_directory(manifest, root, "placement")
        push!(files, joinpath(directory, "aggregate.xyz"))
        for label in iterfragment(manifest)
            append!(files, [joinpath(directory, "$(label)_c1.RunFile"), joinpath(directory, "$(label)_c1.OneInt")])
            for state in fragment_state_labels(states, label)
                push!(files, joinpath(directory, "rotated", "$(label)_$(state).orb"))
            end
        end
    elseif stage == "commonorb"
        directory = workflow_stage_directory(manifest, root, "commonorb")
        append!(files, [joinpath(directory, "commonorb_summary.toml"), joinpath(directory, "gcommon_run", "COMMONORB")])
        for label in iterfragment(manifest), state in fragment_state_labels(states, label), suffix in ("vec", "det")
            push!(files, joinpath(directory, "gcommon_run", "$(label)_$(state).$suffix"))
        end
    elseif stage == "integrals"
        directory = workflow_stage_directory(manifest, root, "integrals")
        append!(files, [joinpath(directory, "integrals.one"), joinpath(directory, "integrals.sys"), joinpath(directory, "factors.jld2")])
    end
    for file in files
        isfile(file) && filesize(file) > 0 || error("missing/empty $stage product: $file")
    end
    return nothing
end

"""
    standard_plan(path; structure_only=false)

Validate and print the ordered fragment bindings, unique intrinsic tasks and backend capabilities without running calculations or writing files.
Coordinates use Angstrom if inspected; only small CPU planning allocations occur, missing inputs/schema errors raise, and reserved adapters are labelled explicitly.
"""
function standard_plan(path; structure_only=false)
    manifest = standard_read(abspath(path); require_files=!structure_only)
    raw = manifest["_standard"]
    if !structure_only && raw["cas"]["backend"] == "openmolcas"
        NOCI.openmolcas4noci_identity(; required_capabilities=["cas", "commonorb", "rotateorb", "chh5", String(raw["integrals"]["method"]), String(raw["cas"]["integral_method"])])
    end
    println("NOCI.jl ", NOCI.noci_version(), " | CPU")
    println("project: ", manifest["project"], " | ahpc_noci | CPU")
    println("molecule symmetries: ", join(["$key=$(raw["molecules"][key]["symmetry"])" for key in sort!(collect(keys(raw["molecules"])))], ", "))
    pair = (String(raw["cas"]["backend"]), String(raw["integrals"]["backend"]), String(raw["integrals"]["method"]))
    println("CAS / integrals: ", join(pair, " / "), " | ", STANDARD_BACKENDS[pair])
    pair[1] == "pyscf" && println("CAS / integral auxiliary bases: ", raw["cas"]["auxiliary_basis"], " / ", raw["integrals"]["auxiliary_basis"])
    println("CASPT2: ", raw["caspt2"]["enabled"] ? "enabled" : "disabled")
    seen = Set{Tuple{String,String,String}}()
    pairs = 0
    for label in iterfragment(manifest)
        b = manifest["_bindings"][label]
        wanted = fragment_state_labels(manifest["electronic_structure"]["states"], label)
        println("  ", label, " -> ", b["molecule"], "/", b["geometry"], " : ", join(wanted, ", "))
        for state in wanted
            push!(seen, (b["molecule"], b["geometry"], state))
            pairs += 1
        end
    end
    println("intrinsic CAS tasks: ", length(seen), "; instance/state bindings: ", pairs, "; MEBFs: ", length(raw["mebf_table"][1]) - 1)
    for task in sort!(collect(seen))
        println("  01_fragment_states/", join(task, "/"))
    end
    println("stages: 00 define -> 01 states -> 02 placement -> 03 commonorb -> 04 integrals -> 05 run")
    println("phases: prepare (pre-run) -> compute (run) -> finalize (postprocess)")
    println("No active-space validation or content hashing; no calculation was launched.")
    return manifest
end
