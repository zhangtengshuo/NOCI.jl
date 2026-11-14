# File: scripts/preparation/validate_case.jl
# Purpose: Validate schema-4 manifests and report the planned preparation tasks.
# SPDX-License-Identifier: Apache-2.0

# validate_case.jl — authoritative AHPC-NOCI case manifest validation and planning report.
#
# Usage: julia --project <NOCI> scripts/preparation/validate_case.jl MANIFEST.yaml
# The script is read-only for the manifest, geometry and initial-orbital files;
# it writes nothing, exits 0 on success and 1 with a clear VALIDATION FAILED
# message on any schema, physics or consistency error.

using YAML
using SHA
using LinearAlgebra

"""
    load_xyz(path)

Read a Cartesian XYZ file and return `(comment, elements, coordinates)` with coordinates in Angstrom as a `Vector{NTuple{3,Float64}}`.
The file is read-only, the atom count line is validated against the parsed body, and malformed files raise before any scientific use.
"""
function load_xyz(path::String)
    lines = readlines(path)
    length(lines) >= 2 || error("XYZ file too short: $path")
    n_atoms = parse(Int, strip(lines[1]))
    elements = String[]
    coordinates = NTuple{3,Float64}[]
    for line in lines[3:end]
        fields = split(strip(line))
        isempty(fields) && continue
        length(fields) == 4 || error("malformed XYZ line in $path: $line")
        push!(elements, String(fields[1]))
        push!(coordinates, (parse(Float64, fields[2]), parse(Float64, fields[3]), parse(Float64, fields[4])))
    end
    length(elements) == n_atoms || error("XYZ atom count mismatch in $path: header $n_atoms, body $(length(elements))")
    return lines[2], elements, coordinates
end

"""
    nuclear_charge(element)

Return the nuclear charge for a supported element symbol.
Only the elements used by standard demonstration cases are tabulated; unknown symbols raise instead of guessing.
"""
function nuclear_charge(element::AbstractString)
    charges = Dict("H" => 1, "C" => 6, "N" => 7, "O" => 8, "S" => 16)
    haskey(charges, element) || error("unsupported element symbol: $element")
    return charges[element]
end

"""
    core_orbital_count(elements)

Return the number of 1s core orbitals of an atom collection (one per non-hydrogen atom).
"""
core_orbital_count(elements) = count(e -> nuclear_charge(e) >= 3, elements)

"""
    spherical_basis_functions(basis, element)

Return the number of spherical Gaussian basis functions for a tabulated basis/element pair.
Only `CC-PVDZ` is tabulated for the standard ethene demonstration; other combinations raise so estimates are never fabricated.
"""
function spherical_basis_functions(basis::AbstractString, element::AbstractString)
    upper = uppercase(basis)
    upper == "CC-PVDZ" || error("no AO-count table for basis $basis (tabulated: CC-PVDZ)")
    counts = Dict("H" => 5, "C" => 14, "N" => 14, "O" => 14, "S" => 18)
    haskey(counts, element) || error("no CC-PVDZ function count for element $element")
    return counts[element]
end

"""
    check_allowed_keys(value, allowed, location)

Recursively reject manifest keys outside the allowed schema tree.
`allowed` maps a key name to `nothing` (leaf) or a nested specification; extra or misplaced keys raise with their full location.
"""
function check_allowed_keys(value::AbstractDict, allowed::Dict{String,Any}, location::String)
    for key in keys(value)
        haskey(allowed, string(key)) || error("unknown key at $location: $key")
        nested = allowed[string(key)]
        if nested !== nothing
            child = value[key]
            child isa AbstractDict || error("key at $location.$key must be a mapping")
            check_allowed_keys(child, nested, "$location.$key")
        end
    end
    return nothing
end

"""
    spins_can_couple(multiplicities, total_multiplicity)

Check whether a set of fragment spin multiplicities can couple to the requested total multiplicity.
Angular-momentum addition ranges are applied successively over half-integer spins; the check is exact and fails closed for unreachable total spins.
"""
function spins_can_couple(multiplicities, total_multiplicity::Int)
    reachable = Set{Float64}([(multiplicities[1] - 1) / 2])
    for multiplicity in multiplicities[2:end]
        s = (multiplicity - 1) / 2
        next = Set{Float64}()
        for j in reachable
            low = abs(j - s)
            for step in 0:Int(round(j + s - low))
                push!(next, low + step)
            end
        end
        reachable = next
    end
    return ((total_multiplicity - 1) / 2) in reachable
end

"""
    parse_frozen_orbital_spec(token)

Parse one per-fragment frozen-orbital specification written to the CommonOrb control and summary records.
`0` selects no frozen orbital, `1-8` freezes orbitals 1 through 8, and `1-3,6-7` freezes explicit blocks (at most eight, matching the largest supported point group); orbitals are numbered from 1 and blocks must be strictly ascending and non-overlapping.
Returns the explicit frozen-orbital numbers; malformed tokens raise with the offending specification.
The same parser lives in `CommonOrbBase.jl` for the control-file reader; keep the two copies identical.
"""
function parse_frozen_orbital_spec(token::AbstractString)::Vector{Int}
    orbitals = Int[]
    token == "0" && return orbitals
    blocks = split(token, ','; keepempty=false)
    length(blocks) <= 8 || error("frozen specification has more than 8 blocks: $token")
    previous = 0
    for block in blocks
        m = match(r"^([0-9]+)(?:-([0-9]+))?$", strip(block))
        m === nothing && error("malformed frozen specification: $token")
        first_orbital = parse(Int, m.captures[1])
        last_orbital = m.captures[2] === nothing ? first_orbital : parse(Int, m.captures[2])
        first_orbital >= 1 || error("frozen orbital numbering starts at 1: $token")
        first_orbital <= last_orbital || error("descending frozen block in $token")
        first_orbital > previous || error("overlapping or unsorted frozen blocks in $token")
        append!(orbitals, first_orbital:last_orbital)
        previous = last_orbital
    end
    return orbitals
end

"""
    expand_fragment_states(manifest, fragments, fragment_geometries)

Return the per-fragment state tables of the manifest as `label => state table`.
With `identical_fragments: true` the single `states` table is validated to apply to fragments that share one intrinsic geometry digest and is shared by every fragment; otherwise `states` must define one table per fragment label so different molecules keep independent state definitions.
"""
function expand_fragment_states(manifest, fragments, fragment_geometries)
    electronics = manifest["electronic_structure"]
    identical_fragments = get(electronics, "identical_fragments", false)
    identical_fragments isa Bool || error("electronic_structure.identical_fragments must be true or false")
    states = electronics["states"]
    states isa AbstractDict && !isempty(states) || error("states must be a non-empty mapping")
    if identical_fragments
        digests = unique([bytes2hex(open(sha256, fragment_geometries[label])) for label in fragments])
        length(digests) == 1 || error("identical_fragments=true requires every fragment to share one intrinsic geometry; write per-fragment states tables instead")
        return Dict(String(label) => states for label in fragments)
    end
    Set(string.(keys(states))) == Set(fragments) ||
        error("states must define one state table per fragment (found $(join(sort(collect(string.(keys(states)))), ",")), need $(join(fragments, ","))); set identical_fragments=true only when one table applies to every fragment")
    tables = Dict{String,Any}()
    for (name, table) in states
        table isa AbstractDict && !isempty(table) || error("states.$(name) must be a non-empty state mapping")
        tables[String(name)] = table
    end
    return tables
end

"""
    validate_case(manifest_path)

Validate an authoritative AHPC-NOCI case manifest and print a planning report.
The manifest is parsed read-only; schema, geometry block structure, per-fragment state tables (`identical_fragments: true` shares one table across fragments with one intrinsic geometry), user initial orbitals, electron and spin bookkeeping, MEBF composition and resource estimates are checked, and the unique monomer science tasks of stage 01 are listed after intrinsic-geometry and initial-orbital deduplication.
The `mebf_table` row order is the canonical fragment order; YAML mappings are unordered, so it defines aggregate block association and every ordered listing.
Returns a summary dictionary carrying the expanded `fragment_states` tables for the chain driver; any violation raises with a precise message.
"""
function validate_case(manifest_path::String)
    isfile(manifest_path) || error("manifest not found: $manifest_path")
    manifest = YAML.load_file(manifest_path)
    manifest isa AbstractDict || error("manifest must be a YAML mapping")
    base = dirname(abspath(manifest_path))

    schema = Dict{String,Any}(
        "project" => nothing, "geometry" => Dict{String,Any}(
            "aggregate" => nothing, "fragments" => nothing),
        "electronic_structure" => Dict{String,Any}(
            "basis" => nothing, "group" => nothing, "total_spin_multiplicity" => nothing,
            "identical_fragments" => nothing, "states" => nothing, "initial_orbitals" => nothing),
        "mebf_table" => nothing,
        "common_orbitals" => Dict{String,Any}("threshold" => nothing, "freeze_core" => nothing),
        "integrals" => Dict{String,Any}("method" => nothing, "generation_threshold" => nothing, "export" => nothing),
        "screening" => Dict{String,Any}("tau_ci" => nothing, "tau_ci_offdiag" => nothing),
        "runtime" => Dict{String,Any}("use_gpu" => nothing, "threads" => nothing, "blas_threads" => nothing,
            "heap_size_hint_mb" => nothing, "scratch_mb" => nothing, "gpu_device" => nothing, "gpu_devices" => nothing),
        "output" => Dict{String,Any}("checkpoint" => nothing))
    check_allowed_keys(manifest, schema, "manifest")

    project = manifest["project"]
    project isa AbstractString && !isempty(project) || error("project must be a non-empty label")

    geometry = manifest["geometry"]
    group = manifest["electronic_structure"]["group"]
    group == "nosym" || error("the standard chain is C1-only; electronic_structure.group must be `nosym`")
    basis = manifest["electronic_structure"]["basis"]

    aggregate_path = joinpath(base, string(geometry["aggregate"]))
    isfile(aggregate_path) || error("aggregate geometry not found: $aggregate_path")
    _, aggregate_elements, _ = load_xyz(aggregate_path)

    table = manifest["mebf_table"]
    table isa AbstractVector && !isempty(table) || error("mebf_table must be a non-empty list")
    row_width = nothing
    fragments = String[]
    for row in table
        row isa AbstractVector || error("mebf_table rows must be lists")
        row_width === nothing ? row_width = length(row) : length(row) == row_width || error("mebf_table rows have unequal widths")
        label = String(row[1])
        label in fragments && error("mebf_table repeats fragment label $label")
        push!(fragments, label)
    end
    row_width >= 3 || error("mebf_table needs a fragment label plus at least two state columns")
    Set(string.(keys(geometry["fragments"]))) == Set(fragments) || error("geometry.fragments labels must match mebf_table fragment labels exactly")

    fragment_geometries = Dict{String,String}()
    for label in fragments
        resolved = joinpath(base, string(geometry["fragments"][label]))
        isfile(resolved) || error("fragment geometry not found: $resolved")
        fragment_geometries[label] = resolved
    end

    total_atoms = 0
    fragment_atoms = Dict{String,Int}()
    for label in fragments
        _, elements, _ = load_xyz(fragment_geometries[label])
        fragment_atoms[label] = length(elements)
        total_atoms += length(elements)
    end
    total_atoms == length(aggregate_elements) || error("aggregate atom count $(length(aggregate_elements)) != sum of fragments $total_atoms")

    offset = 0
    for label in fragments
        _, intrinsic_elements, _ = load_xyz(fragment_geometries[label])
        block_elements = join(aggregate_elements[offset+1:offset+fragment_atoms[label]], ",")
        block_elements == join(intrinsic_elements, ",") || error("fragment $label atom order differs between intrinsic and aggregate geometry")
        offset += fragment_atoms[label]
    end

    fragment_states = expand_fragment_states(manifest, fragments, fragment_geometries)
    total_multiplicity = manifest["electronic_structure"]["total_spin_multiplicity"]
    total_multiplicity isa Integer && total_multiplicity >= 1 || error("total_spin_multiplicity must be a positive integer")
    state_keys = Dict{String,Any}("charge" => nothing, "multiplicity" => nothing, "nactel" => nothing,
        "inactive" => nothing, "ras2" => nothing, "symmetry" => nothing, "ciroot" => nothing, "rlxroot" => nothing)
    for label in fragments
        _, monomer_elements, _ = load_xyz(fragment_geometries[label])
        monomer_electrons = sum(nuclear_charge, monomer_elements)
        for (name, state) in fragment_states[label]
            state isa AbstractDict || error("fragment $label state $name must be a mapping")
            check_allowed_keys(state, state_keys, "states of fragment $label.$name")
            electrons = monomer_electrons - state["charge"]
            electrons >= 0 || error("fragment $label state $name has negative electron count")
            accounted = state["nactel"] + 2 * state["inactive"]
            accounted == electrons || error("fragment $label state $name electron bookkeeping mismatch: nactel + 2*inactive = $accounted, expected $electrons")
            multiplicity = state["multiplicity"]
            multiplicity isa Integer && multiplicity >= 1 || error("fragment $label state $name multiplicity must be a positive integer")
            spin_s = (multiplicity - 1) / 2
            nactel = state["nactel"]
            isinteger(spin_s + nactel / 2) || error("fragment $label state $name: multiplicity $multiplicity is unreachable with nactel $nactel")
            ciroot = string(state["ciroot"])
            records = split(ciroot, ';')
            for record in records
                tokens = split(strip(record))
                isempty(tokens) && error("fragment $label state $name ciroot has an empty record: $ciroot")
                all(t -> occursin(r"^[0-9]+$", t), tokens) || error("fragment $label state $name ciroot is malformed: $ciroot")
            end
        end
    end

    initial = manifest["electronic_structure"]["initial_orbitals"]
    initial isa AbstractDict || error("initial_orbitals must be a mapping of fragment labels to state mappings")
    Set(string.(keys(initial))) == Set(fragments) || error("initial_orbitals must list exactly the mebf_table fragments")
    guess_digests = Dict{String,String}()
    for label in fragments
        block = initial[label]
        block isa AbstractDict || error("initial_orbitals for fragment $label must be a state mapping")
        Set(string.(keys(block))) == Set(string.(keys(fragment_states[label]))) || error("initial_orbitals for fragment $label must cover exactly the defined states of that fragment")
        for (name, path) in block
            resolved = joinpath(base, string(path))
            isfile(resolved) || error("initial orbital file for $label $name not found: $resolved")
            guess_digests["$(label)_$(name)"] = bytes2hex(open(sha256, resolved))
        end
    end

    compositions = Vector{Vector{String}}()
    for column in 2:row_width
        composition = String[]
        multiplicity_list = Int[]
        for row in table
            fragment_label = String(row[1])
            state_label = String(row[column])
            haskey(fragment_states[fragment_label], state_label) || error("mebf_table references state $state_label which is not defined for fragment $fragment_label")
            push!(composition, state_label)
            push!(multiplicity_list, fragment_states[fragment_label][state_label]["multiplicity"])
        end
        spins_can_couple(multiplicity_list, total_multiplicity) || error("column $column ($(join(composition, "+"))) cannot couple to total multiplicity $total_multiplicity")
        push!(compositions, composition)
    end
    allunique(join.(compositions, "+")) || error("mebf_table contains duplicate compositions")
    n_mebf = length(compositions)
    n_pairs = n_mebf * (n_mebf + 1) ÷ 2

    freeze_core = manifest["common_orbitals"]["freeze_core"]
    freeze_core isa Bool || error("common_orbitals.freeze_core must be true or false")
    runtime = manifest["runtime"]
    runtime["use_gpu"] isa Bool || error("runtime.use_gpu must be boolean; GPU execution requires an explicitly installed GPU environment")
    runtime["threads"] isa Integer && runtime["threads"] >= 1 || error("runtime.threads must be a positive integer")
    runtime["blas_threads"] isa Integer && runtime["blas_threads"] >= 1 || error("runtime.blas_threads must be a positive integer")
    runtime["heap_size_hint_mb"] isa Integer && runtime["heap_size_hint_mb"] > 0 || error("runtime.heap_size_hint_mb must be a positive integer")
    runtime["scratch_mb"] isa Integer && runtime["scratch_mb"] > 0 || error("runtime.scratch_mb must be a positive integer")
    output = manifest["output"]
    output["checkpoint"] isa Bool || error("output.checkpoint must be a boolean")
    manifest["integrals"]["method"] in ("ricd", "conventional_cd") || error("integrals.method must be ricd or conventional_cd; select the matching CHH5 template")
    threshold = manifest["integrals"]["generation_threshold"]
    threshold isa Real && isfinite(threshold) && threshold > 0 || error("integrals.generation_threshold must be finite and positive")
    manifest["integrals"]["export"] == "chh5" || error("the standard chain requires integrals.export == `chh5`")

    n_ao = sum(spherical_basis_functions(basis, element) for element in aggregate_elements)
    n_pairs_ao = n_ao * (n_ao + 1) ÷ 2
    packed_bytes = n_pairs_ao * (n_pairs_ao + 1) ÷ 2 * 8
    factor_upper_bytes = n_pairs_ao^2 * 8

    geometry_digests = Dict(label => bytes2hex(open(sha256, fragment_geometries[label])) for label in fragments)
    unique_tasks = Set{String}()
    for label in fragments, name in keys(fragment_states[label])
        push!(unique_tasks, geometry_digests[label] * "|" * String(name) * "|" * guess_digests["$(label)_$(name)"])
    end
    all_state_labels = sort(unique(vcat([collect(string.(keys(fragment_states[label]))) for label in fragments]...)))
    n_fragment_state_entries = sum(length(fragment_states[label]) for label in fragments)

    println("VALIDATION OK: $project")
    println("  manifest: $manifest_path")
    println("  aggregate: $(length(aggregate_elements)) atoms, $n_ao AO ($basis, spherical), $(length(fragments)) fragments ($(join(fragments, ",")))")
    println("  canonical orbital pairs: $n_pairs_ao")
    println("  packed four-index upper bound: $(round(packed_bytes / 1024^2, digits=1)) MiB; ERI factor upper bound: $(round(factor_upper_bytes / 1024^2, digits=1)) MiB (real auxiliary count is measured at stage 04)")
    frozen_line = freeze_core ? "frozen 1s core per fragment: $(join(["$label=$(core_orbital_count(load_xyz(fragment_geometries[label])[2]))" for label in fragments], ", "))" : "freeze_core=false (all-electron)"
    println("  common orbitals: threshold $(manifest["common_orbitals"]["threshold"]), $frozen_line")
    println("  MEBFs: $n_mebf -> $(join(join.(compositions, "|"), ", "))")
    println("  state pairs (including diagonals): $n_pairs")
    println("  unique monomer science tasks for stage 01: $(length(unique_tasks)) (states $(join(all_state_labels, ","))) over $(length(distinct_geometry_groups(geometry_digests, fragments))) distinct intrinsic geometries")
    distinct_guesses = length(Set(values(guess_digests)))
    println("  initial orbital files: $(distinct_guesses) distinct user-prepared file(s) across $(n_fragment_state_entries) fragment-state entries")
    println("  product layout: stage directories beside this manifest (01_fragment_states shared; 02-05 per project subdirectory)")
    return Dict("project" => project, "n_mebf" => n_mebf, "n_pairs" => n_pairs, "n_ao" => n_ao,
        "unique_tasks" => length(unique_tasks), "fragment_states" => fragment_states)
end

"""
    distinct_geometry_groups(geometry_digests, fragments)

Return the distinct intrinsic-geometry digest groups among the fragments.
"""
function distinct_geometry_groups(geometry_digests, fragments)
    return unique([geometry_digests[label] for label in fragments])
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: validate_case.jl MANIFEST.yaml")
    try
        validate_case(abspath(ARGS[1]))
    catch failure
        println(stderr, "VALIDATION FAILED: ", sprint(showerror, failure))
        exit(1)
    end
end
