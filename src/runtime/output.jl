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

#=
 NOCI.jl — Nonorthogonal Configuration Interaction in Julia

 File:            src/runtime/output.jl
 Purpose:         Production .day/.out/.json output contract with matrices, provenance, and atomic writes.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

const NOCI_HARTREE_TO_EV = 27.211386245988
const NOCI_BANNER_PATH = joinpath(@__DIR__, "banner.txt")
const NOCI_OUTPUT_RULE = repeat("-", 112)

"""
    _matrix_json(matrix)

Convert a real matrix to row-major nested vectors suitable for deterministic JSON output.
The matrix is read-only, units are preserved in scalar values, only CPU vectors are allocated, and nonfinite values raise `ArgumentError`.
"""
function _matrix_json(matrix::AbstractMatrix{<:AbstractFloat})
    all(isfinite, matrix) || throw(ArgumentError("output matrix contains a non-finite value"))
    return [collect(@view matrix[row, :]) for row in axes(matrix, 1)]
end

"""
    _telemetry_json(telemetry)

Convert execution telemetry into deterministic JSON-compatible dictionaries of phases and counters.
The telemetry is read-only, seconds and bytes retain explicit field names, no CUDA query occurs, and unsupported metadata values are stringified for diagnostics only.
"""
function _telemetry_json(telemetry::ExecutionTelemetry)
    phases = [Dict{String,Any}(
        "name" => String(phase.name),
        "wall_seconds" => phase.wall_seconds,
        "estimated_host_bytes" => phase.estimated_host_bytes,
        "estimated_device_bytes" => phase.estimated_device_bytes,
        "allocated_bytes" => phase.allocated_bytes,
        "metadata" => Dict{String,Any}(String(key) => (value isa Number || value isa Bool || value isa AbstractString ? value : string(value)) for (key, value) in phase.metadata),
    ) for phase in telemetry.phases]
    counters = Dict{String,Any}(String(key) => value for (key, value) in telemetry.counters)
    return Dict{String,Any}(
        "algorithm" => telemetry.algorithm,
        "backend" => telemetry.backend,
        "preflight_passed" => telemetry.preflight_passed,
        "phases" => phases,
        "counters" => counters,
    )
end

"""
    _solution_json(solution)

Convert a generalized NOCI solution to a JSON-compatible mapping of absolute/relative energies, column eigenvectors, and residual norms.
The solution is read-only, Hartree-to-eV conversion is reporting only, only CPU containers are allocated, and nonfinite values are rejected by final serialization.
"""
function _solution_json(solution::NOCISolution)
    reference = first(solution.energies)
    return Dict{String,Any}(
        "energies" => collect(solution.energies),
        "relative_energies_ev" => [(energy - reference) * NOCI_HARTREE_TO_EV for energy in solution.energies],
        "vectors" => _matrix_json(solution.vectors),
        "vector_layout" => "matrix_rows_with_states_in_columns",
        "residual_norms" => collect(solution.residual_norms),
    )
end

"""
    _write_section!(io, title)

Write one stable report section delimiter without imposing a terminal-width limit.
Only the output stream is changed and stream failures propagate.
"""
function _write_section!(io::IO, title::AbstractString)
    println(io, '\n', NOCI_OUTPUT_RULE)
    println(io, title)
    println(io, NOCI_OUTPUT_RULE)
    return nothing
end

"""
    _write_banner!(io)

Render the package version in the NOCI.jl banner and copy it into a production `.out` stream.
The banner is read-only and missing packaging data raises instead of silently changing program identity.
"""
function _write_banner!(io::IO)
    isfile(NOCI_BANNER_PATH) || throw(ArgumentError("NOCI.jl banner file not found: $NOCI_BANNER_PATH"))
    version_label = "Version $(noci_version())"
    textwidth(version_label) <= 76 || throw(ArgumentError("NOCI.jl version label exceeds banner width"))
    version_line = "# " * rpad(version_label, 76) * " #"
    text = replace(read(NOCI_BANNER_PATH, String), r"(?m)^# Version \{NOCI_VERSION\} *#$" => version_line)
    occursin("{NOCI_VERSION}", text) && throw(ArgumentError("NOCI.jl banner version placeholder is malformed"))
    print(io, text)
    endswith(text, '\n') || println(io)
    return nothing
end

"""
    _read_xyz_geometry(path)

Read a standard XYZ geometry for report provenance and return symbols, Cartesian coordinates in Angstrom, and the comment line.
The source is read-only, fields after xyz are ignored, and malformed counts or nonfinite coordinates raise before output.
"""
function _read_xyz_geometry(path::AbstractString)
    isfile(path) || throw(ArgumentError("geometry file not found: $path"))
    lines = readlines(path)
    length(lines) >= 2 || throw(ArgumentError("XYZ geometry requires atom-count and comment lines: $path"))
    n_atoms = parse(Int, strip(lines[1]))
    n_atoms >= 1 || throw(ArgumentError("XYZ atom count must be positive: $path"))
    length(lines) >= n_atoms + 2 || throw(ArgumentError("XYZ geometry contains fewer atom records than declared: $path"))
    symbols = Vector{String}(undef, n_atoms)
    coordinates = Matrix{Float64}(undef, n_atoms, 3)
    for atom in 1:n_atoms
        fields = split(strip(lines[atom + 2]))
        length(fields) >= 4 || throw(ArgumentError("invalid XYZ atom record $atom in $path"))
        symbols[atom] = String(fields[1])
        isempty(symbols[atom]) && throw(ArgumentError("empty XYZ element label at atom $atom in $path"))
        for axis in 1:3
            value = parse(Float64, fields[axis + 1])
            isfinite(value) || throw(ArgumentError("non-finite XYZ coordinate at atom $atom in $path"))
            coordinates[atom, axis] = value
        end
    end
    return (symbols=symbols, coordinates=coordinates, comment=String(lines[2]))
end

"""
    _write_geometry!(io, geometry_path)

Write complete Cartesian coordinates when report geometry is supplied, otherwise state explicitly that no reporting geometry was provided.
Geometry is provenance only and does not alter the NOCI Hamiltonian or scientific signatures.
"""
function _write_geometry!(io::IO, geometry_path::Union{Nothing,AbstractString})
    _write_section!(io, "MOLECULAR GEOMETRY")
    if isnothing(geometry_path)
        println(io, "status = not supplied")
        return nothing
    end
    geometry = _read_xyz_geometry(geometry_path)
    println(io, "source = ", abspath(geometry_path))
    println(io, "sha256 = ", _sha256_file(geometry_path))
    println(io, "format = XYZ")
    println(io, "units = Angstrom")
    println(io, "atoms = ", length(geometry.symbols))
    println(io, "comment = ", geometry.comment)
    println(io)
    @printf(io, "%6s  %-8s %20s %20s %20s\n", "Index", "Element", "X", "Y", "Z")
    for atom in eachindex(geometry.symbols)
        @printf(io, "%6d  %-8s %20.10f %20.10f %20.10f\n",
            atom,
            geometry.symbols[atom],
            geometry.coordinates[atom, 1],
            geometry.coordinates[atom, 2],
            geometry.coordinates[atom, 3],
        )
    end
    return nothing
end

"""
    _fragment_state_report(specification)

Read only determinant header/first-record and orbital-dimension metadata required for compact fragment-state provenance.
Scientific files are read-only, no determinant expansion or orbital matrix is materialized, and malformed headers propagate.
"""
function _fragment_state_report(specification::FragmentStateSpecification)
    header = open(specification.determinant_path, "r") do io
        eof(io) && throw(ArgumentError("empty determinant file: $(specification.determinant_path)"))
        fields = split(strip(readline(io)))
        length(fields) >= 8 || throw(ArgumentError("invalid determinant header in $(specification.determinant_path)"))
        active_orbitals = nothing
        spin_multiplicity = nothing
        while !eof(io)
            text = strip(readline(io))
            isempty(text) && continue
            record = split(text)
            length(record) >= 2 || throw(ArgumentError("invalid determinant record in $(specification.determinant_path)"))
            active_orbitals = length(record[end])
            _, spin_projection = _occupation_counts(record[end])
            spin_multiplicity = abs(spin_projection) + 1
            break
        end
        isnothing(active_orbitals) && throw(ArgumentError("no determinant record in $(specification.determinant_path)"))
        inactive_before_frozen = parse(Int, fields[1])
        frozen = parse(Int, fields[2])
        inactive_before_frozen >= frozen >= 0 || throw(ArgumentError("invalid inactive/frozen counts in $(specification.determinant_path)"))
        casscf = parse(Float64, fields[5])
        caspt2 = parse(Float64, fields[8])
        tau_mo = parse(Float64, fields[7])
        all(isfinite, (casscf, caspt2, tau_mo)) || throw(ArgumentError("non-finite fragment header value in $(specification.determinant_path)"))
        return (
            determinants=parse(Int, fields[3]),
            electrons=parse(Int, fields[6]),
            tau_mo=tau_mo,
            inactive=inactive_before_frozen - frozen,
            frozen=frozen,
            active=Int(active_orbitals),
            spin_multiplicity=something(spin_multiplicity),
            casscf=casscf,
            caspt2=caspt2,
        )
    end
    return merge(header, (basis=_orbital_dimension(specification.orbital_path),))
end

"""
    _one_electron_header_report(path)

Read only the conventional `.one` header record and return titles, orbital/integral dimensions, and nuclear repulsion.
No operator matrix or ERI payload is loaded and malformed records propagate through the existing Fortran-record helpers.
"""
function _one_electron_header_report(path::AbstractString)
    marker_type = detect_fortran_record_marker(path)
    return open(path, "r") do io
        header = read_fortran_record(io, marker_type)
        length(header) >= 200 || throw(ArgumentError("invalid one-electron header record in $path"))
        return (
            title_1=_decode_fortran_string(view(header, 1:80)),
            title_2=_decode_fortran_string(view(header, 81:160)),
            one_count=Int(_reinterpret_record_scalar(Int64, view(header, 161:168))),
            nuclear_repulsion=_reinterpret_record_scalar(Float64, view(header, 169:176)),
            n_orbitals=Int(_reinterpret_record_scalar(Int64, view(header, 177:184))),
            buffer_length=Int(_reinterpret_record_scalar(Int64, view(header, 185:192))),
            label_mode=Int(_reinterpret_record_scalar(Int64, view(header, 193:200))),
        )
    end
end

"""
    _write_calculation_definition!(io, input)

Write system, screening, integral, fragment-state, electronic-configuration, and resolved complete-spin-manifold definitions in reproducible long-line tables.
Only lightweight determinant/orbital headers are inspected; numerical wave functions and integral matrices are not loaded here, and explicit legacy coupling chains are identified separately from complete-manifold configurations.
"""
function _write_calculation_definition!(io::IO, input::NOCIInput)
    _write_section!(io, "CALCULATION DEFINITION")
    fragment_reports = [_fragment_state_report(specification) for specification in input.fragment_states]
    fragment_index_by_label = Dict(
        specification.label => index
        for (index, specification) in pairs(input.fragment_states)
    )
    configuration_spin_reports = NamedTuple[]
    resolved_spin_basis_count = 0
    for definition in input.mebfs
        local_multiplicities = [
            fragment_reports[fragment_index_by_label[label]].spin_multiplicity
            for label in definition.state_labels
        ]
        descriptors = _spin_coupled_basis_descriptors(
            definition.label,
            local_multiplicities,
            input.spin_multiplicity,
            definition.coupling_multiplicities,
        )
        push!(
            configuration_spin_reports,
            (
                local_multiplicities=local_multiplicities,
                manifold_dimension=first(descriptors).manifold_dimension,
                descriptors=descriptors,
            ),
        )
        resolved_spin_basis_count += length(descriptors)
    end

    println(io, "system_label = ", input.label)
    println(io, "spin_multiplicity = ", input.spin_multiplicity)
    println(io, "algorithm = ", algorithm_name(input.algorithm))
    println(io, "backend = ", backend_name(input.backend))
    println(io, "fragment_states = ", length(input.fragment_states))
    println(io, "electronic_configurations = ", length(input.mebfs))
    println(io, "resolved_spin_basis_functions = ", resolved_spin_basis_count)

    println(io, "\nScreening thresholds (dimensionless)")
    @printf(io, "  tau_ci             = %.6e\n", input.screening.tau_ci)
    @printf(io, "  tau_ci_offdiagonal = %.6e\n", input.screening.tau_ci_offdiagonal)
    @printf(io, "  tau_singularity    = %.6e\n", input.screening.tau_singularity)

    one_path = input.backend.one_path
    integral_header = _one_electron_header_report(one_path)
    println(io, "\nIntegral source summary")
    println(io, "  one_electron = ", abspath(one_path))
    println(io, "  title_1 = ", integral_header.title_1)
    println(io, "  title_2 = ", integral_header.title_2)
    println(io, "  basis_functions = ", integral_header.n_orbitals)
    println(io, "  packed_one_electron_integrals = ", integral_header.one_count)
    @printf(io, "  nuclear_repulsion / Eh = %.12f\n", integral_header.nuclear_repulsion)
    println(io, "  two_electron_buffer_length = ", integral_header.buffer_length)
    println(io, "  two_electron_label_mode = ", integral_header.label_mode)
    if input.backend isa PackedTwoERI
        println(io, "  two_electron_files = ", length(input.backend.two_paths))
        for (index, path) in pairs(input.backend.two_paths)
            println(io, "  two_electron[$index] = ", abspath(path))
        end
    elseif input.backend isa FactorizedERI{ERIFactorBundleSource}
        println(io, "  eri_factor_bundle = ", abspath(input.backend.source.factor_path))
    elseif input.backend isa FactorizedERI{SewardMOTRASource}
        source = input.backend.source
        println(io, "  seward_snapshot = ", abspath(source.snapshot))
        println(io, "  seward_project = ", source.project)
        println(io, "  seward_ranks = ", source.ranks)
        println(io, "  commonorb = ", abspath(source.commonorb))
        println(io, "  frozen_orbitals = ", source.n_frozen)
        println(io, "  deleted_orbitals = ", source.n_deleted)
        println(io, "  julia_motra_backend = ", source.transform_backend)
        println(io, "  julia_motra_gpu_device = ", source.gpu_device)
    end

    println(io, "\nFragment electronic states")
    @printf(io, "%4s %-24s %-10s %-8s %6s %9s %9s %8s %8s %8s %8s %20s %20s %20s %12s\n",
        "#", "Label", "Fragment", "State", "Mult", "Electrons", "DetRec", "Inactive", "Active", "Frozen", "Basis", "E(CASSCF)/Eh", "E(CASPT2)/Eh", "E_corr/Eh", "tau_MO")
    for (index, specification) in pairs(input.fragment_states)
        record = fragment_reports[index]
        @printf(io, "%4d %-24s %-10s %-8s %6d %9d %9d %8d %8d %8d %8d %20.12f %20.12f %20.12f %12.4e\n",
            index,
            specification.label,
            specification.fragment,
            specification.state,
            record.spin_multiplicity,
            record.electrons,
            record.determinants,
            record.inactive,
            record.active,
            record.frozen,
            record.basis,
            record.casscf,
            record.caspt2,
            record.caspt2 - record.casscf,
            record.tau_mo,
        )
    end

    println(io, "\nElectronic configurations and resolved spin manifolds (ordered)")
    @printf(io, "%4s %-20s %-70s %-18s %-14s %10s %10s %20s\n",
        "#", "Configuration", "Fragment-state product", "Local mult.", "Spin basis", "Manifold", "Included", "Manual shift/Eh")
    for (index, definition) in pairs(input.mebfs)
        product = join(definition.state_labels, " * ")
        spin_report = configuration_spin_reports[index]
        local_multiplicities = join(spin_report.local_multiplicities, ",")
        spin_basis = isempty(definition.coupling_multiplicities) ? "complete" : "explicit"
        @printf(io, "%4d %-20s %-70s %-18s %-14s %10d %10d %20.12e\n",
            index,
            definition.label,
            product,
            local_multiplicities,
            spin_basis,
            spin_report.manifold_dimension,
            length(spin_report.descriptors),
            definition.diagonal_energy_shift,
        )
        for descriptor in spin_report.descriptors
            println(
                io,
                "      -> ",
                descriptor.label,
                "  coupling_multiplicities=[",
                join(descriptor.coupling_multiplicities, ","),
                "]",
            )
        end
    end
    return nothing
end

"""
    _write_named_matrix!(io, name, matrix; row_labels=nothing, column_labels=row_labels)

Write one complete labeled matrix with one logical row per line and no terminal-width truncation.
The matrix is read-only, floating values use scientific notation, integer matrices remain exact, and dimension/label mismatches raise.
"""
function _write_named_matrix!(
    io::IO,
    name::AbstractString,
    matrix::AbstractMatrix;
    row_labels::Union{Nothing,AbstractVector{<:AbstractString}}=nothing,
    column_labels::Union{Nothing,AbstractVector{<:AbstractString}}=row_labels,
)
    n_rows, n_columns = size(matrix)
    rows = isnothing(row_labels) ? [string(index) for index in 1:n_rows] : String.(row_labels)
    columns = isnothing(column_labels) ? [string(index) for index in 1:n_columns] : String.(column_labels)
    length(rows) == n_rows || throw(DimensionMismatch("row-label count differs from matrix rows for $name"))
    length(columns) == n_columns || throw(DimensionMismatch("column-label count differs from matrix columns for $name"))
    label_width = 12
    for label in rows
        label_width = max(label_width, length(label) + 2)
    end
    value_width = 20
    println(io, '\n', name)
    print(io, rpad("row\\column", label_width))
    for label in columns
        print(io, lpad(label, value_width))
    end
    println(io)
    for row in 1:n_rows
        print(io, rpad(rows[row], label_width))
        for column in 1:n_columns
            if eltype(matrix) <: Integer
                print(io, lpad(string(matrix[row, column]), value_width))
            else
                @printf(io, "%20.12e", matrix[row, column])
            end
        end
        println(io)
    end
    return nothing
end

"""
    _write_solution_table!(io, title, solution)

Write absolute energies, ground-referenced relative energies, and generalized-eigenproblem residuals for all NOCI states.
Energies are reported in Hartree and eV without mutating the solution.
"""
function _write_solution_table!(io::IO, title::AbstractString, solution::NOCISolution)
    println(io, '\n', title)
    @printf(io, "%7s %22s %22s %22s\n", "State", "Energy / Eh", "Relative / eV", "Residual / Eh")
    reference = first(solution.energies)
    for state in eachindex(solution.energies)
        @printf(io, "%7d %22.12f %22.10f %22.6e\n",
            state,
            solution.energies[state],
            (solution.energies[state] - reference) * NOCI_HARTREE_TO_EV,
            solution.residual_norms[state],
        )
    end
    return nothing
end

"""
    _write_solution_vectors!(io, title, labels, solution)

Write overlap-normalized generalized NOCI eigenvectors with MEBFs in rows and NOCI states in columns.
All coefficients are printed without display truncation and the solution is read-only.
"""
function _write_solution_vectors!(io::IO, title::AbstractString, labels::AbstractVector{<:AbstractString}, solution::NOCISolution)
    state_labels = ["State $state" for state in axes(solution.vectors, 2)]
    _write_named_matrix!(io, title, solution.vectors; row_labels=labels, column_labels=state_labels)
    println(io, "vector_normalization = C' * S * C = I")
    return nothing
end

"""
    _write_telemetry_summary!(io, telemetry)

Aggregate repeated execution phases by phase name and write calls, total/max wall time, peak estimates, and allocation totals.
Detailed per-event chronology remains in `.sec`; missing allocation measurements are reported explicitly instead of coerced to zero.
"""
function _write_telemetry_summary!(io::IO, telemetry::ExecutionTelemetry)
    order = Symbol[]
    statistics = Dict{Symbol,Dict{Symbol,Any}}()
    for phase in telemetry.phases
        if !haskey(statistics, phase.name)
            push!(order, phase.name)
            statistics[phase.name] = Dict{Symbol,Any}(
                :calls => 0,
                :wall_total => 0.0,
                :wall_max => 0.0,
                :host_peak => 0,
                :device_peak => 0,
                :allocated_total => 0,
                :allocations_complete => true,
            )
        end
        item = statistics[phase.name]
        item[:calls] += 1
        item[:wall_total] += phase.wall_seconds
        item[:wall_max] = max(item[:wall_max], phase.wall_seconds)
        item[:host_peak] = max(item[:host_peak], phase.estimated_host_bytes)
        item[:device_peak] = max(item[:device_peak], phase.estimated_device_bytes)
        if isnothing(phase.allocated_bytes)
            item[:allocations_complete] = false
        else
            item[:allocated_total] += phase.allocated_bytes
        end
    end
    @printf(io, "%-40s %7s %14s %14s %18s %18s %20s\n",
        "Phase", "Calls", "Wall total/s", "Wall max/s", "Host peak/B", "Device peak/B", "Allocated total/B")
    for name in order
        item = statistics[name]
        allocated = item[:allocations_complete] ? string(item[:allocated_total]) : "not-collected"
        @printf(io, "%-40s %7d %14.6f %14.6f %18d %18d %20s\n",
            String(name),
            item[:calls],
            item[:wall_total],
            item[:wall_max],
            item[:host_peak],
            item[:device_peak],
            allocated,
        )
    end
    return nothing
end

"""
    _evaluation_json(evaluation)

Convert one unshifted evaluation into the complete machine-readable scientific result mapping.
All source arrays are read-only, overlap is dimensionless and Hamiltonian values retain their energy unit, and only JSON-compatible CPU containers are allocated.
"""
function _evaluation_json(evaluation::NOCIEvaluationResult)
    raw = evaluation.raw
    normalized = evaluation.normalized
    return Dict{String,Any}(
        "label" => evaluation.label,
        "algorithm" => evaluation.algorithm,
        "branch" => evaluation.branch,
        "backend" => evaluation.backend,
        "eri_factor_provenance" => Dict{String,Any}(String(key) => value for phase in evaluation.telemetry.phases if phase.name == :load_eri for (key, value) in phase.metadata),
        "labels" => copy(raw.labels),
        "raw" => Dict{String,Any}(
            "overlap" => _matrix_json(raw.overlap),
            "electronic_one_body" => _matrix_json(raw.electronic_one_body),
            "nuclear_repulsion" => _matrix_json(raw.nuclear_repulsion),
            "one_body_with_nuclear_repulsion" => _matrix_json(raw.one_body_with_nuclear_repulsion),
            "two_body" => _matrix_json(raw.two_body),
            "hamiltonian" => _matrix_json(raw.hamiltonian),
            "determinant_pair_counts" => [collect(@view raw.determinant_pair_counts[row, :]) for row in axes(raw.determinant_pair_counts, 1)],
        ),
        "normalized" => Dict{String,Any}(
            "overlap" => _matrix_json(normalized.overlap),
            "electronic_one_body" => _matrix_json(normalized.electronic_one_body),
            "nuclear_repulsion" => _matrix_json(normalized.nuclear_repulsion),
            "one_body_with_nuclear_repulsion" => _matrix_json(normalized.one_body_with_nuclear_repulsion),
            "two_body" => _matrix_json(normalized.two_body),
            "hamiltonian" => _matrix_json(normalized.hamiltonian),
        ),
        "unshifted_solution" => _solution_json(evaluation.solution),
        "telemetry" => _telemetry_json(evaluation.telemetry),
    )
end

"""
    _write_production_header!(logger, input, runtime, output, checkpoint, input_manifest, scientific_signature, execution_signature)

Write and flush banner, execution provenance, calculation definition, optional geometry, and input manifest before numerical work begins.
Metadata and scientific sources are read-only, only lightweight headers/XYZ text are inspected, human-readable file sizes are printed in megabytes, no CUDA state is opened, and stream errors propagate.
"""
function _write_production_header!(
    logger::ProgressLogger,
    input::NOCIInput,
    runtime::RuntimeOptions,
    output::RunOutputOptions,
    checkpoint::Union{Nothing,CheckpointOptions},
    input_manifest::AbstractVector,
    scientific_signature::AbstractString,
    execution_signature::AbstractString,
)
    io = logger.out_io
    _write_banner!(io)
    _write_section!(io, "RUN AND REPRODUCIBILITY")
    println(io, "started_at_utc = ", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "Z")
    println(io, "user = ", get(ENV, "USER", "unknown"))
    println(io, "host = ", get(ENV, "HOSTNAME", get(ENV, "RUNNER_NAME", "unknown")))
    println(io, "working_directory = ", pwd())
    println(io, "julia_version = ", VERSION)
    println(io, "julia_threads = ", Threads.nthreads())
    println(io, "logical_cpu_threads = ", Sys.CPU_THREADS)
    println(io, "label = ", input.label)
    println(io, "algorithm = ", algorithm_name(input.algorithm))
    println(io, "backend = ", backend_name(input.backend))
    println(io, "output_directory = ", abspath(output.directory))
    println(io, "checkpoint_directory = ", isnothing(checkpoint) || !checkpoint.enabled ? "disabled" : abspath(checkpoint.directory))
    println(io, "scientific_signature_sha256 = ", scientific_signature)
    println(io, "execution_signature_sha256 = ", execution_signature)
    println(io, "gpu_enabled = ", runtime.use_gpu)
    println(io, "gpu_devices = ", runtime.use_gpu ? join(runtime.gpu_devices, ",") : "none")
    println(io, "threads_per_gpu = ", input.algorithm isa AHPCNOCI && runtime.use_gpu ? ahpc_threads_per_gpu(runtime) : 0)

    _write_calculation_definition!(io, input)
    _write_geometry!(io, output.geometry_path)

    _write_section!(io, "INPUT FILE PROVENANCE")
    println(io, "identity = filesystem-stat-v1 (calculation sources; nanosecond modification times stay in the machine-readable manifest records)")
    for record in input_manifest
        @printf(io, "index=%d size_mb=%.6f identity_kind=%s path=%s\n", record["index"], record["bytes"] / 1.0e6, record["identity_kind"], record["path"])
    end
    if !isnothing(output.geometry_path)
        geometry_path = abspath(output.geometry_path)
        geometry_stat = stat(geometry_path)
        @printf(io, "report_geometry size_mb=%.6f sha256=%s path=%s\n", geometry_stat.size / 1.0e6, _sha256_file(geometry_path), geometry_path)
    end
    flush(io)
    return nothing
end

"""
    _write_production_restart_header!(logger, input, runtime, output, checkpoint, scientific_signature, execution_signature)

Write one compact continuation header when a production run appends to an existing `.out` stream.
Only restart provenance and signatures are recorded so the banner, calculation definition, and input manifest appear exactly once per output file; metadata is read-only and stream errors propagate.
"""
function _write_production_restart_header!(
    logger::ProgressLogger,
    input::NOCIInput,
    runtime::RuntimeOptions,
    output::RunOutputOptions,
    checkpoint::Union{Nothing,CheckpointOptions},
    scientific_signature::AbstractString,
    execution_signature::AbstractString,
)
    io = logger.out_io
    _write_section!(io, "PRODUCTION RESTART")
    println(io, "restarted_at_utc = ", Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS"), "Z")
    println(io, "user = ", get(ENV, "USER", "unknown"))
    println(io, "host = ", get(ENV, "HOSTNAME", get(ENV, "RUNNER_NAME", "unknown")))
    println(io, "working_directory = ", pwd())
    println(io, "julia_version = ", VERSION)
    println(io, "label = ", input.label)
    println(io, "algorithm = ", algorithm_name(input.algorithm))
    println(io, "backend = ", backend_name(input.backend))
    println(io, "checkpoint_directory = ", isnothing(checkpoint) || !checkpoint.enabled ? "disabled" : abspath(checkpoint.directory))
    println(io, "scientific_signature_sha256 = ", scientific_signature)
    println(io, "execution_signature_sha256 = ", execution_signature)
    flush(io)
    return nothing
end

"""
    _write_production_summary!(logger, result)

Append NOCI solutions and wave functions, normalized and unnormalized matrices, aggregated telemetry, and output paths.
The completed result is read-only, energy units are Hartree when supplied integrals are Hartree, and tables are emitted without terminal-width truncation.
"""
function _write_production_summary!(logger::ProgressLogger, result::ProductionRunResult)
    io = logger.out_io
    evaluation = result.evaluation
    raw = evaluation.raw
    normalized = evaluation.normalized
    labels = raw.labels

    _write_section!(io, "PRIMARY NOCI RESULTS")
    println(io, "Baseline CASPT2 correlation shifts by MEBF (Hartree)")
    @printf(io, "%4s %-20s %22s\n", "#", "MEBF", "Diagonal shift / Eh")
    for index in eachindex(labels)
        @printf(io, "%4d %-20s %22.12e\n", index, labels[index], result.correlation_shifts[index])
    end
    _write_solution_table!(io, "Unshifted NOCI energies", evaluation.solution)
    _write_solution_table!(io, "Baseline CASPT2-shifted NOCI energies", result.shifted_solution)

    _write_section!(io, "NOCI WAVE FUNCTIONS")
    _write_solution_vectors!(io, "Unshifted overlap-normalized eigenvectors", labels, evaluation.solution)
    _write_solution_vectors!(io, "Baseline CASPT2-shifted overlap-normalized eigenvectors", labels, result.shifted_solution)

    _write_section!(io, "NORMALIZED SCIENTIFIC MATRICES")
    for (name, matrix) in (
        ("Normalized overlap matrix", normalized.overlap),
        ("Normalized electronic one-body matrix", normalized.electronic_one_body),
        ("Normalized nuclear-repulsion matrix", normalized.nuclear_repulsion),
        ("Normalized one-body-plus-nuclear matrix", normalized.one_body_with_nuclear_repulsion),
        ("Normalized two-body matrix", normalized.two_body),
        ("Normalized unshifted Hamiltonian", normalized.hamiltonian),
        ("Baseline shifted Hamiltonian", result.shifted_hamiltonian),
    )
        _write_named_matrix!(io, name, matrix; row_labels=labels)
    end

    _write_section!(io, "UNNORMALIZED NOCI MATRICES")
    for (name, matrix) in (
        ("Raw overlap matrix", raw.overlap),
        ("Raw electronic one-body matrix", raw.electronic_one_body),
        ("Raw nuclear-repulsion matrix", raw.nuclear_repulsion),
        ("Raw one-body-plus-nuclear matrix", raw.one_body_with_nuclear_repulsion),
        ("Raw two-body matrix", raw.two_body),
        ("Raw unshifted Hamiltonian", raw.hamiltonian),
        ("Determinant-pair counts", raw.determinant_pair_counts),
    )
        _write_named_matrix!(io, name, matrix; row_labels=labels)
    end

    _write_section!(io, "PERFORMANCE AND RESOURCE SUMMARY")
    println(io, "detailed_runtime_events = .sec file")
    _write_telemetry_summary!(io, evaluation.telemetry)
    println(io, "\nTelemetry counters")
    for key in sort!(collect(keys(evaluation.telemetry.counters)); by=string)
        println(io, key, " = ", evaluation.telemetry.counters[key])
    end

    _write_section!(io, "OUTPUT ARTIFACTS")
    println(io, "sec = ", result.sec_path)
    println(io, "out = ", result.out_path)
    println(io, "json = ", result.json_path)
    println(io, "matrix_snapshot = ", result.snapshot_path)
    rss_bytes, hwm_bytes = _process_memory_bytes()
    println(io, "final_rss_bytes = ", rss_bytes)
    println(io, "process_hwm_bytes = ", hwm_bytes)
    println(io, "numerical_status = complete")
    flush(io)
    return nothing
end

"""
    _production_json(result)

Build the final JSON-compatible production result including unshifted data, baseline CASPT2 shift, signatures, paths, memory observation, and software identity.
The result is read-only, scientific units are identified explicitly, no files or CUDA resources are accessed except procfs memory sampling, and unsupported nonfinite values are rejected at serialization.
"""
function _production_json(result::ProductionRunResult)
    rss_bytes, hwm_bytes = _process_memory_bytes()
    data = _evaluation_json(result.evaluation)
    data["schema_version"] = 1
    data["software"] = Dict{String,Any}(
        "name" => "NOCI.jl",
        "version" => string(noci_version()),
        "development_status" => "release",
        "author" => "Teng-Shuo Zhang @ZJUT",
    )
    data["units"] = "Hartree"
    data["scientific_signature_sha256"] = result.scientific_signature
    data["execution_signature_sha256"] = result.execution_signature
    data["input_manifest"] = result.input_manifest
    data["baseline_caspt2"] = Dict{String,Any}(
        "state_mapping" => "label_preserving",
        "shift_weighting" => "identity",
        "mebf_shifts" => copy(result.correlation_shifts),
        "shifted_hamiltonian" => _matrix_json(result.shifted_hamiltonian),
        "shifted_solution" => _solution_json(result.shifted_solution),
    )
    data["paths"] = Dict{String,Any}(
        "sec" => result.sec_path,
        "out" => result.out_path,
        "json" => result.json_path,
        "matrix_snapshot" => result.snapshot_path,
    )
    data["memory"] = Dict{String,Any}("final_rss_bytes" => rss_bytes, "process_hwm_bytes" => hwm_bytes)
    return data
end
