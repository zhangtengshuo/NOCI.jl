# Julia replication of preprocess/fortran/gcommon.f90.
#
# Builds a common MO basis from fragment INPORB files, writes COMMONORB,
# generates the per-state .vec files used by rdtraint, and rewrites .det
# headers with determinant counts/labels. Molcas RunFile/OneInt access is
# provided by MolcasIO.jl.

using LinearAlgebra
using Printf
using NOCI
include(joinpath(@__DIR__, "..", "molcas_state_energies.jl"))

include(joinpath(@__DIR__, "MolcasIO.jl"))

mutable struct GcommonConfig
    threshold::Float64
    n_fragments::Int
    n_vec::Vector{Int}
    n_frozen::Vector{Int}
    project::String
    extra_info::Bool
    all_epsilons::Bool
    debug::Bool
    no_avg_core::Bool
    ao_integrals::Bool
    frag_name::Vector{String}
    frag_state::Vector{String}
    frag_label::Vector{String}
    file_map::Dict{String,String}
end

mutable struct FragmentData
    n_electrons::Matrix{Int}
    ener::Matrix{Float64}
    enerpt::Matrix{Float64}
    ener_inporb::Matrix{Float64}
    energy_on_inporb::Bool
end

mutable struct HistogramData
    maxbin::Int
    freq::Vector{Int}
end

# Parse one per-fragment frozen-orbital specification of the Frozen control line:
# `0` selects no frozen orbital, `1-8` freezes orbitals 1 through 8, and
# `1-3,6-7` freezes explicit blocks (at most eight, matching the largest
# supported point group). Orbitals are numbered from 1; blocks must be strictly
# ascending and non-overlapping. The same parser lives in validate_case.jl for
# the chain-side summary records; keep the two copies identical.
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

function read_gcommon_input(io::IO=stdin)::GcommonConfig
    lines = readlines(io)
    threshold = 1.0e-6
    n_fragments = 1
    n_vec = ones(Int, 1)
    n_frozen = zeros(Int, 1)
    project = "unknown."
    extra_info = false
    all_epsilons = false
    debug = false
    no_avg_core = false
    ao_integrals = false
    frag_name = String[]
    frag_state = String[]
    frag_label = String[]

    i = 1
    while i <= length(lines)
        s = strip(lines[i])
        isempty(s) && (i += 1; continue)
        key = uppercase(first(s, min(4, lastindex(s))))
        if key == "THRE"
            i += 1
            threshold = parse(Float64, first(split(strip(lines[i]))))
        elseif key == "FRAG"
            i += 1
            n_fragments = parse(Int, first(split(strip(lines[i]))))
            i += 1
            n_vec = parse.(Int, split(strip(lines[i])))[1:n_fragments]
            n_frozen = zeros(Int, n_fragments)
        elseif key == "PROJ"
            i += 1
            project = first(split(strip(lines[i])))
        elseif key == "EXTR"
            extra_info = true
        elseif key == "ALLE"
            all_epsilons = true
        elseif key == "FROZ"
            i += 1
            specs = split(strip(lines[i]))
            length(specs) >= n_fragments || error("Frozen line needs one specification per fragment, got: $(lines[i])")
            n_frozen = [length(parse_frozen_orbital_spec(String(spec))) for spec in specs[1:n_fragments]]
        elseif key == "NOAV"
            no_avg_core = true
        elseif key == "DEBU"
            debug = true
        elseif key == "AOIN"
            ao_integrals = true
        elseif key == "LABE"
            frag_name = fill("", n_fragments)
            frag_state = String[]
            frag_label = String[]
            for ifrag in 1:n_fragments
                i += 1
                parts = split(strip(lines[i]))
                length(parts) >= 1 + n_vec[ifrag] ||
                    error("Labels line for fragment $ifrag has too few entries")
                frag_name[ifrag] = parts[1]
                for st in parts[2:1+n_vec[ifrag]]
                    push!(frag_state, st)
                    push!(frag_label, parts[1] * "_" * st)
                end
            end
        end
        i += 1
    end

    if isempty(frag_label)
        error("gcommon requires Labels in the input; no default label scheme is implemented")
    end
    return GcommonConfig(threshold, n_fragments, n_vec, n_frozen, project,
                         extra_info, all_epsilons, debug, no_avg_core,
                         ao_integrals, frag_name, frag_state, frag_label,
                         Dict{String,String}())
end

function _logical_filename(cfg::GcommonConfig, ivec_global::Integer, suffix::AbstractString)::String
    return cfg.project * cfg.frag_label[Int(ivec_global)] * "." * suffix
end

function _filename_in(cfg::GcommonConfig, ivec_global::Integer, suffix::AbstractString)::String
    logical = _logical_filename(cfg, ivec_global, suffix)
    return get(cfg.file_map, logical * ".in", get(cfg.file_map, logical, logical))
end

function _filename_out(cfg::GcommonConfig, ivec_global::Integer, suffix::AbstractString)::String
    logical = _logical_filename(cfg, ivec_global, suffix)
    return get(cfg.file_map, logical * ".out", get(cfg.file_map, logical, logical))
end

function _global_vec_start(cfg::GcommonConfig, ifrag::Integer)::Int
    s = 0
    for j in 1:Int(ifrag)-1
        s += cfg.n_vec[j]
    end
    return s
end

function _fortran_e22(x::Real)::String
    xf = Float64(x)
    if xf == 0.0 || !isfinite(xf)
        xf == 0.0 && return "  0.00000000000000E+00"
        return @sprintf("%22.14E", xf)
    end
    expo = floor(Int, log10(abs(xf))) + 1
    mant = xf / 10.0^expo
    return lpad(@sprintf("% .14fE%+03d", mant, expo), 22)
end

function _write_e22_vector(io::IO, v::AbstractVector{<:Real})
    for i in 1:5:length(v)
        jmax = min(i + 4, length(v))
        for j in i:jmax
            print(io, _fortran_e22(v[j]))
        end
        println(io)
    end
end

function _read_numeric_block(lines::Vector{String}, start::Int, n::Int)::Tuple{Vector{Float64},Int}
    vals = Float64[]
    i = start
    while length(vals) < n && i <= length(lines)
        append!(vals, parse.(Float64, split(lines[i])))
        i += 1
    end
    length(vals) >= n || error("Unexpected EOF while reading numeric block")
    return vals[1:n], i
end

function _section_index(lines::Vector{String}, marker::AbstractString)::Int
    idx = findfirst(l -> startswith(l, marker), lines)
    idx === nothing && error("Cannot find $marker in INPORB file")
    return idx
end

function _read_index_labels(lines::Vector{String}, idx::Int, n::Int)::Vector{Char}
    labels = Char[]
    i = idx + 2 # skip "#INDEX" and "* 1234567890"
    while length(labels) < n && i <= length(lines)
        line = lines[i]
        if length(line) >= 3
            payload = line[3:end]
            for c in payload
                c == ' ' && continue
                push!(labels, c)
                length(labels) == n && break
            end
        end
        i += 1
    end
    length(labels) == n || error("Could not read $n orbital labels from #INDEX")
    return labels
end

function _read_orbitals(lines::Vector{String}, idx::Int, n::Int, count::Int)::Matrix{Float64}
    coeff = zeros(Float64, count, n)
    i = idx + 1
    for iorb in 1:count
        i <= length(lines) || error("Unexpected EOF before orbital $iorb")
        startswith(lines[i], "* ORBITAL") || error("Expected * ORBITAL, got: $(lines[i])")
        vals, i = _read_numeric_block(lines, i + 1, n)
        coeff[iorb, :] .= vals
    end
    return coeff
end

function read_vec(cfg::GcommonConfig, data::FragmentData, ifrag::Integer,
                  ivec::Integer, nbas::Integer)
    start = _global_vec_start(cfg, ifrag)
    orbfile = _filename_in(cfg, start + ivec, "orb")
    lines = readlines(orbfile)

    idx_index = _section_index(lines, "#INDEX")
    labels = _read_index_labels(lines, idx_index, nbas)
    nocc_total = count(c -> c in ('f', 'i', '1', '2', '3'), labels)
    nocc = nocc_total - cfg.n_frozen[ifrag]
    nocc >= 0 || error("Negative occupied count in $orbfile")

    idx_orb = _section_index(lines, "#ORB")
    needed = cfg.n_frozen[ifrag] + nocc
    coeff = _read_orbitals(lines, idx_orb, nbas, needed)
    frzvec = cfg.n_frozen[ifrag] == 0 ? zeros(Float64, 0, nbas) :
        coeff[1:cfg.n_frozen[ifrag], :]
    vec = zeros(Float64, nbas, nbas)
    if nocc > 0
        vec[1:nocc, :] .= coeff[cfg.n_frozen[ifrag]+1:needed, :]
    end

    data.energy_on_inporb = true
    idx_info = _section_index(lines, "#INFO")
    info = idx_info + 1 <= length(lines) ? lines[idx_info + 1] : ""
    if occursin("natural orbitals for root", info)
        m = match(r"E=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)", info)
        if m !== nothing
            data.ener_inporb[ifrag, ivec] = parse(Float64, replace(m.captures[1], 'D' => 'E'))
        end
    else
        data.energy_on_inporb = false
    end

    idx_occ = _section_index(lines, "#OCC")
    occ, _ = _read_numeric_block(lines, idx_occ + 2, nbas)
    data.n_electrons[ifrag, ivec] = round(Int, sum(occ))
    return frzvec, vec, nocc
end

function read_atomic_overlap(oneint_name::String, nbas::Integer; lu::Integer=87)::Matrix{Float64}
    opnone_open(oneint_name, Int64(lu))
    try
        return read_overlap_matrix(nbas)
    finally
        clsone(Int64(lu))
    end
end

calculate_smo(a::AbstractMatrix, sao::AbstractMatrix) = Matrix(Symmetric(a * sao * transpose(a)))

function print_dim!(hist::HistogramData, evals_desc::Vector{Float64})
    thrs = 0.1
    bin = 1
    for (i, val) in pairs(evals_desc)
        if val < thrs
            bin <= length(hist.freq) && (hist.freq[bin] += i - 1)
            thrs /= 10
            bin += 1
        end
    end
    hist.maxbin = max(hist.maxbin, bin)
    return nothing
end

function calc_frz_density(nvec::Integer, sao::Matrix{Float64},
                          frovec_first::Matrix{Float64}, frzvec::Matrix{Float64})
    return frzvec * (sao * transpose(frovec_first)) / nvec
end

function average_frozen(frz_density::Matrix{Float64}, frovec_first::Matrix{Float64})
    ev = eigen(Symmetric(frz_density))
    return transpose(ev.vectors) * frovec_first
end

function common_basis(cfg::GcommonConfig, data::FragmentData, hist::HistogramData,
                      ifrag::Integer, nbas_super::Integer, oneint_name::String,
                      nbas_frag::Integer)
    nbas_frag_max = nbas_frag * cfg.n_vec[ifrag]
    lindep = zeros(Float64, nbas_frag_max, nbas_frag)
    sao = read_atomic_overlap(oneint_name, nbas_frag)
    frz_density = zeros(Float64, cfg.n_frozen[ifrag], cfg.n_frozen[ifrag])
    frovec_first = zeros(Float64, cfg.n_frozen[ifrag], nbas_frag)
    frz_frag_orb = zeros(Float64, nbas_super, nbas_super)
    frag_orb_tot = 0

    for ivec in 1:cfg.n_vec[ifrag]
        frzvec, vec, nocc = read_vec(cfg, data, ifrag, ivec, nbas_frag)
        if cfg.n_frozen[ifrag] != 0
            if ivec == 1
                frovec_first .= frzvec
            end
            if !cfg.no_avg_core
                frz_density .+= calc_frz_density(cfg.n_vec[ifrag], sao, frovec_first, frzvec)
            end
        end
        if nocc > 0
            lindep[frag_orb_tot+1:frag_orb_tot+nocc, :] .= vec[1:nocc, :]
            frag_orb_tot += nocc
        end
    end

    lindep2 = lindep[1:frag_orb_tot, :]
    if cfg.n_frozen[ifrag] != 0
        frz_avg = cfg.no_avg_core ? frovec_first : average_frozen(frz_density, frovec_first)
        frz_frag_orb[1:cfg.n_frozen[ifrag], 1:nbas_frag] .= frz_avg
    end

    smo = calculate_smo(lindep2, sao)
    ev = eigen(Symmetric(smo))
    order = sortperm(ev.values; rev=true)
    eigenvalues = ev.values[order]
    eig_rows = transpose(ev.vectors[:, order])

    cfg.extra_info && print_dim!(hist, collect(eigenvalues))
    if cfg.all_epsilons
        println()
        @printf("List of all eigenvalues of the MO overlap matrix of fragment %4d\n", ifrag)
        for j in eachindex(eigenvalues)
            @printf("%5d%20.12E\n", j, eigenvalues[j])
        end
    end

    keep = findall(x -> abs(x) > cfg.threshold, eigenvalues)
    ldim = length(keep)
    sdiag = zeros(Float64, nbas_super)
    sdiag[1:ldim] .= eigenvalues[keep]
    basis = eig_rows[keep, :]
    common_mos_frag = basis * lindep2
    for j in 1:ldim
        common_mos_frag[j, :] ./= sqrt(sdiag[j])
    end
    common_mos = zeros(Float64, nbas_super, nbas_super)
    common_mos[1:ldim, 1:nbas_frag] .= common_mos_frag

    if ifrag == 1
        @printf("Dimension of the common MO basis of fragment  1:%4d\n", ldim)
    else
        @printf("%44s%3d:%4d\n", "", ifrag, ldim)
    end

    start_vec = _global_vec_start(cfg, ifrag)
    for ivec in 1:cfg.n_vec[ifrag]
        vecfile = _filename_out(cfg, start_vec + ivec, "vec")
        _, vec, nocc = read_vec(cfg, data, ifrag, ivec, nbas_frag)
        s_u = vec[1:nocc, :] * sao
        vs_u = common_mos_frag * transpose(s_u)
        tvs_u = zeros(Float64, ldim, ldim)
        ncopy = min(nocc, ldim)
        ncopy > 0 && (tvs_u[1:ncopy, :] .= transpose(vs_u[:, 1:ncopy]))
        open(vecfile, "w") do io
            @printf(io, "%4d\n", ldim)
            for j in 1:ldim
                for k in 1:ldim
                    @printf(io, "%18.14f", tvs_u[j, k])
                    if k % 4 == 0 || k == ldim
                        println(io)
                    end
                end
            end
        end
    end

    return frz_frag_orb, common_mos, sdiag, ldim
end

function ortho_frozen(frozen_orbs::Matrix{Float64}, total_frozen::Integer,
                      nbas::Integer)::Matrix{Float64}
    total_frozen == 0 && return frozen_orbs
    namerun("RUNFIL")
    sao = read_atomic_overlap("ONEINT", nbas; lu=88)
    a = frozen_orbs[1:total_frozen, :]
    smo = calculate_smo(a, sao)
    ev = eigen(Symmetric(smo))
    eig_rows = transpose(ev.vectors)
    out = zeros(Float64, size(frozen_orbs))
    for j in 1:total_frozen
        out[j, :] .= vec(eig_rows[j, :]' * a) ./ sqrt(ev.values[j])
    end
    return out
end

function parse_det_header(line::String)::Tuple{Int,Float64,Float64}
    parts = split(line)
    isempty(parts) && error("Empty determinant header")
    inactm = parse(Int, parts[1])
    floats = Float64[]
    for p in parts[2:end]
        if occursin(r"[\.EeDd]", p)
            try
                push!(floats, parse(Float64, replace(p, 'D' => 'E')))
            catch
            end
        end
    end
    ener = isempty(floats) ? 0.0 : floats[1]
    enerpt = length(floats) >= 2 ? floats[end] : 0.0
    return inactm, ener, enerpt
end

function parse_inactive_count(line::String)::Int
    fields = [line[i:min(i + 3, lastindex(line))] for i in 1:4:min(lastindex(line), 32)]
    if length(fields) >= 8
        vals = Int[]
        ok = true
        for field in fields[1:8]
            s = strip(field)
            if isempty(s)
                push!(vals, 0)
            else
                v = tryparse(Int, s)
                if v === nothing
                    ok = false
                    break
                end
                push!(vals, v)
            end
        end
        ok && return sum(vals)
    end
    return parse(Int, first(split(strip(line))))
end

function _parse_root_energy_line(line::AbstractString)
    m = match(r"RASSCF energy for state\s+([0-9]+)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)", line)
    if m !== nothing
        return parse(Int, m.captures[1]), parse(Float64, replace(m.captures[2], 'D' => 'E'))
    end
    return nothing
end

function _parse_caspt2_total_line(line::AbstractString)
    m = match(r"(?:RASSCF root number|Compute H0 matrices for state|CASPT2 state passed to geometry opt\.)\s+([0-9]+)", line)
    m !== nothing && return (:root, parse(Int, m.captures[1]))
    m = match(r"Total energy:\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)", line)
    m !== nothing && return (:energy, parse(Float64, replace(m.captures[1], 'D' => 'E')))
    return nothing
end

function extract_molcas_energies(path::AbstractString, root::Integer)::Tuple{Float64,Float64}
    ecasscf = 0.0
    ecaspt2 = 0.0
    pending_root = 0
    open(path) do io
        for raw in eachline(io)
            line = rstrip(raw)
            parsed = _parse_root_energy_line(line)
            if parsed !== nothing
                state_root, energy = parsed
                state_root == root && (ecasscf = energy)
                continue
            end

            parsed_pt2 = _parse_caspt2_total_line(line)
            if parsed_pt2 !== nothing
                kind, value = parsed_pt2
                if kind == :root
                    pending_root = Int(value)
                elseif pending_root == root
                    ecaspt2 = Float64(value)
                    pending_root = 0
                end
            end
        end
    end
    return ecasscf, ecaspt2
end

function rewrite_det_files(cfg::GcommonConfig, data::FragmentData)
    ivec_global = 0
    for ifrag in 1:cfg.n_fragments
        for jvec in 1:cfg.n_vec[ifrag]
            ivec_global += 1
            logical = _logical_filename(cfg, ivec_global, "det")
            det_in = _filename_in(cfg, ivec_global, "det")
            det_out = _filename_out(cfg, ivec_global, "det")
            lines = readlines(det_in)
            isempty(lines) && error("Empty determinant file $det_in")
            out_path = get(cfg.file_map, logical * ".molcas_out", "")
            if isempty(out_path)
                inactm, ener, enerpt = parse_det_header(lines[1])
            else
                inactm = parse_inactive_count(lines[1])
                root = parse(Int, get(cfg.file_map, logical * ".root", "1"))
                if haskey(cfg.file_map, logical * ".caspt2")
                    ener, enerpt = read_molcas_state_energies(out_path, root; require_caspt2=parse(Bool, cfg.file_map[logical * ".caspt2"]))
                else
                    ener, enerpt = extract_molcas_energies(out_path, root)
                end
            end
            if ener == 0.0 && data.energy_on_inporb
                ener = data.ener_inporb[ifrag, jvec]
            end
            data.ener[ifrag, jvec] = ener
            data.enerpt[ifrag, jvec] = enerpt

            coeff_by_occ = Dict{String,Float64}()
            for line in lines[2:end]
                parts = split(line)
                length(parts) < 2 && continue
                coeff = parse(Float64, replace(parts[1], 'D' => 'E'))
                occ = parts[2]
                coeff_by_occ[occ] = get(coeff_by_occ, occ, 0.0) + coeff
            end
            occs = sort(collect(keys(coeff_by_occ)); rev=true)
            mkpath(dirname(abspath(det_out)))
            open(det_out, "w") do io
                @printf(io, "%5d%5d%12d    %-6s%22.12f%5d%10.3E%22.12f\n",
                        inactm, cfg.n_frozen[ifrag], length(occs),
                        cfg.frag_state[ivec_global], ener,
                        data.n_electrons[ifrag, jvec], cfg.threshold, enerpt)
                for occ in occs
                    @printf(io, "%15.8E      %s\n", coeff_by_occ[occ], occ)
                end
            end
        end
    end
    return nothing
end

function generate_vecfiles(cfg::GcommonConfig, ifrag::Integer, nbas::Integer)
    start = _global_vec_start(cfg, ifrag)
    ident = Matrix{Float64}(I, nbas, nbas)
    for ivec in 1:cfg.n_vec[ifrag]
        filename = _filename_out(cfg, start + ivec, "vec")
        mkpath(dirname(abspath(filename)))
        open(filename, "w") do io
            @printf(io, "%4d\n", nbas)
            for i in 1:nbas
                for j in 1:nbas
                    @printf(io, "%18.14f", ident[i, j])
                    if j % 4 == 0 || j == nbas
                        println(io)
                    end
                end
            end
        end
    end
end

function write_commonorb(super_basis::Matrix{Float64}, frozen_orbs::Matrix{Float64},
                         occnu::Vector{Float64}, nbas::Integer, total_frozen::Integer,
                         threshold::Float64)
    open("COMMONORB", "w") do io
        println(io, "#INPORB 2.2")
        println(io, "#INFO")
        @printf(io, "* Common molecular orbital basis with tau_MO = %9.3E\n", threshold)
        @printf(io, "%8d%8d%8d\n", 0, 1, 0)
        @printf(io, "%8d\n", nbas)
        @printf(io, "%8d\n", nbas)
        println(io, "#ORB")
        for j in 1:total_frozen
            @printf(io, "* ORBITAL%5d%5d\n", 1, j)
            _write_e22_vector(io, vec(frozen_orbs[j, 1:nbas]))
        end
        for j in 1:(nbas - total_frozen)
            @printf(io, "* ORBITAL%5d%5d\n", 1, j + total_frozen)
            _write_e22_vector(io, vec(super_basis[j, 1:nbas]))
        end
        occ = zeros(Float64, nbas)
        for j in nbas:-1:(total_frozen + 1)
            src = j - total_frozen
            src >= 1 && (occ[j] = occnu[src])
        end
        total_frozen > 0 && (occ[1:total_frozen] .= 2.0)
        println(io, "#OCC")
        println(io, "* OCCUPATION NUMBERS")
        _write_e22_vector(io, occ)
    end
end

function gcommon(cfg::GcommonConfig; workdir::AbstractString=pwd())
    maxvec = maximum(cfg.n_vec)
    data = FragmentData(zeros(Int, cfg.n_fragments, maxvec),
                        zeros(Float64, cfg.n_fragments, maxvec),
                        zeros(Float64, cfg.n_fragments, maxvec),
                        zeros(Float64, cfg.n_fragments, maxvec),
                        false)
    hist = HistogramData(1, zeros(Int, 30))

    @printf("common MO basis with threshold:%10.2E\n", cfg.threshold)
    println(" R. K. Kathir, C. de Graaf, R. Broer, R. W. A. Havenith")
    println(" J. Chem. Theory Comput. 16, 2941-2951 (2020)")
    println()
    println("  program written by Coen de Graaf, URV (2019)")
    println("  frozen orbitals by Aitor Sanchez-Mansilla, URV (2020)")
    println()

    molcas_init(String(workdir), "RUNFIL")
    try
        namerun("RUNFIL")
        nsym = get_iscalar("nSym")
        nsym == 1 || error("Symmetry is not implemented in GronOR")
        nbas = Int(get_iarray("nBas", nsym)[1])

        super_basis = zeros(Float64, nbas, nbas)
        frozen_orbs = zeros(Float64, nbas, nbas)
        occnu = zeros(Float64, nbas)
        start_orb = 0
        start_bas = 0
        total_frozen = 0

        if cfg.ao_integrals
            println(" A set of unit vectors will be written to COMMONORB")
            println()
            super_basis .= Matrix{Float64}(I, nbas, nbas)
            start_bas = nbas
            start_orb = nbas
        end

        for ifrag in 1:cfg.n_fragments
            suffix = cfg.frag_name[ifrag]
            runfile_name = "RUNFIL" * suffix
            oneint_name = "ONEINT" * suffix
            namerun(runfile_name)
            nsym_frag = get_iscalar("nSym")
            nsym_frag == 1 || error("Symmetry is not implemented in GronOR")
            nbas_frag = Int(get_iarray("nBas", nsym_frag)[1])

            if !cfg.ao_integrals
                frz_frag_orb, common_mos, sdiag, ldim =
                    common_basis(cfg, data, hist, ifrag, nbas, oneint_name, nbas_frag)
                for j in 1:ldim, k in 1:nbas_frag
                    super_basis[start_orb+j, start_bas+k] = common_mos[j, k]
                end
                occnu[start_orb+1:start_orb+ldim] .= sdiag[1:ldim]
                for j in 1:cfg.n_frozen[ifrag], k in 1:nbas_frag
                    frozen_orbs[total_frozen+j, start_bas+k] = frz_frag_orb[j, k]
                end
                total_frozen += cfg.n_frozen[ifrag]
                start_orb += ldim
                start_bas += nbas_frag
            else
                generate_vecfiles(cfg, ifrag, nbas_frag)
            end
        end

        if !cfg.no_avg_core && total_frozen != 0
            frozen_orbs = ortho_frozen(frozen_orbs, total_frozen, nbas)
        end

        rewrite_det_files(cfg, data)

        @printf("Final number of orbitals in MO basis    : %4d\n", start_orb)
        @printf("Sum of the number of AO basis functions : %4d\n", start_bas)
        @printf("Frozen orbitals in MOTRA    : %4d\n", total_frozen)
        @printf("Deleted orbitals in MOTRA   : %4d\n", start_bas - start_orb - total_frozen)
        if nbas != start_bas
            println(" Inconsistency detected")
            println(" The number of AO basis functions of the supermolecule is not equal to the sum")
            println(" of the the number of AO basis functions of the fragments")
            println(" $nbas not equal to $start_bas")
        end

        if cfg.extra_info
            println()
            println("Dimension of the common MO basis as function of tau_MO")
            println(" tau_MO # basis functions")
            thrs = 0.1
            for i in 1:hist.maxbin-1
                @printf("%10.2E   %5d\n", thrs, hist.freq[i])
                thrs /= 10
            end
        end

        write_commonorb(super_basis, frozen_orbs, occnu, nbas, total_frozen, cfg.threshold)
    finally
        molcas_final()
    end
    return nothing
end

function read_file_map!(cfg::GcommonConfig, path::AbstractString)
    isfile(path) || error("CommonOrb file map not found: $path")
    for raw in readlines(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        parts = split(line, '\t')
        length(parts) >= 6 || error("file map line needs at least 6 tab-separated fields: $raw")
        logical, orb_in, det_in, molcas_out, vec_out, det_out = parts[1:6]
        cfg.file_map["$logical.orb.in"] = orb_in
        cfg.file_map["$logical.det.in"] = det_in
        cfg.file_map["$logical.det.molcas_out"] = molcas_out
        cfg.file_map["$logical.vec.out"] = vec_out
        cfg.file_map["$logical.det.out"] = det_out
        length(parts) >= 7 && (cfg.file_map["$logical.det.root"] = parts[7])
        length(parts) >= 8 && (cfg.file_map["$logical.det.caspt2"] = parts[8])
    end
    return cfg
end

function gcommon(; workdir::AbstractString=pwd(), file_map::Union{Nothing,String}=nothing)
    cfg = read_gcommon_input(stdin)
    file_map !== nothing && read_file_map!(cfg, file_map)
    return gcommon(cfg; workdir = workdir)
end

function _gcommon_cli(args::Vector{String} = ARGS)
    file_map = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--file-map"
            i += 1
            i <= length(args) || error("--file-map needs a value")
            file_map = args[i]
        else
            error("usage: CommonOrb.jl [--file-map FILE] < input > output")
        end
        i += 1
    end
    gcommon(workdir = pwd(), file_map = file_map)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    NOCI.with_openmolcas4noci() do
        _gcommon_cli()
    end
end
