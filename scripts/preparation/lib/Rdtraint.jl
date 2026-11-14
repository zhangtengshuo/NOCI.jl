# File: scripts/preparation/lib/Rdtraint.jl
# Purpose: Convert OpenMolcas integral records into NOCI .one/.two and related metadata files.
# Algorithm origin: GronOR rdtraint data-conversion workflow; OpenMolcas TRAINT/TRAONE data formats.

using LinearAlgebra
using Printf

const RDTRAINT_NTOC = 64
const RDTRAINT_TRATOC = 106
const RDTRAINT_BUFLEN = 9600

mutable struct RdtraintConfig
    project::String
    filesize_mb::Float64
    print_level::Int
    write_labels::Bool
    almost_zero::Float64
    combas::String
end

struct MotraInfo
    toc::Vector{Int64}
    potnuc::Float64
    nsym::Int
    nbas::Int
    norb::Int
    nfrozen::Int
    ndeleted::Int
end

mutable struct TwoElectronFilePlan
    ltuvx::Int64
    nfiles::Int
    nrecords_on_file::Vector{Int}
    nints_on_file::Vector{Int64}
    filenames::Vector{String}
    total_records::Int
    on_last_file::Int64
    on_last_record::Int
end

mutable struct TwoElectronWriter
    plan::TwoElectronFilePlan
    ntra_buf::Int
    write_labels::Bool
    almost_zero::Float64
    kl1::Vector{Int16}
    kl2::Vector{Int16}
    values::Vector{Float64}
    labels::Vector{Int16}
    pos::Int
    ii::Int
    jj::Int
    record_number::Int
    file_index::Int
    record_in_file::Int
    io::Union{Nothing,IOStream}
    nonzero_small::Int64
    written_integrals::Int64
end

function read_rdtraint_input(io::IO = stdin)::RdtraintConfig
    lines = readlines(io)
    project = "project"
    filesize_mb = 2000.0
    print_level = 1
    write_labels = false
    almost_zero = 1.0e-10
    combas = ""
    nfrags = 0

    i = 1
    while i <= length(lines)
        s = strip(lines[i])
        isempty(s) && (i += 1; continue)
        key = uppercase(first(s, min(4, lastindex(s))))
        if key == "PROJ"
            i += 1
            project = first(split(strip(lines[i])))
        elseif key == "FILE"
            i += 1
            filesize_mb = parse(Float64, first(split(strip(lines[i]))))
        elseif key == "PRIN"
            i += 1
            print_level = parse(Int, first(split(strip(lines[i]))))
        elseif key == "WRIT"
            write_labels = true
        elseif key == "SMAL"
            i += 1
            almost_zero = abs(parse(Float64, first(split(strip(lines[i])))))
        elseif key == "FRAG"
            i += 1
            nfrags = parse(Int, first(split(strip(lines[i]))))
        elseif key == "LABE"
            labels = String[]
            for _ in 1:nfrags
                i += 1
                parts = split(strip(lines[i]))
                isempty(parts) || push!(labels, parts[1])
            end
            combas = join(labels)
        end
        i += 1
    end
    print_level <= 0 && (print_level = 1)
    return RdtraintConfig(project, filesize_mb, print_level, write_labels, almost_zero, combas)
end

function _write_fortran_record(io::IO, parts...)
    payload = IOBuffer()
    for part in parts
        if part isa AbstractString
            write(payload, codeunits(part))
        elseif part isa Number
            write(payload, part)
        elseif part isa AbstractArray
            write(payload, part)
        else
            throw(ArgumentError("Unsupported Fortran record payload part $(typeof(part))"))
        end
    end
    bytes = take!(payload)
    n = Int32(length(bytes))
    write(io, n)
    write(io, bytes)
    write(io, n)
    return nothing
end

_fixed(s::AbstractString, n::Integer) = rpad(first(String(s), min(n, lastindex(String(s)))), n)

function _read_numeric_block(lines::Vector{String}, start::Int, n::Int)::Tuple{Vector{Float64},Int}
    vals = Float64[]
    i = start
    while length(vals) < n && i <= length(lines)
        append!(vals, parse.(Float64, split(lines[i])))
        i += 1
    end
    length(vals) >= n || error("Unexpected EOF while reading COMMONORB numeric block")
    return vals[1:n], i
end

function read_commonorb(commonorb::AbstractString, nbas::Integer, norb::Integer, nfrozen::Integer)
    lines = readlines(commonorb)
    info_idx = findfirst(l -> startswith(l, "#INFO"), lines)
    info_idx === nothing && error("Cannot find #INFO in $commonorb")
    vectit = info_idx < length(lines) ? lines[info_idx + 1] : ""

    orb_idx = findfirst(l -> startswith(l, "#ORB"), lines)
    orb_idx === nothing && error("Cannot find #ORB in $commonorb")
    i = orb_idx + 1
    for _ in 1:nfrozen
        i += 1
        _, i = _read_numeric_block(lines, i, Int(nbas))
    end
    vec = zeros(Float64, Int(norb), Int(nbas))
    for j in 1:Int(norb)
        i <= length(lines) || error("Unexpected EOF before COMMONORB orbital $j")
        startswith(lines[i], "* ORBITAL") || error("Expected COMMONORB orbital header, got $(lines[i])")
        vals, i = _read_numeric_block(lines, i + 1, Int(nbas))
        vec[j, :] .= vals
    end
    return vectit, vec
end

function _packed_symmetric_matrix_from_oneint(label::String, ncomp::Integer, nbas::Integer)
    out = Array{Float64}(undef, Int(nbas), Int(nbas), Int(ncomp))
    for comp in 1:Int(ncomp)
        raw = rdone(label, comp, nbas)
        idx = 0
        @inbounds for j in 1:Int(nbas), i in 1:j
            idx += 1
            v = raw[idx]
            out[j, i, comp] = v
            out[i, j, comp] = v
        end
    end
    return out
end

function _pack_lower_triangle(mat::AbstractMatrix{Float64})
    n = size(mat, 1)
    out = Vector{Float64}(undef, n * (n + 1) ÷ 2)
    idx = 0
    @inbounds for j in 1:n
        for i in 1:j
            idx += 1
            out[idx] = mat[j, i]
        end
    end
    return out
end

transform_packed(vec::Matrix{Float64}, ao::Matrix{Float64}) =
    _pack_lower_triangle(vec * ao * transpose(vec))

function read_motra_info(lu::Int64 = Int64(30))
    info = wr_motra_info(lu; mxsym = Int64(1))
    info.nsym == 1 || error("Symmetry is not implemented in GronOR; nSym=$(info.nsym)")
    return MotraInfo(info.toc, info.potnuc, Int(info.nsym), Int(info.nbas[1]),
                     Int(info.norb[1]), Int(info.nfro[1]), Int(info.ndel[1]))
end

function prepare_one_electron_data(cfg::RdtraintConfig; commonorb::AbstractString = "COMMONORB")
    namerun("RUNFILE")
    nsym = get_iscalar("nSym")
    nsym == 1 || error("Symmetry is not implemented in GronOR; nSym=$nsym")
    nbas_run = Int(get_iarray("nBas", Int64(1))[1])

    opnone_open("ONEINT", Int64(77))
    sAO = _packed_symmetric_matrix_from_oneint("Mltpl  0", 1, nbas_run)[:, :, 1]
    dipole_ao = _packed_symmetric_matrix_from_oneint("Mltpl  1", 3, nbas_run)
    quadrupole_ao = _packed_symmetric_matrix_from_oneint("Mltpl  2", 6, nbas_run)
    clsone(Int64(77))

    lu_one = Int64(30)
    daname(lu_one, "TRAONE")
    info = read_motra_info(lu_one)
    nbas = info.nbas
    nbas == nbas_run || error("RUNFILE nBas=$nbas_run differs from TRAONE nBas=$nbas")
    norb = info.norb
    norbtt = norb * (norb + 1) ÷ 2

    vectit, vec = read_commonorb(commonorb, nbas, norb, info.nfrozen)
    println()
    println(vectit)
    println()

    sMO = transform_packed(vec, sAO)
    fock = Vector{Float64}(undef, norbtt)
    kinint = Vector{Float64}(undef, norbtt)
    ddafile!(lu_one, 2, fock, info.toc[2])
    ddafile!(lu_one, 2, kinint, info.toc[3])

    dipole = Matrix{Float64}(undef, norbtt, 3)
    for comp in 1:3
        dipole[:, comp] .= transform_packed(vec, dipole_ao[:, :, comp])
    end
    quadrupole = Matrix{Float64}(undef, norbtt, 6)
    for comp in 1:6
        quadrupole[:, comp] .= transform_packed(vec, quadrupole_ao[:, :, comp])
    end

    return (
        info = info,
        vectit = vectit,
        norbtt = norbtt,
        sMO = sMO,
        zero = zeros(Float64, norbtt),
        fock = fock,
        dipole = dipole,
        quadrupole = quadrupole,
        lu_one = lu_one,
    )
end

"""
Build the `.one` payload directly from the in-memory Julia MOTRA result.

This is the fused counterpart of `prepare_one_electron_data`: it deliberately
does not create or read `TRAONE`.  `retained` stores the retained CommonOrb
rows (MO × AO), `fock_ao` is the bare/frozen-core effective AO one-electron
matrix, and `core_energy` is the nuclear plus frozen-core scalar.
"""
function prepare_one_electron_data_direct(cfg::RdtraintConfig,
                                          retained::Matrix{Float64},
                                          nbas::Integer, nfrozen::Integer,
                                          ndeleted::Integer,
                                          fock_ao::Matrix{Float64},
                                          core_energy::Real,
                                          vectit::AbstractString)
    namerun("RUNFILE")
    nsym = get_iscalar("nSym")
    nsym == 1 || error("Symmetry is not implemented in GronOR; nSym=$nsym")
    nbas_run = Int(get_iarray("nBas", Int64(1))[1])
    nbas_run == Int(nbas) ||
        error("RUNFILE nBas=$nbas_run differs from fused MOTRA nBas=$nbas")
    size(retained, 2) == nbas_run ||
        throw(DimensionMismatch("retained CommonOrb AO dimension $(size(retained, 2)) != $nbas_run"))
    size(fock_ao) == (nbas_run, nbas_run) ||
        throw(DimensionMismatch("fock_ao must be $nbas_run × $nbas_run"))

    norb = size(retained, 1)
    norb == nbas_run - Int(nfrozen) - Int(ndeleted) ||
        error("retained dimension $norb is inconsistent with nBas=$nbas_run, " *
              "nFrozen=$nfrozen, nDeleted=$ndeleted")
    norbtt = norb * (norb + 1) ÷ 2

    opnone_open("ONEINT", Int64(77))
    sAO = _packed_symmetric_matrix_from_oneint("Mltpl  0", 1, nbas_run)[:, :, 1]
    dipole_ao = _packed_symmetric_matrix_from_oneint("Mltpl  1", 3, nbas_run)
    quadrupole_ao = _packed_symmetric_matrix_from_oneint("Mltpl  2", 6, nbas_run)
    clsone(Int64(77))

    sMO = transform_packed(retained, sAO)
    fock = transform_packed(retained, fock_ao)
    dipole = Matrix{Float64}(undef, norbtt, 3)
    for comp in 1:3
        dipole[:, comp] .= transform_packed(retained, dipole_ao[:, :, comp])
    end
    quadrupole = Matrix{Float64}(undef, norbtt, 6)
    for comp in 1:6
        quadrupole[:, comp] .= transform_packed(retained, quadrupole_ao[:, :, comp])
    end

    info = MotraInfo(zeros(Int64, RDTRAINT_NTOC), Float64(core_energy), 1,
                     nbas_run, norb, Int(nfrozen), Int(ndeleted))
    return (
        info = info,
        vectit = String(vectit),
        norbtt = norbtt,
        sMO = sMO,
        zero = zeros(Float64, norbtt),
        fock = fock,
        dipole = dipole,
        quadrupole = quadrupole,
        lu_one = nothing,
    )
end

function _rdtraint_title(vectit::AbstractString)
    text = strip(replace(vectit, r"^\*\s*" => ""))
    return isempty(text) ? "Integrals in common MO basis" : "Integrals in " * text
end

function build_two_electron_plan(cfg::RdtraintConfig, norb::Integer; ntra_buf::Integer = RDTRAINT_BUFLEN)
    norbtt = Int(norb) * (Int(norb) + 1) ÷ 2
    ltuvx = Int64(norbtt) * Int64(norbtt + 1) ÷ 2
    bytes_per_record = cfg.write_labels ? 2 * 8 * ntra_buf : 8 * ntra_buf
    ntrarec = max(1, Int(floor(cfg.filesize_mb * 1.0e6 / bytes_per_record)))
    full_files = Int(ltuvx ÷ (Int64(ntrarec) * ntra_buf))
    rem = ltuvx - Int64(full_files) * ntrarec * ntra_buf

    nrecords = fill(ntrarec, full_files)
    nints = fill(Int64(ntrarec) * ntra_buf, full_files)
    filenames = [@sprintf("%s%s_%03d.two", cfg.project, cfg.combas, i) for i in 1:full_files]
    if rem > 0
        push!(nrecords, Int(cld(rem, ntra_buf)))
        push!(nints, rem)
        push!(filenames, @sprintf("%s%s_%03d.two", cfg.project, cfg.combas, length(filenames) + 1))
    end
    isempty(nrecords) && begin
        push!(nrecords, 0)
        push!(nints, Int64(0))
        push!(filenames, @sprintf("%s%s_%03d.two", cfg.project, cfg.combas, 1))
    end
    total_records = sum(nrecords)
    on_last_record = rem > 0 ? Int(rem - Int64(max(0, nrecords[end] - 1)) * ntra_buf) : ntra_buf
    return TwoElectronFilePlan(ltuvx, length(nrecords), nrecords, nints, filenames,
                               total_records, rem, on_last_record)
end

function write_one_file(cfg::RdtraintConfig, one_data, plan::TwoElectronFilePlan)
    info = one_data.info
    title1 = _fixed(_rdtraint_title(one_data.vectit), 80)
    title2 = _fixed("from openMolcas to GronOr by rdtraint ", 80)
    filename = string(cfg.project, cfg.combas, ".one")
    mclab = cfg.write_labels ? Int64(0) : Int64(1)
    open(filename, "w") do io
        _write_fortran_record(io, title1, title2, Int64(one_data.norbtt), Float64(info.potnuc),
                              Int64(info.norb), Int64(RDTRAINT_BUFLEN), mclab)
        _write_fortran_record(io, one_data.sMO)
        _write_fortran_record(io, one_data.zero)
        _write_fortran_record(io, one_data.fock)
        for comp in 1:3
            _write_fortran_record(io, view(one_data.dipole, :, comp))
        end
        for comp in 1:6
            _write_fortran_record(io, view(one_data.quadrupole, :, comp))
        end
        _write_fortran_record(io, Int64[8421, -1, 1])
        _write_fortran_record(io, Int64(plan.ltuvx))
        _write_fortran_record(io, Int64(plan.nfiles))
        _write_fortran_record(io, Int64.(plan.nrecords_on_file))
        _write_fortran_record(io, Int64.(plan.nints_on_file))
    end
    println(_rdtraint_title(one_data.vectit))
    println("from openMolcas to GronOr by rdtraint ")
    return filename
end

function _kl_labels(norb::Integer)
    norbtt = Int(norb) * (Int(norb) + 1) ÷ 2
    kl1 = Vector{Int16}(undef, norbtt)
    kl2 = Vector{Int16}(undef, norbtt)
    idx = 0
    @inbounds for l in 1:Int(norb)
        for k in 1:l
            idx += 1
            kl1[idx] = Int16(k)
            kl2[idx] = Int16(l)
        end
    end
    return kl1, kl2
end

@inline function rdtraint_scal(i::Integer, j::Integer, k::Integer, l::Integer)
    s = 1.0
    i == j && (s *= 0.5)
    k == l && (s *= 0.5)
    ((i == k && j == l) || (i == l && j == k)) && (s *= 0.5)
    return s
end

function start_two_writer(cfg::RdtraintConfig, plan::TwoElectronFilePlan, norb::Integer)
    kl1, kl2 = _kl_labels(norb)
    for f in plan.filenames
        isfile(f) && rm(f)
    end
    return TwoElectronWriter(plan, RDTRAINT_BUFLEN, cfg.write_labels, cfg.almost_zero,
                             kl1, kl2, zeros(Float64, RDTRAINT_BUFLEN),
                             zeros(Int16, 4 * RDTRAINT_BUFLEN), 1, 1, 1, 0, 0, 0,
                             nothing, 0, 0)
end

function _open_current_two_file!(w::TwoElectronWriter)
    w.io === nothing || return nothing
    w.file_index += 1
    w.file_index <= w.plan.nfiles || error("Internal rdtraint writer opened too many files")
    w.record_in_file = 0
    w.io = open(w.plan.filenames[w.file_index], "w")
    return nothing
end

function _write_fixed_two_record!(w::TwoElectronWriter, nints::Int, ilast::Int)
    _open_current_two_file!(w)
    w.record_number += 1
    w.record_in_file += 1
    io = something(w.io)

    if nints < w.ntra_buf
        fill!(view(w.values, nints+1:w.ntra_buf), 0.0)
        w.write_labels && fill!(view(w.labels, 4*nints+1:4*w.ntra_buf), Int16(0))
    end
    write(io, w.values)
    if w.write_labels
        write(io, w.labels)
    end
    write(io, Int64(nints))
    write(io, Int64(ilast))

    if w.record_in_file == w.plan.nrecords_on_file[w.file_index]
        close(io)
        w.io = nothing
    end
    return nothing
end

function append_two_integrals!(w::TwoElectronWriter, vals::Vector{Float64},
                               first::Integer = 1, n::Integer = length(vals) - first + 1)
    stop = Int(first + n - 1)
    @inbounds for p in Int(first):stop
        v = vals[p]
        abs(v) < w.almost_zero && (w.nonzero_small += 1)
        i = w.kl1[w.ii]; j = w.kl2[w.ii]
        k = w.kl1[w.jj]; l = w.kl2[w.jj]
        w.values[w.pos] = v * rdtraint_scal(i, j, k, l)
        if w.write_labels
            o = 4 * (w.pos - 1)
            w.labels[o + 1] = i
            w.labels[o + 2] = j
            w.labels[o + 3] = k
            w.labels[o + 4] = l
        end
        w.pos += 1
        w.written_integrals += 1

        w.jj += 1
        if w.jj > length(w.kl1)
            w.ii += 1
            w.jj = w.ii
        end

        if w.pos > w.ntra_buf
            ilast = w.record_number + 1 == w.plan.total_records ? 1 : 0
            _write_fixed_two_record!(w, w.ntra_buf, ilast)
            w.pos = 1
        end
    end
    return nothing
end

function finish_two_writer!(w::TwoElectronWriter)
    if w.pos > 1
        _write_fixed_two_record!(w, w.pos - 1, 1)
        w.pos = 1
    end
    w.io === nothing || (close(w.io); w.io = nothing)
    w.written_integrals == w.plan.ltuvx ||
        error("Wrote $(w.written_integrals) two-electron integrals, expected $(w.plan.ltuvx)")
    return nothing
end

function print_two_plan(cfg::RdtraintConfig, plan::TwoElectronFilePlan)
    println(" Number of 2 el. integrals          :", plan.ltuvx)
    println(" Number of integral files           :", plan.nfiles)
    plan.nfiles > 1 && println(" Number of integrals on last file   :", plan.on_last_file)
    println(" Total number of records            :", plan.total_records)
    plan.nfiles > 1 && println(" Number of records on last file     :", plan.nrecords_on_file[end])
    println(" Number of integrals on last record :", plan.on_last_record)
    println(cfg.write_labels ? " --- Integral labels are written ---" :
                               " --- Integral labels are not written ---")
    println()
end

function write_sys_file(cfg::RdtraintConfig, nbas::Integer)
    filename = string(cfg.project, cfg.combas, ".sys")
    open(filename, "a") do io
        println(io, " * * * General info from OpenMolcas to GronOR")
        header = String(get_carray("Seward Title", Int64(144)))
        if all(==(' '), header[1:min(72, lastindex(header))])
            header = rpad(" (No title provided by user)", 144)
        end
        println(io, first(header, min(72, lastindex(header))))
        println(io, length(header) >= 73 ? header[73:min(144, lastindex(header))] : "")
        natoms = Int(get_iscalar("Unique atoms"))
        println(io, @sprintf("number of atoms: %4d", natoms))
        coord = get_darray("Unique Coordinates", Int64(3 * natoms))
        atom_raw = get_carray("Unique Atom Names", Int64(6 * natoms))
        basis_raw = get_carray("Unique Basis Name", Int64(14 * nbas))
        znuc = get_darray("Nuclear charge", Int64(natoms))
        com = get_darray("Center of Mass", Int64(3))

        atom_labels = [rstrip(String(atom_raw[(i-1)*6+1:i*6])) for i in 1:natoms]
        basis_labels = [String(basis_raw[(i-1)*14+1:i*14]) for i in 1:Int(nbas)]
        ncntr = zeros(Int, natoms, 7)
        shells = Dict('s'=>1, 'p'=>2, 'd'=>3, 'f'=>4, 'g'=>5, 'h'=>6, 'i'=>7)
        iatom = 1
        for ib in 1:Int(nbas)
            if ib > 1 && basis_labels[ib][1:6] != basis_labels[ib-1][1:6]
                iatom += 1
            end
            shell = get(shells, basis_labels[ib][9], 0)
            shell > 0 && iatom <= natoms && (ncntr[iatom, shell] += 1)
        end
        println(io, @sprintf("%6s%-4s%10s%38s%5s", "", "Atom", "x             y             z", "          Znuc", "s  p  d  f  g  h  i"))
        c = 1
        for ia in 1:natoms
            counts = [ncntr[ia, l] ÷ (2 * (l - 1) + 1) for l in 1:7]
            @printf(io, "%4d  %-6s%14.8f%14.8f%14.8f  %7.2f   %3d%3d%3d%3d%3d%3d%3d\n",
                    ia, atom_labels[ia], coord[c], coord[c+1], coord[c+2], znuc[ia], counts...)
            c += 3
        end
        @printf(io, "%18s%14.8f%14.8f%14.8f\n", "Center of Mass:", com[1], com[2], com[3])
    end
    return filename
end

function setup_rdtraint_outputs(cfg::RdtraintConfig)
    one_data = prepare_one_electron_data(cfg)
    plan = build_two_electron_plan(cfg, one_data.info.norb)
    one_file = write_one_file(cfg, one_data, plan)
    print_two_plan(cfg, plan)
    writer = start_two_writer(cfg, plan, one_data.info.norb)
    return one_data, plan, writer, one_file
end

function setup_rdtraint_outputs_direct(cfg::RdtraintConfig, one_data)
    plan = build_two_electron_plan(cfg, one_data.info.norb)
    one_file = write_one_file(cfg, one_data, plan)
    print_two_plan(cfg, plan)
    writer = start_two_writer(cfg, plan, one_data.info.norb)
    return plan, writer, one_file
end

function rdtraint(; input::IO = stdin, workdir::AbstractString = pwd())
    cfg = read_rdtraint_input(input)
    molcas_init(workdir, cfg.project)
    try
        one_data, plan, writer, _ = setup_rdtraint_outputs(cfg)

        lu_tra = Int64(50)
        daname_mf(lu_tra, "TRAINT")
        iad50 = idafile!(lu_tra, 2, zeros(Int64, RDTRAINT_TRATOC), Int64(0))
        buf = Vector{Float64}(undef, RDTRAINT_BUFLEN)
        remaining = plan.ltuvx
        while remaining > 0
            nread = Int(min(Int64(RDTRAINT_BUFLEN), remaining))
            iad50 = ddafile!(lu_tra, 2, buf, 1, nread, iad50)
            append_two_integrals!(writer, buf, 1, nread)
            remaining -= nread
        end
        finish_two_writer!(writer)
        percentage = 100.0 * writer.nonzero_small / plan.ltuvx
        @printf("Number of integrals smaller than %8.1E :  %10d\n", cfg.almost_zero, writer.nonzero_small)
        percentage >= 1.0e-2 ? @printf("( %6.2f %% of the total)\n", percentage) :
                               @printf("(less than %6.2f %% of the total)\n", 1.0e-2)

        daclos(one_data.lu_one)
        daclos(lu_tra)
        write_sys_file(cfg, one_data.info.nbas)
    finally
        molcas_final()
    end
    return nothing
end

function cholesky_integrals(nodes::Integer; input::IO = stdin,
                            workdir::AbstractString = pwd(),
                            use_gpu::Bool = false,
                            gpu_id::Integer = 0)
    cfg = read_rdtraint_input(input)
    molcas_init(workdir, cfg.project)
    try
        one_data, plan, writer, _ = setup_rdtraint_outputs(cfg)
        cho = prepare_rdcho_vectors(nodes)
        if cho.nOrb != one_data.info.norb
            error("rdcho nOrb=$(cho.nOrb) differs from rdtraint/MOTRA nOrb=$(one_data.info.norb)")
        end
        if use_gpu
            println("Fused Cholesky integral reconstruction: GPU GEMM path on device $gpu_id")
            write_cholesky_integrals_gpu!(writer, cho.coeff, cho.numCho, cho.npq, cho.lpqrs; gpu_id = gpu_id)
        else
            println("Fused Cholesky integral reconstruction: CPU exact-dot path")
            write_cholesky_integrals_cpu!(writer, cho.coeff, cho.numCho, cho.npq, cho.lpqrs)
        end
        finish_two_writer!(writer)
        percentage = 100.0 * writer.nonzero_small / plan.ltuvx
        @printf("Number of integrals smaller than %8.1E :  %10d\n", cfg.almost_zero, writer.nonzero_small)
        percentage >= 1.0e-2 ? @printf("( %6.2f %% of the total)\n", percentage) :
                               @printf("(less than %6.2f %% of the total)\n", 1.0e-2)

        daclos(one_data.lu_one)
        write_sys_file(cfg, one_data.info.nbas)
    finally
        molcas_final()
    end
    return nothing
end

function _rdtraint_cli(args::Vector{String} = ARGS)
    isempty(args) || error("usage: Rdtraint.jl < input > output")
    isdefined(@__MODULE__, :molcas_init) || include(joinpath(@__DIR__, "MolcasIO.jl"))
    rdtraint(input = stdin, workdir = pwd())
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    _rdtraint_cli()
end
