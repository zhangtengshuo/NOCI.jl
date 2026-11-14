#!/usr/bin/env julia
# File: scripts/preparation/lib/RotateOrb.jl
# Purpose: Rotate C1 INPORB orbitals using rigid transforms and OpenMolcas RUNFILE shell labels.
# Interface reference: OpenMolcas RUNFILE basis metadata and the GronOR grotate workflow.

using LinearAlgebra
using Printf
using NOCI

include(joinpath(@__DIR__, "MolcasIO.jl"))
include(joinpath(@__DIR__, "InporbIO.jl"))
using .InporbIO

const BASIS_LABEL_LEN = 14

_parse_float_token(token::AbstractString) =
    parse(Float64, replace(token, 'D' => 'E', 'd' => 'E'))

function read_matrix(path::AbstractString)
    nums = Float64[]
    for line in readlines(path)
        startswith(strip(line), "#") && continue
        for tok in split(line)
            try
                push!(nums, _parse_float_token(tok))
            catch
            end
        end
    end
    length(nums) >= 12 || error("$path must contain a 3x3 rotation and a 3-vector translation")
    R = Matrix(reshape(nums[1:9], 3, 3)')
    t = nums[10:12]
    return R, t
end

"""
    matrix_label(path)

Read the ASCII fragment identifier from a rotation-matrix filename or its monomer comment.
Labels such as A, f_99 and 9_A7 are dimensionless; files are read-only, no backend/device is initialized, and a missing identifier raises before orbital rotation.
"""
function matrix_label(path::AbstractString)
    base = basename(path)
    m = match(r"^matrix_([A-Za-z0-9]+(?:_[A-Za-z0-9]+)?)\.txt$", base)
    m !== nothing && return m.captures[1]
    for line in readlines(path)
        m = match(r"monomer\s+([A-Za-z0-9]+(?:_[A-Za-z0-9]+)?)(?:\s|$)", line)
        m !== nothing && return m.captures[1]
    end
    error("cannot infer monomer label from $path; use a matrix_A.txt-style name")
end

"""
    output_orbital_name(label, orb_path)

Return the shared rotation utility's orbital filename for a fragment label and input state path.
The dimensionless naming operation strips an existing Mol prefix, supports numbered labels, changes no input, performs no scientific allocation and raises no backend exceptions.
"""
function output_orbital_name(label::AbstractString, orb_path::AbstractString)
    state = basename(orb_path)
    endswith(lowercase(state), ".orb") && (state = state[1:end-4])
    prefix = ""
    m = match(r"^([0-9]+_)?Mol(?:[A-Za-z0-9]+_[A-Za-z0-9]+|[A-Za-z][A-Za-z0-9]*)_(.+)$", state)
    if m !== nothing
        prefix = something(m.captures[1], "")
        state = m.captures[2]
    end
    return "$(prefix)Mol$(label)_$(state).orb"
end

function read_basis_labels(runfile_path::AbstractString)
    runfile = abspath(runfile_path)
    isfile(runfile) || error("RUNFILE not found: $runfile")
    workdir = dirname(runfile)
    cd(workdir) do
        molcas_init(workdir, "RUNFIL"; module_name = "ROTATEORB")
        try
            namerun(basename(runfile))
            nsym = Int(get_iscalar("nSym"))
            nsym == 1 || error("$runfile has nSym=$nsym; RotateOrb.jl expects a C1 RUNFILE")
            nbas = Int(get_iarray("nBas", Int64(nsym))[1])
            raw = get_carray("Unique Basis Name", Int64(BASIS_LABEL_LEN * nbas))
            return [String(raw[(i - 1) * BASIS_LABEL_LEN + 1:i * BASIS_LABEL_LEN]) for i in 1:nbas]
        finally
            molcas_final()
        end
    end
end

function contraction_counts(labels::Vector{String})
    n = length(labels)
    counts = zeros(Int, n)
    first = 1
    counts[first] = 1
    for j in 2:n
        same_shell = labels[j][1:6] == labels[j - 1][1:6] &&
                     labels[j][9:11] == labels[j - 1][9:11]
        if same_shell
            counts[first] += 1
        else
            first = j
            counts[first] = 1
        end
    end
    for i in 1:n
        counts[i] == 0 && (counts[i] = counts[i - 1])
    end
    return counts
end

_delta(i::Integer, j::Integer) = i == j ? 1 : 0
_idx(m::Integer, lmax::Integer) = m + lmax + 1

function real_harmonic_rotation(row_rotation::Matrix{Float64}, lmax::Int)
    rot = zeros(Float64, lmax, 2 * lmax + 1, 2 * lmax + 1)
    if lmax == 1
        rot[1, :, :] .= row_rotation
        return rot[1, :, :]
    end

    c = lmax + 1
    rot[1, c - 1, c - 1] = row_rotation[2, 2]
    rot[1, c - 1, c    ] = row_rotation[2, 3]
    rot[1, c - 1, c + 1] = row_rotation[2, 1]
    rot[1, c,     c - 1] = row_rotation[3, 2]
    rot[1, c,     c    ] = row_rotation[3, 3]
    rot[1, c,     c + 1] = row_rotation[3, 1]
    rot[1, c + 1, c - 1] = row_rotation[1, 2]
    rot[1, c + 1, c    ] = row_rotation[1, 3]
    rot[1, c + 1, c + 1] = row_rotation[1, 1]

    rotget(layer, mu, m) = begin
        abs(mu) <= layer && abs(m) <= layer || return 0.0
        rot[layer, _idx(mu, lmax), _idx(m, lmax)]
    end
    pfun(i, l, mu, m) = begin
        x = 0.0
        ii = _idx(i, lmax)
        if abs(m) < l
            x = rot[1, ii, _idx(0, lmax)] * rotget(l - 1, mu, m)
        elseif m == l
            x = rot[1, ii, _idx(1, lmax)] * rotget(l - 1, mu, m - 1) -
                rot[1, ii, _idx(-1, lmax)] * rotget(l - 1, mu, -m + 1)
        elseif m == -l
            x = rot[1, ii, _idx(1, lmax)] * rotget(l - 1, mu, m + 1) +
                rot[1, ii, _idx(-1, lmax)] * rotget(l - 1, mu, -m - 1)
        end
        x
    end

    cap_u(l, m1, m2) = pfun(0, l, m1, m2)
    cap_v(l, m1, m2) = begin
        if m1 == 0
            pfun(1, l, 1, m2) + pfun(-1, l, -1, m2)
        elseif m1 > 0
            sqrt(1 + _delta(m1, 1)) * pfun(1, l, m1 - 1, m2) -
                (1 - _delta(m1, 1)) * pfun(-1, l, -m1 + 1, m2)
        else
            (1 - _delta(m1, -1)) * pfun(1, l, m1 + 1, m2) +
                sqrt(1 + _delta(m1, -1)) * pfun(-1, l, -m1 - 1, m2)
        end
    end
    cap_w(l, m1, m2) = begin
        if m1 > 0
            pfun(1, l, m1 + 1, m2) + pfun(-1, l, -m1 - 1, m2)
        elseif m1 < 0
            pfun(1, l, m1 - 1, m2) - pfun(-1, l, -m1 + 1, m2)
        else
            0.0
        end
    end
    uu(l, m1, m2) = sqrt(((l + m1) * (l - m1)) /
                         (l == abs(m2) ? 2 * l * (2 * l - 1) : (l + m2) * (l - m2)))
    vv(l, m1, m2) = 0.5 * sqrt(((1 + _delta(0, m1)) * (l + abs(m1) - 1) * (l + abs(m1))) /
                               (l == abs(m2) ? 2 * l * (2 * l - 1) : (l + m2) * (l - m2))) *
                    (1 - 2 * _delta(0, m1))
    ww(l, m1, m2) = -0.5 * sqrt(((l - abs(m1) - 1) * (l - abs(m1))) /
                                (l == abs(m2) ? 2 * l * (2 * l - 1) : (l + m2) * (l - m2))) *
                    (1 - _delta(0, m1))

    for l in 2:lmax
        for i in 1:(2 * l + 1)
            m1 = i - (l + 1)
            for j in 1:(2 * l + 1)
                m2 = j - (l + 1)
                rot[l, i + lmax - l, j + lmax - l] =
                    uu(l, m1, m2) * cap_u(l, m1, m2) +
                    vv(l, m1, m2) * cap_v(l, m1, m2) +
                    ww(l, m1, m2) * cap_w(l, m1, m2)
            end
        end
    end
    return rot[lmax, :, :]
end

function shell_lmax(label::String)
    length(label) >= 9 || return 0
    length(label) >= 9 && label[9] == 's' && return 0
    length(label) >= 10 && label[9:10] == "px" && return 1
    length(label) >= 12 && label[9:12] == "d02-" && return 2
    length(label) >= 12 && label[9:12] == "f03-" && return 3
    length(label) >= 12 && label[9:12] == "g04-" && return 4
    return -1
end

function rotate_coefficients(coeff::Matrix{Float64}, basis_labels::Vector{String},
                             column_rotation::Matrix{Float64})
    norb, nbas = size(coeff)
    length(basis_labels) == nbas ||
        error("basis label count $(length(basis_labels)) does not match orbital nBas=$nbas")
    ncntr = contraction_counts(basis_labels)
    row_rotation = Matrix(transpose(column_rotation))
    out = copy(coeff)
    rotations = Dict{Int, Matrix{Float64}}(1 => real_harmonic_rotation(row_rotation, 1))
    for l in 2:4
        rotations[l] = real_harmonic_rotation(row_rotation, l)
    end

    for j in 1:nbas
        lmax = shell_lmax(basis_labels[j])
        lmax <= 0 && continue
        width = 2 * lmax + 1
        stride = ncntr[j]
        idxs = [j + (ll - 1) * stride for ll in 1:width]
        maximum(idxs) <= nbas ||
            error("basis shell starting at AO $j ($(strip(basis_labels[j]))) exceeds nBas=$nbas")
        block = coeff[:, idxs]
        out[:, idxs] .= block * rotations[lmax]
    end
    return out
end

function rotate_orbitals(matrix_path::AbstractString, orb_paths::Vector{String};
                         runfile::AbstractString = "RUNFILE", outdir::AbstractString = ".")
    isempty(orb_paths) && error("no orbital files were provided")
    R, _ = read_matrix(matrix_path)
    label = matrix_label(matrix_path)
    basis_labels = read_basis_labels(runfile)
    mkpath(outdir)

    for orb in orb_paths
        data = read_inporb(orb)
        data.nbas == length(basis_labels) ||
            error("$(orb) nBas=$(data.nbas) differs from RUNFILE nBas=$(length(basis_labels))")
        rotated = rotate_coefficients(data.coeff, basis_labels, R)
        out = joinpath(outdir, output_orbital_name(label, orb))
        title = occursin("rotated with RotateOrb", data.title) ?
            data.title : strip(data.title) * " (rotated with RotateOrb)"
        write_inporb(out, data, rotated; title = title)
        println("wrote: $out")
    end
    nothing
end

function parse_cli(args::Vector{String})
    outdir = "."
    runfile = "RUNFILE"
    files = String[]
    i = 1
    while i <= length(args)
        if args[i] == "--out-dir"
            i + 1 <= length(args) || error("--out-dir needs a value")
            outdir = args[i + 1]
            i += 2
        elseif args[i] == "--runfile"
            i + 1 <= length(args) || error("--runfile needs a value")
            runfile = args[i + 1]
            i += 2
        else
            push!(files, args[i])
            i += 1
        end
    end
    length(files) >= 2 ||
        error("usage: RotateOrb.jl matrix_A.txt S0.orb S1.orb ... [--runfile RUNFILE] [--out-dir DIR]")
    return files[1], files[2:end], runfile, outdir
end

function main(args::Vector{String} = ARGS)
    matrix_path, orb_paths, runfile, outdir = parse_cli(args)
    NOCI.with_openmolcas4noci() do
        rotate_orbitals(matrix_path, orb_paths; runfile = runfile, outdir = outdir)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
