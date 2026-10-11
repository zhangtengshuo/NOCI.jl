module InporbIO

using Printf

export InporbData, read_inporb, write_inporb

struct InporbData
    version::String
    title::String
    iuhf::Int
    nsym::Int
    wftype::Int
    nbas::Int
    norb::Int
    coeff::Matrix{Float64}
    occ::Vector{Float64}
    ochr::Vector{Float64}
    one::Vector{Float64}
    index_labels::Vector{Char}
    info_trailer::Vector{String}
    source::String
end

function _inporb_section_index(lines::Vector{String}, keys::AbstractString...)
    for (index, line) in pairs(lines)
        text = strip(line)
        any(key -> startswith(text, key), keys) && return index
    end
    return nothing
end

function _inporb_section_end(lines::Vector{String}, index::Int)
    cursor = index + 1
    while cursor <= length(lines)
        startswith(strip(lines[cursor]), "#") && return cursor - 1
        cursor += 1
    end
    return length(lines)
end

_parse_inporb_float(token::AbstractString) =
    parse(Float64, replace(token, 'D' => 'E', 'd' => 'E'))

function _read_inporb_numeric_block(lines::Vector{String}, index::Int, count::Integer)
    values = Float64[]
    cursor = index
    while length(values) < count && cursor <= length(lines)
        text = strip(lines[cursor])
        startswith(text, "#") && break
        if !isempty(text) && !startswith(text, "*")
            append!(values, _parse_inporb_float.(split(text)))
        end
        cursor += 1
    end
    length(values) >= count || error("expected $count INPORB numeric values, got $(length(values))")
    return values[1:count], cursor
end

function _collect_inporb_numeric_section(lines::Vector{String}, index, count::Integer)
    index === nothing && return Float64[]
    values = Float64[]
    for cursor in (index + 1):_inporb_section_end(lines, index)
        text = strip(lines[cursor])
        isempty(text) && continue
        startswith(text, "*") && continue
        append!(values, _parse_inporb_float.(split(text)))
        length(values) >= count && break
    end
    return values[1:min(length(values), count)]
end

function _read_inporb_index_labels(lines::Vector{String}, index, count::Integer)
    index === nothing && return fill('s', count)
    labels = Char[]
    for cursor in (index + 1):_inporb_section_end(lines, index)
        line = lines[cursor]
        startswith(strip(line), "*") && continue
        payload = length(line) >= 3 ? line[3:end] : line
        for label in payload
            label == ' ' && continue
            push!(labels, label)
            length(labels) == count && return labels
        end
    end
    length(labels) == count || error("could not read $count orbital labels from #INDEX")
    return labels
end

function read_inporb(io::IO; source::AbstractString = "<io>")::InporbData
    lines = readlines(io)
    isempty(lines) && error("empty orbital file: $source")
    info_index = _inporb_section_index(lines, "#INFO")
    orbital_index = _inporb_section_index(lines, "#ORB")
    info_index === nothing && error("missing #INFO in $source")
    orbital_index === nothing && error("missing #ORB in $source")
    info_index + 4 <= length(lines) || error("truncated #INFO in $source")

    title = strip(lines[info_index + 1])
    info_values = split(lines[info_index + 2])
    length(info_values) >= 3 || error("invalid #INFO flags in $source")
    iuhf, nsym, wftype = parse.(Int, info_values[1:3])
    nsym == 1 || error("$source has nSym=$nsym; expected a C1 INPORB file")
    nbas = parse(Int, first(split(lines[info_index + 3])))
    norb = parse(Int, first(split(lines[info_index + 4])))
    nbas > 0 && norb > 0 || error("$source has invalid nBas=$nbas or nOrb=$norb")
    info_trailer = info_index + 5 <= orbital_index - 1 ?
        copy(lines[(info_index + 5):(orbital_index - 1)]) : String[]

    coefficients = zeros(Float64, norb, nbas)
    cursor = orbital_index + 1
    for orbital in 1:norb
        cursor <= length(lines) || error("unexpected EOF before orbital $orbital in $source")
        startswith(strip(lines[cursor]), "* ORBITAL") ||
            error("expected '* ORBITAL' before orbital $orbital in $source")
        values, cursor = _read_inporb_numeric_block(lines, cursor + 1, nbas)
        coefficients[orbital, :] .= values
    end

    occupations = _collect_inporb_numeric_section(
        lines,
        _inporb_section_index(lines, "#OCC"),
        norb,
    )
    readable_occupations = _collect_inporb_numeric_section(
        lines,
        _inporb_section_index(lines, "#OCHR"),
        norb,
    )
    one_electron = _collect_inporb_numeric_section(
        lines,
        _inporb_section_index(lines, "#ONE", "#ENER"),
        norb,
    )
    index_labels = _read_inporb_index_labels(
        lines,
        _inporb_section_index(lines, "#INDEX"),
        norb,
    )

    isempty(occupations) && (occupations = zeros(Float64, norb))
    isempty(readable_occupations) && (readable_occupations = copy(occupations))
    length(occupations) == norb || error("$source has incomplete #OCC data")
    length(readable_occupations) == norb || error("$source has incomplete #OCHR data")
    version = startswith(strip(lines[1]), "#INPORB") ? strip(lines[1]) : "#INPORB 2.2"
    return InporbData(
        version,
        title,
        iuhf,
        nsym,
        wftype,
        nbas,
        norb,
        coefficients,
        occupations,
        readable_occupations,
        one_electron,
        index_labels,
        info_trailer,
        String(source),
    )
end

function read_inporb(path::AbstractString)::InporbData
    open(path, "r") do io
        return read_inporb(io; source = abspath(path))
    end
end

function _write_inporb_values(
    io::IO,
    values::AbstractVector{<:Real};
    columns::Int = 5,
    width::Int = 22,
    precision::Int = 14,
)
    for first_index in 1:columns:length(values)
        last_index = min(first_index + columns - 1, length(values))
        for value in values[first_index:last_index]
            @printf(io, "%*.*E", width, precision, Float64(value))
        end
        println(io)
    end
end

function _write_inporb_index(io::IO, labels::Vector{Char})
    println(io, "#INDEX")
    println(io, "* 1234567890")
    for first_index in 1:10:length(labels)
        last_index = min(first_index + 9, length(labels))
        print(io, mod((first_index - 1) ÷ 10, 10), " ")
        for index in first_index:last_index
            print(io, labels[index])
        end
        println(io)
    end
end

function write_inporb(
    io::IO,
    data::InporbData,
    coefficients::AbstractMatrix{<:Real} = data.coeff;
    title::AbstractString = data.title,
    occupations::AbstractVector{<:Real} = data.occ,
    readable_occupations::AbstractVector{<:Real} = data.ochr,
    one_electron::AbstractVector{<:Real} = data.one,
)
    size(coefficients) == size(data.coeff) || error("INPORB coefficient matrix size mismatch")
    length(occupations) == data.norb || error("INPORB #OCC length mismatch")
    length(readable_occupations) == data.norb || error("INPORB #OCHR length mismatch")
    isempty(one_electron) || length(one_electron) == data.norb ||
        error("INPORB #ONE length mismatch")

    println(io, data.version)
    println(io, "#INFO")
    println(io, title)
    println(io, @sprintf("%8d%8d%8d", data.iuhf, data.nsym, data.wftype))
    println(io, @sprintf("%8d", data.nbas))
    println(io, @sprintf("%8d", data.norb))
    foreach(line -> println(io, line), data.info_trailer)
    println(io, "#ORB")
    for orbital in 1:data.norb
        println(io, @sprintf("* ORBITAL    1%5d", orbital))
        _write_inporb_values(io, view(coefficients, orbital, :))
    end
    println(io, "#OCC")
    println(io, "* OCCUPATION NUMBERS")
    _write_inporb_values(io, occupations)
    println(io, "#OCHR")
    println(io, "* OCCUPATION NUMBERS (HUMAN-READABLE)")
    _write_inporb_values(io, readable_occupations; columns = 10, width = 12, precision = 4)
    if !isempty(one_electron)
        println(io, "#ONE")
        println(io, "* ONE ELECTRON ENERGIES")
        _write_inporb_values(io, one_electron; columns = 10, width = 12, precision = 4)
    end
    _write_inporb_index(io, data.index_labels)
    return nothing
end

function write_inporb(
    path::AbstractString,
    data::InporbData,
    coefficients::AbstractMatrix{<:Real} = data.coeff;
    kwargs...,
)
    open(path, "w") do io
        write_inporb(io, data, coefficients; kwargs...)
    end
    return abspath(path)
end

end # module InporbIO
