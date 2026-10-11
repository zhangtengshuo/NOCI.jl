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

 File:            src/integrals/loaders.jl
 Purpose:         Conventional .one/.two and CHO factor-bundle loaders with validation and weight removal.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    ConventionalIntegralLayout

Validated layout metadata carried by a conventional `.one` file for its companion fixed-record `.two` files.
Counts use host integers, vectors are owned copies, and no numerical integral payload is stored here.
"""
struct ConventionalIntegralLayout
    titles::NTuple{2,String}
    one_electron_integral_count::Int
    n_orbitals::Int
    two_electron_buffer_length::Int
    two_electron_label_mode::Int
    raw_integral_counts::NTuple{3,Int}
    two_electron_integral_count::Int
    two_electron_file_record_counts::Vector{Int}
    two_electron_file_integral_counts::Vector{Int}

    """
        ConventionalIntegralLayout(titles, one_count, n_orbitals, buffer_length, label_mode, raw_counts, two_count, file_records, file_integrals)

    Construct validated conventional file metadata and copy all string/vector storage.
    Nonpositive dimensions, negative counts, inconsistent file arrays, or an inconsistent total integral count raise `ArgumentError` or `DimensionMismatch`.
    """
    function ConventionalIntegralLayout(
        titles::NTuple{2,<:AbstractString},
        one_electron_integral_count::Integer,
        n_orbitals::Integer,
        two_electron_buffer_length::Integer,
        two_electron_label_mode::Integer,
        raw_integral_counts::NTuple{3,<:Integer},
        two_electron_integral_count::Integer,
        two_electron_file_record_counts::AbstractVector{<:Integer},
        two_electron_file_integral_counts::AbstractVector{<:Integer},
    )
        n = Int(n_orbitals)
        n >= 1 || throw(ArgumentError("n_orbitals must be positive"))
        one_count = Int(one_electron_integral_count)
        one_count == canonical_pair_count(n) || throw(ArgumentError("one-electron packed count $one_count is inconsistent with $n orbitals"))
        buffer_length = Int(two_electron_buffer_length)
        buffer_length >= 1 || throw(ArgumentError("two-electron buffer length must be positive"))
        two_count = Int(two_electron_integral_count)
        two_count >= 0 || throw(ArgumentError("two-electron integral count must be nonnegative"))
        file_records = Int.(two_electron_file_record_counts)
        file_integrals = Int.(two_electron_file_integral_counts)
        length(file_records) == length(file_integrals) || throw(DimensionMismatch("two-electron file record/integral count arrays differ in length"))
        all(>=(0), file_records) || throw(ArgumentError("two-electron file record counts must be nonnegative"))
        all(>=(0), file_integrals) || throw(ArgumentError("two-electron file integral counts must be nonnegative"))
        sum(file_integrals) == two_count || throw(ArgumentError("per-file integral counts do not sum to $two_count"))
        raw_counts = Tuple(Int(value) for value in raw_integral_counts)
        return new(
            (String(titles[1]), String(titles[2])),
            one_count,
            n,
            buffer_length,
            Int(two_electron_label_mode),
            raw_counts,
            two_count,
            file_records,
            file_integrals,
        )
    end
end

"""
    OneElectronValues

Physical one-electron values decoded from `.one`: nuclear repulsion, overlap, kinetic, nuclear-attraction, and nine multipole matrices.
All matrices use full orbital dimensions and contain no packed-triangle multiplicity weights.
"""
struct OneElectronValues{T<:AbstractFloat}
    nuclear_repulsion::T
    overlap::Matrix{T}
    kinetic::Matrix{T}
    nuclear_attraction::Matrix{T}
    multipoles::Array{T,3}

    """
        OneElectronValues(nuclear_repulsion, overlap, kinetic, nuclear_attraction, multipoles)

    Construct validated physical one-electron values and copy all array inputs.
    Matrices must share a positive square orbital shape, multipoles must have shape `(n_orbitals,n_orbitals,9)`, and all values must be finite.
    """
    function OneElectronValues(
        nuclear_repulsion::T,
        overlap::AbstractMatrix{T},
        kinetic::AbstractMatrix{T},
        nuclear_attraction::AbstractMatrix{T},
        multipoles::AbstractArray{T,3},
    ) where {T<:AbstractFloat}
        n = size(overlap, 1)
        n >= 1 && size(overlap, 2) == n || throw(DimensionMismatch("overlap must be a positive square matrix"))
        size(kinetic) == (n, n) || throw(DimensionMismatch("kinetic matrix dimension mismatch"))
        size(nuclear_attraction) == (n, n) || throw(DimensionMismatch("nuclear-attraction matrix dimension mismatch"))
        size(multipoles) == (n, n, 9) || throw(DimensionMismatch("multipoles must have shape (n_orbitals,n_orbitals,9)"))
        isfinite(nuclear_repulsion) || throw(ArgumentError("nuclear_repulsion is not finite"))
        all(isfinite, overlap) || throw(ArgumentError("overlap contains a non-finite value"))
        all(isfinite, kinetic) || throw(ArgumentError("kinetic contains a non-finite value"))
        all(isfinite, nuclear_attraction) || throw(ArgumentError("nuclear_attraction contains a non-finite value"))
        all(isfinite, multipoles) || throw(ArgumentError("multipoles contain a non-finite value"))
        return new{T}(
            nuclear_repulsion,
            Matrix{T}(overlap),
            Matrix{T}(kinetic),
            Matrix{T}(nuclear_attraction),
            Array{T,3}(multipoles),
        )
    end
end

"""
    OneElectronFileData

Pair a validated conventional file layout with physical one-electron values decoded from the same `.one` file.
The two fields own their numerical and metadata storage and contain no open file handles.
"""
struct OneElectronFileData{T<:AbstractFloat}
    layout::ConventionalIntegralLayout
    values::OneElectronValues{T}

    """
        OneElectronFileData(layout, values)

    Pair conventional layout and physical values after verifying their orbital dimensions agree.
    Inputs are immutable containers and are retained without additional numerical copies.
    """
    function OneElectronFileData(
        layout::ConventionalIntegralLayout,
        values::OneElectronValues{T},
    ) where {T<:AbstractFloat}
        size(values.overlap, 1) == layout.n_orbitals || throw(DimensionMismatch("one-electron layout/value orbital dimensions differ"))
        return new{T}(layout, values)
    end
end

"""
    ERIFactorBundle

Directly loadable versioned ERI factor bundle with scientific factors and provenance metadata.
The factor matrix uses canonical-pair rows and auxiliary-vector columns.
"""
struct ERIFactorBundle{T<:AbstractFloat}
    version::Int
    factors::ERIFactorValues{T}
    source_kind::String
    motra_nodes::Int
    provenance::ERIFactorProvenance

    """
        ERIFactorBundle(version, factors, source_kind, motra_nodes, provenance)

    Construct a validated version-1 or version-2 factor bundle and copy its provenance string.
    Unsupported versions or negative MOTRA node counts raise `ArgumentError`; scientific factors are retained without another copy.
    """
    function ERIFactorBundle(
        version::Integer,
        factors::ERIFactorValues{T},
        source_kind::AbstractString,
        motra_nodes::Integer,
        provenance::ERIFactorProvenance=ERIFactorProvenance(),
    ) where {T<:AbstractFloat}
        version in (1, 2) || throw(ArgumentError("only ERI factor bundle versions 1 and 2 are supported"))
        motra_nodes >= 0 || throw(ArgumentError("motra_nodes must be nonnegative"))
        return new{T}(Int(version), factors, String(source_kind), Int(motra_nodes), provenance)
    end
end

"""
    one_electron_file_memory(T, n_orbitals)

Estimate peak host numerical payload while decoding and copying a physical `.one` file; device bytes are zero.
The estimate includes twelve full operator matrices, their validated owned copies, and one packed Float64/conversion workspace but excludes object headers and file-library internals.
"""
function one_electron_file_memory(::Type{T}, n_orbitals::Integer) where {T<:AbstractFloat}
    n = Int(n_orbitals)
    n >= 1 || throw(ArgumentError("n_orbitals must be positive"))
    matrix_payload = estimate_array_bytes(T, 24, n, n)
    n_pairs = canonical_pair_count(n)
    record_payload = Base.Checked.checked_add(
        estimate_array_bytes(Float64, n_pairs),
        estimate_array_bytes(T, n_pairs),
    )
    return MemoryEstimate(Base.Checked.checked_add(matrix_payload, record_payload), 0)
end

"""
    packed_two_eri_memory(T, one_file)

Estimate peak host numerical payload for physical `.two` decoding, including source/output value vectors, one Float64 fixed-record buffer, and diagonal-pair flags.
The estimate excludes input metadata, path strings and Julia object headers; device bytes are zero.
"""
function packed_two_eri_memory(::Type{T}, one_file::OneElectronFileData) where {T<:AbstractFloat}
    layout = one_file.layout
    values_bytes = estimate_array_bytes(T, 2, layout.two_electron_integral_count)
    record_bytes = estimate_array_bytes(Float64, Base.Checked.checked_add(layout.two_electron_buffer_length, 2))
    flag_bytes = div(Base.Checked.checked_add(canonical_pair_count(layout.n_orbitals), 7), 8)
    return MemoryEstimate(Base.Checked.checked_add(Base.Checked.checked_add(values_bytes, record_bytes), flag_bytes), 0)
end

"""
    eri_factor_bundle_memory(T, n_orbitals, n_auxiliary; version=1)

Estimate peak host numerical payload for a versioned JLD2 ERI factor load and direct canonical-layout construction; device bytes are zero.
Version 1 includes the loaded pair-major Float64 vector plus one final matrix, version 2 includes its canonical matrix plus a conversion only when `T != Float64`, both include one label matrix, and JLD2 internal metadata are excluded.
"""
function eri_factor_bundle_memory(
    ::Type{T},
    n_orbitals::Integer,
    n_auxiliary::Integer,
    ;
    version::Integer=1,
) where {T<:AbstractFloat}
    n = Int(n_orbitals)
    n_aux = Int(n_auxiliary)
    n >= 1 || throw(ArgumentError("n_orbitals must be positive"))
    n_aux >= 1 || throw(ArgumentError("n_auxiliary must be positive"))
    version in (1, 2) || throw(ArgumentError("unsupported ERI factor bundle version $version"))
    n_coefficients = Base.Checked.checked_mul(canonical_pair_count(n), n_aux)
    raw_bytes = estimate_array_bytes(Float64, n_coefficients)
    output_bytes = version == 1 || T != Float64 ? estimate_array_bytes(T, n_coefficients) : 0
    label_bytes = estimate_array_bytes(Int16, 2, canonical_pair_count(n))
    return MemoryEstimate(
        Base.Checked.checked_add(Base.Checked.checked_add(raw_bytes, output_bytes), label_bytes),
        0,
    )
end

"""
    _unpack_symmetric_values(values, n_orbitals)

Expand physical upper-triangular values in canonical column-major pair order into a full symmetric matrix.
The input is not mutated; invalid packed length raises `DimensionMismatch` before output is returned.
"""
function _unpack_symmetric_values(values::AbstractVector{T}, n_orbitals::Integer) where {T<:AbstractFloat}
    n = Int(n_orbitals)
    length(values) == canonical_pair_count(n) || throw(DimensionMismatch("invalid packed symmetric value count"))
    matrix = Matrix{T}(undef, n, n)
    @inbounds for upper in 1:n, lower in 1:upper
        value = values[canonical_pair_index(lower, upper)]
        matrix[lower, upper] = value
        matrix[upper, lower] = value
    end
    return matrix
end

"""
    _read_physical_symmetric_record(io, marker_type, n_orbitals, label, T)

Read one Float64 Fortran record, validate its canonical packed length and finite values, convert to `T`, and return a full symmetric matrix.
The stream advances once and malformed records raise `ArgumentError`, `DimensionMismatch`, or `EOFError`.
"""
function _read_physical_symmetric_record(
    io::IO,
    marker_type::Type{<:Integer},
    n_orbitals::Int,
    label::AbstractString,
    ::Type{T},
) where {T<:AbstractFloat}
    packed_float64 = _reinterpret_record_values(Float64, read_fortran_record(io, marker_type))
    length(packed_float64) == canonical_pair_count(n_orbitals) || throw(DimensionMismatch("$label record length mismatch"))
    all(isfinite, packed_float64) || throw(ArgumentError("$label record contains a non-finite value"))
    return _unpack_symmetric_values(T.(packed_float64), n_orbitals)
end

"""
    read_one_electron_file(path; T=Float64, runtime=RuntimeOptions())

Read a native-endian conventional `.one` file into validated layout metadata and full physical one-electron matrices.
The operation is CPU/read-only, accepts 32- or 64-bit Fortran record markers, performs no CUDA work, and rejects inconsistent records or trailing bytes.
"""
function read_one_electron_file(
    path::AbstractString;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=RuntimeOptions(),
) where {S<:AbstractFloat}
    marker_type = detect_fortran_record_marker(path)
    return open(path, "r") do io
        header = read_fortran_record(io, marker_type)
        length(header) >= 200 || throw(ArgumentError("invalid one-electron header record in $path"))
        titles = (_decode_fortran_string(view(header, 1:80)), _decode_fortran_string(view(header, 81:160)))
        one_count = Int(_reinterpret_record_scalar(Int64, view(header, 161:168)))
        nuclear_repulsion = S(_reinterpret_record_scalar(Float64, view(header, 169:176)))
        n_orbitals = Int(_reinterpret_record_scalar(Int64, view(header, 177:184)))
        buffer_length = Int(_reinterpret_record_scalar(Int64, view(header, 185:192)))
        label_mode = Int(_reinterpret_record_scalar(Int64, view(header, 193:200)))
        enforce_memory_limits(one_electron_file_memory(S, n_orbitals), runtime)
        overlap = _read_physical_symmetric_record(io, marker_type, n_orbitals, "overlap", S)
        kinetic = _read_physical_symmetric_record(io, marker_type, n_orbitals, "kinetic", S)
        nuclear_attraction = _read_physical_symmetric_record(io, marker_type, n_orbitals, "nuclear-attraction", S)
        multipoles = Array{S,3}(undef, n_orbitals, n_orbitals, 9)
        for component in 1:9
            multipoles[:, :, component] .= _read_physical_symmetric_record(
                io,
                marker_type,
                n_orbitals,
                "multipole $component",
                S,
            )
        end
        raw_count_values = _reinterpret_record_values(Int64, read_fortran_record(io, marker_type))
        length(raw_count_values) == 3 || throw(DimensionMismatch("expected three raw integral counts"))
        raw_counts = Tuple(Int(value) for value in raw_count_values)
        two_count = raw_counts[2] == -1 ?
                    Int(_reinterpret_record_scalar(Int64, read_fortran_record(io, marker_type))) : raw_counts[2]
        n_files = Int(_reinterpret_record_scalar(Int64, read_fortran_record(io, marker_type)))
        n_files >= 0 || throw(ArgumentError("negative two-electron file count"))
        file_records = Int.(_reinterpret_record_values(Int64, read_fortran_record(io, marker_type)))
        file_integrals = Int.(_reinterpret_record_values(Int64, read_fortran_record(io, marker_type)))
        length(file_records) == n_files || throw(DimensionMismatch("two-electron file record-count length mismatch"))
        length(file_integrals) == n_files || throw(DimensionMismatch("two-electron file integral-count length mismatch"))
        eof(io) || throw(ArgumentError("unexpected trailing data in one-electron file $path"))
        layout = ConventionalIntegralLayout(
            titles,
            one_count,
            n_orbitals,
            buffer_length,
            label_mode,
            raw_counts,
            two_count,
            file_records,
            file_integrals,
        )
        values = OneElectronValues(nuclear_repulsion, overlap, kinetic, nuclear_attraction, multipoles)
        return OneElectronFileData(layout, values)
    end
end

"""
    _canonical_pair_diagonal_flags(n_orbitals)

Return flags identifying diagonal orbital pairs in canonical pair order.
The host allocation contains one Boolean per canonical pair and no integral values.
"""
function _canonical_pair_diagonal_flags(n_orbitals::Integer)
    flags = falses(canonical_pair_count(n_orbitals))
    for orbital in 1:Int(n_orbitals)
        flags[canonical_pair_index(orbital, orbital)] = true
    end
    return flags
end

"""
    _read_packed_two_file!(values, pair_is_diagonal, n_pairs, first_pair, second_pair, path, n_records, n_integrals, buffer_length)

Decode one fixed-record `.two` file into preallocated physical pair-of-pairs storage and return the next row-major file position.
The function mutates only `values`, closes the input stream on success or failure, performs no device work, and rejects truncated or trailing file content.
"""
function _read_packed_two_file!(
    values::Vector{T},
    pair_is_diagonal::BitVector,
    n_pairs::Int,
    first_pair::Int,
    second_pair::Int,
    path::AbstractString,
    n_records::Int,
    n_integrals::Int,
    buffer_length::Int,
) where {T<:AbstractFloat}
    record_words = Base.Checked.checked_add(buffer_length, 2)
    record = Vector{Float64}(undef, record_words)
    remaining = n_integrals
    io = open(path, "r")
    try
        for _ in 1:n_records
            read!(io, record)
            take = min(buffer_length, remaining)
            @inbounds for record_index in 1:take
                first_scale = pair_is_diagonal[first_pair] ? 0.5 : 1.0
                second_scale = pair_is_diagonal[second_pair] ? 0.5 : 1.0
                pair_scale = first_pair == second_pair ? 0.5 : 1.0
                physical_value = T(record[record_index] / (first_scale * second_scale * pair_scale))
                values[canonical_pair_index(first_pair, second_pair)] = physical_value
                second_pair += 1
                if second_pair > n_pairs
                    first_pair += 1
                    second_pair = first_pair
                end
            end
            remaining -= take
        end
        eof(io) || throw(ArgumentError("unexpected trailing bytes in two-electron file $path"))
    finally
        close(io)
    end
    remaining == 0 || throw(ArgumentError("two-electron file $path ended before its declared integral count"))
    return first_pair, second_pair
end

"""
    read_packed_two_eri(one_file, paths; T=Float64, runtime=RuntimeOptions())

Read conventional fixed-record `.two` files declared by `one_file`, remove all raw symmetry-folding weights, and reorder values into physical canonical column-major pair-of-pairs storage.
The loader is sequential CPU/read-only, ignores the two non-integral trailer words in each fixed record, performs no CUDA work, and rejects label modes, counts, paths, sizes, or non-finite values that violate the `.one` contract.
"""
function read_packed_two_eri(
    one_file::OneElectronFileData,
    paths::AbstractVector{<:AbstractString};
    T::Type{S}=Float64,
    runtime::RuntimeOptions=RuntimeOptions(),
) where {S<:AbstractFloat}
    layout = one_file.layout
    layout.two_electron_label_mode == 1 || throw(ArgumentError("only canonical two-electron label mode 1 is supported"))
    length(paths) == length(layout.two_electron_file_record_counts) || throw(DimensionMismatch("two-electron file path count mismatch"))
    n_orbitals = layout.n_orbitals
    n_pairs = canonical_pair_count(n_orbitals)
    expected_integrals = canonical_pair_count(n_pairs)
    layout.two_electron_integral_count == expected_integrals || throw(ArgumentError("two-electron integral count is not canonical for $n_orbitals orbitals"))
    enforce_memory_limits(packed_two_eri_memory(S, one_file), runtime)
    values = Vector{S}(undef, expected_integrals)
    pair_is_diagonal = _canonical_pair_diagonal_flags(n_orbitals)
    first_pair = 1
    second_pair = 1
    for file_index in eachindex(paths)
        path = paths[file_index]
        isfile(path) || throw(ArgumentError("two-electron file not found: $path"))
        n_records = layout.two_electron_file_record_counts[file_index]
        n_integrals = layout.two_electron_file_integral_counts[file_index]
        record_words = Base.Checked.checked_add(layout.two_electron_buffer_length, 2)
        record_bytes = Base.Checked.checked_mul(record_words, sizeof(Float64))
        expected_bytes = Base.Checked.checked_mul(n_records, record_bytes)
        filesize(path) == expected_bytes || throw(ArgumentError("unexpected fixed-record file size for $path"))
        first_pair, second_pair = _read_packed_two_file!(
            values,
            pair_is_diagonal,
            n_pairs,
            first_pair,
            second_pair,
            path,
            n_records,
            n_integrals,
            layout.two_electron_buffer_length,
        )
    end
    first_pair == n_pairs + 1 || throw(ArgumentError("two-electron files did not fill the canonical pair-of-pairs domain"))
    return PackedERIValues(n_orbitals, values)
end

"""
    read_packed_two_eri(one_file, path; T=Float64, runtime=RuntimeOptions())

Read one conventional `.two` path using the same validated physical decoding as the multi-file method.
The path is wrapped without mutation and all file/count restrictions are enforced by the primary loader.
"""
function read_packed_two_eri(
    one_file::OneElectronFileData,
    path::AbstractString;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=RuntimeOptions(),
) where {S<:AbstractFloat}
    return read_packed_two_eri(one_file, [String(path)]; T=S, runtime=runtime)
end

"""
    _required_bundle_value(data, key, path)

Return a required string-keyed ERI factor bundle field or raise `ArgumentError` naming the missing key and file.
The source object is not mutated and the returned value retains its loaded Julia type.
"""
function _required_bundle_value(data, key::AbstractString, path::AbstractString)
    haskey(data, key) || throw(ArgumentError("ERI factor bundle $path is missing key $key"))
    return data[key]
end

"""
    _convert_pair_major_eri_factors(pair_major, n_pairs, n_auxiliary, T)

Convert version-1 auxiliary-fast pair-major coefficients directly into canonical pair-row matrix storage while validating finiteness in the same threaded pass.
The source is read-only, workers own disjoint pair rows, exactly one molecular-scale output is allocated, CUDA is not touched, and non-finite values raise `ArgumentError` after all tasks join.
"""
function _convert_pair_major_eri_factors(
    pair_major::Vector{Float64},
    n_pairs::Int,
    n_auxiliary::Int,
    ::Type{T},
) where {T<:AbstractFloat}
    coefficients = Matrix{T}(undef, n_pairs, n_auxiliary)
    valid = fill(true, min(Threads.nthreads(), n_pairs))
    ranges = _balanced_auxiliary_ranges(n_pairs, length(valid))
    tasks = map(enumerate(ranges)) do (worker, pair_range)
        Threads.@spawn begin
            local_valid = true
            @inbounds for pair in pair_range, auxiliary in 1:n_auxiliary
                value = pair_major[auxiliary + (pair - 1) * n_auxiliary]
                local_valid &= isfinite(value)
                coefficients[pair, auxiliary] = T(value)
            end
            valid[worker] = local_valid
            nothing
        end
    end
    fetch.(tasks)
    all(valid) || throw(ArgumentError("ERI factor bundle coefficients contain a non-finite value"))
    return coefficients
end

"""
    write_eri_factor_bundle_v2(path, bundle)

Atomically write one loaded ERI factor bundle in canonical pair-row version-2 layout for copy-free Float64 production loading.
The bundle is read-only, the destination is replaced atomically, canonical labels and validation metadata are generated on CPU, CUDA is not touched, and filesystem errors preserve any previous complete file.
"""
function write_eri_factor_bundle_v2(path::AbstractString, bundle::ERIFactorBundle{T}) where {T<:AbstractFloat}
    factors = bundle.factors
    n_orbitals = factors.n_orbitals
    n_pairs, n_auxiliary = size(factors.coefficients)
    labels = Matrix{Int16}(undef, 2, n_pairs)
    pair = 0
    @inbounds for upper in 1:n_orbitals, lower in 1:upper
        pair += 1
        labels[1, pair] = Int16(lower)
        labels[2, pair] = Int16(upper)
    end
    values = Dict{String,Any}(
        "version" => 2,
        "nbas" => n_orbitals,
        "npairs" => n_pairs,
        "naux" => n_auxiliary,
        "labels" => labels,
        "coefficients" => Matrix{Float64}(factors.coefficients),
        "layout" => "canonical-pair-rows_auxiliary-columns",
        "coefficients_validated" => true,
        "source_kind" => bundle.source_kind,
        "motra_nodes" => bundle.motra_nodes,
        "factor_provenance" => eri_factor_provenance_dict(bundle.provenance),
    )
    return _atomic_jld2_write(path, values)
end

"""
    read_eri_factor_bundle(path; T=Float64, runtime=RuntimeOptions())

Read version-1 pair-major or version-2 canonical-layout JLD2 metric-whitened ERI factors, validate labels/provenance, and return pair-by-auxiliary physical factors.
The loader is CPU/read-only, version 1 performs one fused threaded conversion/finite pass, version 2 transfers validated Float64 matrix storage without a molecular-scale copy, CUDA is not initialized, and malformed contracts raise before return.
"""
function read_eri_factor_bundle(
    path::AbstractString;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=RuntimeOptions(),
) where {S<:AbstractFloat}
    isfile(path) || throw(ArgumentError("ERI factor bundle not found: $path"))
    return JLD2.jldopen(path, "r") do data
        provenance = haskey(data, "factor_provenance") ? _read_eri_factor_provenance(data["factor_provenance"]) : ERIFactorProvenance()
        version = Int(_required_bundle_value(data, "version", path))
        version in (1, 2) || throw(ArgumentError("unsupported ERI factor bundle version $version in $path"))
        n_orbitals = Int(_required_bundle_value(data, "nbas", path))
        n_pairs = Int(_required_bundle_value(data, "npairs", path))
        n_auxiliary = Int(_required_bundle_value(data, "naux", path))
        n_orbitals >= 1 || throw(ArgumentError("ERI factor bundle orbital count must be positive"))
        n_pairs == canonical_pair_count(n_orbitals) || throw(DimensionMismatch("ERI factor bundle pair count is not canonical"))
        n_auxiliary >= 1 || throw(ArgumentError("ERI factor bundle requires at least one auxiliary vector"))
        enforce_memory_limits(eri_factor_bundle_memory(S, n_orbitals, n_auxiliary; version), runtime)
        labels = Matrix{Int16}(_required_bundle_value(data, "labels", path))
        size(labels) == (2, n_pairs) || throw(DimensionMismatch("ERI factor bundle label matrix shape mismatch"))
        pair_index = 0
        for upper in 1:n_orbitals, lower in 1:upper
            pair_index += 1
            labels[1, pair_index] == lower && labels[2, pair_index] == upper || throw(ArgumentError("ERI factor bundle canonical-pair labels or ordering are invalid"))
        end
        coefficients = if version == 1
            loaded_pair_major = _required_bundle_value(data, "coeff", path)
            loaded_pair_major isa Vector{Float64} || throw(ArgumentError("version-1 ERI factor bundle coeff must be a Vector{Float64}"))
            pair_major = loaded_pair_major::Vector{Float64}
            length(pair_major) == Base.Checked.checked_mul(n_pairs, n_auxiliary) || throw(DimensionMismatch("ERI factor bundle coefficient length mismatch"))
            _convert_pair_major_eri_factors(pair_major, n_pairs, n_auxiliary, S)
        else
            String(_required_bundle_value(data, "layout", path)) == "canonical-pair-rows_auxiliary-columns" || throw(ArgumentError("version-2 ERI factor bundle layout is invalid"))
            _required_bundle_value(data, "coefficients_validated", path) === true || throw(ArgumentError("version-2 ERI factor bundle lacks validated coefficient metadata"))
            loaded = _required_bundle_value(data, "coefficients", path)
            loaded isa Matrix{Float64} || throw(ArgumentError("version-2 ERI factor bundle coefficients must be a Matrix{Float64}"))
            size(loaded) == (n_pairs, n_auxiliary) || throw(DimensionMismatch("ERI factor bundle coefficient matrix shape mismatch"))
            S == Float64 ? loaded : Matrix{S}(loaded)
        end
        source_kind = String(haskey(data, "source_kind") ? data["source_kind"] : "unknown")
        motra_nodes = Int(haskey(data, "motra_nodes") ? data["motra_nodes"] : 0)
        motra_nodes >= 0 || throw(ArgumentError("ERI factor bundle motra_nodes must be nonnegative"))
        return ERIFactorBundle(version, ERIFactorValues{S}(n_orbitals, coefficients, Val(:validated_owned)), source_kind, motra_nodes, provenance)
    end
end

"""
    core_hamiltonian(one_electron)

Return a newly allocated physical one-electron core Hamiltonian as kinetic plus nuclear attraction.
The input is not mutated, units follow the source `.one` file, and no device memory is allocated.
"""
function core_hamiltonian(one_electron::OneElectronValues)
    return one_electron.kinetic + one_electron.nuclear_attraction
end
