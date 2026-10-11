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

 File:            src/core/wavefunctions.jl
 Purpose:         Fragment wavefunctions with packed active occupations and cross-overlap block preparation.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    PackedActiveOccupations

Store active alpha and beta occupations as one `UInt64` bit mask per determinant while presenting a read-only matrix of canonical `Int8` occupation values.
Rows are dimensionless active-orbital indices, columns are determinants, storage is CPU-owned, active domains are limited to 64 orbitals, and no CUDA allocation is retained.
"""
struct PackedActiveOccupations <: AbstractMatrix{Int8}
    n_active_orbitals::Int
    alpha_bits::Vector{UInt64}
    beta_bits::Vector{UInt64}

    """
        PackedActiveOccupations(n_active_orbitals, alpha_bits, beta_bits; copy_data=true)

    Construct validated packed active occupations from equal-length alpha/beta bit-mask vectors.
    Inputs are copied by default, `copy_data=false` transfers already-owned `Vector{UInt64}` storage for internal builders, bits outside the active domain are rejected, and no CUDA work occurs.
    """
    function PackedActiveOccupations(
        n_active_orbitals::Integer,
        alpha_bits::AbstractVector{UInt64},
        beta_bits::AbstractVector{UInt64},
        ;
        copy_data::Bool=true,
    )
        rank = Int(n_active_orbitals)
        0 <= rank <= 64 || throw(ArgumentError("packed active occupations support 0:64 active orbitals"))
        length(alpha_bits) == length(beta_bits) || throw(DimensionMismatch("packed alpha/beta determinant counts differ"))
        valid_mask = rank == 64 ? typemax(UInt64) : (UInt64(1) << rank) - UInt64(1)
        all(bits -> bits & ~valid_mask == 0, alpha_bits) || throw(ArgumentError("packed alpha occupations contain bits outside the active domain"))
        all(bits -> bits & ~valid_mask == 0, beta_bits) || throw(ArgumentError("packed beta occupations contain bits outside the active domain"))
        if copy_data
            return new(rank, collect(alpha_bits), collect(beta_bits))
        end
        alpha_bits isa Vector{UInt64} && beta_bits isa Vector{UInt64} || throw(ArgumentError("copy_data=false requires owned Vector{UInt64} inputs"))
        return new(rank, alpha_bits, beta_bits)
    end
end

"""
    PackedActiveOccupations(active_occupations)

Pack a dense or abstract active-occupation matrix into alpha and beta `UInt64` bit planes.
The input is read-only, returned vectors are CPU-owned, no CUDA work occurs, and ranks above 64 or values outside `-1,0,1,2` raise.
"""
function PackedActiveOccupations(active_occupations::AbstractMatrix{<:Integer})
    n_active, n_determinants = size(active_occupations)
    n_active <= 64 || throw(ArgumentError("packed active occupations support at most 64 active orbitals"))
    alpha_bits = zeros(UInt64, n_determinants)
    beta_bits = zeros(UInt64, n_determinants)
    @inbounds for determinant in 1:n_determinants, active in 1:n_active
        value = active_occupations[active, determinant]
        value in (-1, 0, 1, 2) || throw(ArgumentError("active occupations must use only -1, 0, 1, or 2"))
        bit = UInt64(1) << (active - 1)
        value in (1, 2) && (alpha_bits[determinant] |= bit)
        value in (-1, 2) && (beta_bits[determinant] |= bit)
    end
    return PackedActiveOccupations(n_active, alpha_bits, beta_bits)
end

"""
    PackedActiveOccupations(active_occupations::PackedActiveOccupations)

Copy packed alpha/beta occupation planes without decoding their logical matrix entries.
The source is read-only, returned CPU vectors are owned, units are dimensionless, no CUDA work occurs, and validated source invariants are preserved.
"""
function PackedActiveOccupations(active_occupations::PackedActiveOccupations)
    return PackedActiveOccupations(
        active_occupations.n_active_orbitals,
        active_occupations.alpha_bits,
        active_occupations.beta_bits,
    )
end

"""
    size(occupations::PackedActiveOccupations)

Return active-orbital rows and determinant columns for packed occupation storage.
The query allocates nothing, mutates no input, and has no CUDA side effects.
"""
Base.size(occupations::PackedActiveOccupations) = (occupations.n_active_orbitals, length(occupations.alpha_bits))

"""
    getindex(occupations::PackedActiveOccupations, active, determinant)

Decode one canonical active occupation value from packed alpha/beta bit masks.
The input is read-only, the return value is dimensionless `Int8`, no allocation or CUDA work occurs, and invalid indices raise `BoundsError`.
"""
@inline function Base.getindex(occupations::PackedActiveOccupations, active::Int, determinant::Int)
    @boundscheck checkbounds(occupations, active, determinant)
    bit = UInt64(1) << (active - 1)
    alpha = !iszero(occupations.alpha_bits[determinant] & bit)
    beta = !iszero(occupations.beta_bits[determinant] & bit)
    return alpha ? (beta ? Int8(2) : Int8(1)) : (beta ? Int8(-1) : Int8(0))
end

"""
    getindex(occupations::PackedActiveOccupations, :, determinants)

Return a packed determinant-column subset without expanding active occupations.
The source is read-only, selected bit vectors are copied on CPU, no CUDA work occurs, and invalid determinant indices raise `BoundsError`.
"""
function Base.getindex(
    occupations::PackedActiveOccupations,
    ::Colon,
    determinants::AbstractVector{<:Integer},
)
    indices = Int.(determinants)
    checkbounds(occupations.alpha_bits, indices)
    return PackedActiveOccupations(
        occupations.n_active_orbitals,
        occupations.alpha_bits[indices],
        occupations.beta_bits[indices];
        copy_data=false,
    )
end

"""
    IndexStyle(PackedActiveOccupations)

Declare Cartesian indexing for the read-only packed occupation matrix interface.
The trait query allocates nothing, mutates no input, and has no CUDA side effects.
"""
Base.IndexStyle(::Type{PackedActiveOccupations}) = IndexCartesian()

"""
    DeterminantExpansion(coefficients, active_occupations, inactive_orbitals)

Store one fragment CI expansion with columns as determinants and active occupation codes `2`, `1`, `-1`, and `0` for double, alpha, beta, and empty.
Construction copies inputs, requires at least one determinant, validates finite coefficients and occupation codes, and performs no device allocation.
"""
struct DeterminantExpansion{T<:AbstractFloat}
    coefficients::Vector{T}
    active_occupations::PackedActiveOccupations
    inactive_orbitals::Int

    """
        DeterminantExpansion(coefficients, active_occupations, inactive_orbitals)

    Construct a validated determinant expansion with one coefficient per occupation column.
    Inputs are not mutated; invalid dimensions, empty expansions, negative inactive count, non-finite coefficients, or unsupported occupation values raise before return.
    """
    function DeterminantExpansion(
        coefficients::AbstractVector{T},
        active_occupations::AbstractMatrix{<:Integer},
        inactive_orbitals::Integer,
    ) where {T<:AbstractFloat}
        n_determinants = length(coefficients)
        n_determinants >= 1 || throw(ArgumentError("a determinant expansion must contain at least one determinant"))
        size(active_occupations, 2) == n_determinants || throw(DimensionMismatch("occupation column count must match coefficient count"))
        inactive_orbitals >= 0 || throw(ArgumentError("inactive_orbitals must be nonnegative"))
        all(isfinite, coefficients) || throw(ArgumentError("determinant coefficients contain a non-finite value"))
        all(value -> value in (-1, 0, 1, 2), active_occupations) || throw(ArgumentError("active occupations must use only -1, 0, 1, or 2"))
        return new{T}(Vector{T}(coefficients), PackedActiveOccupations(active_occupations), Int(inactive_orbitals))
    end

    """
        DeterminantExpansion(coefficients, active_occupations, inactive_orbitals, Val(:owned))

    Construct a validated determinant expansion by transferring internal owned coefficient and packed-occupation arrays without another molecular-scale copy.
    Inputs must be concrete CPU-owned vectors produced by a package builder, no CUDA work occurs, and invalid counts, energies, or inactive-orbital metadata raise before return.
    """
    function DeterminantExpansion(
        coefficients::Vector{T},
        active_occupations::PackedActiveOccupations,
        inactive_orbitals::Integer,
        ::Val{:owned},
    ) where {T<:AbstractFloat}
        n_determinants = length(coefficients)
        n_determinants >= 1 || throw(ArgumentError("a determinant expansion must contain at least one determinant"))
        size(active_occupations, 2) == n_determinants || throw(DimensionMismatch("occupation column count must match coefficient count"))
        inactive_orbitals >= 0 || throw(ArgumentError("inactive_orbitals must be nonnegative"))
        all(isfinite, coefficients) || throw(ArgumentError("determinant coefficients contain a non-finite value"))
        return new{T}(coefficients, active_occupations, Int(inactive_orbitals))
    end
end

"""
    DeterminantFileData

Validated scientific header and canonicalized CI expansion decoded from one fragment `.det` file.
Energies are in Hartree, `tau_mo` is dimensionless, and frozen/inactive orbital counts retain their distinct meanings.
"""
struct DeterminantFileData{T<:AbstractFloat}
    label::String
    casscf_energy::T
    caspt2_energy::T
    correlation_energy::T
    n_electrons::Int
    spin_multiplicity::Int
    n_frozen_orbitals::Int
    tau_mo::T
    expansion::DeterminantExpansion{T}

    """
        DeterminantFileData(label, casscf_energy, caspt2_energy, n_electrons, spin_multiplicity, n_frozen_orbitals, tau_mo, expansion)

    Construct validated determinant-file data and derive the CASPT2-minus-CASSCF correlation energy.
    The expansion is retained, the label is copied, and invalid energies, counts, multiplicity, or threshold raise `ArgumentError`.
    """
    function DeterminantFileData(
        label::AbstractString,
        casscf_energy::T,
        caspt2_energy::T,
        n_electrons::Integer,
        spin_multiplicity::Integer,
        n_frozen_orbitals::Integer,
        tau_mo::T,
        expansion::DeterminantExpansion{T},
    ) where {T<:AbstractFloat}
        isempty(strip(label)) && throw(ArgumentError("determinant label must not be empty"))
        isfinite(casscf_energy) || throw(ArgumentError("CASSCF energy is not finite"))
        isfinite(caspt2_energy) || throw(ArgumentError("CASPT2 energy is not finite"))
        n_electrons >= 0 || throw(ArgumentError("n_electrons must be nonnegative"))
        spin_multiplicity >= 1 || throw(ArgumentError("spin_multiplicity must be positive"))
        n_frozen_orbitals >= 0 || throw(ArgumentError("n_frozen_orbitals must be nonnegative"))
        isfinite(tau_mo) && tau_mo >= zero(T) || throw(ArgumentError("tau_mo must be finite and nonnegative"))
        return new{T}(
            String(label),
            casscf_energy,
            caspt2_energy,
            caspt2_energy - casscf_energy,
            Int(n_electrons),
            Int(spin_multiplicity),
            Int(n_frozen_orbitals),
            tau_mo,
            expansion,
        )
    end
end

"""
    OrbitalCoefficients(coefficients)

Store one square common-basis fragment orbital coefficient matrix with basis functions in rows and molecular orbitals in columns.
Construction copies the matrix, validates positive square dimensions and finite values, and performs no GPU work.
"""
struct OrbitalCoefficients{T<:AbstractFloat}
    coefficients::Matrix{T}

    """
        OrbitalCoefficients(coefficients)

    Construct validated square orbital coefficients as an owned matrix.
    The input is not mutated; empty, nonsquare, or non-finite matrices raise before the object is returned.
    """
    function OrbitalCoefficients(coefficients::AbstractMatrix{T}) where {T<:AbstractFloat}
        n_orbitals = size(coefficients, 1)
        n_orbitals >= 1 && size(coefficients, 2) == n_orbitals || throw(DimensionMismatch("orbital coefficients must be a positive square matrix"))
        all(isfinite, coefficients) || throw(ArgumentError("orbital coefficients contain a non-finite value"))
        return new{T}(Matrix{T}(coefficients))
    end
end

"""
    FragmentWavefunction

Pair fragment determinant data with its common-basis orbital coefficients after validating the referenced occupied/active orbital domain.
The two components are retained without additional numerical copies.
"""
struct FragmentWavefunction{T<:AbstractFloat}
    determinants::DeterminantFileData{T}
    orbitals::OrbitalCoefficients{T}

    """
        FragmentWavefunction(determinants, orbitals)

    Construct a fragment wavefunction whose retained inactive plus active orbitals fit inside the orbital coefficient dimension.
    Inputs are retained and inconsistent orbital domains raise `DimensionMismatch`.
    """
    function FragmentWavefunction(
        determinants::DeterminantFileData{T},
        orbitals::OrbitalCoefficients{T},
    ) where {T<:AbstractFloat}
        referenced = determinants.expansion.inactive_orbitals + size(determinants.expansion.active_occupations, 1)
        referenced <= size(orbitals.coefficients, 1) || throw(DimensionMismatch("determinant orbital domain exceeds orbital coefficient dimension"))
        return new{T}(determinants, orbitals)
    end
end

"""
    active_orbital_count(expansion)

Return the number of active occupation rows in a determinant expansion.
The query allocates no storage and does not mutate the expansion.
"""
function active_orbital_count(expansion::DeterminantExpansion)
    return size(expansion.active_occupations, 1)
end

"""
    determinant_count(expansion)

Return the number of determinant columns and coefficients in an expansion.
The query allocates no storage and does not mutate the expansion.
"""
function determinant_count(expansion::DeterminantExpansion)
    return length(expansion.coefficients)
end

"""
    _occupation_value(character)

Map one canonical `.det` occupation character to `Int8`: `2=>2`, `a=>1`, `b=>-1`, and `0=>0`.
Unsupported characters raise `ArgumentError`; the function allocates no array storage.
"""
function _occupation_value(character::Char)
    character == '2' && return Int8(2)
    character == 'a' && return Int8(1)
    character == 'b' && return Int8(-1)
    character == '0' && return Int8(0)
    throw(ArgumentError("unsupported determinant occupation character $(repr(character))"))
end

"""
    _occupation_character(value)

Map one validated active occupation code back to the canonical `.det` character.
Unsupported values raise `ArgumentError`; the function allocates no array storage.
"""
function _occupation_character(value::Integer)
    value == 2 && return '2'
    value == 1 && return 'a'
    value == -1 && return 'b'
    value == 0 && return '0'
    throw(ArgumentError("unsupported active occupation value $value"))
end

"""
    occupation_string(expansion, determinant)

Return the canonical `0/2/a/b` active occupation string for one one-based determinant column.
The expansion is not mutated and an out-of-range determinant raises `BoundsError`.
"""
function occupation_string(expansion::DeterminantExpansion, determinant::Integer)
    1 <= determinant <= determinant_count(expansion) || throw(BoundsError(1:determinant_count(expansion), determinant))
    characters = Vector{Char}(undef, active_orbital_count(expansion))
    @inbounds for orbital in eachindex(characters)
        characters[orbital] = _occupation_character(expansion.active_occupations[orbital, determinant])
    end
    return String(characters)
end

"""
    _occupation_counts(occupation)

Return active electron count and alpha-minus-beta spin projection for one canonical occupation string.
The string is validated through `_occupation_value` and is not mutated.
"""
function _occupation_counts(occupation::AbstractString)
    n_electrons = 0
    spin_projection = 0
    for character in occupation
        value = _occupation_value(character)
        value == 2 && (n_electrons += 2)
        value == 1 && (n_electrons += 1; spin_projection += 1)
        value == -1 && (n_electrons += 1; spin_projection -= 1)
    end
    return n_electrons, spin_projection
end

"""
    _active_occupations(occupation_strings)

Convert equal-length canonical occupation strings into an `Int8` matrix with one determinant per column.
Inputs are not mutated; empty input, inconsistent lengths, or unsupported characters raise `ArgumentError`.
"""
function _active_occupations(occupation_strings::AbstractVector{<:AbstractString})
    isempty(occupation_strings) && throw(ArgumentError("occupation string collection must not be empty"))
    n_active = length(first(occupation_strings))
    occupations = Matrix{Int8}(undef, n_active, length(occupation_strings))
    for (determinant, occupation) in pairs(occupation_strings)
        length(occupation) == n_active || throw(ArgumentError("determinant occupation strings have inconsistent lengths"))
        for orbital in 1:n_active
            occupations[orbital, determinant] = _occupation_value(occupation[orbital])
        end
    end
    return occupations
end

"""
    read_determinant_file(path; T=Float64)

Read, validate, merge duplicate occupations, and lexicographically canonicalize one fragment `.det` expansion.
The operation is CPU/read-only, validates electron/spin sectors and finite values, performs no CUDA work, and raises on malformed headers or record counts.
"""
function read_determinant_file(path::AbstractString; T::Type{S}=Float64) where {S<:AbstractFloat}
    isfile(path) || throw(ArgumentError("determinant file not found: $path"))
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("empty determinant file: $path"))
    header = split(strip(lines[1]))
    length(header) >= 8 || throw(ArgumentError("invalid determinant header in $path"))
    inactive_before_frozen = parse(Int, header[1])
    n_frozen = parse(Int, header[2])
    n_records = parse(Int, header[3])
    label = header[4]
    casscf_energy = parse(S, header[5])
    n_electrons = parse(Int, header[6])
    tau_mo = parse(S, header[7])
    caspt2_energy = parse(S, header[8])
    inactive_before_frozen >= n_frozen >= 0 || throw(ArgumentError("invalid inactive/frozen orbital counts in $path"))
    n_records >= 1 || throw(ArgumentError("determinant record count must be positive in $path"))
    coefficient_by_occupation = Dict{String,S}()
    active_electron_count = nothing
    spin_projection = nothing
    records_read = 0
    for line in @view lines[2:end]
        text = strip(line)
        isempty(text) && continue
        fields = split(text)
        length(fields) >= 2 || throw(ArgumentError("invalid determinant record in $path: $line"))
        coefficient = parse(S, fields[1])
        isfinite(coefficient) || throw(ArgumentError("non-finite determinant coefficient in $path"))
        occupation = String(fields[end])
        electron_count, projection = _occupation_counts(occupation)
        if active_electron_count === nothing
            active_electron_count = electron_count
            spin_projection = projection
        else
            electron_count == active_electron_count || throw(ArgumentError("determinants in $path have inconsistent active electron counts"))
            projection == spin_projection || throw(ArgumentError("determinants in $path have inconsistent alpha/beta sectors"))
        end
        coefficient_by_occupation[occupation] = get(coefficient_by_occupation, occupation, zero(S)) + coefficient
        records_read += 1
    end
    records_read == n_records || throw(ArgumentError("determinant count mismatch in $path: expected $n_records, got $records_read"))
    active_electron_count === nothing && throw(ArgumentError("no determinant records found in $path"))
    n_electrons == 2 * inactive_before_frozen + active_electron_count || throw(ArgumentError("header electron count is inconsistent with determinant occupations in $path"))
    occupation_strings = sort!(collect(keys(coefficient_by_occupation)))
    coefficients = S[coefficient_by_occupation[occupation] for occupation in occupation_strings]
    occupations = _active_occupations(occupation_strings)
    expansion = DeterminantExpansion(coefficients, occupations, inactive_before_frozen - n_frozen)
    multiplicity = abs(something(spin_projection)) + 1
    return DeterminantFileData(
        label,
        casscf_energy,
        caspt2_energy,
        n_electrons,
        multiplicity,
        n_frozen,
        tau_mo,
        expansion,
    )
end

"""
    orbital_coefficient_memory(T, n_orbitals)

Estimate peak host numerical payload for parsed and owned square orbital coefficient matrices; device bytes are zero.
The estimate includes the parsed vector and final owned matrix but excludes source text, token strings, and Julia object headers.
"""
function orbital_coefficient_memory(::Type{T}, n_orbitals::Integer) where {T<:AbstractFloat}
    n = Int(n_orbitals)
    n >= 1 || throw(ArgumentError("n_orbitals must be positive"))
    return MemoryEstimate(estimate_array_bytes(T, 2, n, n), 0)
end

"""
    read_orbital_coefficients(path; T=Float64, runtime=RuntimeOptions())

Read a text `.vec` file whose first token is the square orbital dimension and return validated common-basis coefficients.
The operation is CPU/read-only, enforces numerical output memory before parsing matrix values, performs no CUDA work, and rejects missing, malformed, non-finite, or incorrectly sized files.
"""
function read_orbital_coefficients(
    path::AbstractString;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=RuntimeOptions(),
) where {S<:AbstractFloat}
    isfile(path) || throw(ArgumentError("orbital coefficient file not found: $path"))
    return open(path, "r") do io
        eof(io) && throw(ArgumentError("empty orbital coefficient file: $path"))
        header = split(strip(readline(io)))
        length(header) == 1 || throw(ArgumentError("orbital coefficient header must contain exactly one dimension in $path"))
        n_orbitals = parse(Int, header[1])
        enforce_memory_limits(orbital_coefficient_memory(S, n_orbitals), runtime)
        tokens = split(read(io, String))
        expected = Base.Checked.checked_mul(n_orbitals, n_orbitals)
        length(tokens) == expected || throw(DimensionMismatch("orbital coefficient value count in $path is $(length(tokens)), expected $expected"))
        values = Vector{S}(undef, expected)
        for index in eachindex(tokens)
            values[index] = parse(S, tokens[index])
        end
        return OrbitalCoefficients(reshape(values, n_orbitals, n_orbitals))
    end
end

"""
    read_fragment_wavefunction(det_path, vec_path; T=Float64, runtime=RuntimeOptions())

Read and cross-validate one fragment determinant expansion and common-basis orbital coefficient file.
Both files are read-only, no CUDA state is initialized, and inconsistent scientific dimensions or malformed inputs raise before return.
"""
function read_fragment_wavefunction(
    det_path::AbstractString,
    vec_path::AbstractString;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=RuntimeOptions(),
) where {S<:AbstractFloat}
    determinants = read_determinant_file(det_path; T=S)
    orbitals = read_orbital_coefficients(vec_path; T=S, runtime=runtime)
    return FragmentWavefunction(determinants, orbitals)
end
