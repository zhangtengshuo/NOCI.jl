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

 File:            src/core/mebf.jl
 Purpose:         Two-fragment MEBF construction from spin microstate expansions and Clebsch-Gordan sector paths.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    SpinMicrostateExpansion

Store normalized determinant coefficients and canonical two-bit active-occupation codes grouped from highest to lowest spin projection.
Sector counts, active-orbital count, and multiplicity are dimensionless metadata; the object owns CPU arrays, supports at most 32 active orbitals, and contains no strings or CUDA storage.
"""
struct SpinMicrostateExpansion{T<:AbstractFloat}
    coefficients::Vector{T}
    occupation_codes::Vector{UInt64}
    sector_counts::Vector{Int}
    spin_multiplicity::Int
    n_active_orbitals::Int
end

"""
    PreparedFragmentMicrostates

Cache one fragment's compact spin microstates together with stable descending-magnitude index orders and sector maxima for screened dimer construction.
All arrays are CPU-owned and treated as read-only after construction; values are dimensionless and no CUDA storage is retained.
"""
struct PreparedFragmentMicrostates{T<:AbstractFloat}
    microstates::SpinMicrostateExpansion{T}
    sector_ranges::Vector{UnitRange{Int}}
    sorted_indices::Vector{Vector{Int}}
    maximum_magnitudes::Vector{T}
end

"""
    MolecularBasisFunction

Store one spin-adapted molecular many-electron basis function with block-assembled orbitals, active determinant expansion, and additive fragment correlation-energy shift.
Orbital coefficients and energies use the floating-point type `T`, arrays are CPU-owned, and no integral or CUDA storage is retained.
"""
struct MolecularBasisFunction{T<:AbstractFloat}
    label::String
    fragment_state_indices::Vector{Int}
    orbital_coefficients::Matrix{T}
    expansion::DeterminantExpansion{T}
    correlation_energy_shift::T
end

"""
    _normalized_coefficients(coefficients)

Return a normalized owned copy of determinant coefficients using their Euclidean norm.
The input is not mutated, units are unchanged, no CUDA work occurs, and empty, non-finite, or zero-norm inputs raise `ArgumentError`.
"""
function _normalized_coefficients(coefficients::AbstractVector{T}) where {T<:AbstractFloat}
    isempty(coefficients) && throw(ArgumentError("coefficient vector must not be empty"))
    all(isfinite, coefficients) || throw(ArgumentError("coefficients must be finite"))
    norm_squared = sum(abs2, coefficients)
    isfinite(norm_squared) && norm_squared > zero(T) || throw(ArgumentError("determinant expansion has zero or non-finite norm"))
    return collect(coefficients) ./ sqrt(norm_squared)
end

"""
    _occupation_sort_digit(value)

Map one active occupation value to its canonical two-bit sorting digit `0 < 2 < alpha < beta`.
The dimensionless scalar input is not mutated, no allocation or CUDA work occurs, and unsupported values raise `ArgumentError`.
"""
@inline function _occupation_sort_digit(value::Integer)
    value == 0 && return UInt64(0)
    value == 2 && return UInt64(1)
    value == 1 && return UInt64(2)
    value == -1 && return UInt64(3)
    throw(ArgumentError("unsupported active occupation value $value"))
end

"""
    _occupation_sort_code(expansion, determinant)

Encode one determinant column as a canonical `UInt64` with two bits per active orbital and the first orbital in the most significant occupied field.
The expansion is read-only, the result is dimensionless, no heap or CUDA storage is allocated, and active domains above 32 orbitals or invalid indices raise.
"""
function _occupation_sort_code(expansion::DeterminantExpansion, determinant::Integer)
    n_active = active_orbital_count(expansion)
    n_active <= 32 || throw(ArgumentError("production MEBF construction supports at most 32 active orbitals"))
    1 <= determinant <= determinant_count(expansion) || throw(BoundsError(1:determinant_count(expansion), determinant))
    code = UInt64(0)
    @inbounds for orbital in 1:n_active
        code = (code << 2) | _occupation_sort_digit(expansion.active_occupations[orbital, determinant])
    end
    return code
end

"""
    _occupation_digit_from_code(code, orbital, n_active_orbitals)

Return the canonical two-bit digit for one one-based active orbital in a compact occupation code.
Arguments are dimensionless and read-only, no allocation or CUDA work occurs, and invalid orbital counts or indices raise `ArgumentError` or `BoundsError`.
"""
@inline function _occupation_digit_from_code(code::UInt64, orbital::Integer, n_active_orbitals::Integer)
    0 <= n_active_orbitals <= 32 || throw(ArgumentError("active-orbital count must lie in 0:32"))
    1 <= orbital <= n_active_orbitals || throw(BoundsError(1:n_active_orbitals, orbital))
    shift = 2 * (Int(n_active_orbitals) - Int(orbital))
    return (code >> shift) & UInt64(3)
end

"""
    _replace_occupation_digit(code, orbital, n_active_orbitals, digit)

Return a compact occupation code with one two-bit orbital digit replaced.
Arguments are dimensionless and read-only, no allocation or CUDA work occurs, and invalid indices, counts, or digits raise before modification.
"""
@inline function _replace_occupation_digit(code::UInt64, orbital::Integer, n_active_orbitals::Integer, digit::UInt64)
    digit <= UInt64(3) || throw(ArgumentError("occupation digit must lie in 0:3"))
    shift = 2 * (Int(n_active_orbitals) - Int(orbital))
    _occupation_digit_from_code(code, orbital, n_active_orbitals)
    mask = UInt64(3) << shift
    return (code & ~mask) | (digit << shift)
end

"""
    _spin_microstates(expansion, spin_multiplicity)

Generate normalized spin-projection sectors directly from a highest-projection determinant expansion using compact occupation codes and repeated spin lowering.
The expansion is read-only, returned CPU arrays are owned, no occupation strings or CUDA storage are created, and active domains above 32 orbitals or impossible multiplicities raise `ArgumentError`.
"""
function _spin_microstates(
    expansion::DeterminantExpansion{T},
    spin_multiplicity::Integer,
) where {T<:AbstractFloat}
    spin_multiplicity >= 1 || throw(ArgumentError("spin_multiplicity must be positive"))
    n_active = active_orbital_count(expansion)
    n_active <= 32 || throw(ArgumentError("production MEBF construction supports at most 32 active orbitals"))
    current_coefficients = _normalized_coefficients(expansion.coefficients)
    current_codes = UInt64[_occupation_sort_code(expansion, determinant) for determinant in 1:determinant_count(expansion)]
    sector_counts = zeros(Int, spin_multiplicity)
    sector_counts[1] = length(current_coefficients)
    all_coefficients = copy(current_coefficients)
    all_codes = copy(current_codes)
    for sector in 2:spin_multiplicity
        coefficient_by_code = Dict{UInt64,T}()
        for (coefficient, code) in zip(current_coefficients, current_codes)
            for orbital in 1:n_active
                _occupation_digit_from_code(code, orbital, n_active) == UInt64(2) || continue
                lowered = _replace_occupation_digit(code, orbital, n_active, UInt64(3))
                coefficient_by_code[lowered] = get(coefficient_by_code, lowered, zero(T)) + coefficient
            end
        end
        isempty(coefficient_by_code) && throw(ArgumentError("cannot generate spin sector $sector for multiplicity $spin_multiplicity"))
        current_codes = sort!(collect(keys(coefficient_by_code)))
        current_coefficients = _normalized_coefficients(T[coefficient_by_code[code] for code in current_codes])
        sector_counts[sector] = length(current_coefficients)
        append!(all_coefficients, current_coefficients)
        append!(all_codes, current_codes)
    end
    return SpinMicrostateExpansion(all_coefficients, all_codes, sector_counts, Int(spin_multiplicity), n_active)
end

"""
    _spin_sector_ranges(expansion)

Return one contiguous one-based coefficient range for each stored spin-projection sector.
The expansion is read-only, only the small range vector is allocated on CPU, and no CUDA state is touched.
"""
function _spin_sector_ranges(expansion::SpinMicrostateExpansion)
    ranges = Vector{UnitRange{Int}}(undef, length(expansion.sector_counts))
    first_index = 1
    for sector in eachindex(expansion.sector_counts)
        last_index = first_index + expansion.sector_counts[sector] - 1
        ranges[sector] = first_index:last_index
        first_index = last_index + 1
    end
    return ranges
end

"""
    _factorial_integer(n)

Return exact arbitrary-precision factorial for a nonnegative integer used by the Clebsch-Gordan recurrence.
The function allocates only scalar `BigInt` storage on CPU, touches no device state, and raises `ArgumentError` for negative input.
"""
function _factorial_integer(n::Integer)
    n >= 0 || throw(ArgumentError("factorial input must be nonnegative"))
    result = BigInt(1)
    for value in 2:Int(n)
        result *= value
    end
    return result
end

"""
    _clebsch_recurrence_factor(s, m, s1, s2)

Return the positive recurrence factor for adjacent total-spin Clebsch-Gordan coefficients.
Arguments are dimensionless integer or half-integer spin quantum numbers, no inputs are mutated, no CPU/GPU arrays are allocated, and invalid zero denominators propagate as non-finite values.
"""
function _clebsch_recurrence_factor(s::Float64, m::Float64, s1::Float64, s2::Float64)
    value = (s * s - m * m) * ((s1 + s2 + 1)^2 - s * s) * (s * s - (s1 - s2)^2)
    value /= s * s * (4 * s * s - 1)
    return sqrt(value)
end

"""
    _clebsch_diagonal_factor(s, m, s1, s2, m1, m2)

Return the diagonal factor used by the total-spin Clebsch-Gordan recurrence.
Arguments are dimensionless spin quantum numbers, inputs are read-only, no arrays or CUDA state are allocated, and invalid zero denominators propagate as non-finite values.
"""
function _clebsch_diagonal_factor(s::Float64, m::Float64, s1::Float64, s2::Float64, m1::Float64, m2::Float64)
    return m1 - m2 + m * (s2 * (s2 + 1) - s1 * (s1 + 1)) / (s * (s + 1))
end

"""
    clebsch_gordan_coefficient(s1, s2, m1, m2, s, m)

Return the real Condon-Shortley Clebsch-Gordan coefficient for dimensionless integer or half-integer spin quantum numbers.
Inputs are not mutated, the scalar calculation uses exact `BigInt` factorials and no CPU/GPU arrays, forbidden couplings return zero, and invalid quantum numbers raise `ArgumentError`.
"""
function clebsch_gordan_coefficient(
    s1::Real,
    s2::Real,
    m1::Real,
    m2::Real,
    s::Real,
    m::Real,
)
    values = Float64.((s1, s2, m1, m2, s, m))
    spin1, spin2, projection1, projection2, total_spin, total_projection = values
    all(isfinite, values) || throw(ArgumentError("spin quantum numbers must be finite"))
    spin1 >= 0 && spin2 >= 0 && total_spin >= 0 || throw(ArgumentError("spin quantum numbers must be nonnegative"))
    all(value -> abs(2 * value - round(2 * value)) <= 1.0e-12, values) || throw(ArgumentError("spin quantum numbers must be integers or half-integers"))
    abs(projection1) <= spin1 && abs(projection2) <= spin2 || throw(ArgumentError("invalid component spin projection"))
    abs((projection1 + projection2) - total_projection) <= 1.0e-12 || return 0.0
    abs(spin1 - spin2) <= total_spin <= spin1 + spin2 || return 0.0
    numerator = _factorial_integer(round(Int, 2 * spin1)) * _factorial_integer(round(Int, 2 * spin2))
    numerator *= _factorial_integer(round(Int, spin1 + spin2 + total_projection))
    numerator *= _factorial_integer(round(Int, spin1 + spin2 - total_projection))
    denominator = _factorial_integer(round(Int, 2 * spin1 + 2 * spin2))
    denominator *= _factorial_integer(round(Int, spin1 + projection1))
    denominator *= _factorial_integer(round(Int, spin1 - projection1))
    denominator *= _factorial_integer(round(Int, spin2 + projection2))
    denominator *= _factorial_integer(round(Int, spin2 - projection2))
    coefficient = sqrt(Float64(numerator) / Float64(denominator))
    previous_two = coefficient
    recurrence_spin = spin1 + spin2
    recurrence_spin == total_spin && return coefficient
    coefficient *= _clebsch_diagonal_factor(recurrence_spin, total_projection, spin1, spin2, projection1, projection2) /
                   _clebsch_recurrence_factor(recurrence_spin, total_projection, spin1, spin2)
    previous_one = coefficient
    recurrence_spin -= 1
    while recurrence_spin != total_spin
        coefficient = (
            previous_one * _clebsch_diagonal_factor(recurrence_spin, total_projection, spin1, spin2, projection1, projection2) -
            previous_two * _clebsch_recurrence_factor(recurrence_spin + 1, total_projection, spin1, spin2)
        ) / _clebsch_recurrence_factor(recurrence_spin, total_projection, spin1, spin2)
        recurrence_spin -= 1
        previous_two = previous_one
        previous_one = coefficient
    end
    return coefficient
end

"""
    _coupling_targets(n_states, total_spin_multiplicity, coupling_multiplicities)

Return the target multiplicity after each left-to-right fragment coupling, appending the system target when only intermediate targets are supplied.
Inputs are read-only, only a small integer vector is allocated, no CUDA work occurs, and invalid counts or final multiplicities raise `ArgumentError`.
"""
function _coupling_targets(
    n_states::Integer,
    total_spin_multiplicity::Integer,
    coupling_multiplicities::AbstractVector{<:Integer},
)
    n_states >= 1 || throw(ArgumentError("at least one fragment state is required"))
    total_spin_multiplicity >= 1 || throw(ArgumentError("total_spin_multiplicity must be positive"))
    n_states == 1 && return Int[]
    if length(coupling_multiplicities) == n_states - 2
        return vcat(Int.(coupling_multiplicities), Int(total_spin_multiplicity))
    end
    length(coupling_multiplicities) == n_states - 1 || throw(ArgumentError("invalid number of coupling multiplicities"))
    last(coupling_multiplicities) == total_spin_multiplicity || throw(ArgumentError("final coupling multiplicity must match the system multiplicity"))
    return Int.(coupling_multiplicities)
end

"""
    _prepare_fragment_microstates(entry)

Build the immutable compact spin-microstate cache entry used by every MEBF that references one fragment state.
The fragment entry is read-only, returned CPU arrays are owned, sector indices use stable descending coefficient magnitude, no CUDA work occurs, and active domains above 32 orbitals raise.
"""
function _prepare_fragment_microstates(entry::FragmentStateEntry{T}) where {T<:AbstractFloat}
    determinant_data = entry.wavefunction.determinants
    microstates = _spin_microstates(determinant_data.expansion, determinant_data.spin_multiplicity)
    sector_ranges = _spin_sector_ranges(microstates)
    sorted_indices = [sort!(collect(range); by=index -> abs(microstates.coefficients[index]), rev=true) for range in sector_ranges]
    maximum_magnitudes = T[isempty(indices) ? zero(T) : abs(microstates.coefficients[first(indices)]) for indices in sorted_indices]
    return PreparedFragmentMicrostates(microstates, sector_ranges, sorted_indices, maximum_magnitudes)
end

"""
    _prepare_fragment_microstate_cache(problem)

Prepare each referenced fragment state exactly once for production compact MEBF construction and return a state-indexed cache.
The problem is read-only, cache construction is serial before worker tasks begin, all returned CPU arrays are immutable by convention, no CUDA work occurs, and unsupported non-dimer or above-32-active MEBFs raise instead of falling back.
"""
function _prepare_fragment_microstate_cache(problem::NOCIProblemDefinition{T}) where {T<:AbstractFloat}
    referenced_indices = Set{Int}()
    for composition in problem.mebfs
        length(composition.fragment_state_indices) == 2 || throw(ArgumentError("production MEBF construction requires exactly two fragment states per MEBF"))
        n_active = sum(active_orbital_count(problem.fragment_states[index].wavefunction.determinants.expansion) for index in composition.fragment_state_indices)
        n_active <= 32 || throw(ArgumentError("production MEBF construction supports at most 32 total active orbitals per MEBF"))
        union!(referenced_indices, composition.fragment_state_indices)
    end
    cache = Dict{Int,PreparedFragmentMicrostates{T}}()
    for index in sort!(collect(referenced_indices))
        cache[index] = _prepare_fragment_microstates(problem.fragment_states[index])
    end
    return cache
end

"""
    _count_injective_spin_products(left, right, target_projection_twice, tau_ci)

Count surviving spin-compatible fragment products in the same magnitude-screened traversal used by production MEBF assembly.
Inputs are read-only and dimensionless, no occupation product or hash table is materialized, the result is an overflow-checked `Int`, and forbidden couplings contribute zero candidates.
"""
function _count_injective_spin_products(
    left::PreparedFragmentMicrostates{T},
    right::PreparedFragmentMicrostates{T},
    target_projection_twice::Int,
    tau_ci::T,
) where {T<:AbstractFloat}
    left_microstates = left.microstates
    right_microstates = right.microstates
    count = 0
    for left_sector in 1:left_microstates.spin_multiplicity, right_sector in 1:right_microstates.spin_multiplicity
        left_projection_twice = left_microstates.spin_multiplicity - 1 - 2 * (left_sector - 1)
        right_projection_twice = right_microstates.spin_multiplicity - 1 - 2 * (right_sector - 1)
        left_projection_twice + right_projection_twice == target_projection_twice || continue
        coupling = T(clebsch_gordan_coefficient(
            (left_microstates.spin_multiplicity - 1) / 2,
            (right_microstates.spin_multiplicity - 1) / 2,
            left_projection_twice / 2,
            right_projection_twice / 2,
            target_projection_twice / 2,
            target_projection_twice / 2,
        ))
        iszero(coupling) && continue
        right_indices = right.sorted_indices[right_sector]
        right_maximum = right.maximum_magnitudes[right_sector]
        for left_index in left.sorted_indices[left_sector]
            left_coefficient = left_microstates.coefficients[left_index]
            abs(left_coefficient) * right_maximum >= tau_ci || break
            for right_index in right_indices
                right_coefficient = right_microstates.coefficients[right_index]
                abs(left_coefficient) * abs(right_coefficient) >= tau_ci || break
                count = Base.Checked.checked_add(count, 1)
            end
        end
    end
    return count
end

"""
    _combine_spin_expansions(left, right, target_multiplicity, tau_ci)

Spin-couple two cached compact fragment expansions through injective bit-field products, stable magnitude screening, exact-sized direct arrays, and final code ordering.
Inputs are read-only, `tau_ci` is a dimensionless inclusive pre-product cutoff, left/right occupation bit fields are disjoint so each surviving pair owns one molecular code, duplicate codes raise instead of being silently merged, and returned CPU arrays preserve the established lexicographic code order without hash-table or explicit-GC overhead.
"""
function _combine_spin_expansions(
    left::PreparedFragmentMicrostates{T},
    right::PreparedFragmentMicrostates{T},
    target_multiplicity::Integer,
    tau_ci::T,
) where {T<:AbstractFloat}
    left_microstates = left.microstates
    right_microstates = right.microstates
    n_active = left_microstates.n_active_orbitals + right_microstates.n_active_orbitals
    n_active <= 32 || throw(ArgumentError("production MEBF construction supports at most 32 total active orbitals"))
    target_projection_twice = Int(target_multiplicity) - 1
    right_shift = 2 * right_microstates.n_active_orbitals
    product_count = _count_injective_spin_products(left, right, target_projection_twice, tau_ci)
    product_count > 0 || throw(ArgumentError("no determinant products survived spin coupling"))
    occupation_codes = Vector{UInt64}(undef, product_count)
    coefficients = Vector{T}(undef, product_count)
    product = 0
    for left_sector in 1:left_microstates.spin_multiplicity, right_sector in 1:right_microstates.spin_multiplicity
        left_projection_twice = left_microstates.spin_multiplicity - 1 - 2 * (left_sector - 1)
        right_projection_twice = right_microstates.spin_multiplicity - 1 - 2 * (right_sector - 1)
        left_projection_twice + right_projection_twice == target_projection_twice || continue
        coupling = T(clebsch_gordan_coefficient(
            (left_microstates.spin_multiplicity - 1) / 2,
            (right_microstates.spin_multiplicity - 1) / 2,
            left_projection_twice / 2,
            right_projection_twice / 2,
            target_projection_twice / 2,
            target_projection_twice / 2,
        ))
        iszero(coupling) && continue
        right_indices = right.sorted_indices[right_sector]
        right_maximum = right.maximum_magnitudes[right_sector]
        for left_index in left.sorted_indices[left_sector]
            left_coefficient = left_microstates.coefficients[left_index]
            abs(left_coefficient) * right_maximum >= tau_ci || break
            left_code = left_microstates.occupation_codes[left_index] << right_shift
            for right_index in right_indices
                right_coefficient = right_microstates.coefficients[right_index]
                abs(left_coefficient) * abs(right_coefficient) >= tau_ci || break
                product += 1
                @inbounds occupation_codes[product] = left_code | right_microstates.occupation_codes[right_index]
                @inbounds coefficients[product] = left_coefficient * right_coefficient * coupling
            end
        end
    end
    product == product_count || throw(ArgumentError("spin-product count changed between exact sizing and direct fill"))
    order = sortperm(occupation_codes)
    ordered_codes = occupation_codes[order]
    @inbounds for index in 2:length(ordered_codes)
        ordered_codes[index] != ordered_codes[index - 1] || throw(ArgumentError("fragment occupation bit fields produced a duplicate molecular determinant code"))
    end
    return coefficients[order], ordered_codes, n_active
end

"""
    _alpha_beta_permutation_sign_with_scratch!(spins, occupation)

Return the alpha/beta canonicalization sign while reusing caller-owned spin scratch.
The occupation is read-only, `spins` is overwritten only within the required occupied-spin prefix, no heap or CUDA allocation is performed by the helper, and malformed spin sectors raise `ArgumentError`.
"""
function _alpha_beta_permutation_sign_with_scratch!(
    spins::AbstractVector{Int8},
    occupation::AbstractVector{<:Integer},
)
    required = 2 * length(occupation)
    length(spins) >= required || throw(DimensionMismatch("alpha/beta sign scratch is too small"))
    n_spins = 0
    projection_twice = 0
    @inbounds for value in occupation
        if value == 2
            n_spins += 1; spins[n_spins] = Int8(1)
            n_spins += 1; spins[n_spins] = Int8(-1)
        elseif value == 1
            n_spins += 1; spins[n_spins] = Int8(1)
        elseif value == -1
            n_spins += 1; spins[n_spins] = Int8(-1)
        end
        projection_twice += value == 1 ? 1 : value == -1 ? -1 : 0
    end
    permutations = 0
    if projection_twice > 0
        for position in (n_spins - projection_twice + 1):n_spins
            @inbounds spins[position] == 1 && continue
            source = 0
            @inbounds for candidate in 1:(position - 1)
                if spins[candidate] == 1
                    source = candidate
                    break
                end
            end
            source > 0 || throw(ArgumentError("invalid spin occupation ordering"))
            @inbounds spins[source], spins[position] = spins[position], spins[source]
            permutations += position - source
        end
    end
    for position in 1:(n_spins - projection_twice - 1)
        expected = isodd(position) ? Int8(1) : Int8(-1)
        @inbounds spins[position] == expected && continue
        source = 0
        @inbounds for candidate in (position + 1):n_spins
            if spins[candidate] == expected
                source = candidate
                break
            end
        end
        source > 0 || throw(ArgumentError("invalid spin occupation ordering"))
        @inbounds spins[position], spins[source] = spins[source], spins[position]
        permutations += source - position
    end
    return isodd(permutations) ? -1 : 1
end

"""
    _alpha_beta_permutation_sign(occupation)

Return the fermionic sign for reordering occupied active spin orbitals into alternating alpha/beta order with excess alpha orbitals last.
The occupation vector is read-only, the compatibility wrapper owns a bounded temporary scratch vector, no CUDA state is touched, and malformed spin sectors raise `ArgumentError`.
"""
function _alpha_beta_permutation_sign(occupation::AbstractVector{<:Integer})
    spins = Vector{Int8}(undef, 2 * length(occupation))
    return _alpha_beta_permutation_sign_with_scratch!(spins, occupation)
end

"""
    _closed_shell_permutation_sign_with_scratch!(occupied, occupation)

Return the closed-shell canonicalization sign while reusing caller-owned occupation scratch.
The source occupation is read-only, the scratch prefix is overwritten deterministically, no heap or CUDA allocation is performed by the helper, and malformed occupations raise `ArgumentError`.
"""
function _closed_shell_permutation_sign_with_scratch!(
    occupied::AbstractVector{Int8},
    occupation::AbstractVector{<:Integer},
)
    length(occupied) >= length(occupation) || throw(DimensionMismatch("closed-shell sign scratch is too small"))
    n_occupied = 0
    n_closed = 0
    @inbounds for value in occupation
        value == 0 && continue
        n_occupied += 1
        occupied[n_occupied] = Int8(value)
        n_closed += value == 2
    end
    permutations = 0
    for position in 1:n_closed
        @inbounds occupied[position] == 2 && continue
        source = 0
        @inbounds for candidate in (position + 1):n_occupied
            if occupied[candidate] == 2
                source = candidate
                break
            end
        end
        source > 0 || throw(ArgumentError("invalid closed-shell occupation ordering"))
        @inbounds occupied[position], occupied[source] = occupied[source], occupied[position]
        permutations += source - position
    end
    return isodd(permutations) ? -1 : 1
end

"""
    _closed_shell_permutation_sign(occupation)

Return the fermionic sign for moving doubly occupied active orbitals ahead of open-shell orbitals.
The input is read-only, the compatibility wrapper owns one bounded temporary vector, no CUDA state is touched, and malformed occupations raise `ArgumentError`.
"""
function _closed_shell_permutation_sign(occupation::AbstractVector{<:Integer})
    occupied = Vector{Int8}(undef, length(occupation))
    return _closed_shell_permutation_sign_with_scratch!(occupied, occupation)
end

"""
    _molecular_orbitals(entries, indices)

Block-assemble selected fragment orbital coefficients with all inactive columns first and all active columns second.
Inputs are read-only, the returned CPU matrix has dimension `n_basis` by `n_inactive+n_active`, no CUDA work occurs, and inconsistent source domains raise `DimensionMismatch`.
"""
function _molecular_orbitals(entries::AbstractVector{FragmentStateEntry{T}}, indices::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    states = [entries[index] for index in indices]
    n_basis = sum(size(state.wavefunction.orbitals.coefficients, 1) for state in states)
    n_inactive = sum(state.wavefunction.determinants.expansion.inactive_orbitals for state in states)
    n_active = sum(active_orbital_count(state.wavefunction.determinants.expansion) for state in states)
    orbitals = zeros(T, n_basis, n_inactive + n_active)
    row_start = 1
    inactive_start = 1
    active_start = n_inactive + 1
    for state in states
        source = state.wavefunction.orbitals.coefficients
        expansion = state.wavefunction.determinants.expansion
        n_rows = size(source, 1)
        local_inactive = expansion.inactive_orbitals
        local_active = active_orbital_count(expansion)
        local_inactive + local_active <= size(source, 2) || throw(DimensionMismatch("fragment orbital matrix does not contain its inactive and active columns"))
        rows = row_start:(row_start + n_rows - 1)
        local_inactive > 0 && (orbitals[rows, inactive_start:(inactive_start + local_inactive - 1)] .= source[:, 1:local_inactive])
        local_active > 0 && (orbitals[rows, active_start:(active_start + local_active - 1)] .= source[:, (local_inactive + 1):(local_inactive + local_active)])
        row_start += n_rows
        inactive_start += local_inactive
        active_start += local_active
    end
    return orbitals, n_inactive
end

"""
    _finalize_molecular_expansion(coefficients, occupation_codes, n_active_orbitals, inactive_orbitals)

Magnitude-order, phase-correct, and materialize screened molecular coefficients without post-screen renormalization into the standard active-occupation matrix representation.
Inputs are read-only, returned CPU arrays are owned, no intermediate occupation strings or CUDA storage are created, and invalid counts, codes, or zero-norm expansions raise.
"""
function _finalize_molecular_expansion(
    coefficients::AbstractVector{T},
    occupation_codes::AbstractVector{UInt64},
    n_active_orbitals::Integer,
    inactive_orbitals::Integer,
) where {T<:AbstractFloat}
    length(coefficients) == length(occupation_codes) || throw(DimensionMismatch("coefficient and occupation-code counts differ"))
    0 <= n_active_orbitals <= 32 || throw(ArgumentError("active-orbital count must lie in 0:32"))
    molecular_coefficients = collect(coefficients)
    isempty(molecular_coefficients) && throw(ArgumentError("molecular determinant coefficient vector must not be empty"))
    all(isfinite, molecular_coefficients) || throw(ArgumentError("molecular determinant coefficients must be finite"))
    any(coefficient -> !iszero(coefficient), molecular_coefficients) || throw(ArgumentError("molecular determinant coefficient vector is identically zero"))
    order = sortperm(abs.(molecular_coefficients); rev=true)
    n_active = Int(n_active_orbitals)
    ordered_coefficients = Vector{T}(undef, length(order))
    alpha_bits = zeros(UInt64, length(order))
    beta_bits = zeros(UInt64, length(order))
    occupation = Vector{Int8}(undef, n_active)
    spin_scratch = Vector{Int8}(undef, 2 * n_active)
    occupied_scratch = Vector{Int8}(undef, n_active)
    @inbounds for (determinant, source_index) in enumerate(order)
        code = occupation_codes[source_index]
        for orbital in 1:n_active
            digit = _occupation_digit_from_code(code, orbital, n_active_orbitals)
            value = digit == UInt64(0) ? Int8(0) : digit == UInt64(1) ? Int8(2) : digit == UInt64(2) ? Int8(1) : Int8(-1)
            occupation[orbital] = value
            bit = UInt64(1) << (orbital - 1)
            value in (1, 2) && (alpha_bits[determinant] |= bit)
            value in (-1, 2) && (beta_bits[determinant] |= bit)
        end
        alpha_beta_sign = _alpha_beta_permutation_sign_with_scratch!(spin_scratch, occupation)
        closed_shell_sign = _closed_shell_permutation_sign_with_scratch!(occupied_scratch, occupation)
        ordered_coefficients[determinant] = molecular_coefficients[source_index] * T(alpha_beta_sign * closed_shell_sign)
    end
    packed = PackedActiveOccupations(n_active, alpha_bits, beta_bits; copy_data=false)
    return DeterminantExpansion(ordered_coefficients, packed, inactive_orbitals, Val(:owned))
end

"""
    _build_molecular_basis_function(problem, composition, prepared_fragments, tau_ci)

Build one production dimer MEBF from cached compact fragment microstates, screened code products, block-assembled orbitals, phase correction, and fragment correlation shifts.
Inputs are read-only, returned CPU arrays are owned, `tau_ci` is dimensionless, no string or CUDA work occurs, and unsupported topology, missing cache entries, forbidden coupling, or complete screening raises instead of falling back.
"""
function _build_molecular_basis_function(
    problem::NOCIProblemDefinition{T},
    composition::MEBFComposition,
    prepared_fragments::AbstractDict{Int,PreparedFragmentMicrostates{T}},
    tau_ci::T,
) where {T<:AbstractFloat}
    length(composition.fragment_state_indices) == 2 || throw(ArgumentError("production MEBF construction requires exactly two fragment states"))
    entries = [problem.fragment_states[index] for index in composition.fragment_state_indices]
    targets = _coupling_targets(length(entries), problem.spin_multiplicity, composition.coupling_multiplicities)
    left_index, right_index = composition.fragment_state_indices
    haskey(prepared_fragments, left_index) || throw(ArgumentError("missing prepared fragment microstates for index $left_index"))
    haskey(prepared_fragments, right_index) || throw(ArgumentError("missing prepared fragment microstates for index $right_index"))
    coefficients, occupation_codes, n_active = _combine_spin_expansions(prepared_fragments[left_index], prepared_fragments[right_index], first(targets), tau_ci)
    orbital_coefficients, inactive_orbitals = _molecular_orbitals(problem.fragment_states, composition.fragment_state_indices)
    expansion = _finalize_molecular_expansion(coefficients, occupation_codes, n_active, inactive_orbitals)
    correlation_energy_shift = sum(entry.wavefunction.determinants.correlation_energy for entry in entries)
    return MolecularBasisFunction(composition.label, copy(composition.fragment_state_indices), orbital_coefficients, expansion, correlation_energy_shift)
end

"""
    _count_screened_coefficient_products(left_coefficients, sorted_right_magnitudes, tau_ci)

Count coefficient pairs satisfying `abs(left * right) >= tau_ci` by exact monotone product comparisons against sorted nonnegative right magnitudes.
Inputs are read-only dimensionless CPU vectors, the result is an overflow-checked `Int`, no determinant products or strings are materialized, and invalid thresholds or magnitudes raise `ArgumentError`.
"""
function _count_screened_coefficient_products(
    left_coefficients::AbstractVector{T},
    sorted_right_magnitudes::AbstractVector{T},
    tau_ci::T,
) where {T<:AbstractFloat}
    isfinite(tau_ci) && tau_ci >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    issorted(sorted_right_magnitudes) || throw(ArgumentError("right coefficient magnitudes must be sorted"))
    all(value -> isfinite(value) && value >= zero(T), sorted_right_magnitudes) || throw(ArgumentError("right coefficient magnitudes must be finite and nonnegative"))
    all(isfinite, left_coefficients) || throw(ArgumentError("left coefficients must be finite"))
    pair_count = 0
    n_right = length(sorted_right_magnitudes)
    for left_coefficient in left_coefficients
        left_magnitude = abs(left_coefficient)
        first_passing = 1
        last_failing = n_right
        while first_passing <= last_failing
            middle = first_passing + ((last_failing - first_passing) >>> 1)
            if left_magnitude * sorted_right_magnitudes[middle] >= tau_ci
                last_failing = middle - 1
            else
                first_passing = middle + 1
            end
        end
        pair_count = Base.Checked.checked_add(pair_count, n_right - first_passing + 1)
    end
    return pair_count
end

"""
    _screened_spin_product_count(left, right, target_multiplicity, tau_ci)

Count the spin-sector-compatible determinant products that can survive the same pre-product CI threshold used by `_combine_spin_expansions`.
Microstate inputs are read-only and dimensionless, only sector-sized magnitude vectors are allocated, no concatenated occupations or product dictionary is built, and invalid spin coupling or thresholds raise.
"""
function _screened_spin_product_count(
    left::SpinMicrostateExpansion{T},
    right::SpinMicrostateExpansion{T},
    target_multiplicity::Integer,
    tau_ci::T,
) where {T<:AbstractFloat}
    target_multiplicity >= 1 || throw(ArgumentError("target_multiplicity must be positive"))
    isfinite(tau_ci) && tau_ci >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    left_ranges = _spin_sector_ranges(left)
    right_ranges = _spin_sector_ranges(right)
    right_magnitudes = [sort!(abs.(right.coefficients[range])) for range in right_ranges]
    target_projection_twice = Int(target_multiplicity) - 1
    count = 0
    for left_sector in 1:left.spin_multiplicity, right_sector in 1:right.spin_multiplicity
        left_projection_twice = left.spin_multiplicity - 1 - 2 * (left_sector - 1)
        right_projection_twice = right.spin_multiplicity - 1 - 2 * (right_sector - 1)
        left_projection_twice + right_projection_twice == target_projection_twice || continue
        coupling = clebsch_gordan_coefficient(
            (left.spin_multiplicity - 1) / 2,
            (right.spin_multiplicity - 1) / 2,
            left_projection_twice / 2,
            right_projection_twice / 2,
            target_projection_twice / 2,
            target_projection_twice / 2,
        )
        iszero(coupling) && continue
        count = Base.Checked.checked_add(
            count,
            _count_screened_coefficient_products(
                @view(left.coefficients[left_ranges[left_sector]]),
                right_magnitudes[right_sector],
                tau_ci,
            ),
        )
    end
    return count
end

"""
    _mebf_assembly_memory(problem, prepared_fragments; tau_ci=0)

Estimate a conservative peak host numerical payload for molecular orbital matrices and spin-coupled determinant arrays after the production pre-product CI screen; device bytes are zero.
The problem and prepared cache are read-only, products are counted in `O(n log n)` without materializing occupations, the default zero threshold preserves an unscreened upper bound, and checked integer overflow or invalid production-domain data raise.
"""
function _mebf_assembly_memory(
    problem::NOCIProblemDefinition{T},
    prepared_fragments::AbstractDict{Int,PreparedFragmentMicrostates{T}};
    tau_ci::Real=zero(T),
) where {T<:AbstractFloat}
    threshold = T(tau_ci)
    isfinite(threshold) && threshold >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    orbital_bytes_total = 0
    determinant_bytes_total = 0
    for composition in problem.mebfs
        entries = [problem.fragment_states[index] for index in composition.fragment_state_indices]
        n_inactive = sum(entry.wavefunction.determinants.expansion.inactive_orbitals for entry in entries)
        n_active = sum(active_orbital_count(entry.wavefunction.determinants.expansion) for entry in entries)
        orbital_bytes_total = Base.Checked.checked_add(
            orbital_bytes_total,
            estimate_array_bytes(T, problem.n_basis, n_inactive + n_active),
        )
        length(entries) == 2 || throw(ArgumentError("production MEBF construction requires exactly two fragment states"))
        targets = _coupling_targets(2, problem.spin_multiplicity, composition.coupling_multiplicities)
        left_index, right_index = composition.fragment_state_indices
        candidate_count = _screened_spin_product_count(
            prepared_fragments[left_index].microstates,
            prepared_fragments[right_index].microstates,
            first(targets),
            threshold,
        )
        determinant_bytes = Base.Checked.checked_add(
            estimate_array_bytes(T, candidate_count),
            estimate_array_bytes(UInt64, 2, candidate_count),
        )
        determinant_bytes_total = Base.Checked.checked_add(determinant_bytes_total, determinant_bytes)
    end
    temporary_bytes = Base.Checked.checked_mul(2, determinant_bytes_total)
    return MemoryEstimate(Base.Checked.checked_add(Base.Checked.checked_add(orbital_bytes_total, determinant_bytes_total), temporary_bytes), 0)
end

"""
    mebf_assembly_memory(problem; tau_ci=0)

Prepare the compact fragment cache and estimate conservative peak host numerical storage for production molecular-basis-function assembly; device bytes are zero.
The problem is read-only, the temporary cache is CPU-owned and released after estimation, no strings or CUDA storage are created, and unsupported topology, active domains above 32 orbitals, invalid thresholds, or checked overflow raise.
"""
function mebf_assembly_memory(
    problem::NOCIProblemDefinition{T};
    tau_ci::Real=zero(T),
) where {T<:AbstractFloat}
    prepared_fragments = _prepare_fragment_microstate_cache(problem)
    return _mebf_assembly_memory(problem, prepared_fragments; tau_ci)
end

"""
    _build_molecular_basis_function_collection(problem, prepared_fragments, tau_ci)

Build every production compact molecular MEBF concurrently on available Julia threads from one shared fragment cache and return results in the original composition order.
The problem and cache are shared read-only, each task owns its coupling dictionary and output arrays, result order is deterministic, every launched task is fetched before an error propagates, and no fallback, strings, or CUDA work occurs.
"""
function _build_molecular_basis_function_collection(
    problem::NOCIProblemDefinition{T},
    prepared_fragments::AbstractDict{Int,PreparedFragmentMicrostates{T}},
    tau_ci::T,
) where {T<:AbstractFloat}
    tasks = map(problem.mebfs) do composition
        Threads.@spawn _build_molecular_basis_function(problem, composition, prepared_fragments, tau_ci)
    end
    basis_functions = Vector{Union{Nothing,MolecularBasisFunction{T}}}(nothing, length(tasks))
    first_error = nothing
    for index in eachindex(tasks)
        try
            basis_functions[index] = fetch(tasks[index])
        catch error
            isnothing(first_error) && (first_error = error)
        end
    end
    isnothing(first_error) || throw(first_error)
    return MolecularBasisFunction{T}[basis::MolecularBasisFunction{T} for basis in basis_functions]
end

"""
    build_molecular_basis_functions(problem; screening=ScreeningOptions(), runtime=RuntimeOptions(), telemetry=nothing)

Build independent molecular MEBFs concurrently on Julia threads, then apply the shared global CI cutoff `tau_ci/maxabs(C)` without renormalizing retained coefficients.
The problem is read-only, returned scientific arrays retain input order and are CPU-owned, a concurrency-aware numerical memory preflight runs before product assembly, optional telemetry records wall time and observed allocations, and resource failure or complete screening raises.
"""
function build_molecular_basis_functions(
    problem::NOCIProblemDefinition{T};
    screening::ScreeningOptions{T}=ScreeningOptions(T(1.0e-6), T(1.0e-5), T(1.0e-12)),
    runtime::RuntimeOptions=RuntimeOptions(),
    telemetry::Union{Nothing,ExecutionTelemetry}=nothing,
) where {T<:AbstractFloat}
    prepared_fragments = _prepare_fragment_microstate_cache(problem)
    estimate = _mebf_assembly_memory(problem, prepared_fragments; tau_ci=screening.tau_ci)
    enforce_memory_limits(estimate, runtime)
    work = function ()
        basis_functions = _build_molecular_basis_function_collection(problem, prepared_fragments, screening.tau_ci)
        maximum_coefficient = maximum(maximum(abs, basis.expansion.coefficients) for basis in basis_functions)
        maximum_coefficient > zero(T) || throw(ArgumentError("MEBF collection has zero coefficient norm"))
        cutoff = screening.tau_ci / maximum_coefficient
        for index in eachindex(basis_functions)
            basis = basis_functions[index]
            keep = findall(coefficient -> abs(coefficient) > cutoff, basis.expansion.coefficients)
            isempty(keep) && throw(ArgumentError("MEBF $(basis.label) lost every determinant at cutoff $cutoff"))
            reduced_coefficients = basis.expansion.coefficients[keep]
            reduced_occupations = basis.expansion.active_occupations[:, keep]
            reduced_expansion = DeterminantExpansion(
                reduced_coefficients,
                reduced_occupations,
                basis.expansion.inactive_orbitals,
                Val(:owned),
            )
            basis_functions[index] = MolecularBasisFunction(
                basis.label,
                basis.fragment_state_indices,
                basis.orbital_coefficients,
                reduced_expansion,
                basis.correlation_energy_shift,
            )
        end
        return basis_functions
    end
    isnothing(telemetry) && return work()
    return record_phase!(
        work,
        telemetry,
        :build_molecular_basis_functions;
        estimate=estimate,
        metadata=Dict(
            :n_mebfs => length(problem.mebfs),
            :tau_ci => screening.tau_ci,
            :julia_threads => Threads.nthreads(),
            :mebf_tasks => length(problem.mebfs),
            :prepared_fragment_states => length(prepared_fragments),
            :occupation_representation => :compact_uint64,
        ),
        collect_allocations=runtime.collect_allocations,
    )
end
