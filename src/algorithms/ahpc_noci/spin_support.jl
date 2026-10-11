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

 File:            src/algorithms/ahpc_noci/spin_support.jl
 Purpose:         Canonical occupied-active spin-string identities and support enumeration over packed masks.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    CanonicalSpinStringPair

Identify one ordered left/right pair of canonical occupied-active spin strings using two 64-bit spatial-orbital masks.
Bit positions are dimensionless active spatial-orbital indices, the immutable key occupies sixteen inline bytes, and the production representation supports at most 64 active orbitals.
"""
struct CanonicalSpinStringPair
    left_bits::UInt64
    right_bits::UInt64

    """
        CanonicalSpinStringPair(left_bits, right_bits)

    Construct one compact ordered pair from nonnegative integer masks that fit the 64-active-orbital production domain.
    Inputs are dimensionless and unmodified, the result occupies sixteen inline bytes, no allocation or CUDA work occurs, and wider masks raise `OverflowError`.
    """
    function CanonicalSpinStringPair(left_bits::Integer, right_bits::Integer)
        0 <= left_bits <= typemax(UInt64) && 0 <= right_bits <= typemax(UInt64) || throw(OverflowError("canonical spin-pair masks exceed the production 64-orbital domain"))
        return new(UInt64(left_bits), UInt64(right_bits))
    end
end

"""
    CanonicalMEBFSpinStrings

Store borrowed canonical alpha/beta occupied-active bit planes and one owned canonicalization-parity vector for every determinant in one MEBF.
The source MEBF must outlive this read-only view, parities are dimensionless `+1/-1`, and no orbital coefficients, ERIs, or CUDA state are retained.
"""
struct CanonicalMEBFSpinStrings
    active_rank::Int
    alpha_bits::Vector{UInt64}
    beta_bits::Vector{UInt64}
    determinant_phase::Vector{Int8}
end

"""
    SpinPairSupportDomain

Store first-encounter-ordered alpha, beta, and union supports of canonical single-spin string pairs for one MEBF pair.
Union IDs map sector-local supports into one reusable kernel domain, lookup metadata is diagnostic, and all arrays reside on CPU.
"""
struct SpinPairSupportDomain
    alpha_support::Vector{CanonicalSpinStringPair}
    beta_support::Vector{CanonicalSpinStringPair}
    union_support::Vector{CanonicalSpinStringPair}
    alpha_union_ids::Vector{Int32}
    beta_union_ids::Vector{Int32}
    intersection_count::Int
    requested_lookup::SupportLookup
    selected_lookup::SupportLookup
    lookup_rehashes::Int
end

"""
    _spin_bits(active_occupation, alpha)

Pack one canonical occupied-active alpha or beta string into a `UInt64` bit mask and return it.
The occupation vector is read-only and dimensionless, no allocation or CUDA work occurs, and active ranks above 64 or invalid codes raise `ArgumentError`.
"""
function _spin_bits(active_occupation::AbstractVector{<:Integer}, alpha::Bool)
    length(active_occupation) <= 64 || throw(ArgumentError("canonical spin strings support at most 64 active orbitals"))
    bits = UInt64(0)
    for (active, value) in pairs(active_occupation)
        value in (-1, 0, 1, 2) || throw(ArgumentError("unsupported active occupation value $value"))
        occupied = alpha ? value in (1, 2) : value in (-1, 2)
        occupied && (bits |= UInt64(1) << (active - 1))
    end
    return bits
end

"""
    _canonical_determinant_phase(active_occupation)

Return the product of alpha/beta parities needed to reorder each closed-then-open occupied-active list into increasing canonical order.
The occupation vector is read-only and dimensionless, no storage or device state is allocated, and invalid codes raise `ArgumentError`.
"""
function _canonical_determinant_phase(active_occupation::AbstractVector{<:Integer})
    inversions = 0
    for (open_active, open_value) in pairs(active_occupation)
        open_value in (-1, 0, 1, 2) || throw(ArgumentError("unsupported active occupation value $open_value"))
        open_value in (-1, 1) || continue
        for later_active in (open_active + 1):length(active_occupation)
            active_occupation[later_active] == 2 && (inversions += 1)
        end
    end
    return isodd(inversions) ? Int8(-1) : Int8(1)
end

"""
    _canonical_determinant_phase(alpha_bits, beta_bits, active_rank)

Return the closed-before-open to canonical active-orbital reorder parity directly from packed alpha/beta occupation masks.
Inputs are dimensionless and read-only, bit operations avoid occupation-vector expansion, no allocation or CUDA work occurs, and bits outside `active_rank` raise `ArgumentError`.
"""
function _canonical_determinant_phase(alpha_bits::UInt64, beta_bits::UInt64, active_rank::Integer)
    rank = Int(active_rank)
    0 <= rank <= 64 || throw(ArgumentError("active_rank must lie in 0:64"))
    valid_mask = rank == 64 ? typemax(UInt64) : (UInt64(1) << rank) - UInt64(1)
    (alpha_bits | beta_bits) & ~valid_mask == 0 || throw(ArgumentError("packed occupation contains an orbital outside active_rank"))
    open_bits = alpha_bits ⊻ beta_bits
    closed_bits = alpha_bits & beta_bits
    inversions = 0
    while !iszero(open_bits)
        open_position = trailing_zeros(open_bits)
        later_mask = open_position == 63 ? UInt64(0) : ~((UInt64(1) << (open_position + 1)) - UInt64(1))
        inversions += count_ones(closed_bits & later_mask)
        open_bits &= open_bits - UInt64(1)
    end
    return isodd(inversions) ? Int8(-1) : Int8(1)
end

"""
    build_canonical_mebf_spin_strings(basis_function)

Build a compact canonical spin-string view by borrowing the MEBF's packed alpha/beta planes and computing only determinant reorder parities.
The basis function is read-only and must outlive the returned view, only the parity vector is allocated on CPU, no AO/ERI/CUDA work occurs, and active ranks above 64 or malformed occupations raise.
"""
function build_canonical_mebf_spin_strings(basis_function::MolecularBasisFunction)
    expansion = basis_function.expansion
    n_determinants = determinant_count(expansion)
    active_rank = active_orbital_count(expansion)
    active_rank <= 64 || throw(ArgumentError("production canonical spin strings support at most 64 active orbitals"))
    alpha_bits = expansion.active_occupations.alpha_bits
    beta_bits = expansion.active_occupations.beta_bits
    determinant_phase = Vector{Int8}(undef, n_determinants)
    for determinant in 1:n_determinants
        determinant_phase[determinant] = _canonical_determinant_phase(alpha_bits[determinant], beta_bits[determinant], active_rank)
    end
    return CanonicalMEBFSpinStrings(active_rank, alpha_bits, beta_bits, determinant_phase)
end

"""
    spin_string_indices(bits, active_rank)

Decode one canonical occupied-active bit mask into an increasing vector of one-based spatial-orbital indices.
Inputs are dimensionless, the returned vector is CPU-owned, no device work occurs, and bits outside `active_rank` or ranks outside `0:64` raise `ArgumentError`.
"""
function spin_string_indices(bits::UInt64, active_rank::Integer)
    rank = Int(active_rank)
    0 <= rank <= 64 || throw(ArgumentError("active_rank must lie in 0:64"))
    valid_mask = rank == 64 ? typemax(UInt64) : (UInt64(1) << rank) - UInt64(1)
    bits & ~valid_mask == 0 || throw(ArgumentError("spin string contains an orbital outside active_rank"))
    indices = Int[]
    remaining = bits
    while !iszero(remaining)
        bit = trailing_zeros(remaining)
        push!(indices, bit + 1)
        remaining &= remaining - UInt64(1)
    end
    return indices
end

"""
    _spin_pair_code(left_bits, right_bits)

Encode two 64-bit canonical spin masks injectively into one CPU `UInt128` lookup key with the left mask in the high word.
Inputs are dimensionless and read-only, no allocation or CUDA work occurs, and all 64-bit mask pairs map uniquely.
"""
@inline function _spin_pair_code(left_bits::UInt64, right_bits::UInt64)
    return (UInt128(left_bits) << 64) | UInt128(right_bits)
end

"""
    _spin_pair_code(pair)

Encode a canonical left/right spin-string pair as one injective `UInt128` CPU lookup key.
The immutable pair is not mutated, no allocation occurs, and the encoding is injective throughout the production 64-active-orbital domain.
"""
@inline function _spin_pair_code(pair::CanonicalSpinStringPair)
    return _spin_pair_code(pair.left_bits, pair.right_bits)
end

"""
    SpinPairFlatMap(expected_entries=0)

Private open-address map from exact `UInt64` or `UInt128` spin-pair codes to positive local IDs.
Capacity is a power of two with load factor at most one half, arrays are CPU-owned, and invalid expected sizes raise an exception before allocation.
"""
mutable struct SpinPairFlatMap{K<:Unsigned}
    keys::Vector{K}
    values::Vector{Int32}
    occupied::BitVector
    count::Int
    rehashes::Int
end

"""
    _spin_pair_flat_capacity(expected_entries)

Return a power-of-two flat-map capacity that keeps the requested entry count at load factor at most one half.
The count is dimensionless, no array is allocated, and negative or overflowing sizes raise an exception.
"""
function _spin_pair_flat_capacity(expected_entries::Integer)
    expected = Int(expected_entries)
    expected >= 0 || throw(ArgumentError("expected_entries must be nonnegative"))
    minimum_capacity = max(16, expected == 0 ? 16 : Base.Checked.checked_mul(2, expected))
    capacity = 16
    while capacity < minimum_capacity
        capacity = Base.Checked.checked_mul(capacity, 2)
    end
    return capacity
end

"""
    SpinPairFlatMap(expected_entries=0)

Construct an empty open-address spin-pair map with optional exact-support pre-sizing.
Only lookup arrays are allocated on CPU, no scientific ordering is derived from table slots, and invalid or overflowing sizes raise an exception.
"""
function SpinPairFlatMap(expected_entries::Integer=0, ::Type{K}=UInt128) where {K<:Union{UInt64,UInt128}}
    capacity = _spin_pair_flat_capacity(expected_entries)
    return SpinPairFlatMap(Vector{K}(undef, capacity), Vector{Int32}(undef, capacity), falses(capacity), 0, 0)
end

"""
    _mix_spin_pair_word(value)

Mix one 64-bit word with the SplitMix64 finalizer for CPU hash-slot selection.
The input is dimensionless and read-only, unsigned overflow is intentional, no allocation occurs, and the return value has no scientific ordering role.
"""
@inline function _mix_spin_pair_word(value::UInt64)
    mixed = value
    mixed = (mixed ⊻ (mixed >> 30)) * UInt64(0xbf58476d1ce4e5b9)
    mixed = (mixed ⊻ (mixed >> 27)) * UInt64(0x94d049bb133111eb)
    return mixed ⊻ (mixed >> 31)
end

"""
    _spin_pair_hash_slot(key, capacity)

Map an exact 128-bit spin-pair key to a one-based initial slot in a power-of-two capacity by mixing both 64-bit words.
Arguments are read-only and dimensionless, hash collisions are resolved by exact integer equality, no allocation occurs, and callers must provide a valid key/capacity.
"""
@inline function _spin_pair_hash_slot(key::UInt128, capacity::Int)
    low = UInt64(key & UInt128(typemax(UInt64)))
    high = UInt64(key >> 64)
    mixed = _mix_spin_pair_word(low ⊻ _mix_spin_pair_word(high + UInt64(0x9e3779b97f4a7c15)))
    return Int((mixed & UInt64(capacity - 1)) + UInt64(1))
end

"""
    _spin_pair_flat_get(map, key)

Return the positive local ID for an exact 64- or 128-bit key or zero when absent from an open-address map.
The map is read-only, no allocation or device work occurs, and zero-valued keys are supported through a separate occupancy bitmap.
"""
@inline function _spin_pair_flat_get(map::SpinPairFlatMap{K}, key::K) where {K}
    capacity = length(map.keys)
    slot = _spin_pair_hash_slot(key, capacity)
    @inbounds while true
        map.occupied[slot] || return 0
        map.keys[slot] == key && return map.values[slot]
        slot = slot == capacity ? 1 : slot + 1
    end
end

"""
    _spin_pair_flat_insert!(map, key, value)

Insert one absent exact 64- or 128-bit key and positive local ID, rehashing before load factor exceeds one half, and return the value.
The CPU map is mutated, scientific order remains caller-defined, and duplicate keys or invalid values raise an exception.
"""
function _spin_pair_flat_insert!(map::SpinPairFlatMap{K}, key::K, value::Integer) where {K}
    0 < value <= typemax(Int32) || throw(ArgumentError("flat-map values must fit a positive Int32 local ID"))
    _spin_pair_flat_get(map, key) == 0 || throw(ArgumentError("flat-map key already exists"))
    if map.count + 1 > length(map.keys) ÷ 2
        old_keys = map.keys
        old_values = map.values
        old_occupied = map.occupied
        map.keys = Vector{K}(undef, Base.Checked.checked_mul(2, length(old_keys)))
        map.values = Vector{Int32}(undef, length(map.keys))
        map.occupied = falses(length(map.keys))
        map.count = 0
        map.rehashes += 1
        for old_slot in eachindex(old_keys)
            old_occupied[old_slot] || continue
            _spin_pair_flat_insert!(map, old_keys[old_slot], old_values[old_slot])
        end
    end
    capacity = length(map.keys)
    slot = _spin_pair_hash_slot(key, capacity)
    @inbounds while map.occupied[slot]
        slot = slot == capacity ? 1 : slot + 1
    end
    map.keys[slot] = key
    local_id = Int32(value)
    map.values[slot] = local_id
    map.occupied[slot] = true
    map.count += 1
    return local_id
end

"""
    _exact_nested_spin_support_count(left_bits, right_bits, pair_block)

Count distinct ordered spin-string pairs exactly from nested cutoff windows without replaying screened determinant edges.
Inputs are read-only and dimensionless, work is linear in determinant and distinct-spin counts, temporary dictionaries scale with spin support rather than pair count, and a nonnested pair block raises `ArgumentError`.
"""
function _exact_nested_spin_support_count(
    left_bits::Vector{UInt64},
    right_bits::Vector{UInt64},
    pair_block::RowCompressedDeterminantPairs,
)
    length(left_bits) == pair_block.n_left || throw(DimensionMismatch("left spin-string count differs from pair-block rows"))
    length(right_bits) == pair_block.n_right || throw(DimensionMismatch("right spin-string count differs from pair-block columns"))
    seen_left = Set{UInt64}()
    right_frequencies = Dict{UInt64,Int}()
    initialized = false
    current_first = 0
    current_last = 0
    distinct_right = 0
    total_support = 0
    for left_index in 1:pair_block.n_left
        left_key = left_bits[left_index]
        left_key in seen_left && continue
        push!(seen_left, left_key)
        first_right = pair_block.diagonal ? left_index : 1
        last_right = Int(pair_block.last_right[left_index])
        first_right <= last_right || continue
        if !initialized
            current_first = first_right
            current_last = last_right
            for right_index in current_first:current_last
                right_key = right_bits[right_index]
                previous = get(right_frequencies, right_key, 0)
                right_frequencies[right_key] = previous + 1
                previous == 0 && (distinct_right += 1)
            end
            initialized = true
        else
            first_right >= current_first || throw(ArgumentError("screened determinant windows are not nested on the left boundary"))
            last_right <= current_last || throw(ArgumentError("screened determinant windows are not nested on the right boundary"))
            while current_first < first_right
                right_key = right_bits[current_first]
                remaining = right_frequencies[right_key] - 1
                remaining == 0 ? (delete!(right_frequencies, right_key); distinct_right -= 1) : (right_frequencies[right_key] = remaining)
                current_first += 1
            end
            while current_last > last_right
                right_key = right_bits[current_last]
                remaining = right_frequencies[right_key] - 1
                remaining == 0 ? (delete!(right_frequencies, right_key); distinct_right -= 1) : (right_frequencies[right_key] = remaining)
                current_last -= 1
            end
        end
        total_support = Base.Checked.checked_add(total_support, distinct_right)
    end
    return total_support
end

"""
    _support_counts(left, right, pair_block)

Count exact alpha and beta support sizes with the production nested-window planner before first-encounter ID construction.
Inputs are read-only, no screened edge is enumerated, temporary CPU dictionaries scale with unique determinant spin strings, and invalid or nonnested domains raise explicitly.
"""
function _support_counts(
    left::CanonicalMEBFSpinStrings,
    right::CanonicalMEBFSpinStrings,
    pair_block::RowCompressedDeterminantPairs,
)
    alpha_count = _exact_nested_spin_support_count(left.alpha_bits, right.alpha_bits, pair_block)
    beta_count = _exact_nested_spin_support_count(left.beta_bits, right.beta_bits, pair_block)
    return alpha_count, beta_count
end

"""
    _selected_support_lookup(requested, expected_entries, screened_pairs)

Resolve the adaptive support-lookup policy to one concrete backend or return an explicitly requested backend unchanged.
Counts are dimensionless, no storage or device state are allocated, and callers must provide nonnegative counts.
"""
function _selected_support_lookup(requested::SupportLookup, expected_entries::Int, screened_pairs::Int)
    expected_entries >= 0 && screened_pairs >= 0 || throw(ArgumentError("support and screened-pair counts must be nonnegative"))
    return requested == AdaptiveLookup ? (
        expected_entries >= 2_000_000 && screened_pairs >= 50_000_000 ? FlatHashPresizedLookup : DictPresizedLookup
    ) : requested
end

"""
    _support_lookup(requested, expected_entries, screened_pairs)

Construct a presized exact-key Dict or flat-hash lookup and return it with the selected typed backend, applying the conservative adaptive crossover policy.
Counts are dimensionless, only CPU lookup payload is allocated, and invalid or nonproduction selector values raise before construction.
"""
function _support_lookup(requested::SupportLookup, expected_entries::Int, screened_pairs::Int, ::Type{K}=UInt128) where {K<:Union{UInt64,UInt128}}
    selected = _selected_support_lookup(requested, expected_entries, screened_pairs)
    if selected == DictPresizedLookup
        lookup = Dict{K,Int32}()
        sizehint!(lookup, expected_entries)
        return lookup, selected
    end
    selected == FlatHashPresizedLookup || throw(ArgumentError("unsupported production support lookup: $selected"))
    return SpinPairFlatMap(expected_entries, K), selected
end

"""
    _lookup_id(lookup, key)

Return a spin-pair local ID or zero from a Julia exact-key dictionary.
The lookup and integer key are read-only, no allocation occurs, and missing entries return zero.
"""
@inline function _lookup_id(lookup::Dict{K,Int32}, key::K) where {K}
    return get(lookup, key, Int32(0))
end

"""
    _lookup_id(lookup::SpinPairFlatMap, key)

Return a spin-pair local ID or zero from the private exact-key flat-hash representation without allocation.
The lookup is read-only and the key is a dimensionless exact integer code.
"""
@inline function _lookup_id(lookup::SpinPairFlatMap{K}, key::K) where {K}
    return _spin_pair_flat_get(lookup, key)
end

"""
    _insert_lookup!(lookup, key, value)

Insert an absent exact spin-pair code and local ID into a Julia dictionary, mutating only the lookup and returning the value.
Keys and values are dimensionless, no device work occurs, and duplicate-key checks remain the caller's responsibility.
"""
@inline function _insert_lookup!(lookup::Dict{K,Int32}, key::K, value::Integer) where {K}
    0 < value <= typemax(Int32) || throw(OverflowError("spin-pair support exceeds Int32 local-ID storage"))
    local_id = Int32(value)
    lookup[key] = local_id
    return local_id
end

"""
    _insert_lookup!(lookup::SpinPairFlatMap, key, value)

Insert an absent exact spin-pair code and local ID into the private flat map and return the value.
The CPU lookup is mutated, no device work occurs, and invalid or duplicate inputs raise through `_spin_pair_flat_insert!`.
"""
@inline function _insert_lookup!(lookup::SpinPairFlatMap{K}, key::K, value::Integer) where {K}
    return _spin_pair_flat_insert!(lookup, key, value)
end

"""
    _spin_lookup_key_type(left, right)

Return `UInt64` for two validated spin-string domains with at most 32 active orbitals each, otherwise `UInt128`.
Inputs are read-only canonical spin strings with dimensionless ranks; masks are validated once in linear determinant work before selecting the lossless edge-loop representation, invalid ranks or out-of-domain masks raise `ArgumentError`, no device work occurs, and the wide route retains every mask bit through orbital 64 independently of occupied rank.
"""
function _spin_lookup_key_type(left::CanonicalMEBFSpinStrings, right::CanonicalMEBFSpinStrings)
    for strings in (left, right)
        rank = strings.active_rank
        0 <= rank <= 64 || throw(ArgumentError("canonical active rank must lie in 0:64"))
        valid_mask = rank == 64 ? typemax(UInt64) : (UInt64(1) << rank) - UInt64(1)
        for plane in (strings.alpha_bits, strings.beta_bits)
            any(bits -> !iszero(bits & ~valid_mask), plane) && throw(ArgumentError("canonical spin mask exceeds its declared active rank"))
        end
    end
    return max(left.active_rank, right.active_rank) <= 32 ? UInt64 : UInt128
end

"""
    _spin_pair_code(K, left_bits, right_bits)

Encode read-only dimensionless canonical masks in the state-pair-selected exact key type, returning an injective CPU integer without allocation.
The compact method requires both masks to fit 32 bits, a condition validated by the canonical active rank before edge traversal; wider domains use the separate 128-bit method.
"""
@inline function _spin_pair_code(::Type{UInt64}, left_bits::UInt64, right_bits::UInt64)
    return (left_bits << 32) | right_bits
end

"""
    _spin_pair_code(UInt128, left_bits, right_bits)

Encode two read-only 64-bit orbital masks into an injective 128-bit CPU key, retaining the large-domain production contract without allocation or device work.
All dimensionless UInt64 inputs are valid and no scientific data are mutated.
"""
@inline function _spin_pair_code(::Type{UInt128}, left_bits::UInt64, right_bits::UInt64)
    return _spin_pair_code(left_bits, right_bits)
end

"""
    _spin_pair_hash_slot(key::UInt64, capacity)

Return a one-based slot for an exact compact spin-pair key in a power-of-two CPU table using a 64-bit mixer.
The dimensionless key is read-only, capacity must be positive and a power of two, collisions use exact equality, and no allocation occurs.
"""
@inline function _spin_pair_hash_slot(key::UInt64, capacity::Int)
    return Int((_mix_spin_pair_word(key) & UInt64(capacity - 1)) + UInt64(1))
end

"""
    _numbered_spin_pair_domain(left, right)

Select exact spin-string numbering for active-orbital domains above 32 only when
all alpha/beta occupied counts are below 32. Physical masks remain UInt64 and
are validated before this dimensionless, allocation-free CPU decision. Occupied
count 32 retains the manuscript representation.
"""
function _numbered_spin_pair_domain(left::CanonicalMEBFSpinStrings, right::CanonicalMEBFSpinStrings)
    physical_key_type = _spin_lookup_key_type(left, right)
    physical_key_type === UInt64 && return false
    return all(strings -> all(plane -> all(bits -> count_ones(bits) < 32, plane), (strings.alpha_bits, strings.beta_bits)), (left, right))
end

"""
    _canonical_spin_number_bytes(strings)

Bound CPU storage for two UInt32 determinant spin-number planes and a shared
alpha/beta mask-to-number table. The capacity uses at most two strings per
determinant, checked arithmetic raises before allocation, units are bytes,
and inputs are not modified. This bound excludes the borrowed physical masks.
"""
function _canonical_spin_number_bytes(strings::CanonicalMEBFSpinStrings)
    n_strings = Base.Checked.checked_add(length(strings.alpha_bits), length(strings.beta_bits))
    capacity = _spin_pair_flat_capacity(n_strings)
    return Base.Checked.checked_add(estimate_array_bytes(UInt32, n_strings),
        Base.Checked.checked_add(estimate_array_bytes(UInt64, capacity),
        Base.Checked.checked_add(estimate_array_bytes(Int32, capacity), cld(capacity, 8))))
end

"""
    _canonical_spin_numbers(strings, numbered)

Return determinant-indexed alpha/beta spin numbers and their exact mask table,
or borrow physical masks when numbering is disabled. Numbering is shared across
alpha and beta on this MEBF side, with positive UInt32 IDs assigned by first
encounter; physical masks and phases are read-only. Only linear determinant
storage is allocated, no Cartesian pair table or CUDA state is created, and
more than typemax(Int32) distinct strings raises before assigning an ID.
"""
function _canonical_spin_numbers(strings::CanonicalMEBFSpinStrings, numbered::Bool)
    numbered || return (alpha=strings.alpha_bits, beta=strings.beta_bits, numbers=nothing)
    numbers = SpinPairFlatMap(length(strings.alpha_bits) + length(strings.beta_bits), UInt64)
    alpha = Vector{UInt32}(undef, length(strings.alpha_bits))
    beta = Vector{UInt32}(undef, length(strings.beta_bits))
    for (physical, numbered_plane) in ((strings.alpha_bits, alpha), (strings.beta_bits, beta))
        for index in eachindex(physical)
            bits = physical[index]
            spin_number = _lookup_id(numbers, bits)
            if spin_number == 0
                numbers.count < typemax(Int32) || throw(OverflowError("canonical spin-string count exceeds Int32"))
                spin_number = Int32(numbers.count + 1)
                _insert_lookup!(numbers, bits, spin_number)
            end
            numbered_plane[index] = UInt32(spin_number)
        end
    end
    return (alpha=alpha, beta=beta, numbers=numbers)
end

"""
    _spin_pair_code(UInt64, left_number::UInt32, right_number::UInt32)

Encode two exact canonical spin-string numbers into a 64-bit pair number.
Arguments are dimensionless, read-only and already validated; no mask truncation,
allocation, orbital reordering, or CUDA work occurs.
"""
@inline function _spin_pair_code(::Type{UInt64}, left_number::UInt32, right_number::UInt32)
    return (UInt64(left_number) << 32) | UInt64(right_number)
end

"""
    _spin_pair_union_code(K, pair, left_numbers, right_numbers)

Return the exact lookup code for one physical spin pair during alpha/beta union
construction, using the same per-side numbering as determinant-pair traversal.
All arguments are read-only, absent numbered masks raise, and no allocation or
CUDA work occurs. The nothing/nothing case preserves physical-mask encoding.
"""
@inline function _spin_pair_union_code(::Type{K}, pair::CanonicalSpinStringPair, left_numbers, right_numbers) where {K}
    if left_numbers === nothing && right_numbers === nothing
        return _spin_pair_code(K, pair.left_bits, pair.right_bits)
    end
    left_number = _lookup_id(left_numbers, pair.left_bits)
    right_number = _lookup_id(right_numbers, pair.right_bits)
    left_number > 0 && right_number > 0 || throw(ArgumentError("union spin string absent from canonical numbering"))
    return _spin_pair_code(UInt64, UInt32(left_number), UInt32(right_number))
end
