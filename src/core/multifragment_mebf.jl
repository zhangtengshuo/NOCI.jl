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

 File:            src/core/multifragment_mebf.jl
 Purpose:         Packed 64-active-orbital spin microstate expansions for one fragment state.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    PackedSpinMicrostateExpansion64

Store normalized spin-projection-sector coefficients together with packed alpha/beta active occupations for one fragment state in a domain of at most 64 active spatial orbitals.
All arrays are CPU-owned and dimensionless, sector counts sum to the determinant count, no occupation strings or CUDA storage are retained, and malformed bit domains or coefficients raise during construction.
"""
struct PackedSpinMicrostateExpansion64{T<:AbstractFloat}
    coefficients::Vector{T}
    alpha_bits::Vector{UInt64}
    beta_bits::Vector{UInt64}
    sector_counts::Vector{Int}
    spin_multiplicity::Int
    n_active_orbitals::Int

    function PackedSpinMicrostateExpansion64(
        coefficients::AbstractVector{T},
        alpha_bits::AbstractVector{UInt64},
        beta_bits::AbstractVector{UInt64},
        sector_counts::AbstractVector{<:Integer},
        spin_multiplicity::Integer,
        n_active_orbitals::Integer,
    ) where {T<:AbstractFloat}
        rank = Int(n_active_orbitals)
        0 <= rank <= 64 || throw(ArgumentError("packed multifragment microstates support active rank 0:64"))
        multiplicity = Int(spin_multiplicity)
        multiplicity >= 1 || throw(ArgumentError("spin_multiplicity must be positive"))
        counts = Int.(sector_counts)
        length(counts) == multiplicity || throw(DimensionMismatch("sector count length must equal spin multiplicity"))
        all(>=(0), counts) || throw(ArgumentError("sector counts must be nonnegative"))
        count = length(coefficients)
        length(alpha_bits) == count && length(beta_bits) == count || throw(DimensionMismatch("coefficient and packed occupation counts differ"))
        sum(counts) == count || throw(DimensionMismatch("sector counts do not sum to the microstate count"))
        all(isfinite, coefficients) || throw(ArgumentError("microstate coefficients contain a non-finite value"))
        valid_mask = rank == 64 ? typemax(UInt64) : (UInt64(1) << rank) - UInt64(1)
        all(bits -> bits & ~valid_mask == 0, alpha_bits) || throw(ArgumentError("alpha microstate bits exceed the active domain"))
        all(bits -> bits & ~valid_mask == 0, beta_bits) || throw(ArgumentError("beta microstate bits exceed the active domain"))
        return new{T}(
            Vector{T}(coefficients),
            Vector{UInt64}(alpha_bits),
            Vector{UInt64}(beta_bits),
            counts,
            multiplicity,
            rank,
        )
    end
end

"""
    PreparedPackedFragment64

Cache one fragment's 64-active packed spin microstates together with sector ranges, stable descending-magnitude determinant orders, and sector maxima for exact multifragment branch-and-bound traversal.
The value owns CPU metadata and microstate arrays, is treated as read-only after construction, and contains no molecular product or CUDA state.
"""
struct PreparedPackedFragment64{T<:AbstractFloat}
    microstates::PackedSpinMicrostateExpansion64{T}
    sector_ranges::Vector{UnitRange{Int}}
    sorted_indices::Vector{Vector{Int}}
    maximum_magnitudes::Vector{T}
end

"""
    MultifragmentSectorPath

Store one legal ordered fragment spin-sector path and the product of Clebsch-Gordan coefficients for the user-specified left-to-right intermediate-spin coupling basis.
Sector indices and the coefficient are dimensionless, the path owns its small CPU vector, and no determinant products or CUDA storage are retained.
"""
struct MultifragmentSectorPath{T<:AbstractFloat}
    sectors::Vector{Int}
    clebsch_product::T
end

"""
    MultifragmentProduct64

Store one exactly screened molecular determinant product before final normalization/fermionic phase correction using global alpha/beta active masks in at most 64 spatial orbitals.
Coefficients are dimensionless, arrays are CPU-owned and exact-sized after a count pass, and no molecular orbital or CUDA data are retained.
"""
struct MultifragmentProduct64{T<:AbstractFloat}
    coefficients::Vector{T}
    alpha_bits::Vector{UInt64}
    beta_bits::Vector{UInt64}
    n_active_orbitals::Int
end

"""
    _packed_spin_pair_key64(alpha_bits, beta_bits)

Encode one alpha/beta occupation-mask pair injectively into a `UInt128` CPU dictionary key.
The 64-bit masks are read-only and no allocation or CUDA work occurs.
"""
@inline function _packed_spin_pair_key64(alpha_bits::UInt64, beta_bits::UInt64)
    return (UInt128(alpha_bits) << 64) | UInt128(beta_bits)
end

"""
    _unpack_spin_pair_key64(key)

Decode one injective `UInt128` alpha/beta occupation key into its two `UInt64` masks.
The key is read-only, the return value is dimensionless, and no allocation or CUDA work occurs.
"""
@inline function _unpack_spin_pair_key64(key::UInt128)
    return UInt64(key >> 64), UInt64(key & UInt128(typemax(UInt64)))
end

"""
    _canonical_occupation_sort_key64(alpha_bits, beta_bits, n_active_orbitals)

Return the canonical two-bit lexicographic occupation key `0 < 2 < alpha < beta` for packed alpha/beta masks over at most 64 active orbitals.
Inputs are dimensionless and read-only, the `UInt128` result is injective over the validated active domain, no allocation occurs, and out-of-domain bits raise `ArgumentError`.
"""
function _canonical_occupation_sort_key64(
    alpha_bits::UInt64,
    beta_bits::UInt64,
    n_active_orbitals::Integer,
)
    rank = Int(n_active_orbitals)
    0 <= rank <= 64 || throw(ArgumentError("active rank must lie in 0:64"))
    valid_mask = rank == 64 ? typemax(UInt64) : (UInt64(1) << rank) - UInt64(1)
    (alpha_bits | beta_bits) & ~valid_mask == 0 || throw(ArgumentError("occupation bits exceed the active domain"))
    key = UInt128(0)
    @inbounds for orbital in 1:rank
        bit = UInt64(1) << (orbital - 1)
        alpha = !iszero(alpha_bits & bit)
        beta = !iszero(beta_bits & bit)
        digit = alpha ? (beta ? UInt128(1) : UInt128(2)) : (beta ? UInt128(3) : UInt128(0))
        key = (key << 2) | digit
    end
    return key
end

"""
    _spin_sector_ranges64(expansion)

Return contiguous one-based ranges for every stored spin-projection sector of a packed 64-active microstate expansion.
The input is read-only, the returned small CPU vector is owned, and inconsistent sector metadata raises `DimensionMismatch`.
"""
function _spin_sector_ranges64(expansion::PackedSpinMicrostateExpansion64)
    ranges = Vector{UnitRange{Int}}(undef, length(expansion.sector_counts))
    first_index = 1
    for sector in eachindex(expansion.sector_counts)
        last_index = first_index + expansion.sector_counts[sector] - 1
        ranges[sector] = first_index:last_index
        first_index = last_index + 1
    end
    first_index == length(expansion.coefficients) + 1 || throw(DimensionMismatch("packed microstate sector ranges are incomplete"))
    return ranges
end

"""
    _spin_microstates64(expansion, spin_multiplicity)

Generate all spin-projection sectors from one highest-projection determinant expansion using packed alpha/beta masks and repeated exact spin lowering over at most 64 active orbitals.
The input is read-only, each generated sector is normalized independently as in the established dimer builder, duplicate lowered occupations are deterministically accumulated on CPU, no occupation strings/CUDA work occur, and impossible multiplicities raise `ArgumentError`.
"""
function _spin_microstates64(
    expansion::DeterminantExpansion{T},
    spin_multiplicity::Integer,
) where {T<:AbstractFloat}
    multiplicity = Int(spin_multiplicity)
    multiplicity >= 1 || throw(ArgumentError("spin_multiplicity must be positive"))
    n_active = active_orbital_count(expansion)
    n_active <= 64 || throw(ArgumentError("multifragment MEBF construction supports at most 64 active orbitals"))

    current_coefficients = _normalized_coefficients(expansion.coefficients)
    current_alpha = copy(expansion.active_occupations.alpha_bits)
    current_beta = copy(expansion.active_occupations.beta_bits)
    sector_counts = zeros(Int, multiplicity)
    sector_counts[1] = length(current_coefficients)
    all_coefficients = copy(current_coefficients)
    all_alpha = copy(current_alpha)
    all_beta = copy(current_beta)

    for sector in 2:multiplicity
        coefficient_by_key = Dict{UInt128,T}()
        @inbounds for determinant in eachindex(current_coefficients)
            coefficient = current_coefficients[determinant]
            alpha_bits = current_alpha[determinant]
            beta_bits = current_beta[determinant]
            alpha_only = alpha_bits & ~beta_bits
            while !iszero(alpha_only)
                orbital = trailing_zeros(alpha_only)
                bit = UInt64(1) << orbital
                lowered_alpha = alpha_bits & ~bit
                lowered_beta = beta_bits | bit
                key = _packed_spin_pair_key64(lowered_alpha, lowered_beta)
                coefficient_by_key[key] = get(coefficient_by_key, key, zero(T)) + coefficient
                alpha_only &= alpha_only - UInt64(1)
            end
        end
        isempty(coefficient_by_key) && throw(ArgumentError("cannot generate spin sector $sector for multiplicity $multiplicity"))
        keys_ordered = collect(keys(coefficient_by_key))
        sort!(keys_ordered; by=key -> begin
            alpha_bits, beta_bits = _unpack_spin_pair_key64(key)
            _canonical_occupation_sort_key64(alpha_bits, beta_bits, n_active)
        end)
        current_coefficients = _normalized_coefficients(T[coefficient_by_key[key] for key in keys_ordered])
        current_alpha = Vector{UInt64}(undef, length(keys_ordered))
        current_beta = Vector{UInt64}(undef, length(keys_ordered))
        @inbounds for index in eachindex(keys_ordered)
            current_alpha[index], current_beta[index] = _unpack_spin_pair_key64(keys_ordered[index])
        end
        sector_counts[sector] = length(current_coefficients)
        append!(all_coefficients, current_coefficients)
        append!(all_alpha, current_alpha)
        append!(all_beta, current_beta)
    end

    return PackedSpinMicrostateExpansion64(
        all_coefficients,
        all_alpha,
        all_beta,
        sector_counts,
        multiplicity,
        n_active,
    )
end

"""
    _prepare_packed_fragment64(expansion, spin_multiplicity)

Build the immutable 64-active packed microstate cache for one fragment determinant expansion and stable sector-local magnitude orders used by exact multifragment screening.
The determinant expansion is read-only, returned arrays are CPU-owned, no molecular products/CUDA state are created, and invalid spin metadata propagates from `_spin_microstates64`.
"""
function _prepare_packed_fragment64(
    expansion::DeterminantExpansion{T},
    spin_multiplicity::Integer,
) where {T<:AbstractFloat}
    microstates = _spin_microstates64(expansion, spin_multiplicity)
    sector_ranges = _spin_sector_ranges64(microstates)
    sorted_indices = [sort!(collect(range); by=index -> abs(microstates.coefficients[index]), rev=true) for range in sector_ranges]
    maximum_magnitudes = T[isempty(indices) ? zero(T) : abs(microstates.coefficients[first(indices)]) for indices in sorted_indices]
    return PreparedPackedFragment64(microstates, sector_ranges, sorted_indices, maximum_magnitudes)
end

"""
    _prepare_fragment_microstate_cache64(problem)

Prepare each fragment state referenced by any MEBF exactly once using the 64-active packed multifragment representation without imposing the legacy dimer-only topology restriction.
The loaded problem is read-only, returned CPU dictionary entries are owned and treated as immutable, no molecular determinant products/CUDA work occur, and per-fragment active ranks above 64 raise before assembly.
"""
function _prepare_fragment_microstate_cache64(problem::NOCIProblemDefinition{T}) where {T<:AbstractFloat}
    referenced_indices = Set{Int}()
    for composition in problem.mebfs
        union!(referenced_indices, composition.fragment_state_indices)
        total_active = sum(active_orbital_count(problem.fragment_states[index].wavefunction.determinants.expansion) for index in composition.fragment_state_indices)
        total_active <= 64 || throw(ArgumentError("multifragment MEBF construction supports at most 64 total active orbitals per MEBF"))
    end
    cache = Dict{Int,PreparedPackedFragment64{T}}()
    for index in sort!(collect(referenced_indices))
        determinant_data = problem.fragment_states[index].wavefunction.determinants
        cache[index] = _prepare_packed_fragment64(determinant_data.expansion, determinant_data.spin_multiplicity)
    end
    return cache
end

"""
    _sector_projection_twice(spin_multiplicity, sector)

Return twice the spin projection for one one-based sector ordered from highest to lowest projection.
Arguments and result are dimensionless integers, no allocation occurs, and invalid multiplicities or sector indices raise immediately.
"""
@inline function _sector_projection_twice(spin_multiplicity::Integer, sector::Integer)
    multiplicity = Int(spin_multiplicity)
    multiplicity >= 1 || throw(ArgumentError("spin multiplicity must be positive"))
    1 <= sector <= multiplicity || throw(BoundsError(1:multiplicity, sector))
    return multiplicity - 1 - 2 * (Int(sector) - 1)
end

"""
    _multifragment_sector_paths(prepared, coupling_multiplicities)

Enumerate all legal local spin-projection-sector paths for an ordered multifragment product under fixed left-to-right intermediate coupling multiplicities.
The prepared fragment caches are read-only, returned small CPU paths own sector indices and Clebsch-Gordan products, final projection is the highest projection of the requested total spin, no determinant Cartesian product/CUDA work occurs, and inconsistent coupling counts raise `ArgumentError`.
"""
function _multifragment_sector_paths(
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    coupling_multiplicities::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    fragment_count = length(prepared)
    fragment_count >= 2 || throw(ArgumentError("multifragment sector traversal requires at least two fragments"))
    targets = Int.(coupling_multiplicities)
    length(targets) == fragment_count - 1 || throw(ArgumentError("one coupling multiplicity is required for each multifragment coupling step"))
    all(>(0), targets) || throw(ArgumentError("coupling multiplicities must be positive"))

    sectors = zeros(Int, fragment_count)
    paths = MultifragmentSectorPath{T}[]
    first_multiplicity = prepared[1].microstates.spin_multiplicity
    first_spin = (first_multiplicity - 1) / 2

    function descend(fragment::Int, previous_spin::Float64, previous_projection::Float64, clebsch_product::T)
        local_multiplicity = prepared[fragment].microstates.spin_multiplicity
        local_spin = (local_multiplicity - 1) / 2
        target_spin = (targets[fragment - 1] - 1) / 2
        for sector in 1:local_multiplicity
            local_projection = _sector_projection_twice(local_multiplicity, sector) / 2
            target_projection = previous_projection + local_projection
            abs(target_projection) <= target_spin + 1.0e-12 || continue
            coupling = T(clebsch_gordan_coefficient(
                previous_spin,
                local_spin,
                previous_projection,
                local_projection,
                target_spin,
                target_projection,
            ))
            iszero(coupling) && continue
            sectors[fragment] = sector
            product = clebsch_product * coupling
            if fragment == fragment_count
                abs(target_projection - target_spin) <= 1.0e-12 || continue
                push!(paths, MultifragmentSectorPath(copy(sectors), product))
            else
                descend(fragment + 1, target_spin, target_projection, product)
            end
        end
        return nothing
    end

    for first_sector in 1:first_multiplicity
        sectors[1] = first_sector
        first_projection = _sector_projection_twice(first_multiplicity, first_sector) / 2
        descend(2, first_spin, first_projection, one(T))
    end
    isempty(paths) && throw(ArgumentError("no legal multifragment spin-sector path reaches the requested total spin"))
    return paths
end

"""
    _visit_multifragment_path_products!(visitor, prepared, path, tau_ci)

Traverse determinant tuples for one legal spin-sector path with an exact monotone coefficient-product branch-and-bound and invoke `visitor(indices, coefficient_product)` for every survivor.
Prepared caches and path metadata are read-only, `tau_ci` uses the established pre-Clebsch local-CI-product semantics, determinant indices are supplied through one reused CPU vector, no full Cartesian product/CUDA storage is created, and invalid thresholds raise `ArgumentError`.
"""
function _visit_multifragment_path_products!(
    visitor,
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    path::MultifragmentSectorPath{T},
    tau_ci::T,
) where {T<:AbstractFloat}
    isfinite(tau_ci) && tau_ci >= zero(T) || throw(ArgumentError("tau_ci must be finite and nonnegative"))
    fragment_count = length(prepared)
    length(path.sectors) == fragment_count || throw(DimensionMismatch("sector path length differs from fragment count"))
    selected = zeros(Int, fragment_count)
    determinant_lists = Vector{Vector{Int}}(undef, fragment_count)
    maximum_magnitudes = Vector{T}(undef, fragment_count)
    for fragment in 1:fragment_count
        sector = path.sectors[fragment]
        determinant_lists[fragment] = prepared[fragment].sorted_indices[sector]
        maximum_magnitudes[fragment] = prepared[fragment].maximum_magnitudes[sector]
    end
    suffix_bound = ones(T, fragment_count)
    running = one(T)
    for fragment in fragment_count:-1:1
        suffix_bound[fragment] = running
        running *= maximum_magnitudes[fragment]
    end
    tau_ci > zero(T) && running < tau_ci && return nothing

    function descend(fragment::Int, prefix_magnitude::T, prefix_coefficient::T)
        indices = determinant_lists[fragment]
        microstates = prepared[fragment].microstates
        @inbounds for determinant in indices
            coefficient = microstates.coefficients[determinant]
            magnitude = prefix_magnitude * abs(coefficient)
            magnitude * suffix_bound[fragment] >= tau_ci || break
            selected[fragment] = determinant
            product = prefix_coefficient * coefficient
            if fragment == fragment_count
                magnitude >= tau_ci && visitor(selected, product)
            else
                descend(fragment + 1, magnitude, product)
            end
        end
        return nothing
    end

    descend(1, one(T), one(T))
    return nothing
end

"""
    multifragment_product_count(prepared, coupling_multiplicities, tau_ci)

Return the exact number of molecular determinant tuples surviving `tau_ci` across all legal spin-sector paths without materializing any molecular determinant product.
Inputs are read-only and dimensionless, counting uses checked CPU integer arithmetic and exact branch-and-bound traversal, no CUDA work occurs, and malformed coupling/threshold data raise before return.
"""
function multifragment_product_count(
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    coupling_multiplicities::AbstractVector{<:Integer},
    tau_ci::T,
) where {T<:AbstractFloat}
    paths = _multifragment_sector_paths(prepared, coupling_multiplicities)
    count = Ref(0)
    for path in paths
        _visit_multifragment_path_products!(prepared, path, tau_ci) do _, _
            count[] = Base.Checked.checked_add(count[], 1)
        end
    end
    return count[]
end

"""
    multifragment_product_memory(T, determinant_count)

Return the exact retained numerical payload for one materialized multifragment product containing a coefficient plus alpha/beta `UInt64` masks per determinant.
The estimate reports host bytes only, allocates no arrays, and rejects negative counts or checked overflow.
"""
function multifragment_product_memory(
    ::Type{T},
    determinant_count::Integer,
) where {T<:AbstractFloat}
    count = Int(determinant_count)
    count >= 0 || throw(ArgumentError("multifragment determinant count must be nonnegative"))
    bytes = Base.Checked.checked_add(
        estimate_array_bytes(T, count),
        estimate_array_bytes(UInt64, 2, count),
    )
    return MemoryEstimate(bytes, 0)
end

"""
    build_multifragment_product64(prepared, coupling_multiplicities, tau_ci; runtime=RuntimeOptions())

Count, memory-preflight, and exactly materialize all screened molecular determinant products for ordered fragment states in a total active domain of at most 64 orbitals.
The prepared fragment caches are read-only, screening is based only on the product of local CI magnitudes before Clebsch factors, global alpha/beta masks preserve fragment orbital order, outputs are CPU-owned, no CUDA work occurs, and insufficient host limits fail before molecular-product allocation.
"""
function build_multifragment_product64(
    prepared::AbstractVector{PreparedPackedFragment64{T}},
    coupling_multiplicities::AbstractVector{<:Integer},
    tau_ci::T;
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    fragment_count = length(prepared)
    fragment_count >= 2 || throw(ArgumentError("multifragment product construction requires at least two fragments"))
    total_active = sum(fragment.microstates.n_active_orbitals for fragment in prepared)
    total_active <= 64 || throw(ArgumentError("multifragment product exceeds the 64-active-orbital domain"))
    paths = _multifragment_sector_paths(prepared, coupling_multiplicities)
    count = Ref(0)
    for path in paths
        _visit_multifragment_path_products!(prepared, path, tau_ci) do _, _
            count[] = Base.Checked.checked_add(count[], 1)
        end
    end
    count[] > 0 || throw(ArgumentError("no molecular determinant products survived multifragment screening"))
    enforce_memory_limits(multifragment_product_memory(T, count[]), runtime)

    coefficients = Vector{T}(undef, count[])
    alpha_bits = Vector{UInt64}(undef, count[])
    beta_bits = Vector{UInt64}(undef, count[])
    offsets = zeros(Int, fragment_count)
    for fragment in 2:fragment_count
        offsets[fragment] = offsets[fragment - 1] + prepared[fragment - 1].microstates.n_active_orbitals
    end

    position = Ref(0)
    for path in paths
        _visit_multifragment_path_products!(prepared, path, tau_ci) do selected, local_product
            position[] += 1
            alpha = UInt64(0)
            beta = UInt64(0)
            @inbounds for fragment in 1:fragment_count
                microstates = prepared[fragment].microstates
                determinant = selected[fragment]
                shift = offsets[fragment]
                alpha |= microstates.alpha_bits[determinant] << shift
                beta |= microstates.beta_bits[determinant] << shift
            end
            @inbounds coefficients[position[]] = local_product * path.clebsch_product
            @inbounds alpha_bits[position[]] = alpha
            @inbounds beta_bits[position[]] = beta
        end
    end
    position[] == count[] || throw(ArgumentError("multifragment product count changed between exact count and fill passes"))
    return MultifragmentProduct64(coefficients, alpha_bits, beta_bits, total_active)
end

"""
    _multifragment_product_to_expansion(product, inactive_orbitals)

Stable magnitude-order, fermion-phase-correct, and convert one screened multifragment packed product without post-screen renormalization into the standard production `DeterminantExpansion` representation.
The product is read-only, output arrays are CPU-owned, phase conventions reuse the established alpha/beta and closed-shell canonicalization helpers, no CUDA work occurs, and zero-norm or invalid inactive metadata raise before return.
"""
function _multifragment_product_to_expansion(
    product::MultifragmentProduct64{T},
    inactive_orbitals::Integer,
) where {T<:AbstractFloat}
    inactive_orbitals >= 0 || throw(ArgumentError("inactive_orbitals must be nonnegative"))
    molecular_coefficients = product.coefficients
    isempty(molecular_coefficients) && throw(ArgumentError("multifragment product must contain at least one determinant"))
    all(isfinite, molecular_coefficients) || throw(ArgumentError("multifragment molecular coefficients must be finite"))
    any(coefficient -> !iszero(coefficient), molecular_coefficients) || throw(ArgumentError("multifragment molecular coefficient vector is identically zero"))
    order = sortperm(abs.(molecular_coefficients); rev=true)
    n_active = product.n_active_orbitals
    ordered_coefficients = Vector{T}(undef, length(order))
    ordered_alpha = Vector{UInt64}(undef, length(order))
    ordered_beta = Vector{UInt64}(undef, length(order))
    occupation = Vector{Int8}(undef, n_active)
    spin_scratch = Vector{Int8}(undef, 2 * n_active)
    occupied_scratch = Vector{Int8}(undef, n_active)

    @inbounds for (determinant, source_index) in enumerate(order)
        alpha_bits = product.alpha_bits[source_index]
        beta_bits = product.beta_bits[source_index]
        for orbital in 1:n_active
            bit = UInt64(1) << (orbital - 1)
            alpha = !iszero(alpha_bits & bit)
            beta = !iszero(beta_bits & bit)
            occupation[orbital] = alpha ? (beta ? Int8(2) : Int8(1)) : (beta ? Int8(-1) : Int8(0))
        end
        alpha_beta_sign = _alpha_beta_permutation_sign_with_scratch!(spin_scratch, occupation)
        closed_shell_sign = _closed_shell_permutation_sign_with_scratch!(occupied_scratch, occupation)
        ordered_coefficients[determinant] = molecular_coefficients[source_index] * T(alpha_beta_sign * closed_shell_sign)
        ordered_alpha[determinant] = alpha_bits
        ordered_beta[determinant] = beta_bits
    end

    packed = PackedActiveOccupations(n_active, ordered_alpha, ordered_beta; copy_data=false)
    return DeterminantExpansion(ordered_coefficients, packed, inactive_orbitals, Val(:owned))
end

"""
    _build_multifragment_molecular_basis_function(problem, composition, prepared_fragments, tau_ci; runtime=RuntimeOptions())

Build one arbitrary-fragment molecular basis function through fixed-order spin coupling, exact lazy determinant-product screening, 64-active packed occupations, established fermionic phase correction, and block-assembled fragment orbitals.
Scientific inputs and prepared caches are read-only, the returned MEBF owns CPU arrays, molecular product allocation is exact-count preflighted, no CUDA work occurs, and unsupported total active domains or forbidden spin couplings raise explicitly.
"""
function _build_multifragment_molecular_basis_function(
    problem::NOCIProblemDefinition{T},
    composition::MEBFComposition,
    prepared_fragments::AbstractDict{Int,PreparedPackedFragment64{T}},
    tau_ci::T;
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    length(composition.fragment_state_indices) >= 2 || throw(ArgumentError("multifragment MEBF construction requires at least two fragment states"))
    entries = [problem.fragment_states[index] for index in composition.fragment_state_indices]
    targets = _coupling_targets(length(entries), problem.spin_multiplicity, composition.coupling_multiplicities)
    prepared = PreparedPackedFragment64{T}[]
    for index in composition.fragment_state_indices
        haskey(prepared_fragments, index) || throw(ArgumentError("missing packed multifragment microstates for fragment-state index $index"))
        push!(prepared, prepared_fragments[index])
    end
    product = build_multifragment_product64(prepared, targets, tau_ci; runtime)
    orbital_coefficients, inactive_orbitals = _molecular_orbitals(problem.fragment_states, composition.fragment_state_indices)
    expansion = _multifragment_product_to_expansion(product, inactive_orbitals)
    correlation_energy_shift = sum(entry.wavefunction.determinants.correlation_energy for entry in entries)
    return MolecularBasisFunction(
        composition.label,
        copy(composition.fragment_state_indices),
        orbital_coefficients,
        expansion,
        correlation_energy_shift,
    )
end

"""
    _build_multifragment_molecular_basis_functions(problem; screening=ScreeningOptions(), runtime=RuntimeOptions())

Build every loaded problem MEBF serially through the new arbitrary-fragment exact screened product engine while reusing one packed microstate cache across all compositions.
The problem is read-only, outputs are CPU-owned and preserve MEBF order, each molecular product is exact-count memory-preflighted, no CUDA work occurs, and this internal characterization entry point does not replace the established production dimer builder yet.
"""
function _build_multifragment_molecular_basis_functions(
    problem::NOCIProblemDefinition{T};
    screening::ScreeningOptions{T}=ScreeningOptions(tau_ci=T(1.0e-6), tau_ci_offdiagonal=T(1.0e-5), tau_singularity=T(1.0e-12)),
    runtime::RuntimeOptions=RuntimeOptions(),
) where {T<:AbstractFloat}
    prepared = _prepare_fragment_microstate_cache64(problem)
    basis_functions = Vector{MolecularBasisFunction{T}}(undef, length(problem.mebfs))
    for index in eachindex(problem.mebfs)
        basis_functions[index] = _build_multifragment_molecular_basis_function(
            problem,
            problem.mebfs[index],
            prepared,
            screening.tau_ci;
            runtime,
        )
    end
    return basis_functions
end
