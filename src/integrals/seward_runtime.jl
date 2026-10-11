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

 File:            src/integrals/seward_runtime.jl
 Purpose:         Retained-MO Cholesky products and Julia-MOTRA provenance from serial or MPI snapshots.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    SewardMOTRAProduct

Hold transformed retained-MO metric-whitened ERI factors and aggregate Julia-MOTRA provenance from one serial or MPI snapshot.
Factors are CPU-owned in square-root ERI units, time is seconds, and no OpenMolcas/CUDA handle or temporary shard storage is retained.
"""
struct SewardMOTRAProduct{T<:AbstractFloat}
    factors::ERIFactorValues{T}
    source_ranks::Int
    transform_backend::Symbol
    transform_gpu_device::Int
    transform_seconds::Float64
end

"""
    seward_orbital_dimensions(source)

Return `(n_ao, n_orbitals)` from the complete CommonOrb record count and configured frozen/deleted spaces.
The CommonOrb file is read-only, dimensions are unitless, and inconsistent spaces raise before OpenMolcas initialization or large allocation.
"""
function seward_orbital_dimensions(source::SewardMOTRASource)
    n_ao = count_commonorb_orbitals(source.commonorb)
    n_orbitals = n_ao - source.n_frozen - source.n_deleted
    n_orbitals > 0 || error(
        "SEWARD/CommonOrb settings imply no retained orbitals: n_ao=$n_ao, n_frozen=$(source.n_frozen), n_deleted=$(source.n_deleted)",
    )
    return (; n_ao, n_orbitals)
end

"""
    seward_motra_memory(T, source, n_orbitals)

Estimate peak Julia-owned host/device numerical storage for SEWARD decoding and AO-to-retained-MO transformation.
The inspection reads only small shard metadata and CommonOrb record counts, reports bytes, and does not initialize OpenMolcas or CUDA.
"""
function seward_motra_memory(
    ::Type{T},
    source::SewardMOTRASource,
    n_orbitals::Integer,
) where {T<:AbstractFloat}
    n_orbitals >= 1 || throw(ArgumentError("retained orbital count must be positive"))
    layout = inspect_seward_source(source)
    n_ao = Int(n_orbitals) + source.n_frozen + source.n_deleted
    n_pairs = canonical_pair_count(Int(n_orbitals))
    n_reduced_upper = canonical_pair_count(n_ao)
    batch = source.batch_size
    factor_bytes = estimate_array_bytes(T, n_pairs, layout.n_auxiliary)
    commonorb_bytes = estimate_array_bytes(Float64, source.n_frozen + Int(n_orbitals), n_ao)
    reduced_bytes = estimate_array_bytes(Float64, n_reduced_upper, batch)
    decode_bytes = estimate_array_bytes(Float64, n_reduced_upper * (batch + 1) + 1)
    mapping_bytes = estimate_array_bytes(Int64, 2, n_reduced_upper)
    ao_bytes = estimate_array_bytes(Float64, n_ao, n_ao, batch)
    mo_bytes = estimate_array_bytes(Float64, Int(n_orbitals), Int(n_orbitals), batch)
    half_bytes = source.transform_backend == :cpu ? estimate_array_bytes(Float64, Int(n_orbitals), n_ao, batch) : 0
    host_bytes = sum((factor_bytes, commonorb_bytes, reduced_bytes, decode_bytes, mapping_bytes, ao_bytes, mo_bytes, half_bytes))
    device_bytes = if source.transform_backend == :gpu
        sum((
            estimate_array_bytes(Float64, 2, Int(n_orbitals), n_ao),
            ao_bytes,
            estimate_array_bytes(Float64, Int(n_orbitals), n_ao, batch),
            mo_bytes,
        ))
    else
        0
    end
    return MemoryEstimate(host_bytes, device_bytes)
end

"""
    _stage_seward_rank!(stage, layout, shard)

Materialize one task-owned logical-name staging directory for the OpenMolcas Cholesky decoder.
The function copies only mutable `RUNFILE/CHORST` metadata, links large read-only source files, returns `stage`, and never modifies the original SEWARD snapshot.
"""
function _stage_seward_rank!(stage::AbstractString, layout::SewardSourceLayout, shard::SewardShardInfo)
    source = layout.source
    cp(_seward_source_file(source, 0, "RunFile"), joinpath(stage, "RUNFILE"); force=true)
    _replace_stage_link(_seward_source_file(source, 0, "OneInt"), joinpath(stage, "ONEINT"))
    _replace_stage_link(_seward_source_file(source, 0, "ChRed"), joinpath(stage, "CHRED"))
    _replace_stage_link(_seward_source_file(source, 0, "ChMap"), joinpath(stage, "CHOMAP"))
    _replace_stage_link(shard.chvec_path, joinpath(stage, "CHVEC1"))
    if source.ranks == 1
        _replace_stage_link(shard.chrst_path, joinpath(stage, "CHORST"))
    else
        local_chrst = read_seward_chrst(shard.chrst_path)
        write_normalized_shard_chrst(joinpath(stage, "CHORST"), local_chrst, layout.global_chrst, shard)
    end
    return stage
end

"""
    _assemble_seward_factor_columns!(global_eri_factors, shard_eri_factors, global_auxiliary_ids)

Copy one transformed SEWARD shard into its exact positions on the global ERI-factor auxiliary axis.
Both factor matrices are in canonical-orbital-pair rows and square-root ERI units; `global_auxiliary_ids`
contains one-based global Cholesky-vector indices in the shard's local column order. The destination is
mutated only at those columns, inputs are otherwise read-only, and row, column, range, or duplicate-ID
mismatches raise before copying. No CUDA/OpenMolcas state is accessed and no molecular-scale temporary
matrix is allocated.
"""
function _assemble_seward_factor_columns!(
    global_eri_factors::AbstractMatrix{T},
    shard_eri_factors::AbstractMatrix{T},
    global_auxiliary_ids::AbstractVector{<:Integer},
) where {T<:AbstractFloat}
    size(shard_eri_factors, 1) == size(global_eri_factors, 1) || throw(DimensionMismatch(
        "SEWARD shard/global ERI factor orbital-pair dimensions differ",
    ))
    size(shard_eri_factors, 2) == length(global_auxiliary_ids) || throw(DimensionMismatch(
        "SEWARD shard ERI factor columns differ from global auxiliary-ID count",
    ))
    n_auxiliary = size(global_eri_factors, 2)
    auxiliary_ids = Int.(global_auxiliary_ids)
    all(global_auxiliary_index -> 1 <= global_auxiliary_index <= n_auxiliary, auxiliary_ids) || throw(BoundsError(
        axes(global_eri_factors, 2), auxiliary_ids,
    ))
    length(unique(auxiliary_ids)) == length(auxiliary_ids) || throw(ArgumentError(
        "SEWARD shard contains duplicate global auxiliary IDs",
    ))
    for (local_auxiliary_index, global_auxiliary_index) in enumerate(auxiliary_ids)
        copyto!(
            view(global_eri_factors, :, global_auxiliary_index),
            view(shard_eri_factors, :, local_auxiliary_index),
        )
    end
    return global_eri_factors
end

"""
    transform_seward_eri_factors(source; T=Float64, runtime=RuntimeOptions())

Transform serial or MPI-sharded C1 SEWARD AO Cholesky vectors directly into retained CommonOrb/MO factors using CPU BLAS or one CUDA device.
Source files are read-only, only task-owned temporary metadata is created and automatically removed, memory limits are enforced before large allocation, and no fallback occurs when the configured GPU path fails.
"""
function transform_seward_eri_factors(
    source::SewardMOTRASource;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=RuntimeOptions(),
) where {S<:AbstractFloat}
    layout = inspect_seward_source(source)
    dimensions = seward_orbital_dimensions(source)
    n_orbitals = dimensions.n_orbitals
    estimate = seward_motra_memory(S, source, n_orbitals)
    enforce_memory_limits(estimate, runtime)
    source.transform_backend == :gpu && !runtime.use_gpu && throw(ArgumentError(
        "SEWARD transform_backend=:gpu requires runtime.use_gpu=true",
    ))
    source.transform_backend == :gpu && source.gpu_device != runtime.gpu_device && throw(ArgumentError(
        "SEWARD gpu_device=$(source.gpu_device) must equal runtime primary gpu_device=$(runtime.gpu_device)",
    ))
    global_retained_mo_eri_factors = Matrix{Float64}(undef, canonical_pair_count(n_orbitals), layout.n_auxiliary)
    elapsed_seconds = 0.0
    lock(_MOLCAS_RUNTIME_LOCK)
    try
        mktempdir(source.scratch_parent; prefix="noci_seward_motra_") do stage
            for shard in layout.shards
                _stage_seward_rank!(stage, layout, shard)
                shard_motra_result = cd(stage) do
                    initialize_molcas_runtime!(
                        stage,
                        "NOCI_MOTRA_RANK_$(shard.rank)";
                        molcas_root=source.molcas_root,
                        omc_root=source.omc_root,
                        module_name="MOTRA",
                    )
                    try
                        molcas_namerun!("RUNFILE")
                        original_auxiliary = Int(only(molcas_get_iarray("NumCho", 1)))
                        original_auxiliary == layout.n_auxiliary || error(
                            "SEWARD RunFile NumCho=$original_auxiliary but shards contain $(layout.n_auxiliary)",
                        )
                        source.ranks == 1 || molcas_put_iarray!("NumCho", Int64[shard.local_count])
                        transform_staged_seward_eri_factors(
                            source.commonorb;
                            n_auxiliary=shard.local_count,
                            n_frozen=source.n_frozen,
                            n_deleted=source.n_deleted,
                            batch_size=source.batch_size,
                            backend=source.transform_backend,
                            gpu_device=source.gpu_device,
                        )
                    finally
                        finalize_molcas_runtime!()
                    end
                end
                shard_motra_result.dimensions.n_orbitals == n_orbitals || throw(DimensionMismatch("SEWARD shard retained dimension changed"))
                _assemble_seward_factor_columns!(
                    global_retained_mo_eri_factors,
                    shard_motra_result.retained_mo_eri_factors,
                    shard.global_ids,
                )
                elapsed_seconds += shard_motra_result.elapsed_seconds
            end
        end
    finally
        _MOLCAS_LIBRARY_HANDLE[] == C_NULL || finalize_molcas_runtime!()
        unlock(_MOLCAS_RUNTIME_LOCK)
    end
    typed_retained_mo_eri_factors = S == Float64 ? global_retained_mo_eri_factors : Matrix{S}(global_retained_mo_eri_factors)
    factors = ERIFactorValues{S}(n_orbitals, typed_retained_mo_eri_factors, Val(:validated_owned))
    return SewardMOTRAProduct(
        factors,
        source.ranks,
        source.transform_backend,
        source.transform_backend == :gpu ? source.gpu_device : -1,
        elapsed_seconds,
    )
end

"""
    load_seward_eri_factors(source; T=Float64, runtime=RuntimeOptions())

Return only the retained-MO metric-whitened ERI factors from an embedded SEWARD/Julia-MOTRA transformation.
The source remains read-only, returned factors are CPU-owned, and all validation, memory, OpenMolcas, CPU/GPU, and cleanup constraints of `transform_seward_eri_factors` apply.
"""
function load_seward_eri_factors(
    source::SewardMOTRASource;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=RuntimeOptions(),
) where {S<:AbstractFloat}
    return transform_seward_eri_factors(source; T, runtime).factors
end
