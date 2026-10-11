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

 File:            src/integral_sources/JuliaMotra.jl
 Purpose:         Embedded Julia-MOTRA MO transformation against task-owned SEWARD snapshots.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    JuliaMOTRADimensions

Record AO, retained-MO, frozen, deleted, auxiliary, reduced-vector, and canonical-pair dimensions for one Julia-MOTRA transform.
All fields are dimensionless host integers and the value contains no array ownership or OpenMolcas handle.
"""
struct JuliaMOTRADimensions
    n_ao::Int
    n_orbitals::Int
    n_frozen::Int
    n_deleted::Int
    n_auxiliary::Int
    n_reduced::Int
    n_orbital_pairs::Int
end

"""
    JuliaMOTRAResult

Hold one shard's canonical-pair-by-auxiliary MO metric-whitened ERI factors and transform telemetry.
Retained-MO metric-whitened ERI factors are CPU-owned in square-root ERI units, elapsed time is seconds, and no OpenMolcas or CUDA handle is retained.
"""
struct JuliaMOTRAResult
    dimensions::JuliaMOTRADimensions
    retained_mo_eri_factors::Matrix{Float64}
    elapsed_seconds::Float64
    backend::Symbol
    gpu_device::Int
end

"""
    count_commonorb_orbitals(path)

Count `* ORBITAL` records in an OpenMolcas ASCII INPORB/CommonOrb file.
The file is read-only, the result is dimensionless, and missing files propagate an I/O error.
"""
function count_commonorb_orbitals(path::AbstractString)
    return count(line -> startswith(line, "* ORBITAL"), eachline(path))
end

"""
    read_commonorb_rows(path, n_ao, n_used)

Read the first `n_used` CommonOrb coefficient rows for an `n_ao`-function C1 basis.
The returned dense `Float64` matrix has orbitals in rows, the file is not modified, and malformed headers, nonnumeric coefficients, or truncated rows raise an error.
"""
function read_commonorb_rows(path::AbstractString, n_ao::Integer, n_used::Integer)
    n_ao >= 1 || throw(ArgumentError("CommonOrb AO dimension must be positive"))
    1 <= n_used <= n_ao || throw(ArgumentError("CommonOrb used-orbital count must lie in 1:n_ao"))
    lines = readlines(path)
    orbital_section = findfirst(line -> startswith(line, "#ORB"), lines)
    orbital_section === nothing && error("CommonOrb file has no #ORB section: $path")
    rows = Matrix{Float64}(undef, Int(n_used), Int(n_ao))
    line_index = orbital_section + 1
    for orbital in 1:Int(n_used)
        line_index <= length(lines) || error("CommonOrb ended before orbital $orbital")
        startswith(lines[line_index], "* ORBITAL") || error("expected CommonOrb orbital header at line $line_index")
        values = Float64[]
        line_index += 1
        while length(values) < n_ao && line_index <= length(lines)
            append!(values, parse.(Float64, split(lines[line_index])))
            line_index += 1
        end
        length(values) == n_ao || error("CommonOrb orbital $orbital has $(length(values)) coefficients, expected $n_ao")
        rows[orbital, :] .= values
    end
    return rows
end

"""
    scatter_reduced_eri_factors_c1!(ao, reduced, mapping, n_vectors)

Expand reduced C1 Cholesky vectors into symmetric AO matrices for one batch.
The function mutates the leading `n_vectors` slices of `ao`, leaves inputs unchanged, returns `ao`, and rejects invalid AO pair labels or dimensions.
"""
function scatter_reduced_eri_factors_c1!(
    ao::Array{Float64,3},
    reduced::Matrix{Float64},
    mapping::Matrix{Int64},
    n_vectors::Integer,
)
    n = Int(n_vectors)
    1 <= n <= size(ao, 3) || throw(ArgumentError("batch vector count is outside AO workspace"))
    size(reduced, 1) == size(mapping, 2) || throw(DimensionMismatch("reduced vector and mapping dimensions differ"))
    size(reduced, 2) >= n || throw(DimensionMismatch("reduced vector workspace has too few columns"))
    fill!(view(ao, :, :, 1:n), 0.0)
    n_ao = size(ao, 1)
    size(ao, 2) == n_ao || throw(DimensionMismatch("AO ERI-factor workspace must be square"))
    @inbounds for reduced_index in axes(mapping, 2)
        first_ao = Int(mapping[1, reduced_index])
        second_ao = Int(mapping[2, reduced_index])
        1 <= first_ao <= n_ao && 1 <= second_ao <= n_ao || error("invalid reduced-to-AO pair ($first_ao,$second_ao)")
        for vector_index in 1:n
            value = reduced[reduced_index, vector_index]
            ao[first_ao, second_ao, vector_index] = value
            ao[second_ao, first_ao, vector_index] = value
        end
    end
    return ao
end

"""
    transform_ao_eri_factors_cpu!(mo, half, orbitals, ao, n_vectors)

Apply the two-sided AO-to-MO transform to a batch using BLAS `mul!` calls.
The function mutates `half` and `mo`, leaves input orbitals/AO factors unchanged, and uses the caller-configured BLAS thread count for CPU parallelism.
"""
function transform_ao_eri_factors_cpu!(
    mo::Array{Float64,3},
    half::Array{Float64,3},
    orbitals::Matrix{Float64},
    ao::Array{Float64,3},
    n_vectors::Integer,
)
    orbital_transpose = transpose(orbitals)
    for vector_index in 1:Int(n_vectors)
        mul!(view(half, :, :, vector_index), orbitals, view(ao, :, :, vector_index))
        mul!(view(mo, :, :, vector_index), view(half, :, :, vector_index), orbital_transpose)
    end
    return mo
end

"""
    JuliaMOTRAGPUWorkspace(orbitals, n_ao, n_orbitals, batch_size, gpu_device)

Allocate reusable CUDA arrays for batched two-sided AO-to-MO Cholesky transformation.
The selected device is zero based, allocations are in bytes implied by the dense dimensions, and unavailable CUDA or inconsistent shapes raise before the workspace is returned.
"""
mutable struct JuliaMOTRAGPUWorkspace
    orbitals
    orbitals_transpose
    ao
    half
    mo
end

"""
    pack_mo_eri_factors!(retained_mo_eri_factors, first_auxiliary, mo, n_vectors)

Pack symmetric retained-MO matrices into canonical column-major orbital-pair rows and consecutive auxiliary columns.
The function mutates `retained_mo_eri_factors`, leaves `mo` unchanged, returns the destination, and rejects insufficient matrix or column capacity.
"""
function pack_mo_eri_factors!(
    retained_mo_eri_factors::Matrix{Float64},
    first_auxiliary::Integer,
    mo::Array{Float64,3},
    n_vectors::Integer,
)
    n_orbitals = size(mo, 1)
    size(mo, 2) == n_orbitals || throw(DimensionMismatch("MO ERI-factor matrices must be square"))
    size(retained_mo_eri_factors, 1) == canonical_pair_count(n_orbitals) || throw(DimensionMismatch("canonical-pair destination row count mismatch"))
    last_auxiliary = Int(first_auxiliary) + Int(n_vectors) - 1
    1 <= first_auxiliary <= last_auxiliary <= size(retained_mo_eri_factors, 2) || throw(BoundsError(retained_mo_eri_factors, (:, first_auxiliary:last_auxiliary)))
    pair = 0
    @inbounds for upper in 1:n_orbitals, lower in 1:upper
        pair += 1
        for local_auxiliary in 1:Int(n_vectors)
            retained_mo_eri_factors[pair, Int(first_auxiliary) + local_auxiliary - 1] = mo[upper, lower, local_auxiliary]
        end
    end
    return retained_mo_eri_factors
end

"""
    transform_staged_seward_eri_factors(commonorb; n_auxiliary, n_frozen, n_deleted, batch_size, backend, gpu_device)

Decode the active staged C1 SEWARD Cholesky source and transform it into canonical retained-MO factors.
The caller owns OpenMolcas initialization and staging; returned retained-MO metric-whitened ERI factors are CPU-owned, the staged RunFile is updated only with frozen/deleted counts, and GPU requests use cuBLAS without fallback.
"""
function transform_staged_seward_eri_factors(
    commonorb::AbstractString;
    n_auxiliary::Integer,
    n_frozen::Integer,
    n_deleted::Integer,
    batch_size::Integer,
    backend::Symbol,
    gpu_device::Integer,
)
    backend in (:cpu, :gpu) || throw(ArgumentError("Julia-MOTRA backend must be :cpu or :gpu"))
    molcas_namerun!("RUNFILE")
    Int(molcas_get_iscalar("nSym")) == 1 || error("Julia-MOTRA currently supports C1 only")
    n_ao = Int(only(molcas_get_iarray("nBas", 1)))
    available = count_commonorb_orbitals(commonorb)
    n_orbitals = n_ao - Int(n_frozen) - Int(n_deleted)
    n_orbitals > 0 || error("invalid retained dimension from n_ao=$n_ao, n_frozen=$n_frozen, n_deleted=$n_deleted")
    available >= Int(n_frozen) + n_orbitals || error("CommonOrb contains $available orbitals but $(Int(n_frozen) + n_orbitals) are required")
    molcas_put_iarray!("nFro", Int64[Int(n_frozen)])
    molcas_put_iarray!("nDel", Int64[Int(n_deleted)])
    rows = read_commonorb_rows(commonorb, n_ao, Int(n_frozen) + n_orbitals)
    retained = Matrix(view(rows, Int(n_frozen)+1:Int(n_frozen)+n_orbitals, :))
    status = molcas_cholesky_initialize!(0.0)
    status == 0 || error("OpenMolcas Cho_X_Init failed with status $status")
    gpu_workspace = nothing
    try
        n_reduced = molcas_reduced_cholesky_dimension()
        n_reduced > 0 || error("invalid reduced ERI-factor dimension $n_reduced")
        mapping = Matrix{Int64}(undef, 2, n_reduced)
        molcas_reduced_to_ao_map!(mapping, 1)
        batch = Int(batch_size)
        reduced = Matrix{Float64}(undef, n_reduced, batch)
        decode_scratch = Vector{Float64}(undef, n_reduced * (batch + 1) + 1)
        ao = Array{Float64}(undef, n_ao, n_ao, batch)
        mo = Array{Float64}(undef, n_orbitals, n_orbitals, batch)
        half = backend == :cpu ? Array{Float64}(undef, n_orbitals, n_ao, batch) : Array{Float64}(undef, 0, 0, 0)
        retained_mo_eri_factors = Matrix{Float64}(undef, canonical_pair_count(n_orbitals), Int(n_auxiliary))
        backend == :gpu && (gpu_workspace = _cuda_dispatch(:JuliaMOTRAGPUWorkspace,retained, n_ao, n_orbitals, batch, Int(gpu_device)))
        dimensions = JuliaMOTRADimensions(
            n_ao, n_orbitals, Int(n_frozen), Int(n_deleted), Int(n_auxiliary),
            n_reduced, canonical_pair_count(n_orbitals),
        )
        started = time_ns()
        first_auxiliary = 1
        while first_auxiliary <= n_auxiliary
            n_vectors = min(batch, Int(n_auxiliary) - first_auxiliary + 1)
            molcas_read_cholesky_vectors!(view(reduced, :, 1:n_vectors), first_auxiliary, 1, decode_scratch)
            scatter_reduced_eri_factors_c1!(ao, reduced, mapping, n_vectors)
            if backend == :gpu
                transform_ao_eri_factors_gpu!(mo, something(gpu_workspace), ao, n_vectors)
            else
                transform_ao_eri_factors_cpu!(mo, half, retained, ao, n_vectors)
            end
            pack_mo_eri_factors!(retained_mo_eri_factors, first_auxiliary, mo, n_vectors)
            first_auxiliary += n_vectors
        end
        elapsed = (time_ns() - started) / 1.0e9
        return JuliaMOTRAResult(dimensions, retained_mo_eri_factors, elapsed, backend, backend == :gpu ? Int(gpu_device) : -1)
    finally
        gpu_workspace === nothing || release_julia_motra_gpu_workspace!(gpu_workspace)
        final_status = molcas_cholesky_finalize!()
        final_status == 0 || @warn "OpenMolcas Cho_X_Final returned nonzero" status=final_status
    end
end
