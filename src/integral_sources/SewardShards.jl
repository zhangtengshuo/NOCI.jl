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

 File:            src/integral_sources/SewardShards.jl
 Purpose:         Decoded SEWARD RunFile, OneInt, ChRed, and MPI-shard Cholesky vector metadata.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    SewardChRst

Decoded metadata from one SEWARD `ChRst` file, including the original bytes and five ILP64 vector-information columns.
Offsets are one-based byte indices, counts are dimensionless, and construction itself performs no file or device operation.
"""
struct SewardChRst
    bytes::Vector{UInt8}
    numcho_offset::Int
    infvec_offset::Int
    nrows::Int
    columns::Matrix{Int64}
end

"""
    SewardShardInfo

Describe one validated rank-local `ChVec1/ChRst` pair and its global Cholesky-vector IDs.
All paths and arrays are CPU-owned, rank is zero based, and no OpenMolcas handle is retained.
"""
struct SewardShardInfo
    rank::Int
    chvec_path::String
    chrst_path::String
    local_count::Int
    global_ids::Vector{Int}
end

"""
    SewardSourceLayout

Summarize a validated serial or MPI-sharded SEWARD snapshot before OpenMolcas initialization.
The structure owns shard metadata, reports the total auxiliary-vector count, and contains no scientific factor payload.
"""
struct SewardSourceLayout
    source::SewardMOTRASource
    global_chrst::SewardChRst
    shards::Vector{SewardShardInfo}
    n_auxiliary::Int
end

"""
    _read_chrst_ints(data, position, count, path)

Decode `count` native-endian ILP64 integers from a `ChRst` byte vector and advance `position`.
The byte array is read-only, the returned vector is owned, and truncated input raises an error naming the source path.
"""
function _read_chrst_ints(
    data::Vector{UInt8},
    position::Base.RefValue{Int},
    count::Integer,
    path::AbstractString,
)
    count >= 0 || throw(ArgumentError("ChRst integer count must be nonnegative"))
    first_byte = position[]
    last_byte = first_byte + 8 * Int(count) - 1
    last_byte <= length(data) || error("truncated ChRst file: $path")
    values = copy(reinterpret(Int64, view(data, first_byte:last_byte)))
    position[] = last_byte + 1
    return values
end

"""
    read_seward_chrst(path)

Read and structurally decode one native-endian ExactDenEmb SEWARD `ChRst` file.
The file is read-only, returned bytes/columns are CPU-owned, and malformed headers, dimensions, or truncation raise before OpenMolcas is loaded.
"""
function read_seward_chrst(path::AbstractString)
    isfile(path) || throw(ArgumentError("SEWARD ChRst file not found: $path"))
    data = read(path)
    position = Ref(1)
    header = _read_chrst_ints(data, position, 4, path)
    n_symmetry = Int(header[1])
    n_symmetry > 0 || error("invalid nSym=$n_symmetry in $path")
    _read_chrst_ints(data, position, n_symmetry, path)
    _read_chrst_ints(data, position, 2, path)
    position[] += 64
    position[] <= length(data) + 1 || error("truncated ChRst character header: $path")
    max_reduced_set = Int(only(_read_chrst_ints(data, position, 1, path)))
    max_reduced_set >= 0 || error("invalid maxred=$max_reduced_set in $path")
    _read_chrst_ints(data, position, max_reduced_set, path)
    numcho_offset = position[]
    nrows = Int(only(_read_chrst_ints(data, position, 1, path)))
    nrows > 0 || error("invalid ChRst row count $nrows in $path")
    infvec_offset = position[]
    columns = reshape(_read_chrst_ints(data, position, 5 * nrows, path), nrows, 5)
    return SewardChRst(data, numcho_offset, infvec_offset, nrows, columns)
end

"""
    seward_local_vector_count(chrst, chvec_path)

Infer the number of physical vectors in a rank-local `ChVec1` from its byte length and `ChRst` disk addresses.
Inputs are read-only, the count is dimensionless, and non-ILP64-aligned or empty shard files raise an error.
"""
function seward_local_vector_count(chrst::SewardChRst, chvec_path::AbstractString)
    isfile(chvec_path) || throw(ArgumentError("SEWARD ChVec1 file not found: $chvec_path"))
    words, remainder = divrem(filesize(chvec_path), 8)
    remainder == 0 || error("ChVec1 byte size is not a multiple of eight: $chvec_path")
    words > 0 || error("empty SEWARD ChVec1 shard: $chvec_path")
    first_outside = findfirst(>=(words), view(chrst.columns, :, 3))
    return first_outside === nothing ? chrst.nrows : first_outside - 1
end

"""
    shard_global_vector_ids(rank, ranks, local_count, chrst)

Return the global one-based auxiliary-vector IDs owned by one zero-based SEWARD rank.
Authoritative nonzero `ChRst` column-five IDs are preferred; legacy zero columns use round-robin placement, and invalid or duplicate IDs raise without modifying metadata.
"""
function shard_global_vector_ids(
    rank::Integer,
    ranks::Integer,
    local_count::Integer,
    chrst::SewardChRst,
)
    0 <= rank < ranks || throw(ArgumentError("SEWARD rank must satisfy 0 <= rank < ranks"))
    0 <= local_count <= chrst.nrows || throw(ArgumentError("SEWARD local vector count is outside ChRst rows"))
    ids = Int.(view(chrst.columns, 1:Int(local_count), 5))
    if any(!=(0), ids)
        all(>(0), ids) || error("rank $rank ChRst column five mixes zero and nonzero global IDs")
        length(unique(ids)) == length(ids) || error("rank $rank ChRst contains duplicate global vector IDs")
        return ids
    end
    return [Int(rank) + (index - 1) * Int(ranks) + 1 for index in 1:Int(local_count)]
end

"""
    _seward_rank_directory(source, rank)

Return the directory containing one zero-based SEWARD rank, using the snapshot root for rank zero and `tmp_rank` for MPI worker ranks.
The source is not mutated, no path is created, and an out-of-range rank raises `ArgumentError`.
"""
function _seward_rank_directory(source::SewardMOTRASource, rank::Integer)
    0 <= rank < source.ranks || throw(ArgumentError("SEWARD rank is outside configured range"))
    return rank == 0 ? abspath(source.snapshot) : joinpath(abspath(source.snapshot), "tmp_$(rank)")
end

"""
    _seward_source_file(source, rank, suffix)

Return the absolute path of one project-prefixed file in a serial or MPI-sharded SEWARD snapshot.
The function allocates only the path string and performs no existence check or file mutation.
"""
function _seward_source_file(source::SewardMOTRASource, rank::Integer, suffix::AbstractString)
    return joinpath(_seward_rank_directory(source, rank), "$(source.project).$(suffix)")
end

"""
    inspect_seward_source(source)

Validate all serial or MPI `ChVec1/ChRst` shards, rank-zero metadata files, and the exact global auxiliary-vector partition.
The inspection is read-only and CUDA/OpenMolcas-free; missing files, inconsistent row counts, gaps, or duplicate global IDs raise before scientific allocation.
"""
function inspect_seward_source(source::SewardMOTRASource)
    snapshot = abspath(source.snapshot)
    isdir(snapshot) || throw(ArgumentError("SEWARD snapshot directory not found: $snapshot"))
    isfile(source.commonorb) || throw(ArgumentError("SEWARD CommonOrb file not found: $(source.commonorb)"))
    isdir(source.scratch_parent) || throw(ArgumentError("SEWARD scratch parent does not exist: $(source.scratch_parent)"))
    for suffix in ("RunFile", "OneInt", "ChRed", "ChMap")
        path = _seward_source_file(source, 0, suffix)
        isfile(path) || throw(ArgumentError("missing rank-zero SEWARD file: $path"))
    end
    global_chrst = read_seward_chrst(_seward_source_file(source, 0, "ChRst"))
    preliminary = Tuple{Int,String,String,Int,SewardChRst}[]
    for rank in 0:(source.ranks - 1)
        chvec_path = _seward_source_file(source, rank, "ChVec1")
        chrst_path = _seward_source_file(source, rank, "ChRst")
        chrst = read_seward_chrst(chrst_path)
        chrst.nrows == global_chrst.nrows || error("rank-$rank ChRst row count differs from rank zero")
        local_count = seward_local_vector_count(chrst, chvec_path)
        push!(preliminary, (rank, chvec_path, chrst_path, local_count, chrst))
    end
    total = sum(entry[4] for entry in preliminary)
    total > 0 || error("SEWARD snapshot contains no Cholesky vectors")
    assigned = falses(total)
    shards = SewardShardInfo[]
    for (rank, chvec_path, chrst_path, local_count, chrst) in preliminary
        ids = shard_global_vector_ids(rank, source.ranks, local_count, chrst)
        for global_id in ids
            1 <= global_id <= total || error("rank $rank global vector ID $global_id is outside 1:$total")
            assigned[global_id] && error("SEWARD global vector ID $global_id is assigned more than once")
            assigned[global_id] = true
        end
        push!(shards, SewardShardInfo(rank, chvec_path, chrst_path, local_count, ids))
    end
    all(assigned) || error("SEWARD shard mapping does not cover every global vector ID")
    global_chrst.nrows == total || error("rank-zero ChRst declares $(global_chrst.nrows) vectors but shards contain $total")
    return SewardSourceLayout(source, global_chrst, shards, total)
end

"""
    write_normalized_shard_chrst(path, local_chrst, global_chrst, shard)

Write a compact temporary `ChRst` whose metadata describes only one rank-local vector set while preserving local disk addresses.
The destination is created or replaced inside caller-owned scratch, source metadata is unchanged, and no original snapshot file is modified.
"""
function write_normalized_shard_chrst(
    path::AbstractString,
    local_chrst::SewardChRst,
    global_chrst::SewardChRst,
    shard::SewardShardInfo,
)
    columns = copy(local_chrst.columns)
    for (local_index, global_index) in enumerate(shard.global_ids)
        columns[local_index, 1] = global_chrst.columns[global_index, 1]
        columns[local_index, 2] = global_chrst.columns[global_index, 2]
        columns[local_index, 4] = global_chrst.columns[global_index, 4]
        columns[local_index, 5] = 0
    end
    open(path, "w") do io
        write(io, view(local_chrst.bytes, 1:(local_chrst.numcho_offset - 1)))
        write(io, Int64(shard.local_count))
        for column in 1:5
            write(io, view(columns, 1:shard.local_count, column))
        end
    end
    return abspath(path)
end

"""
    _replace_stage_link(source, destination)

Create an absolute symbolic link in temporary staging storage, replacing an existing staging entry only.
The source is read-only, the destination path is mutated, and callers must ensure the destination lies within task-owned scratch.
"""
function _replace_stage_link(source::AbstractString, destination::AbstractString)
    ispath(destination) && rm(destination; force=true)
    symlink(abspath(source), destination)
    return destination
end
