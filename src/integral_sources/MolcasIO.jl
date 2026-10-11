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

 File:            src/integral_sources/MolcasIO.jl
 Purpose:         Libdl loading of the OpenMolcas ExactDenEmb library and runtime-directory staging.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

using Libdl

const _MOLCAS_LIBRARY_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _MOLCAS_LABEL_LENGTH = Int64(16)
const _MOLCAS_RUNTIME_LOCK = ReentrantLock()

"""
    _required_runtime_directory(path, label)

Return an absolute existing runtime directory for an OpenMolcas dependency.
The path is read-only, no library is loaded, and an empty or missing directory raises `ArgumentError`.
"""
function _required_runtime_directory(path::AbstractString, label::AbstractString)
    isempty(path) && throw(ArgumentError("$label path is empty"))
    absolute = abspath(path)
    isdir(absolute) || throw(ArgumentError("$label directory does not exist: $absolute"))
    return absolute
end

"""
    _optional_molcas_symbol(handle, name)

Return an optional symbol address from an already loaded library or `C_NULL` when absent.
The handle is not closed or mutated, and the function performs no scientific allocation.
"""
function _optional_molcas_symbol(handle::Ptr{Cvoid}, name::Symbol)
    try
        return Libdl.dlsym(handle, name)
    catch
        return C_NULL
    end
end

"""
    _initialize_molcas_memory!()

Initialize the OpenMolcas memory manager when the loaded library exposes `inimem_`.
The routine mutates OpenMolcas process-global state, returns `nothing`, and requires an initialized library handle.
"""
function _initialize_molcas_memory!()
    symbol = _optional_molcas_symbol(_MOLCAS_LIBRARY_HANDLE[], :inimem_)
    symbol != C_NULL && ccall(symbol, Cvoid, ())
    return nothing
end

"""
    _load_runtime_dependency!(names, directories)

Load the first existing shared-library name from the ordered runtime directories with global symbol visibility.
Inputs are read-only, the loaded dependency remains process-resident, and missing or unloadable libraries raise an error listing searched paths.
"""
function _load_runtime_dependency!(names::Tuple{Vararg{String}}, directories::Vector{String})
    searched = String[]
    for name in names, directory in directories
        path = joinpath(directory, name)
        push!(searched, path)
        if isfile(path)
            try
                Libdl.dlopen(path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
                return path
            catch error_value
                error("failed to load OpenMolcas dependency $path: $(sprint(showerror, error_value))")
            end
        end
    end
    error("OpenMolcas dependency $(join(names, " or ")) not found; searched $(join(searched, ", "))")
end

"""
    _prime_openmolcas_runtime!(molcas_root, omc_root)

Preload the shared libraries required by the ExactDenEmb OpenMolcas build before `libmolcas.so` is opened.
MKL 2025 and 2026 use distinct supported library filenames.
The function mutates Julia's dynamic-loader search state, leaves dependencies loaded, and raises immediately for a missing dependency.
"""
function _prime_openmolcas_runtime!(molcas_root::String, omc_root::String)
    directories = filter(isdir, unique(String[joinpath(molcas_root, "lib"), joinpath(omc_root, "lib")]))
    for directory in reverse(directories)
        directory in Libdl.DL_LOAD_PATH || pushfirst!(Libdl.DL_LOAD_PATH, directory)
    end
    for names in (
        ("libhdf5.so.320", "libhdf5.so"),
        ("libxc.so.15",),
        ("libxcf03.so.15",),
        ("libmpi.so.40",),
        ("libmpi_mpifh.so.40",),
        ("libmpi_usempi_ignore_tkr.so.40",),
        ("libmpi_usempif08.so.40",),
        ("libmkl_core.so.3", "libmkl_core.so.2"),
        ("libmkl_sequential.so.3", "libmkl_sequential.so.2"),
        ("libmkl_gf_ilp64.so.3", "libmkl_gf_ilp64.so.2"),
        ("libmkl_blacs_openmpi_ilp64.so.2",),
        ("libmkl_scalapack_ilp64.so.2",),
    )
        _load_runtime_dependency!(names, directories)
    end
    return nothing
end

"""
    initialize_molcas_runtime!(workdir, project; molcas_root, omc_root, module_name="MOTRA")

Initialize the ExactDenEmb `libmolcas.so` runtime for one staged SEWARD shard.
The function mutates process-global OpenMolcas environment and library state, allocates only runtime metadata, and must be paired with `finalize_molcas_runtime!`; concurrent calls are unsupported and serialized by the caller.
"""
function initialize_molcas_runtime!(
    workdir::AbstractString,
    project::AbstractString;
    molcas_root::AbstractString,
    omc_root::AbstractString,
    module_name::AbstractString="MOTRA",
)
    _MOLCAS_LIBRARY_HANDLE[] == C_NULL || finalize_molcas_runtime!()
    molcas = _required_runtime_directory(molcas_root, "OpenMolcas")
    omc = _required_runtime_directory(omc_root, "OpenMolcas dependency environment")
    library = joinpath(molcas, "lib", "libmolcas.so")
    isfile(library) || error("OpenMolcas library not found: $library")
    validate_openmolcas4noci_library(library)
    stage = abspath(workdir)
    isdir(stage) || throw(ArgumentError("OpenMolcas work directory does not exist: $stage"))
    ENV["MOLCAS"] = molcas
    ENV["MOLCAS_LIBMOLCAS_SO"] = library
    ENV["OMC"] = omc
    ENV["MOLCAS_MEM"] = get(ENV, "MOLCAS_MEM", "2048")
    ENV["MOLCAS_WORKDIR"] = stage
    ENV["MOLCAS_PROJECT"] = String(project)
    ENV["WorkDir"] = stage
    ENV["Project"] = String(project)
    _prime_openmolcas_runtime!(molcas, omc)
    _MOLCAS_LIBRARY_HANDLE[] = Libdl.dlopen(library, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    symbol = _optional_molcas_symbol(_MOLCAS_LIBRARY_HANDLE[], :molcas_io_init)
    symbol == C_NULL ? _initialize_molcas_memory!() : ccall(symbol, Cvoid, (Ptr{UInt8},), String(module_name) * "\0")
    return nothing
end

"""
    finalize_molcas_runtime!()

Finalize and close the active OpenMolcas library context.
The function mutates process-global library state, is idempotent, returns `nothing`, and performs no scientific file deletion.
"""
function finalize_molcas_runtime!()
    handle = _MOLCAS_LIBRARY_HANDLE[]
    if handle != C_NULL
        symbol = _optional_molcas_symbol(handle, :molcas_io_final)
        symbol != C_NULL && ccall(symbol, Cvoid, ())
        Libdl.dlclose(handle)
        _MOLCAS_LIBRARY_HANDLE[] = C_NULL
    end
    return nothing
end

"""
    _molcas_label(label)

Return a null-terminated 16-character OpenMolcas runfile label.
The input string is not mutated and labels longer than 16 characters are rejected before any foreign call.
"""
function _molcas_label(label::AbstractString)
    ncodeunits(label) <= _MOLCAS_LABEL_LENGTH || throw(ArgumentError("OpenMolcas label is longer than 16 bytes: $label"))
    return rpad(String(label), Int(_MOLCAS_LABEL_LENGTH)) * "\0"
end

"""
    molcas_namerun!(name)

Select the logical RunFile name in the active OpenMolcas context.
The call mutates OpenMolcas global state, returns `nothing`, and requires `initialize_molcas_runtime!` to have completed.
"""
function molcas_namerun!(name::AbstractString)
    ccall(Libdl.dlsym(_MOLCAS_LIBRARY_HANDLE[], :namerun_), Cvoid, (Ptr{UInt8}, Int64), _molcas_label(name), _MOLCAS_LABEL_LENGTH)
    return nothing
end

"""
    molcas_get_iscalar(label)

Read one ILP64 scalar from the active OpenMolcas RunFile.
The RunFile is not modified, the result is dimensionless unless the label defines units, and missing labels are reported by the foreign runtime.
"""
function molcas_get_iscalar(label::AbstractString)
    value = Ref{Int64}(0)
    ccall(
        Libdl.dlsym(_MOLCAS_LIBRARY_HANDLE[], :get_iscalar_),
        Cvoid,
        (Ptr{UInt8}, Ptr{Int64}, Int64),
        _molcas_label(label),
        value,
        _MOLCAS_LABEL_LENGTH,
    )
    return value[]
end

"""
    molcas_get_iarray(label, count)

Read `count` ILP64 integers from the active OpenMolcas RunFile.
The returned vector is CPU-owned, the RunFile is not modified, and negative counts or foreign-library failures propagate as errors.
"""
function molcas_get_iarray(label::AbstractString, count::Integer)
    count >= 0 || throw(ArgumentError("OpenMolcas integer-array count must be nonnegative"))
    data = Vector{Int64}(undef, Int(count))
    ccall(
        Libdl.dlsym(_MOLCAS_LIBRARY_HANDLE[], :get_iarray_),
        Cvoid,
        (Ptr{UInt8}, Ptr{Int64}, Ref{Int64}, Int64),
        _molcas_label(label),
        data,
        Ref(Int64(count)),
        _MOLCAS_LABEL_LENGTH,
    )
    return data
end

"""
    molcas_put_iarray!(label, data)

Write one ILP64 integer array into the staged active RunFile.
Only the temporary staged RunFile is mutated, the input vector is retained unchanged, and the caller must avoid using this function on original snapshot files.
"""
function molcas_put_iarray!(label::AbstractString, data::Vector{Int64})
    count = Ref{Int64}(length(data))
    ccall(
        Libdl.dlsym(_MOLCAS_LIBRARY_HANDLE[], :put_iarray_),
        Cvoid,
        (Ptr{UInt8}, Ptr{Int64}, Ref{Int64}, Int64),
        _molcas_label(label),
        data,
        count,
        _MOLCAS_LABEL_LENGTH,
    )
    return nothing
end

"""
    molcas_cholesky_initialize!(memory_fraction=0.0)

Initialize the OpenMolcas Cholesky-vector decoder for the staged `CHRED/CHORST/CHOMAP/CHVEC1` files.
The function mutates decoder state and returns its integer status code; the caller must finalize it even when subsequent transformation fails.
"""
function molcas_cholesky_initialize!(memory_fraction::Float64=0.0)
    status = Ref{Int64}(0)
    ccall(Libdl.dlsym(_MOLCAS_LIBRARY_HANDLE[], :cho_x_init_), Cvoid, (Ref{Int64}, Ref{Float64}), status, Ref(memory_fraction))
    return status[]
end

"""
    molcas_cholesky_finalize!()

Finalize the active OpenMolcas Cholesky-vector decoder and return its integer status code.
The routine mutates decoder state, allocates no scientific arrays, and is valid only after successful initialization.
"""
function molcas_cholesky_finalize!()
    status = Ref{Int64}(0)
    ccall(Libdl.dlsym(_MOLCAS_LIBRARY_HANDLE[], :cho_x_final_), Cvoid, (Ref{Int64},), status)
    return status[]
end

"""
    molcas_reduced_cholesky_dimension()

Return the active reduced-set-1 row count from the ExactDenEmb Cholesky module.
The result is dimensionless, no storage is allocated, and an incompatible OpenMolcas library raises during symbol resolution.
"""
function molcas_reduced_cholesky_dimension()
    pointer_value = Libdl.dlsym(_MOLCAS_LIBRARY_HANDLE[], :__cholesky_MOD_mmbstrt)
    return Int(unsafe_load(Ptr{Int64}(pointer_value)))
end

"""
    molcas_reduced_to_ao_map!(mapping, reduced_set=1)

Fill a two-row ILP64 matrix with the reduced-Cholesky-to-AO pair mapping.
The function mutates `mapping`, returns it, performs no device work, and rejects matrices with fewer than two rows.
"""
function molcas_reduced_to_ao_map!(mapping::Matrix{Int64}, reduced_set::Integer=1)
    size(mapping, 1) >= 2 || throw(ArgumentError("reduced-to-AO mapping needs at least two rows"))
    ccall(
        Libdl.dlsym(_MOLCAS_LIBRARY_HANDLE[], :cho_rstof_),
        Cvoid,
        (Ptr{Int64}, Ref{Int64}, Ref{Int64}, Ref{Int64}),
        mapping,
        Ref(Int64(size(mapping, 1))),
        Ref(Int64(size(mapping, 2))),
        Ref(Int64(reduced_set)),
    )
    return mapping
end

"""
    molcas_read_cholesky_vectors!(vectors, first_vector, symmetry, scratch)

Decode consecutive reduced Cholesky vectors into the contiguous columns of `vectors`.
The function mutates `vectors` and OpenMolcas decoder state, uses caller-owned CPU scratch, and rejects noncontiguous destination columns.
"""
function molcas_read_cholesky_vectors!(
    vectors::StridedMatrix{Float64},
    first_vector::Integer,
    symmetry::Integer,
    scratch::Vector{Float64},
)
    stride(vectors, 1) == 1 || throw(ArgumentError("OpenMolcas CHO_GETVEC requires contiguous columns"))
    GC.@preserve vectors scratch begin
        ccall(
            Libdl.dlsym(_MOLCAS_LIBRARY_HANDLE[], :cho_getvec_),
            Cvoid,
            (Ptr{Float64}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}),
            pointer(vectors),
            Ref(Int64(size(vectors, 1))),
            Ref(Int64(size(vectors, 2))),
            Ref(Int64(first_vector)),
            Ref(Int64(symmetry)),
            pointer(scratch),
            Ref(Int64(length(scratch))),
        )
    end
    return vectors
end
