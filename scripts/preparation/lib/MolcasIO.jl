# ccall wrappers for OpenMolcas libmolcas.so (ILP64 build) — read-only side.
#
# Used by the Class B preprocess tools (grotate / gcommon / rdcho / rdtraint),
# which only *consume* Molcas data (RunFile / OneInt / DaFile). Write paths
# (Put_* / WrOne / OpnOne-create) are intentionally omitted — add them back if
# a future tool needs to write. Tool-specific wrappers (Cho_X_*, DANAME_MF,
# DACLOS, WR_MOTRA_Info) are added alongside the tool that needs them.

using Libdl
import NOCI

const _lib = Ref{Ptr{Cvoid}}(C_NULL)
const _label_len = Int64(16)
const _dafile_len = Int64(16)

function _required_env_dir(name::String)::String
    value = get(ENV, name, "")
    isempty(value) && error("Required environment variable $name is not set; source the NOCI/OpenMolcas environment first")
    path = abspath(value)
    isdir(path) || error("$name does not name an existing directory: $path")
    return path
end

"""
    _optional_env_dir(name)

Return the absolute directory named by an optional environment variable, or `nothing` when it is unset.
No files are changed, and a nonempty value that is not a directory raises an error before OpenMolcas is loaded.
"""
function _optional_env_dir(name::String)::Union{Nothing,String}
    value = get(ENV, name, "")
    isempty(value) && return nothing
    path = abspath(value)
    isdir(path) || error("$name does not name an existing directory: $path")
    return path
end

function _optional_symbol(handle::Ptr{Cvoid}, name::Symbol)
    try
        return dlsym(handle, name)
    catch
        return C_NULL
    end
end

function _initialize_molcas_memory!()
    # omc13's xsetmem_ints takes an integer memory size, not the older string
    # shim signature. inimem_ reads MOLCAS_MEM and initializes the global MMA
    # state without depending on that tool-local shim.
    sym = _optional_symbol(_lib[], :inimem_)
    sym != C_NULL && ccall(sym, Cvoid, ())
    nothing
end

# --- Library resolution ---

function _resolve_libmolcas_path()
    explicit = get(ENV, "MOLCAS_LIBMOLCAS_SO", "")
    if !isempty(explicit)
        libpath = abspath(explicit)
        isfile(libpath) || error("MOLCAS_LIBMOLCAS_SO does not name a file: $libpath")
        return libpath
    end
    molcas_root = _required_env_dir("MOLCAS")
    libpath = joinpath(molcas_root, "lib", "libmolcas.so")
    isfile(libpath) || error("Cannot find libmolcas.so at $libpath; set MOLCAS or MOLCAS_LIBMOLCAS_SO")
    return libpath
end

function _prime_omc_runtime!(molcas_root::String, omc_root::Union{Nothing,String})
    dirs = String[joinpath(molcas_root, "lib")]
    isnothing(omc_root) || push!(dirs, joinpath(omc_root, "lib"))
    dirs = filter(isdir, unique(dirs))
    for dir in reverse(dirs)
        if !(dir in DL_LOAD_PATH)
            pushfirst!(DL_LOAD_PATH, dir)
        end
    end
    nothing
end

# --- Init / Final ---

function molcas_init(workdir::String, project::String; module_name::String="CASPT2")
    if _lib[] != C_NULL
        molcas_final()
    end
    so_path = NOCI.validate_openmolcas4noci_library(_resolve_libmolcas_path())
    molcas_root = dirname(dirname(so_path))  # lib/libmolcas.so -> install root
    omc_root = _optional_env_dir("OMC")
    ENV["MOLCAS"] = molcas_root
    isnothing(omc_root) || (ENV["OMC"] = omc_root)
    ENV["MOLCAS_MEM"] = get(ENV, "MOLCAS_MEM", "2048")
    ENV["MOLCAS_WORKDIR"] = workdir
    ENV["MOLCAS_PROJECT"] = project
    ENV["WorkDir"] = workdir
    ENV["Project"] = project

    _prime_omc_runtime!(molcas_root, omc_root)
    _lib[] = dlopen(so_path, RTLD_LAZY | RTLD_GLOBAL)
    if _lib[] == C_NULL
        error("Failed to dlopen libmolcas.so at $so_path")
    end
    sym = _optional_symbol(_lib[], :molcas_io_init)
    if sym != C_NULL
        ccall(sym, Cvoid, (Ptr{UInt8},), module_name * "\0")
    else
        _initialize_molcas_memory!()
    end
    nothing
end

function molcas_final()
    if _lib[] != C_NULL
        sym = _optional_symbol(_lib[], :molcas_io_final)
        sym != C_NULL && ccall(sym, Cvoid, ())
        dlclose(_lib[])
        _lib[] = C_NULL
    end
end

# Switch the active RunFile context (NameRun) by re-initializing against a copy
# of the given RunFile path.
function molcas_namerun(runfile_path::String)
    ENV["Project"] = basename(runfile_path)
    workdir = dirname(runfile_path)
    ENV["WorkDir"] = workdir
    molcas_final()
    molcas_init(workdir, basename(runfile_path))
end

# --- Helpers ---

_flabel(label::AbstractString) = rpad(label, 16) * "\0"

# NameRun: set the active RunFile name (lightweight switch, no re-init).
# Mirrors `call NameRun('RUNFILE')` in the fortran tools.
function namerun(name::AbstractString)
    ccall(dlsym(_lib[], :namerun_), Cvoid, (Ptr{UInt8}, Int64), _flabel(name), _label_len)
    nothing
end

# --- RunFile existence checks (qpg_*; avoid abort on missing field) ---

function qpg_iscalar(label::String)::Tuple{Bool, Int64}
    found = Ref{Int64}(0); ndata = Ref{Int64}(0)
    ccall(dlsym(_lib[], :qpg_iscalar_), Cvoid,
          (Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Int64),
          _flabel(label), found, ndata, _label_len)
    return (found[] != 0, ndata[])
end

function qpg_iarray(label::String)::Tuple{Bool, Int64}
    found = Ref{Int64}(0); ndata = Ref{Int64}(0)
    ccall(dlsym(_lib[], :qpg_iarray_), Cvoid,
          (Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Int64),
          _flabel(label), found, ndata, _label_len)
    return (found[] != 0, ndata[])
end

function qpg_carray(label::String)::Tuple{Bool, Int64}
    found = Ref{Int64}(0); ndata = Ref{Int64}(0)
    ccall(dlsym(_lib[], :qpg_carray_), Cvoid,
          (Ptr{UInt8}, Ptr{Int64}, Ptr{Int64}, Int64),
          _flabel(label), found, ndata, _label_len)
    return (found[] != 0, ndata[])
end

# --- RunFile reads (ILP64: integers Int64) ---

function get_iscalar(label::String)::Int64
    val = Ref{Int64}(0)
    ccall(dlsym(_lib[], :get_iscalar_), Cvoid,
          (Ptr{UInt8}, Ptr{Int64}, Int64), _flabel(label), val, _label_len)
    return val[]
end

function get_iscalar_safe(label::String, default::Int64=Int64(0))::Int64
    found, _ = qpg_iscalar(label)
    return found ? get_iscalar(label) : default
end

function get_dscalar(label::String)::Float64
    val = Ref{Float64}(0.0)
    ccall(dlsym(_lib[], :get_dscalar_), Cvoid,
          (Ptr{UInt8}, Ptr{Float64}, Int64), _flabel(label), val, _label_len)
    return val[]
end

function get_iarray(label::String, n::Int64)::Vector{Int64}
    data = Vector{Int64}(undef, n)
    ccall(dlsym(_lib[], :get_iarray_), Cvoid,
          (Ptr{UInt8}, Ptr{Int64}, Ref{Int64}, Int64),
          _flabel(label), data, Ref(Int64(n)), _label_len)
    return data
end

function get_darray(label::String, n::Int64)::Vector{Float64}
    data = Vector{Float64}(undef, n)
    ccall(dlsym(_lib[], :get_darray_), Cvoid,
          (Ptr{UInt8}, Ptr{Float64}, Ref{Int64}, Int64),
          _flabel(label), data, Ref(Int64(n)), _label_len)
    return data
end

function get_carray(label::String, n::Int64)::Vector{UInt8}
    buf = Vector{UInt8}(undef, Int(n) + 1)
    ccall(dlsym(_lib[], :get_carray_), Cvoid,
          (Ptr{UInt8}, Ptr{UInt8}, Ref{Int64}, Int64),
          _flabel(label), buf, Ref(Int64(n)), _label_len)
    return buf[1:Int(n)]
end

# RunFile writes used by the standalone Julia MOTRA replacement.  These are
# deliberately kept small: MOTRA only needs to publish the selected frozen and
# deleted dimensions before downstream rdcho/rdtraint inspect RUNFILE.
function put_iarray(label::String, data::Vector{Int64})
    n = Ref{Int64}(length(data))
    ccall(dlsym(_lib[], :put_iarray_), Cvoid,
          (Ptr{UInt8}, Ptr{Int64}, Ref{Int64}, Int64),
          _flabel(label), data, n, _label_len)
    nothing
end

# --- OneInt read (OpnOne / RdOne / ClsOne) ---

function opnone_open(fname::String, lu::Int64=Int64(2))::Int64
    rc = Ref{Int64}(-1)
    iopt = Int64(0)
    ccall(dlsym(_lib[], :opnone_), Cvoid,
          (Ptr{Int64}, Ref{Int64}, Ptr{UInt8}, Ref{Int64}, Int64),
          rc, Ref(iopt), rpad(fname, 8), Ref(lu), Int64(8))
    rc[] == 0 || error("OpnOne (open) failed, rc=$(rc[])")
    return lu
end

function clsone(lu::Int64=Int64(2))
    rc = Ref{Int64}(0)
    ccall(dlsym(_lib[], :clsone_), Cvoid, (Ptr{Int64}, Ref{Int64}), rc, Ref(Int64(0)))
    rc[] == 0 || error("ClsOne failed, rc=$(rc[])")
    nothing
end

# RdOne: reads a packed lower-triangular OneInt field (label, comp) into a
# buffer of nBas*(nBas+1)/2 + 4 doubles.
function rdone(label::String, comp::Integer, nBas::Integer)::Vector{Float64}
    nTri = Int(nBas) * (Int(nBas) + 1) ÷ 2
    buf = Vector{Float64}(undef, nTri + 4)
    rc = Ref{Int64}(0)
    ccall(dlsym(_lib[], :rdone_), Cvoid,
          (Ptr{Int64}, Ref{Int64}, Ptr{UInt8}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Int64),
          rc, Ref(Int64(0)), rpad(label, 8), Ref(Int64(comp)), buf, Ref(Int64(1)), Int64(8))
    rc[] == 0 || error("RdOne failed for '$label' comp=$comp, rc=$(rc[])")
    return buf
end

# AO overlap matrix from OneInt 'Mltpl  0' comp 1 (symmetrized lower triangle).
function read_overlap_matrix(nBas::Integer)::Matrix{Float64}
    buf = rdone("Mltpl  0", 1, nBas)
    S = zeros(Float64, nBas, nBas)
    idx = 0
    for j in 1:nBas, i in 1:j
        idx += 1
        S[i, j] = buf[idx]
        S[j, i] = buf[idx]
    end
    return S
end

# --- DaFile I/O (direct-access unformatted; ILP64) ---

function daname(lu::Int64, name::String)
    ccall(dlsym(_lib[], :daname_), Cvoid,
          (Ref{Int64}, Ptr{UInt8}, Int64),
          Ref(Int64(lu)), rpad(name, 16) * "\0", _dafile_len)
end

function daname_wa(lu::Int64, name::String)
    ccall(dlsym(_lib[], :daname_wa_), Cvoid,
          (Ref{Int64}, Ptr{UInt8}, Int64),
          Ref(Int64(lu)), rpad(name, 16) * "\0", _dafile_len)
end

function daname_mf(lu::Int64, name::String)
    ccall(dlsym(_lib[], :daname_mf_), Cvoid,
          (Ref{Int64}, Ptr{UInt8}, Int64),
          Ref(Int64(lu)), rpad(name, 16) * "\0", _dafile_len)
end

function wr_motra_info(lu::Int64; mxsym::Int64 = Int64(1), label_size::Int64 = Int64(80000))
    idisk = Ref{Int64}(0)
    toc = Vector{Int64}(undef, 64)
    ecor = Ref{Float64}(0.0)
    nsym = Ref{Int64}(0)
    nbas = Vector{Int64}(undef, mxsym)
    norb = Vector{Int64}(undef, mxsym)
    nfro = Vector{Int64}(undef, mxsym)
    ndel = Vector{Int64}(undef, mxsym)
    bslbl = Vector{UInt8}(undef, label_size)
    ccall(dlsym(_lib[], :wr_motra_info_), Cvoid,
          (Ref{Int64}, Ref{Int64}, Ref{Int64}, Ptr{Int64}, Ref{Int64},
           Ref{Float64}, Ref{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
           Ptr{Int64}, Ref{Int64}, Ptr{UInt8}, Ref{Int64}),
          Ref(Int64(lu)), Ref(Int64(2)), idisk, toc, Ref(Int64(length(toc))),
          ecor, nsym, nbas, norb, nfro, ndel, Ref(mxsym), bslbl, Ref(label_size))
    return (
        idisk = idisk[],
        toc = toc,
        potnuc = ecor[],
        nsym = nsym[],
        nbas = nbas,
        norb = norb,
        nfro = nfro,
        ndel = ndel,
        bslbl = bslbl,
    )
end

# Read/write the TRAONE header with caller-owned values.  OpenMolcas' helper is
# symmetric in iopt (1=write, 2=read), but the older Julia wrapper above was a
# read-only convenience function.
function wr_motra_info!(lu::Int64, iopt::Int64, idisk::Int64,
                        toc::Vector{Int64}, potnuc::Float64, nsym::Int64,
                        nbas::Vector{Int64}, norb::Vector{Int64},
                        nfro::Vector{Int64}, ndel::Vector{Int64},
                        bslbl::Vector{UInt8})
    length(nbas) == length(norb) == length(nfro) == length(ndel) ||
        throw(ArgumentError("MOTRA symmetry arrays must have equal lengths"))
    idisk_ref = Ref{Int64}(idisk)
    ecor_ref = Ref{Float64}(potnuc)
    nsym_ref = Ref{Int64}(nsym)
    mxsym_ref = Ref{Int64}(length(nbas))
    ccall(dlsym(_lib[], :wr_motra_info_), Cvoid,
          (Ref{Int64}, Ref{Int64}, Ref{Int64}, Ptr{Int64}, Ref{Int64},
           Ref{Float64}, Ref{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
           Ptr{Int64}, Ref{Int64}, Ptr{UInt8}, Ref{Int64}),
          Ref(lu), Ref(iopt), idisk_ref, toc, Ref(Int64(length(toc))),
          ecor_ref, nsym_ref, nbas, norb, nfro, ndel, mxsym_ref, bslbl,
          Ref(Int64(length(bslbl))))
    return idisk_ref[]
end

# iopt: 1 = write, 2 = read. Returns the updated disk address (Molcas advances
# idisk by length(buf); the deprecated version discarded this and was unusable
# for sequential writes like rdcho's TRAINT output).
function idafile!(lu::Int64, iopt::Int64, buf::Vector{Int64}, idisk::Int64)::Int64
    ref = Ref{Int64}(idisk)
    ccall(dlsym(_lib[], :idafile_), Cvoid,
          (Ref{Int64}, Ref{Int64}, Ptr{Int64}, Ref{Int64}, Ref{Int64}),
          Ref(Int64(lu)), Ref(Int64(iopt)), buf, Ref(Int64(length(buf))), ref)
    return ref[]
end

function ddafile!(lu::Int64, iopt::Int64, buf::Vector{Float64}, idisk::Int64)::Int64
    ref = Ref{Int64}(idisk)
    ccall(dlsym(_lib[], :ddafile_), Cvoid,
          (Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ref{Int64}),
          Ref(Int64(lu)), Ref(Int64(iopt)), buf, Ref(Int64(length(buf))), ref)
    return ref[]
end

function ddafile!(lu::Int64, iopt::Int64, buf::StridedVector{Float64}, idisk::Int64)::Int64
    stride(buf, 1) == 1 || throw(ArgumentError("ddafile! requires contiguous data"))
    ref = Ref{Int64}(idisk)
    GC.@preserve buf begin
        ccall(dlsym(_lib[], :ddafile_), Cvoid,
              (Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ref{Int64}),
              Ref(lu), Ref(iopt), pointer(buf), Ref(Int64(length(buf))), ref)
    end
    return ref[]
end

function ddafile!(lu::Int64, iopt::Int64, buf::Vector{Float64},
                  first::Integer, n::Integer, idisk::Int64)::Int64
    n64 = Int64(n)
    n64 >= 0 || throw(ArgumentError("ddafile! length must be non-negative"))
    first64 = Int64(first)
    first64 >= 1 || throw(ArgumentError("ddafile! first index must be 1-based"))
    first64 + n64 - 1 <= length(buf) ||
        throw(BoundsError(buf, first64:first64+n64-1))
    ref = Ref{Int64}(idisk)
    GC.@preserve buf begin
        ptr = pointer(buf, first64)
        ccall(dlsym(_lib[], :ddafile_), Cvoid,
              (Ref{Int64}, Ref{Int64}, Ptr{Float64}, Ref{Int64}, Ref{Int64}),
              Ref(Int64(lu)), Ref(Int64(iopt)), ptr, Ref(n64), ref)
    end
    return ref[]
end

# Close a DaFile unit (DACLOS).
function daclos(lu::Int64)
    ccall(dlsym(_lib[], :daclos_), Cvoid, (Ref{Int64},), Ref(Int64(lu)))
    nothing
end

# --- Cholesky interface (Cho_X_*) — used by rdcho ---

# Cho_X_Init(irc, BufFrac): initialize the Cholesky infrastructure
# (reads CHRED/CHVEC1/CHORST/CHOMAP from the active RunFile context).
# irc must be 0 on entry; returns the Molcas return code.
function cho_x_init(fracmem::Float64 = 0.0)::Int64
    irc = Ref{Int64}(0)
    ccall(dlsym(_lib[], :cho_x_init_), Cvoid, (Ref{Int64}, Ref{Float64}), irc, Ref(fracmem))
    return irc[]
end

# Cho_X_nVecRS(iRed, iSym, iVec, nVec): returns (iVec=start, nVec=count) for the
# reduced Cholesky vector set iRed in symmetry iSym.
function cho_x_nvecrs(iRed::Integer, iSym::Integer = 1)::Tuple{Int64,Int64}
    iVec = Ref{Int64}(0); nVec = Ref{Int64}(0)
    ccall(dlsym(_lib[], :cho_x_nvecrs_), Cvoid,
          (Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{Int64}),
          Ref(Int64(iRed)), Ref(Int64(iSym)), iVec, nVec)
    return (iVec[], nVec[])
end

function cho_x_final()::Int64
    irc = Ref{Int64}(0)
    ccall(dlsym(_lib[], :cho_x_final_), Cvoid, (Ref{Int64},), irc)
    return irc[]
end

# Dimension of reduced set 1 after Cho_X_Init.  This is a public module
# variable in libmolcas; C1 uses it as the row count returned by CHO_GETVEC.
function cho_mmbstrt()::Int64
    ptr = dlsym(_lib[], :__cholesky_MOD_mmbstrt)
    return unsafe_load(Ptr{Int64}(ptr))
end

# Map reduced-set-1 elements to global AO indices.  For C1, mapping[:,r]
# contains the (a,b) pair represented by reduced element r.
function cho_rstof!(mapping::Matrix{Int64}, ired::Integer = 1)
    size(mapping, 1) >= 2 || throw(ArgumentError("CHO_RSTOF mapping needs at least two rows"))
    n = Ref{Int64}(size(mapping, 1))
    lrdim = Ref{Int64}(size(mapping, 2))
    red = Ref{Int64}(ired)
    ccall(dlsym(_lib[], :cho_rstof_), Cvoid,
          (Ptr{Int64}, Ref{Int64}, Ref{Int64}, Ref{Int64}),
          mapping, n, lrdim, red)
    nothing
end

# Read vectors and expand every on-disk reduced set into the current reduced
# set (location 2, initialized to reduced set 1 by Cho_X_Init).
function cho_getvec!(vectors::StridedMatrix{Float64}, first_vector::Integer,
                     symmetry::Integer, scratch::Vector{Float64})
    lenvec = Ref{Int64}(size(vectors, 1))
    numvec = Ref{Int64}(size(vectors, 2))
    ivec1 = Ref{Int64}(first_vector)
    isym = Ref{Int64}(symmetry)
    lscr = Ref{Int64}(length(scratch))
    stride(vectors, 1) == 1 || throw(ArgumentError("CHO_GETVEC requires contiguous columns"))
    GC.@preserve vectors scratch begin
        ccall(dlsym(_lib[], :cho_getvec_), Cvoid,
              (Ptr{Float64}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{Int64},
               Ptr{Float64}, Ref{Int64}),
              pointer(vectors), lenvec, numvec, ivec1, isym, pointer(scratch), lscr)
    end
    nothing
end
