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

 File:            src/runtime/evaluate.jl
 Purpose:         High-level evaluate(input) dispatch to the production AHPC and conventional DP paths.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    validate_backend(algorithm::DPNOCI, backend::PackedTwoERI)

Validate the supported DP-NOCI/conventional-ERI combination and return `nothing` before file loading.
"""
validate_backend(::DPNOCI, ::PackedTwoERI) = nothing

"""
    validate_backend(algorithm::DPNOCI, backend::FactorizedERI{ERIFactorBundleSource})

Reject DP-NOCI with directly supplied factorized ERIs by raising `UnsupportedERIBackend` before large allocation or GPU initialization.
"""
function validate_backend(algorithm::DPNOCI, backend::FactorizedERI{ERIFactorBundleSource})
    throw(UnsupportedERIBackend(algorithm_name(algorithm), backend_name(backend)))
end

"""
    validate_backend(algorithm::DPNOCI, backend::FactorizedERI{SewardMOTRASource})

Allow a SEWARD source for DP-NOCI because evaluation explicitly materializes physical four-index ERIs before the DP kernel.
The function performs no source loading, OpenMolcas initialization, host allocation, or CUDA work.
"""
validate_backend(::DPNOCI, ::FactorizedERI{SewardMOTRASource}) = nothing

"""
    validate_backend(algorithm::AHPCNOCI, backend::PackedTwoERI)

Validate the supported AHPC-NOCI/conventional-ERI combination and return `nothing` before file loading.
"""
validate_backend(::AHPCNOCI, ::PackedTwoERI) = nothing

"""
    validate_backend(algorithm::AHPCNOCI, backend::FactorizedERI)

Validate the supported AHPC-NOCI/factorized-ERI combination and return `nothing` before factor loading.
"""
validate_backend(::AHPCNOCI, ::FactorizedERI) = nothing

"""
    _eri_source_metadata(backend)

Read only small factor-provenance metadata for production telemetry, returning a CPU dictionary of scalar values.
Packed ERIs return an empty mapping and legacy SEWARD temporary files return an explicitly unknown origin; no scientific factor payload or device is loaded, inputs remain read-only, and malformed declared provenance raises before allocation.
"""
function _eri_source_metadata(backend::ERIBackend)
    backend isa FactorizedERI || return Dict{Symbol,Any}()
    provenance = if backend isa FactorizedERI{ERIFactorBundleSource}
        JLD2.jldopen(backend.source.factor_path, "r") do data
            haskey(data, "factor_provenance") ? _read_eri_factor_provenance(data["factor_provenance"]) : ERIFactorProvenance()
        end
    else
        ERIFactorProvenance()
    end
    return Dict{Symbol,Any}(Symbol(key) => value for (key, value) in eri_factor_provenance_dict(provenance))
end

"""
    _eri_loading_memory(T, backend::PackedTwoERI, one_electron)

Return the packed `.two` loader peak host-memory estimate after validating its `.one` layout dependency.
Inputs are read-only, units are bytes, no ERI values or CUDA arrays are allocated, and inconsistent metadata raise through `packed_two_eri_memory`.
"""
function _eri_loading_memory(
    ::Type{T},
    ::PackedTwoERI,
    one_electron::OneElectronFileData,
) where {T<:AbstractFloat}
    return packed_two_eri_memory(T, one_electron)
end

"""
    _eri_loading_memory(T, backend::FactorizedERI{ERIFactorBundleSource}, one_electron)

Read only small JLD2 dimension/version metadata and return the matching ERI factor bundle peak host-memory estimate.
The backend and one-electron data are read-only, no factor coefficients or CUDA arrays are loaded, and missing or inconsistent dimensions raise an exception.
"""
function _eri_loading_memory(
    ::Type{T},
    backend::FactorizedERI{ERIFactorBundleSource},
    one_electron::OneElectronFileData,
) where {T<:AbstractFloat}
    factor_path = backend.source.factor_path
    return JLD2.jldopen(factor_path, "r") do data
        n_orbitals = Int(_required_bundle_value(data, "nbas", factor_path))
        n_auxiliary = Int(_required_bundle_value(data, "naux", factor_path))
        version = Int(_required_bundle_value(data, "version", factor_path))
        n_orbitals == one_electron.layout.n_orbitals || throw(DimensionMismatch("ERI-factor and one-electron orbital dimensions differ"))
        return eri_factor_bundle_memory(T, n_orbitals, n_auxiliary; version)
    end
end

"""
    _eri_loading_memory(T, backend::FactorizedERI{SewardMOTRASource}, one_electron)

Inspect small SEWARD shard metadata and estimate Julia-MOTRA host/device storage for the retained one-electron orbital dimension.
Inputs and source files are read-only, OpenMolcas and CUDA are not initialized, and malformed source metadata raises before scientific allocation.
"""
function _eri_loading_memory(
    ::Type{T},
    backend::FactorizedERI{SewardMOTRASource},
    one_electron::OneElectronFileData,
) where {T<:AbstractFloat}
    return seward_motra_memory(T, backend.source, one_electron.layout.n_orbitals)
end

"""
    _load_eri_values(T, backend::PackedTwoERI, one_electron, runtime)

Load physical conventional packed ERIs using the backend paths and validated `.one` layout.
Inputs are read-only, returned values are CPU-owned in energy units, memory limits are enforced by the loader, and no CUDA state is initialized.
"""
function _load_eri_values(
    ::Type{T},
    backend::PackedTwoERI,
    one_electron::OneElectronFileData,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    return read_packed_two_eri(one_electron, backend.two_paths; T, runtime)
end

"""
    _load_eri_values(T, backend::FactorizedERI{ERIFactorBundleSource}, one_electron, runtime)

Load and validate one physical ERI factor bundle and return only its pair-by-auxiliary factor values.
Inputs are read-only, factors are CPU-owned in square-root ERI units, memory limits are enforced by the loader, and no CUDA state is initialized.
"""
function _load_eri_values(
    ::Type{T},
    backend::FactorizedERI{ERIFactorBundleSource},
    one_electron::OneElectronFileData,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    bundle = read_eri_factor_bundle(backend.source.factor_path; T, runtime)
    bundle.factors.n_orbitals == one_electron.layout.n_orbitals || throw(DimensionMismatch("ERI-factor and one-electron orbital dimensions differ"))
    return bundle.factors
end


"""
    _load_eri_values(T, backend::FactorizedERI{SewardMOTRASource}, one_electron, runtime)

Transform one serial or MPI-sharded SEWARD source into retained-MO metric-whitened ERI factors and verify its dimension against the independently loaded `.one` file.
Source files remain read-only, returned factors are CPU-owned, configured CPU/GPU execution never falls back, and temporary staging is removed on success or failure.
"""
function _load_eri_values(
    ::Type{T},
    backend::FactorizedERI{SewardMOTRASource},
    one_electron::OneElectronFileData,
    runtime::RuntimeOptions,
) where {T<:AbstractFloat}
    factors = load_seward_eri_factors(backend.source; T, runtime)
    factors.n_orbitals == one_electron.layout.n_orbitals || throw(DimensionMismatch("SEWARD/Julia-MOTRA and one-electron orbital dimensions differ"))
    return factors
end

"""
    _evaluate_noci(input; T=Float64, runtime=input.runtime, checkpoint=nothing, progress=nothing)

Execute the shared unshifted NOCI pipeline with optional production progress and restart contexts.
Scientific inputs remain read-only, returned results are CPU-owned in integral units, optional host-file writes are confined to the supplied contexts, and DP-NOCI rejects direct ERI factor bundles while explicitly materializing an embedded SEWARD source into four-index ERIs.
"""
function _evaluate_noci(
    input::NOCIInput;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=input.runtime,
    checkpoint::Union{Nothing,CheckpointSession}=nothing,
    progress::Union{Nothing,ProgressLogger}=nothing,
) where {S<:AbstractFloat}
    runtime.use_gpu && !noci_gpu_supported() && throw(ArgumentError("NOCI.jl $(noci_version()) supports CPU execution only"))
    validate_backend(input.algorithm, input.backend)
    if runtime.use_gpu && isnothing(Base.get_extension(@__MODULE__, :NOCICUDAExt))
        _activate_cuda_device(runtime)
        return Base.invokelatest(_evaluate_noci, input; T, runtime, checkpoint, progress)
    end
    runtime.use_gpu && _activate_cuda_device(runtime)
    if input.algorithm isa AHPCNOCI
        ahpc_threads_per_gpu(runtime)
        input.algorithm.branch isa AHPCSpinStringBranch || throw(AlgorithmNotImplemented(
            algorithm_name(input.algorithm),
            branch_name(input.algorithm.branch),
        ))
    end
    telemetry = ExecutionTelemetry(algorithm_name(input.algorithm), backend_name(input.backend))
    progress_event!(progress, :load_problem_started; algorithm=algorithm_name(input.algorithm), backend=backend_name(input.backend))
    problem = load_problem_definition(input; T=S, runtime, telemetry)
    progress_event!(progress, :load_problem_complete; n_fragment_states=length(problem.fragment_states), n_basis=problem.n_basis)
    basis_functions = build_molecular_basis_functions(problem; screening=input.screening, runtime, telemetry)
    progress_event!(progress, :build_mebfs_complete; n_mebfs=length(basis_functions))
    one_electron_estimate = one_electron_file_memory(S, problem.n_basis)
    one_electron = record_phase!(
        () -> read_one_electron_file(input.backend.one_path; T=S, runtime),
        telemetry,
        :load_one_electron;
        estimate=one_electron_estimate,
        collect_allocations=runtime.collect_allocations,
    )
    progress_event!(progress, :load_one_electron_complete; n_basis=one_electron.layout.n_orbitals)
    eri_estimate = _eri_loading_memory(S, input.backend, one_electron)
    integrals = record_phase!(
        () -> _load_eri_values(S, input.backend, one_electron, runtime),
        telemetry,
        :load_eri;
        estimate=eri_estimate,
        metadata=_eri_source_metadata(input.backend),
        collect_allocations=runtime.collect_allocations,
    )
    progress_event!(progress, :load_eri_complete; backend=backend_name(input.backend))
    if input.algorithm isa DPNOCI && integrals isa ERIFactorValues
        conversion_estimate = packed_eri_from_factors_memory(S, integrals.n_orbitals, size(integrals.coefficients, 2); runtime)
        auxiliary_count = size(integrals.coefficients, 2)
        integrals = record_phase!(
            () -> packed_eri_from_factors(integrals; runtime),
            telemetry,
            :materialize_four_index_eri;
            estimate=conversion_estimate,
            metadata=Dict(:source => :seward_eri_factors, :n_auxiliary => auxiliary_count),
            collect_allocations=runtime.collect_allocations,
        )
        progress_event!(progress, :materialize_four_index_eri_complete; n_orbitals=integrals.n_orbitals)
    end
    raw = if input.algorithm isa DPNOCI
        assemble_dp_noci_matrices(basis_functions, one_electron.values, integrals, input.screening; runtime, telemetry, checkpoint, progress)
    else
        assemble_ahpc_spin_string_matrices(
            basis_functions,
            one_electron.values,
            integrals,
            input.screening,
            input.algorithm.branch.acceleration;
            runtime,
            telemetry,
            checkpoint,
            progress,
        )
    end
    progress_event!(progress, :raw_matrix_complete; n_mebfs=length(raw.labels))
    normalized = normalize_noci_matrices(raw; runtime, telemetry)
    progress_event!(progress, :normalization_complete; n_mebfs=length(normalized.labels))
    solution = solve_generalized_noci(normalized; runtime, telemetry)
    progress_event!(progress, :unshifted_solve_complete; n_states=length(solution.energies), ground_energy=first(solution.energies))
    return NOCIEvaluationResult(
        input.label,
        algorithm_name(input.algorithm),
        input.algorithm isa DPNOCI ? "direct-determinant-pair" : branch_name(input.algorithm.branch),
        backend_name(input.backend),
        raw,
        normalized,
        solution,
        telemetry,
    )
end

"""
    evaluate(input::NOCIInput; T=Float64, runtime=input.runtime)

Execute one repository-local unshifted NOCI calculation from validated input through problem/integral loading, matrix assembly, normalization, and generalized eigensolution.
Inputs and source files are read-only, results and telemetry are CPU-owned, DP-NOCI and production AHPC-NOCI support their declared CPU/CUDA paths, direct factor-bundle DP requests fail before loading, and only an explicitly selected embedded SEWARD source initializes the configured ExactDenEmb OpenMolcas library against task-owned staging.
"""
function evaluate(
    input::NOCIInput;
    T::Type{S}=Float64,
    runtime::RuntimeOptions=input.runtime,
) where {S<:AbstractFloat}
    return _evaluate_noci(input; T, runtime)
end
