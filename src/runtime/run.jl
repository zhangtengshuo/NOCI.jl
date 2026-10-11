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

 File:            src/runtime/run.jl
 Purpose:         Production run orchestration: paths, logging, checkpoints, and final output assembly.
 Copyright:       (C) 2026 Teng-Shuo Zhang
 SPDX-License-Identifier: Apache-2.0
 Original author: Teng-Shuo Zhang
=#

"""
    _production_paths(output)

Return absolute `.sec`, `.out`, `.json`, and matrix-snapshot paths for one production output contract.
The output metadata is read-only, no directory or file is created, and malformed roots have already been rejected by `RunOutputOptions`.
"""
function _production_paths(output::RunOutputOptions)
    directory = abspath(output.directory)
    return (
        sec=joinpath(directory, string(output.root, ".sec")),
        out=joinpath(directory, string(output.root, ".out")),
        json=joinpath(directory, string(output.root, ".json")),
        snapshot=joinpath(directory, string(output.root, ".matrix_snapshot.jld2")),
    )
end

"""
    _write_checkpoint_status(session, status; fields...)

Atomically update the machine-readable checkpoint status JSON with signatures, wall time, and caller-provided diagnostics.
The session is read-only except its directory contents, values are host metadata only, no CUDA state is touched, and serialization or filesystem errors propagate.
"""
function _write_checkpoint_status(session::CheckpointSession, status::AbstractString; fields...)
    data = Dict{String,Any}(
        "schema_version" => NOCI_CHECKPOINT_VERSION,
        "status" => String(status),
        "updated_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "scientific_signature_sha256" => session.scientific_signature,
        "execution_signature_sha256" => session.execution_signature,
    )
    for (key, value) in pairs(fields)
        data[String(key)] = value
    end
    return _atomic_json_write(joinpath(session.directory, "status.json"), data)
end

"""
    run_noci(input; output, checkpoint=CheckpointOptions(...), T=Float64, runtime=input.runtime, test_mode=false)

Execute one production NOCI calculation with real-time `.sec` progress, a scientific `.out` containing scalar pairwise/full-multistate couplings, atomic checkpoint/restart, a self-contained matrix snapshot, baseline CASPT2 scalar shifts, and final atomic JSON.
A run that appends to existing `.sec`/`.out` files writes one compact restart header instead of repeating the banner and input manifest, so each output file carries them exactly once.
When `test_mode=true`, additional progress-only events are emitted to `.sec` and stdout without changing scientific inputs, screening, CUDA topology, arithmetic, reduction order, or checkpoint semantics.
Scientific sources remain read-only, all reported energies use the integral unit, DP checkpoint batching never changes pair traversal or reduction order, CUDA use follows `runtime`, scalar coupling reporting performs only host postprocessing of completed matrices, and an existing final JSON is never overwritten.
"""
function run_noci(
    input::NOCIInput;
    output::RunOutputOptions,
    checkpoint::Union{Nothing,CheckpointOptions}=CheckpointOptions(directory=joinpath(output.directory, string(output.root, ".checkpoint"))),
    T::Type{S}=Float64,
    runtime::RuntimeOptions=input.runtime,
    test_mode::Bool=false,
) where {S<:AbstractFloat}
    input.algorithm isa AHPCNOCI && ahpc_threads_per_gpu(runtime)
    paths = _production_paths(output)
    mkpath(abspath(output.directory))
    isfile(paths.json) && throw(ArgumentError("refusing to overwrite completed production JSON: $(paths.json)"))
    input_manifest = build_input_manifest(input)
    scientific_signature = build_scientific_signature(input, S; input_manifest)
    execution_signature = build_execution_signature(runtime, S)
    checkpoint_session = isnothing(checkpoint) || !checkpoint.enabled ? nothing : open_checkpoint_session(checkpoint, scientific_signature, execution_signature)
    append_logs = isfile(paths.sec) || isfile(paths.out)
    logger = try
        open_progress_logger(paths.sec, paths.out; append=append_logs, test_mode)
    catch
        !isnothing(checkpoint_session) && close_checkpoint_session!(checkpoint_session)
        rethrow()
    end
    try
        if append_logs
            _write_production_restart_header!(logger, input, runtime, output, checkpoint, scientific_signature, execution_signature)
        else
            _write_production_header!(logger, input, runtime, output, checkpoint, input_manifest, scientific_signature, execution_signature)
        end
        progress_event!(
            logger,
            append_logs ? :production_restart_started : :production_run_started;
            scientific_signature=scientific_signature,
            execution_signature=execution_signature,
            test_mode,
            sec_path=paths.sec,
        )
        test_progress_event!(logger, :test_mode_progress_enabled; sec_path=paths.sec)
        !isnothing(checkpoint_session) && _write_checkpoint_status(checkpoint_session, "running"; sec_path=paths.sec, out_path=paths.out, test_mode)
        evaluation = _evaluate_noci(input; T=S, runtime, checkpoint=checkpoint_session, progress=logger)
        metadata = _fragment_energy_metadata(input, S)
        shifts = _correlation_shifts(metadata, metadata.caspt2_energies)
        shifted_hamiltonian = record_phase!(
            () -> apply_diagonal_shifts(evaluation.normalized.hamiltonian, evaluation.normalized.overlap, shifts; weighting=:identity),
            evaluation.telemetry,
            :apply_baseline_caspt2_shifts;
            collect_allocations=runtime.collect_allocations,
        )
        shifted_solution = solve_generalized_noci(
            _normalized_with_hamiltonian(evaluation.normalized, shifted_hamiltonian);
            runtime,
            telemetry=evaluation.telemetry,
        )
        progress_event!(logger, :baseline_caspt2_solve_complete; ground_energy=first(shifted_solution.energies))
        snapshot_path = save_matrix_snapshot(
            paths.snapshot,
            input,
            evaluation,
            metadata,
            shifts,
            shifted_hamiltonian,
            shifted_solution,
            input_manifest,
            scientific_signature,
            execution_signature,
        )
        progress_event!(logger, :matrix_snapshot_committed; path=snapshot_path)
        result = ProductionRunResult(
            evaluation,
            shifts,
            shifted_hamiltonian,
            shifted_solution,
            input_manifest,
            scientific_signature,
            execution_signature,
            paths.sec,
            paths.out,
            paths.json,
            snapshot_path,
        )
        _write_production_summary!(logger, result)
        _write_scalar_coupling_summary!(logger, result)
        json_data = _production_json(result)
        _attach_scalar_coupling_json!(json_data, result)
        _atomic_json_write(paths.json, json_data)
        !isnothing(checkpoint_session) && _write_checkpoint_status(checkpoint_session, "complete"; json_path=paths.json, snapshot_path=snapshot_path, sec_path=paths.sec)
        progress_event!(logger, :production_run_complete; json=paths.json, snapshot=snapshot_path, sec=paths.sec, ground_energy=first(shifted_solution.energies))
        return result
    catch error
        try
            progress_event!(logger, :production_run_failed; error=sprint(showerror, error), sec=paths.sec)
        catch
        end
        if !isnothing(checkpoint_session)
            try
                _write_checkpoint_status(checkpoint_session, "failed"; error=sprint(showerror, error), sec_path=paths.sec)
            catch
            end
        end
        rethrow()
    finally
        try
            close_progress_logger!(logger)
        finally
            !isnothing(checkpoint_session) && close_checkpoint_session!(checkpoint_session)
        end
    end
end
