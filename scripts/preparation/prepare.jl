include("case_driver.jl")

"""
    preparation_main(arguments)

Dispatch recommended check/plan/stage commands or the internal production worker from the installed preparation entry.
Arguments carry paths and dimensionless thread/checkpoint settings; the worker sets CPU BLAS threads, reads the generated input and writes production outputs through NOCI APIs.
No input is mutated; unsupported commands raise before calculations, and scientific memory/resource constraints remain enforced by NOCI.
"""
function preparation_main(arguments::Vector{String})
    isempty(arguments) && error("usage: noci-prepare check|plan MANIFEST [--structure-only], or stage STAGE MANIFEST [prepare|compute|finalize|all]")
    command = first(arguments)
    if command == "stage"
        main(arguments[2:end])
    elseif command in ("check", "plan")
        main(arguments)
    elseif command == "execute"
        length(arguments) == 7 || error("internal execute requires INPUT OUTDIR PREFIX XYZ BLAS_THREADS CHECKPOINT")
        _, input, directory, prefix, xyz, blas, checkpoint = arguments
        LinearAlgebra.BLAS.set_num_threads(parse(Int, blas))
        NOCI.run_noci(NOCI.read_noci_input(input);
            output=NOCI.RunOutputOptions(directory, prefix; geometry_path=xyz),
            checkpoint=NOCI.CheckpointOptions(enabled=parse(Bool, checkpoint), directory=joinpath(directory, prefix * ".checkpoint")))
    else
        error("unknown preparation command: $command")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        preparation_main(ARGS)
    catch failure
        println(stderr, "PREPARATION FAILED: ", sprint(showerror, failure))
        exit(1)
    end
end
