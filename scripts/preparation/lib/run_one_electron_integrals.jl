#!/usr/bin/env julia

push!(LOAD_PATH, @__DIR__)
include(joinpath(@__DIR__, "MolcasIO.jl"))
include(joinpath(@__DIR__, "Rdtraint.jl"))

function _usage(io::IO = stderr)
    println(io, "usage: run_one_electron_integrals.jl")
    println(io, "       reads the normal rdtraint input from stdin and writes only .one/.sys")
    println(io, "       no four-index .two reconstruction is performed")
end

function main(args = ARGS)
    if any(arg -> arg in ("-h", "--help"), args)
        _usage(stdout)
        return nothing
    end
    isempty(args) || begin
        _usage()
        throw(ArgumentError("run_one_electron_integrals.jl takes no positional arguments"))
    end

    cfg = read_rdtraint_input(stdin)
    molcas_init(pwd(), cfg.project)
    try
        one_data = prepare_one_electron_data(cfg)

        # Preserve the established .one binary layout. Its header contains the
        # historical four-index file-plan metadata, but no .two files are
        # created and the Cholesky_ERI runtime does not consume that metadata.
        plan = build_two_electron_plan(cfg, one_data.info.norb)
        one_file = write_one_file(cfg, one_data, plan)

        println("Two-electron integral reconstruction: skipped (Cholesky_ERI uses the external factor source directly)")
        println("NOCI one-electron integral file: ", one_file)

        daclos(one_data.lu_one)
        sys_file = write_sys_file(cfg, one_data.info.nbas)
        println("NOCI system file: ", sys_file)
    finally
        molcas_final()
    end
    return nothing
end

main()
