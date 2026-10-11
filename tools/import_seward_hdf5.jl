#!/usr/bin/env julia

using NOCI

"""
    parse_options(arguments)

Parse the compact portable-SEWARD import command line into validated paths and typed transform options.
The input vector is read-only, counts are dimensionless, GPU indices are zero based, and unknown/missing options raise `ArgumentError` before scientific files are opened.
"""
function parse_options(arguments::Vector{String})
    length(arguments) >= 3 || throw(ArgumentError("usage: import_seward_hdf5.jl EXPORT_DIRECTORY COMMONORB OUTPUT_PREFIX --frozen N --deleted N [--batch-size N] [--backend cpu|gpu] [--gpu-device N]"))
    export_directory, commonorb, output_prefix = arguments[1:3]
    options = Dict("frozen" => "0", "deleted" => "0", "batch-size" => "8", "backend" => "cpu", "gpu-device" => "0")
    index = 4
    while index <= length(arguments)
        startswith(arguments[index], "--") || throw(ArgumentError("expected an option, got $(arguments[index])"))
        key = arguments[index][3:end]
        haskey(options, key) || throw(ArgumentError("unknown option --$key"))
        index < length(arguments) || throw(ArgumentError("option --$key requires a value"))
        options[key] = arguments[index + 1]
        index += 2
    end
    backend = Symbol(options["backend"])
    backend in (:cpu, :gpu) || throw(ArgumentError("--backend must be cpu or gpu"))
    return (
        export_directory=abspath(export_directory), commonorb=abspath(commonorb), output_prefix=abspath(output_prefix),
        n_frozen=parse(Int, options["frozen"]), n_deleted=parse(Int, options["deleted"]),
        batch_size=parse(Int, options["batch-size"]), backend=backend, gpu_device=parse(Int, options["gpu-device"]),
    )
end

"""
    main(arguments)

Import one portable SEWARD HDF5 directory and write `<prefix>.one`, `<prefix>.sys`, and `<prefix>.two.jld2`.
Source files are read-only, output replacement is atomic, timing is reported in seconds, and schema/resource/transform errors propagate with a nonzero process exit.
"""
function main(arguments::Vector{String})
    options = parse_options(arguments)
    result = NOCI.import_seward_hdf5(
        options.export_directory,
        options.commonorb;
        n_frozen=options.n_frozen,
        n_deleted=options.n_deleted,
        batch_size=options.batch_size,
        backend=options.backend,
        gpu_device=options.gpu_device,
    )
    paths = NOCI.write_seward_hdf5_products(
        result;
        one_path=string(options.output_prefix, ".one"),
        system_path=string(options.output_prefix, ".sys"),
        bundle_path=string(options.output_prefix, ".two.jld2"),
    )
    println("one: ", paths.one)
    println("system: ", paths.system)
    println("factor bundle: ", paths.bundle)
    println("AO-to-MO import seconds: ", result.elapsed_seconds)
    return nothing
end

main(ARGS)
