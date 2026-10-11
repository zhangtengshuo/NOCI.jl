#!/usr/bin/env julia

using NOCI

"""
    main(arguments)

Run one input with the installed OpenMolcas4NOCI distribution.
Arguments are `INPUT [SCRATCH_DIRECTORY]`, output is written beside the input as `<stem>.out`, and invalid arity or runtime failure exits nonzero without changing the input.
"""
function main(arguments::Vector{String})
    1 <= length(arguments) <= 2 || error("usage: openmolcas4noci.jl INPUT.in [SCRATCH_DIRECTORY]")
    input = abspath(arguments[1])
    scratch = length(arguments) == 2 ? abspath(arguments[2]) : nothing
    output = joinpath(dirname(input), string(splitext(basename(input))[1], ".out"))
    NOCI.run_openmolcas4noci(input; scratch, output)
    println(output)
    return nothing
end

main(ARGS)
