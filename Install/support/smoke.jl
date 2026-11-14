# File: Install/support/smoke.jl
# Purpose: Run the installation smoke check with OpenMolcas and NOCI.
# SPDX-License-Identifier: Apache-2.0

using NOCI
using Libdl
using HDF5

"""
    check_installation(workdir)

Check managed runtime discovery, shared-library loading and a serial H2/STO-3G CHH5 export, then return nothing.
The supplied task-owned directory receives a tiny input, scratch and output; AO operators and integrals use atomic units, only CPU work is requested, the existing source and accepted dependencies remain unchanged, and process, library, export-format, dimension or fixed-reference failures propagate without invoking CUDA.functional or scientific production tests.
"""
function check_installation(workdir::String)
    input = joinpath(workdir, "h2_chh5.in")
    output = joinpath(workdir, "h2_chh5.out")
    NOCI.openmolcas4noci_identity(; required_capabilities=["cas", "cd", "ricd", "chh5", "commonorb", "rotateorb", "library"])
    NOCI.with_openmolcas4noci() do
        program_prefix = dirname(dirname(NOCI.openmolcas4noci_executable()))
        NOCI._prime_openmolcas_runtime!(ENV["MOLCAS"], program_prefix)
        handle = Libdl.dlopen(ENV["MOLCAS_LIBMOLCAS_SO"], Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
        Libdl.dlclose(handle)
    end
    NOCI.run_openmolcas4noci(input; scratch=joinpath(workdir, "scratch"), output)
    manifest = NOCI.inspect_seward_hdf5(joinpath(workdir, "h2_chh5.NOCI_SEWARD_H5"))
    (manifest.n_ao, manifest.n_reduced_pairs, manifest.n_auxiliary, manifest.n_mpi_ranks) == (2, 3, 3, 1) || error("unexpected H2 export dimensions")
    overlap = [1.0 0.6598731211014597; 0.6598731211014597 1.0]
    core = [-1.1209594575132389 -0.9593757705058865; -0.9593757705058865 -1.1209594575132389]
    maximum(abs, manifest.overlap - overlap) <= 1.0e-9 || error("H2 overlap reference mismatch")
    maximum(abs, manifest.core_hamiltonian - core) <= 1.0e-8 || error("H2 core-Hamiltonian reference mismatch")
    factors = zeros(Float64, 3, 3)
    for (path, ids) in zip(manifest.shard_paths, manifest.shard_global_auxiliary_indices)
        HDF5.h5open(path, "r") do file
            factors[:, ids] .= read(file["cholesky/factors"])
        end
    end
    gram = [0.7746059442114877 0.44459112429456915 0.5699948829139262;
            0.44459112429456915 0.29759055150774666 0.44459112429456915;
            0.5699948829139262 0.44459112429456915 0.7746059442114877]
    maximum(abs, factors * transpose(factors) - gram) <= 2.0e-9 || error("H2 integral reference mismatch")
    ricd_input = joinpath(workdir, "h2_ricd_chh5.in")
    NOCI.run_openmolcas4noci(ricd_input; scratch=joinpath(workdir, "ricd-scratch"), output=joinpath(workdir, "h2_ricd_chh5.out"))
    ricd = NOCI.inspect_seward_hdf5(joinpath(workdir, "h2_ricd_chh5.NOCI_SEWARD_H5"))
    ricd.n_ao == 2 && ricd.n_mpi_ranks == 1 || error("unexpected RICD export dimensions")
    ricd.provenance.origin == :ricd || error("RICD capability check returned a different factor origin")
    return nothing
end

length(ARGS) == 1 || error("usage: smoke.jl WORK_DIRECTORY")
check_installation(abspath(ARGS[1]))
