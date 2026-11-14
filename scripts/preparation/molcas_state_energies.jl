# File: scripts/preparation/molcas_state_energies.jl
# Purpose: Read selected CASSCF and CASPT2 energies from OpenMolcas outputs.
# SPDX-License-Identifier: Apache-2.0

"""
    read_molcas_state_energies(path, root; require_caspt2=false)

Read the selected CASSCF root and optional CASPT2 total energy, returning `(casscf_energy, caspt2_energy)` in hartree.
CASPT2 totals are accepted only inside the CASPT2 module and for the requested root; absent optional CASPT2 returns the CASSCF energy, giving zero correlation correction.
Files are read-only, CPU parsing streams text without scientific arrays, and missing/nonfinite selected energies or a missing required CASPT2 result raise before completion is recorded.
"""
function read_molcas_state_energies(path::AbstractString, root::Integer; require_caspt2::Bool=false)
    casscf = nothing
    caspt2 = nothing
    module_name = ""
    pending_root = 0
    number = "[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[EeDd][-+]?\\d+)?"
    casscf_pattern = Regex("RASSCF energy for state\\s+([0-9]+)\\s+(" * number * ")")
    total_pattern = Regex("Total energy:\\s*(" * number * ")")
    for line in eachline(path)
        start = match(r"Start Module:\s*([A-Za-z0-9]+)"i, line)
        if start !== nothing
            module_name = lowercase(start.captures[1])
            pending_root = 0
        end
        value = match(casscf_pattern, line)
        if value !== nothing && parse(Int, value.captures[1]) == root
            casscf = parse(Float64, replace(value.captures[2], 'D' => 'E', 'd' => 'e'))
        end
        selected = match(r"RASSCF root number\s+([0-9]+)", line)
        total = match(total_pattern, line)
        if module_name == "caspt2"
            selected !== nothing && (pending_root = parse(Int, selected.captures[1]))
            fallback = match(r"(?:Compute H0 matrices for state|CASPT2 state passed to geometry opt\.)\s+([0-9]+)", line)
            fallback !== nothing && (pending_root = parse(Int, fallback.captures[1]))
            summary = match(r"CASPT2 Root\s+([0-9]+)"i, line)
            if summary !== nothing && total !== nothing
                # OpenMolcas summarizes SS totals with fewer decimal places than PCG.
                parse(Int, summary.captures[1]) == root && caspt2 === nothing &&
                    (caspt2 = parse(Float64, replace(total.captures[1], 'D' => 'E', 'd' => 'e')))
            elseif total !== nothing && pending_root == root
                caspt2 = parse(Float64, replace(total.captures[1], 'D' => 'E', 'd' => 'e'))
            end
        elseif selected !== nothing && total !== nothing && parse(Int, selected.captures[1]) == root
            casscf = parse(Float64, replace(total.captures[1], 'D' => 'E', 'd' => 'e'))
        end
    end
    casscf !== nothing && isfinite(casscf) || error("missing/nonfinite CASSCF energy for root $root in $path")
    require_caspt2 && caspt2 === nothing && error("missing CASPT2 energy for selected root $root in $path")
    caspt2 === nothing && (caspt2 = casscf)
    isfinite(caspt2) || error("nonfinite CASPT2 energy for root $root in $path")
    return casscf, caspt2
end
