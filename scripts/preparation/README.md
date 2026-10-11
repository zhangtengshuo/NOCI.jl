# AHPC-NOCI preparation commands

This directory contains the single maintained implementation of the public AHPC-NOCI preparation commands.

The shell entry points are intentionally short: they resolve the NOCI.jl installation, accept explicit inputs, and call either OpenMolcas, the shared geometry/orbital utilities, CommonOrb, or the portable factor-bundle converter.

No command contains a cluster name, scheduler, account, partition, username, fixed CPU/GPU count, or fixed installation path.

Users call these commands through the selected installation's `bin/noci-prepare` dispatcher and `prepare.jl`; recommended check/plan/stage commands use one shared driver and the schema-4 species registry, rather than copying preparation code into cases.
The recommended copyable skeleton and detailed operation order are in [`examples/ahpc_noci_standard`](../../examples/ahpc_noci_standard/) and its [tutorial](../../examples/ahpc_noci_standard/TUTORIAL.md).
`noci-prepare init` copies that clean skeleton; prior direct preparation commands and legacy manifests still use the shared implementations.
The schema validator and adapter capability registry reserve native PySCF DFCD while rejecting execution before any tasks are created; no optional Python, CUDA, or CondaPkg dependency is added to the CPU package.

Schema-4 quantum-chemistry inputs are rendered from workflow-owned templates first, with installed defaults in `templates/`; the monomer symmetry, effective PySCF auxiliary basis, enabled CASPT2 settings and selected template enter task definitions.
Schema-4 CASPT2 is enabled by default, including when the block or its enabled flag is omitted; explicitly set `caspt2.enabled: false` to disable it.
The standard OpenMolcas path records selected-root CAS/CASPT2 energies through `molcas_state_energies.jl`, shared with CommonOrb, while legacy manifests keep their existing energy parser.
