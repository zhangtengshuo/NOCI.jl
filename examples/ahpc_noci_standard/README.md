# Standard AHPC-NOCI workflow

[English](README.md) | [中文](README.zh.md)

Start with [the detailed tutorial](TUTORIAL.md) and the annotated [ethene dimer definition](00_system_define/ethene_dimer.yaml).
This release provides the schema-4 C1 route: OpenMolcas monomer CAS, optional single-state CASPT2, OpenMolcas RICD/CD integrals and AHPC-NOCI.
Preparation uses CPU; stage 05 defaults to CPU and supports an explicitly installed GPU environment selected with `runtime.use_gpu: true`.
PySCF DFCD is a reserved adapter with structure templates; its full numerical workflow is unavailable.

For one molecule with multiple state wavefunctions, see [the ethene monomer definition](00_system_define/ethene_monomer.yaml) and the single-molecule tutorial section.

```text
00_system_define/       Complete independent YAML definitions, XYZ and user seeds
  templates/            Editable calculation inputs
01_fragment_states/     Shared intrinsic tasks, such as f/GS0/S1
02_fragment_placement/  <project>/ positioned geometry and rotated orbitals
03_common_orbitals/     <project>/ COMMONORB and .vec/.det
04_integrals/           <project>/ .one/.sys/factors.jld2
05_noci/                <project>/ input, results and optional checkpoints
```

Only stage 01 shares calculations across projects; stages 02–05 belong to each project independently.
Each molecule declares its own C1 symmetry, carrier geometries, seeds and states; each fragment instance binds to one molecule and carrier.
MEBF row order determines fragment order; labels are never sorted numerically.
Users provide chemically appropriate orbitals and CAS settings; the workflow does not select or assess active spaces.
CASPT2 is enabled by default and can be disabled with `caspt2.enabled: false`.
Workflow-owned [templates](00_system_define/templates/README.md) take priority over installed defaults.

```bash
noci-prepare check 00_system_define/ethene_dimer.yaml
noci-prepare plan 00_system_define/ethene_trimer.yaml
```

These commands read inputs and backend identity without launching calculations.
`run_all.sh` launches real calculations; run it only in a user-owned copy.
See the [backend contract](BACKEND_CONTRACT.md) for supported and reserved routes.
