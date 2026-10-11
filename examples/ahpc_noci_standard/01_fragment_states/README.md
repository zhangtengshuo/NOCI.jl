# 01: Shared monomer CAS calculations

[中文](README.zh.md)

Before execution, check stage 00 and seeds, generate one molecule/carrier/state task and render CAS plus optional CASPT2 inputs from user templates.
Calculate each intrinsic task once; differently placed instances share its original RasOrb and VecDet.
When enabled, CASPT2 runs on the selected root after the original CAS wavefunction is exported.
After execution, require normal completion, selected-root energies and real products, write the completion record and expose instance/state copies under `bindings/<project>/`.
Task directories such as `f/GS0/S1/` can be reused across YAML definitions when scientific inputs and validated backend identity agree.
Changed coordinates, seeds, basis, CAS settings or backend identity cause an explicit conflict.

```bash
bash run.sh ethene_dimer.yaml prepare
bash run.sh ethene_dimer.yaml compute
bash run.sh ethene_dimer.yaml finalize
```

`prepare` writes inputs; `compute` launches OpenMolcas; `finalize` validates and extracts completed results.
Stage 01 is the only shared calculation stage and has no donor-project or execution-order dependency.
