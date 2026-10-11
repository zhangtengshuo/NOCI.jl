# 05: Run AHPC-NOCI

[中文](README.zh.md)

Require this project's stage-03 `.vec/.det` and stage-04 `.one/.sys/factors.jld2`.
Generate `<project>/<project>.yaml` with `algorithm.name: ahpc-noci` and `integrals.backend: factorized`, referencing short instance/state labels.
Change scientific settings in stage 00 rather than editing this generated YAML.
Run the CPU calculation and retain progress, `.out`, final `.json`, matrix snapshot and optional checkpoints.
`finalize` requires the actual completion event and final JSON; an input file or placeholder result is insufficient.

```bash
bash run.sh ethene_dimer.yaml all
```

Stage-05 `compute` is a Julia calculation, not an OpenMolcas input to submit through a quantum-chemistry launcher.
Use a new project/output identity when changing the MEBF table, screening or other scientific inputs.
