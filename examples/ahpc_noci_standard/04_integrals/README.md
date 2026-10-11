# 04: Generate and transform factorized integrals

[中文](README.zh.md)

Require the actual stage-03 COMMONORB and frozen/deleted counts.
For RICD, set RICD/CDThreshold in GATEWAY and only CHH5 in SEWARD; its threshold controls atomic auxiliary-basis construction.
For molecular CD, omit GATEWAY RICD and set Cholesky/ThrCholesky/CHH5 in SEWARD.
Run the aggregate C1 export, validate CHH5 layout and provenance, and transform the factors into the actual common basis with consistent frozen-core corrections.
Outputs are `<project>/integrals.one`, `integrals.sys` and `factors.jld2`; no conventional `.two` or dense four-index ERI is generated.

```bash
bash run.sh ethene_dimer.yaml all
```

`finalize` performs numerical transformation and can require substantial resources.
The PySCF DFCD route remains reserved and is rejected before execution.
