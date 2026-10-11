# 00: Define the system

[中文](README.zh.md)

Choose one YAML as the authoritative scientific input and fill in molecule identities, per-molecule symmetry, carrier XYZ, user seeds, CAS states, fragment bindings and the MEBF table.
Paths are relative to that YAML; independent definitions use distinct `project` labels and never refer to another project.

| Definition | Purpose |
| --- | --- |
| `ethene_dimer.yaml` | Six MEBFs: S0S0, S1S0, S0S1, D+D-, D-D+, T1T1. |
| `ethene_trimer.yaml` | Six explicitly declared trimer columns on GS0S0S0 geometry. |
| `ethene2_butadiene.yaml` | Two ethene instances and one butadiene instance. |
| `ethene_dimer_subset.yaml` | Three neutral MEBFs, independent CommonOrb and integrals. |
| `ethene_dimer_cd.yaml` | Molecular CD aggregate integrals with the same monomer CAS definition. |
| `pyscf_dfcd.yaml` | Reserved native PySCF seeds and explicit auxiliary basis; structure review only. |

```bash
bash run.sh ethene_dimer.yaml
noci-prepare plan 00_system_define/pyscf_dfcd.yaml --structure-only
```

`run.sh` checks inputs without launching a calculation.
Complete geometry optimization and any required frequency calculations beforehand; they are outside stages 00–05.
Single-state CASPT2 is enabled by default for the selected OpenMolcas root; set `enabled: false` to disable it.
Edit the actual [input templates](templates/README.md) in the user-owned case; explicit missing paths fail instead of falling back.
For the reserved PySCF route, CAS and integral auxiliary bases inherit from each other when only one is specified.

Each `.zh.yaml` counterpart has the same scientific definition with Chinese annotations; select either language for a calculation.
