# Reserved native PySCF templates

[中文](README.zh.md)

`cas.py.in` defines native DF-CASSCF with an explicit CAS auxiliary basis, a user-supplied MO `.npy`, electron/spin/root settings and original `.npz` output.
`integrals_dfcd.py.in` defines blocked AO DF/CDERI with an explicit integral auxiliary basis and Coulomb-metric threshold, producing HDF5 factors and one-electron AO data without four-index ERIs.
These are editable solver templates; the standard workflow rejects PySCF execution because the complete NOCI adapter is unavailable.
See the [DF-CASSCF interface](https://pyscf.org/_modules/pyscf/mcscf/df.html#density_fit) and [blocked DF interface](https://pyscf.org/_modules/pyscf/df/outcore.html#cholesky_eri_b).
CAS and integral auxiliary bases inherit when only one is specified; users may explicitly choose different bases.
The PySCF CASPT2 adapter is unavailable and is never replaced implicitly by another PT2 method.

| Placeholders | Meaning |
| --- | --- |
| `XYZ_LITERAL`, `BASIS_LITERAL`, `AUXILIARY_BASIS_LITERAL`, `INITIAL_ORBITALS_LITERAL` | Safely quoted Python string literals supplied by a future adapter. |
| `CHARGE`, `SPIN_2S`, `NCAS`, `NELECAS_ALPHA`, `NELECAS_BETA`, `NCORE`, `ROOT` | User-specified native CAS parameters. |
| `THREADS`, `MEMORY_MB` | Adapter-provided resource settings. |
| `METRIC_LINDEP` | Small-eigenvalue rejection threshold for the auxiliary Coulomb metric. |
| `STATE_OUTPUT_LITERAL`, `CDERI_PATH_LITERAL`, `ONE_AO_PATH_LITERAL`, `OVERLAP_PATH_LITERAL` | Adapter-selected raw product paths. |

State-bundle conversion, rotations, CommonOrb, frozen-core handling, factor provenance, resource preflight and numerical acceptance remain future adapter work.
