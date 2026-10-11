# CAS and integral backend contract

[中文](BACKEND_CONTRACT.zh.md)

`cas.backend` and `integrals.backend` are separate declarations; stage names and directory responsibilities are independent of the quantum-chemistry program.
The executable C1 CPU route uses the explicitly accepted OpenMolcas4NOCI 26.06.2 build, its CHH5 export and the existing CommonOrb/RotateOrb tools.
PySCF native DFCD is a reserved adapter and must not be described as numerically implemented.

| CAS | Integrals | Status |
| --- | --- | --- |
| OpenMolcas | OpenMolcas RICD | Implemented preparation route; validate the full case in the target environment. |
| OpenMolcas | OpenMolcas CD | Implemented preparation route; validate the full case in the target environment. |
| PySCF | PySCF DFCD | Schema, auxiliary-basis, seed and output contracts reserved; execution is rejected. |
| OpenMolcas | PySCF DFCD | Future mixed adapter; rejected by current input validation. |
| Any | Exact four-index ERI | Outside this recommended factor workflow. |

## Stable stage interfaces

Adapters provide `prepare`, `compute` and `finalize` and preserve molecule/carrier/instance identities and stages 00–05.
`STANDARD_BACKENDS` declares combinations; `PREPARATION_STAGE_ADAPTERS` registers handlers in the installation.
Add new adapters there rather than copying scientific implementations into case Bash wrappers.
Each molecule declares its own point group; executable symmetry is C1 only.
New adapters still require implementation, parameter validation, provenance and numerical acceptance.

| Stage | Required products |
| --- | --- |
| 01 | Original orbitals, CI, energies and backend identity tied to user seed, carrier XYZ, basis and state; intrinsic tasks are shared. |
| 02 | Instance transforms, rotated orbitals, positioned XYZ and aggregate XYZ in fixed instance order. |
| 03 | Actual retained common basis, instance/state `.vec/.det`, frozen/deleted counts and dimensions. |
| 04 | `integrals.one`, `integrals.sys`, `factors.jld2` in that same actual common basis. |
| 05 | AHPC-NOCI with factorized integrals and short instance/state labels; backend differences do not alter algorithm dispatch. |

## Reserved native PySCF CAS

Seeds use native PySCF AO MO coefficients with center/basis, spherical/Cartesian, coefficient orientation/order and normalization metadata.
Users provide orbitals and active-space settings; no AVAS, automatic reordering or automatic active-space choice substitutes for them.
`spin_2S`, `nelecas: [n_alpha, n_beta]`, `ncore`, `ncas` and one-based `root` retain native meanings.
`ncore` is the number of CAS inactive orbitals and is not the NOCI frozen-core count.
Native state-bundle export, rotation and rectangular CommonOrb still require implementation and acceptance.
A rectangular common basis must carry actual AO/retained dimensions and a frozen subspace without inventing OpenMolcas deleted counts.
Templates and unavailable user `.npy` seeds are not a completed runnable example.

## DFCD and metric whitening

DFCD means three-center integrals and an auxiliary Coulomb metric whitened into unit-metric Gram factors.
It differs from molecular ERI CD and OpenMolcas atomic RICD.
`cas.auxiliary_basis` and `integrals.auxiliary_basis` inherit when only one is set, or may be specified separately.
At least one auxiliary basis and `integrals.metric_lindep` must be explicit; no automatic basis selection or molecular-CD threshold interpretation is allowed.
AO-to-common-basis transformation must be blocked over the auxiliary dimension and must not form a pair-by-pair four-index matrix or `.two`.
The effective frozen-core one-electron operator and constant energy must be derived consistently from the same factors/subspace, not just removed CI electrons.
DFCD needs its own explicit origin, auxiliary-basis, metric-method/threshold/units provenance; the current conventional-CD/RICD provenance contract cannot be relabelled as accepted DFCD.
The factorized numerical kernel can be shared, while source metadata and read/write compatibility need separate implementation and tests.

## Future mixed OpenMolcas CAS and PySCF DFCD

Convert the original OpenMolcas AO convention into PySCF representation at the stage-02/03 bridge before generating one-electron operators and factors in the same common basis.
Atom order, contraction, normalization, real-spherical order/sign and coefficient orientation require an explicit mapping with numerical closure; equal basis names are insufficient.
Representation validation does not assess the user's active-space chemistry or choose new orbitals.
Stage-01 intrinsic tasks depend on molecule/carrier/state/CAS/seed definitions; changing a later integral backend must not automatically repeat identical monomer CAS.
Stages 03/04 reuse must include their actual common basis and backend definition.
Only stage 01 is shared across projects; stages 02–05 process their own data and never register a donor project.

## Energies and templates

OpenMolcas stage 01 enables single-state CASPT2 by default, accepts `caspt2.enabled: false`, and carries selected-root energies into stage 03.
With CASPT2 disabled, the `.det` CASPT2 energy equals CASSCF energy, producing zero fragment correlation correction.
CAS wavefunctions remain unchanged by energy-only CASPT2 reporting.
All supported calculations render user-selected templates; local templates take priority over installed defaults and their identities participate in reuse.
PySCF CASPT2 is unavailable and is never replaced implicitly by another PT2 method.
