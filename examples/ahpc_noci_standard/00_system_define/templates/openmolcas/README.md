# OpenMolcas input templates

[中文](README.zh.md)

`cas.in`, `caspt2.in`, `fragment_reference.in`, `aggregate_reference.in`, `integrals_ricd.in` and `integrals_cd.in` are editable inputs used to render real calculations.
C1 references use `group = nosym`; the defaults contain no EXPBAS.

| Placeholders | Source |
| --- | --- |
| `XYZ`, `BASIS`, `POINT_GROUP` | Current geometry, `cas.basis` and per-molecule point group; C1 becomes nosym. |
| `GATEWAY_INTEGRAL_OPTIONS`, `SEWARD_INTEGRAL_OPTIONS` | Mutually consistent monomer RICD/CD options. |
| `GENERATION_THRESHOLD` | Atomic auxiliary-basis or molecular-CD threshold for stage 04. |
| `CHARGE`, `NACTEL`, `INACTIVE`, `RAS2`, `STATE_SYMMETRY`, `MULTIPLICITY`, `CIROOT`, `RLXROOT` | Explicit user state settings. |
| `INITIAL_ORBITALS`, `STATE`, `MOLECULE`, `CARRIER_GEOMETRY` | Intrinsic seed/state/molecule/carrier identity, independent of placement. |
| `FRAGMENT` | Instance identity for stage-02 references. |
| `CASPT2_BLOCK` | Rendered CASPT2 input, or empty when disabled. |
| `IPEA_SHIFT`, `IMAGINARY_SHIFT`, `MAX_ITERATIONS` | CASPT2 settings; shifts are in Eh. |

CASPT2 uses `Multistate = 1 {{RLXROOT}}` for the explicitly selected CASSCF root; see the [OpenMolcas manual](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html).
RasOrb/VecDet are exported before CASPT2; NOCI uses the original CAS wavefunction and separate CAS/CASPT2 energies, not PT2ORB as a replacement seed.
Template changes participate in stage-01 task identity.
