# Editable calculation templates

[中文](README.zh.md)

The `openmolcas/` files render the supported CAS/CASPT2, C1 reference and RICD/CD tasks; `pyscf/` contains reserved native solver templates.
An explicit YAML template path has highest priority and a missing explicit path fails.
Without an override, preparation looks first for the matching file under stage 00, then uses the installed default in `scripts/preparation/templates/`.
Only `{{UPPER_CASE_KEY}}` placeholders are replaced; OpenMolcas `$Project` and `$CurrDir` are preserved.
Template identity participates in reuse checks; edit templates in a user-owned copy, not a shared installation.

| Role | Template |
| --- | --- |
| Monomer CAS | `openmolcas/cas.in` |
| Selected-root CASPT2 | `openmolcas/caspt2.in` |
| Per-instance C1 reference | `openmolcas/fragment_reference.in` |
| Aggregate C1 reference | `openmolcas/aggregate_reference.in` |
| RICD factors | `openmolcas/integrals_ricd.in` |
| Molecular CD factors | `openmolcas/integrals_cd.in` |
| Reserved native DF-CASSCF | `pyscf/cas.py.in` |
| Reserved AO DF/CDERI | `pyscf/integrals_dfcd.py.in` |

CASPT2 is enabled by default, including an omitted block or flag; set `caspt2.enabled: false` to disable it.
Original CAS orbitals and CI are retained, and selected-root CAS/CASPT2 energies are recorded separately.
The shift parameters are in Eh and remain the user's scientific choice.
The PySCF adapter and its CASPT2 integration are unavailable; template structure review does not certify an executable workflow.
