# Detailed standard AHPC-NOCI tutorial

[中文](TUTORIAL.zh.md)

This tutorial accompanies the schema-4 YAML definitions and stages 00–05 in this directory.
The executable route is C1, CPU, OpenMolcas monomer CAS and OpenMolcas RICD/CD factors for AHPC-NOCI.
PySCF DFCD is reserved; solver templates and structure review do not certify its numerical adapter.
Validate real calculations in a user-owned copy rather than writing products into source directories.

## 1. What users supply

Complete required monomer geometry optimization, frequency checks and aggregate structure preparation beforehand; these are outside stages 00–05.
For each molecule/carrier geometry, provide XYZ, matching initial RasOrb and every required state's CAS parameters.
Seeds must match geometry, basis and AO center order; states may share one seed or use separate files.
The workflow does not generate seeds, select/reorder active orbitals, or judge the physical character of roots and orbitals.
Users are responsible for their scientific choices; software checks explicit references, counts, completion and usable products.
XYZ coordinates are in Angstrom.
Aggregate fragment blocks follow MEBF row order and each block preserves its monomer/seed atom order, including identical-element atoms.

## 2. Molecules, carriers, instances and states

`molecules.f` declares a molecule, for example ethene; `molecules.g` can declare butadiene.
Short f/g/h names are convenient, but registry keys may be other nonempty ASCII alphanumeric identifiers.
Molecular identity is explicit and is not inferred from a chemical graph or content hash.
Declare each point group under `molecules.<id>.symmetry`; executable symmetry is C1.
State-level OpenMolcas `symmetry: 1` is an irrep index, distinct from the molecular point group.
One molecule can have multiple carriers such as `f/GS0` and `f/GS1`, each with its own XYZ and seeds.
Changing a carrier's contents cannot preserve an old same-name cached calculation automatically.

`f_99`, `f_03` and `g_05` identify positioned instances, each explicitly referencing a molecule/carrier under `fragments`.
Labels contain exactly one underscore with a nonempty ASCII alphanumeric string on each side; they are case-sensitive and allow neither spaces, hyphens nor path characters.
The label prefix does not infer the molecule; the explicit `molecule` reference is authoritative.
MEBF row order alone determines fragment order; numeric labels are never sorted.
`f/GS0/S1` is one original monomer CAS task, shared when several instances request it.
Stage 02 rotates its orbitals into each instance's position; an instance's resulting `.vec` cannot be substituted for another instance's representation.

## 3. Select the installation and create a case

Install this release using `Install/install.sh`, including its OpenMolcas component for preparation.
Backend compatibility comes from `OpenMolcasCompatibility.toml`; unknown newer versions are not automatically accepted.
Scientific execution never runs a package manager, resolves an environment or downloads packages.
The standalone micromamba executable is used only for installing OpenMolcas native dependencies.

```bash
NOCI_INSTALL="$HOME/bin/NOCI.jl/v26.10.0"
source "$NOCI_INSTALL/bin/noci-environment.sh"
export PATH="$NOCI_INSTALL/bin:$PATH"
command -v noci-prepare
noci-prepare init "$HOME/noci_cases/ethene"
cd "$HOME/noci_cases/ethene"
```

Use the preparation command from the exact selected version; there is no root-level preparation dispatcher.
`init` copies the standard inputs, seeds, wrappers and documentation, while new results stay in the user's case.
`JULIA_BIN`, `NOCI_JULIA_PROJECT` and `NOCI_OPENMOLCAS4NOCI` come from installation context, not scientific YAML.
Implementations remain under installed `scripts/preparation/`, sharing the driver, registry and library utilities; cases contain inputs and short wrappers.

## 4. The stage-00 definition

Open `00_system_define/ethene_dimer.yaml`; relative paths are resolved from its directory rather than the caller's current directory.
Each YAML is a complete independent definition with no implicit inheritance from another YAML.

| Block | Meaning |
| --- | --- |
| `schema_version`, `project`, `method` | Version 4, unique output identity, fixed `ahpc_noci`. |
| `geometry` | Aggregate XYZ specifying target fragment poses. |
| `cas` | Backend, orbital basis and monomer integral approximation. |
| `caspt2` | Enable flag, IPEA/imaginary shifts and iterations. |
| `templates` | Explicit workflow-owned input templates. |
| `molecules` | Per-molecule symmetry, carrier XYZ, seeds and states. |
| `fragments` | Instance bindings and automatic frozen-core flags. |
| `system` | Aggregate charge and target total spin multiplicity. |
| `mebf_table` | Fragment rows and MEBF columns. |
| `common_orbitals` | This project's orbital-union retention threshold. |
| `integrals` | Aggregate RICD/CD method and its own generation threshold. |
| `screening` | Diagonal/off-diagonal CI screening. |
| `runtime` | Julia/BLAS threads, blocked workspace and GC hint. |
| `output` | Optional checkpoint selection. |

Monomer CAS integral settings are independent of aggregate stage-04 settings.
The CD aggregate example preserves the monomer CAS definition, allowing stage-01 reuse while constructing independent CommonOrb/integral products.
Changing the monomer method or threshold also changes its task definition.
OpenMolcas `nactel`, `inactive`, `ras2`, `multiplicity`, `ciroot` and `rlxroot` retain their native active-electron, inactive-orbital, active-orbital, spin and root meanings.
S1 uses `ciroot: "1 2; 2"` and `rlxroot: 2`; energy extraction retains that selected root.

```yaml
initial_orbitals:
  default: orbitals/ethene_ccpvdz_seed.RasOrb
  S1: orbitals/ethene_s1special.RasOrb
```

This illustrative S1 override requires a user-provided file; supplied examples reference only their available seeds.
Other requested states use `default`; alternatively list every state explicitly without a default.
Unused defaults are not read, misspelled state names fail, and expanded effective per-state seeds determine task identity.

`fragments.<label>.frozen_orbitals` is Boolean, defaults to false and must not contain an orbital-number string.
True freezes the automatic all-electron chemical core, using the [PySCF chemcore convention](https://pyscf.org/_modules/pyscf/data/elements.html#chemcore) implemented in Julia without requiring PySCF.
Ethene has two core spatial orbitals and butadiene four; all inactive orbitals must not be treated as core.
Users must place core orbitals first in the inactive block; the workflow expands the flag into `1-Ncore` without interpreting orbital chemistry or reordering active spaces.
ECP subtraction and special core selections are outside this automatic contract.
Structure-only review defers XYZ-based core counting; normal checks determine actual counts before execution.
Stage 03 supplies actual frozen/deleted counts to stage 04 automatically.
`common_orbitals.threshold` defaults to `1.0e-5` and is independent of integral thresholds.

### CASPT2 and templates

CASPT2 is enabled when its block or `enabled` flag is omitted.

```yaml
caspt2:
  enabled: true
  ipea_shift: 0.0
  imaginary_shift: 0.2
  max_iterations: 80
```

Shifts are in Eh and are user scientific choices, not universal recommendations.
Stage 01 inserts single-state CASPT2 after RASSCF using `Multistate = 1 <rlxroot>`; see the [OpenMolcas manual](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html).
Original RasOrb/VecDet are exported before CASPT2; selected-root CAS/CASPT2 energies are recorded separately in completion records and `.det`.
With CASPT2 disabled, both energies are equal and the fragment correlation correction is zero.
NOCI retains unshifted results alongside its existing fragment-energy correction outputs.
CASPT2 flag/parameter changes invalidate incompatible task reuse.
The reserved PySCF route cannot silently skip CASPT2 or substitute a different PT2 method.

Explicit YAML [template paths](00_system_define/templates/README.md) have priority and missing explicit paths fail.
Otherwise select the matching stage-00 template before the installed default.
`{{UPPER_CASE_KEY}}` placeholders use text substitution while `$Project`/`$CurrDir` remain native OpenMolcas variables.
Template changes enter reuse definitions.
Native PySCF solver templates do not provide the still-missing bundle/rotation/CommonOrb/factor adapters.

### MEBF columns and total spin

The dimer columns are S0S0, S1S0, S0S1, D+D-, D-D+ and T1T1:

```yaml
mebf_table:
  - [f_99, S0, S1, S0, D+, D-, T1]
  - [f_03, S0, S0, S1, D-, D+, T1]
```

The trimer uses exactly these six declared columns on GS0S0S0 geometry:

```yaml
mebf_table:
  - [f_99, S0, S1, D+, D-, T1, T1]
  - [f_03, S0, S0, D-, D+, T1, S0]
  - [f_12, S0, S0, S0, S0, S0, T1]
```

`system.spin_multiplicity` declares total spin; two triplet fragments need not form a total triplet.
The examples use supported total-singlet coupling paths.

### Integral origins

RICD uses atomic aCD/acCD auxiliary bases; its threshold controls atomic basis construction rather than molecular ERI truncation.
Molecular CD decomposes molecular ERIs and has a molecular-CD tolerance.
Reserved PySCF DFCD uses three-center integrals and metric whitening into unit-metric Gram factors.
Its CAS/integral auxiliary bases inherit when only one is specified, or can be chosen separately.
At least one auxiliary basis and `integrals.metric_lindep` must be explicit; do not reuse RICD/CD threshold names for a metric cutoff.
Origins and threshold semantics accompany factors even when the AHPC kernel shares their unit-metric representation.
See the [backend contract](BACKEND_CONTRACT.md) and [native DF implementation](https://pyscf.org/_modules/pyscf/df/incore.html).

## 5. Check and plan without calculations

```bash
bash 00_system_define/run.sh ethene_dimer.yaml
noci-prepare plan 00_system_define/ethene_trimer.yaml
noci-prepare plan 00_system_define/ethene2_butadiene.yaml
noci-prepare plan 00_system_define/pyscf_dfcd.yaml --structure-only
```

Normal checks validate fields, references, atom/block order, seed existence, charge/spin combinations, parameter types/ranges and declared backend identity/capabilities.
They neither judge active-space chemistry nor run OpenMolcas, PySCF or NOCI energies.
The dimer requires five intrinsic tasks and ten instance/state bindings; the trimer requires five tasks and eleven bindings; the f/f/g example requires seven tasks.
Unused declared states are not computed.
`--structure-only` skips file/backend availability checks and does not certify executability.

## 6. Prepare, compute and finalize

| Mode | Responsibility |
| --- | --- |
| `prepare` | Check prerequisites, write tasks/inputs; stage 02 also fits geometry. |
| `compute` | Run the current stage's external tasks or final NOCI calculation. |
| `finalize` | Validate completion and perform numerical rotation/CommonOrb/factor conversion as required. |
| `all` | Execute the three phases for this stage. |

`finalize` can perform substantial numerical work and is not a file-copy-only operation.
For a scheduler, prepare tasks, submit outstanding OpenMolcas inputs from `tasks_to_submit.txt`, then finalize actual completed results.
This workflow embeds no cluster/account/scheduler/SSH settings; those belong to deployment.
Stage-05 compute is a Julia job, and subsequent stages require genuine completed preceding products.

The following commands launch real calculations:

```bash
bash run_all.sh ethene_dimer.yaml
```

Equivalent per-stage execution is:

```bash
bash 01_fragment_states/run.sh ethene_dimer.yaml all
bash 02_fragment_placement/run.sh ethene_dimer.yaml all
bash 03_common_orbitals/run.sh ethene_dimer.yaml all
bash 04_integrals/run.sh ethene_dimer.yaml all
bash 05_noci/run.sh ethene_dimer.yaml all
```

Each wrapper also accepts `prepare`, `compute` or `finalize` instead of `all`.
Julia and BLAS threads are applied separately to final NOCI; `scratch_mb` is a blocked-workspace budget and `heap_size_hint_mb` is a GC hint, neither an operating-system total-memory limit.

## 7. Stage products

Stage 01 stores intrinsic tasks under `f/GS0/S1/` and instance/state copies under `bindings/<project>/`.
Reuse compares molecule/carrier/state, parsed XYZ/atom order, CAS/CASPT2 settings, templates, seed path/size/mtime and validated backend identity.
Large scientific files are not fully hashed during routine startup; preserving size/mtime while replacing content cannot be detected by these lightweight records.
Use a new carrier identifier or explicitly retire affected tasks/downstream products after changing scientific inputs.

Stage 02 writes `matrix_f_99.txt`, `placed_f_99.xyz` and the reconstructed `aggregate.xyz` using proper rigid transforms in declared atom order.
RMSD is reported but does not assess chemical suitability or exchange identical-element centers.
Internal carrier geometry is preserved; aggregate input defines pose, not an orbital deformation.
The same rotation applies to orbitals, and stages 03/04/final reports use this actual aggregate geometry.

Stage 03 writes COMMONORB and short `f_99_S1.vec/.det` names in the project's retained basis.
The internal map hides tool-required project prefixes; selected-root energies and frozen/deleted counts must come from real outputs.
Missing deleted counts fail rather than defaulting to zero.

Stage 04 uses mutually consistent C1 RICD/CD templates, with no EXPBAS, and transforms CHH5 into `integrals.one`, `integrals.sys` and `factors.jld2` using the actual common basis and counts.
It neither writes conventional `.two` nor reconstructs a full four-index matrix.

Stage 05 writes its generated project YAML, fixed AHPC-NOCI/factorized input, progress `.sec`, scientific `.out`, final `.json`, matrix snapshot and optional checkpoint.
Change inputs in stage 00 and choose a new project rather than editing generated YAML.
Real completion and final JSON are required before claiming numerical results or acceptance.

## 8. Independent projects and shared stage 01

Every YAML fully defines its own calculation, with no donor-project name or execution-order dependency.
Dimer and trimer share their five identical intrinsic ethene tasks regardless of which runs first.
Stages 02–05 remain independent, including the actual common basis and integrals.
The three-neutral-MEBF subset needs only S0/S1 and can run first in an empty case root; its CommonOrb uses its own state set.
The aggregate-CD example can reuse unchanged monomer tasks while making independent downstream products.
A new project isolates aggregate outputs but cannot bypass a changed same-name shared monomer task; use a new carrier or an independent case root when appropriate.

## 9. Restart and troubleshooting

Completed monomer tasks can be reused, and valid current-project products/checkpoints support continuation.
Finished NOCI JSON is not overwritten in place when scientific settings change.
An exclusive stage lock directory requires checking whether its owning process is still alive before manually removing a stale lock.

| Symptom | Check |
| --- | --- |
| Missing preparation command | Source the exact installation environment and put its bin first on PATH. |
| Missing seed | Resolve the YAML path from stage 00 and supply the requested state's actual seed. |
| XYZ mismatch | Verify MEBF row order, fragment atom counts/order and orbital centers. |
| Task identity conflict | Check geometry, basis, CAS/CASPT2 settings, seed, templates and backend build. |
| Missing prerequisites | Complete the preceding real `finalize`; never use empty placeholder products. |
| Missing deleted count | Inspect actual CommonOrb output rather than inventing zero. |
| RICD CHH5 failure | Check backend compatibility and avoid SEWARD Cholesky options with GATEWAY RICD. |
| PySCF unavailable | Use structure review only; renaming a backend cannot enable an absent adapter. |

## 10. Review without modifying source data

Review molecule/carrier/instance distinctions, atom/seed correspondence, MEBF columns, shared intrinsic tasks, independent aggregate stages and threshold meanings.
Check every stage's prerequisites, actions and outputs and confirm wrappers use the selected installed preparation implementation.
Use normal check/plan when the backend is configured, or structure-only review for schema inspection.
Run input generation and calculations only in a user-owned copy; simulated products are never scientific acceptance evidence.
