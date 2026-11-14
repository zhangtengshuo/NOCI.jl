# NOCI.jl

[中文](README.zh.md)

**NOCI.jl** is a Julia package for nonorthogonal configuration interaction (NOCI) based on molecular fragment wavefunctions. At present, method development is led by Teng-Shuo Zhang (College of Chemical Engineering, Zhejiang University of Technology) and Prof. Ganglong Cui (College of Chemistry, Beijing Normal University).

It constructs molecular electronic basis functions (MEBFs) from fragment states, evaluates the overlap and Hamiltonian matrix elements, and solves the resulting generalized eigenvalue problem.

The package provides two NOCI implementations: **AHPC-NOCI**, developed in this project, and **G-NOCI**, a Julia implementation of the original GronOR NOCI-F method.

## Methods

### AHPC-NOCI

AHPC-NOCI combines **active-Hamiltonian precontraction (AHPC)**, **polynomial determinant-derivative contraction (PDDC)**, and **spin-string-pair reuse**. These techniques eliminate repeated processing of large-dimensional overlap matrices and the SVD-based branching in cofactor evaluation. Compared with the conventional NOCI-F workflow, AHPC-NOCI improves computational performance by **more than four orders of magnitude**.

### G-NOCI

G-NOCI is a Julia reimplementation of the original determinant-pair NOCI-F method used in GronOR. The implementation is highly optimized for **single-node CPU/GPU parallel execution**.

G-NOCI retains the direct determinant-pair formulation of the original method. Its straightforward computational structure makes it the primary implementation base for developing additional functionality and the reference implementation for checking the correctness of AHPC-NOCI.

## Installation and preparation

See the [installation guide](Install/README.md), then follow the [schema-4 standard workflow](examples/ahpc_noci_standard/TUTORIAL.md).

The default installation includes OpenMolcas4NOCI 26.06.2 as the CAS and integral backend. Its version is managed independently and checked against `OpenMolcasCompatibility.toml`. Existing compatible integral data can also be used directly.

```bash
# Display the installation plan
bash Install/install.sh --plan

# Install the CPU environment and OpenMolcas backend
bash Install/install.sh --execute
```

The recommended C1 workflow shares intrinsic monomer CAS tasks, rotates each fragment instance, constructs its CommonOrb basis, and imports RICD or molecular-CD factors for AHPC-NOCI. CASPT2 is enabled by default for selected monomer roots and can be disabled explicitly.

## Integral data

- **G-NOCI** evaluates determinant-pair matrix elements using physical four-index electron-repulsion integrals from packed `.two` files. An explicitly configured SEWARD source can also be used to generate the four-index integrals.
- **AHPC-NOCI** accepts packed four-index integrals or factorized integrals from molecular Cholesky decomposition (CD) and metric-whitened resolution-of-identity Cholesky decomposition (RICD).
- Portable OpenMolcas SEWARD CHH5 data are validated and transformed into the actual CommonOrb basis, including frozen-core contributions.
- The standard factorized-integral input uses `integrals.backend: factorized`.

The standard case writes `factors.jld2`. See the [standard workflow](examples/ahpc_noci_standard/TUTORIAL.md) and [backend contract](examples/ahpc_noci_standard/BACKEND_CONTRACT.md) for input details.

## Running calculations

The standard tutorial describes fragment-state preparation, fragment placement, CommonOrb construction, integral preparation, and NOCI matrix assembly.

```bash
# Check the installed version
noci --version

# Prepare and inspect a calculation
noci-prepare init NEW_CASE_DIRECTORY
noci-prepare check MANIFEST.yaml
noci-prepare plan MANIFEST.yaml

# Run NOCI
noci run INPUT.yaml OUTPUT_DIRECTORY OUTPUT_ROOT

# Re-evaluate a saved calculation with overrides
noci override SNAPSHOT.jld2 OVERRIDE.yaml
```

See the [preparation and run guide](docs/tutorials/AHPC_NOCI_Preparation_and_Run.md) for the complete workflow and input requirements.

## GPU execution

The default installation provides CPU execution. To install the optional CUDA environment:

```bash
bash Install/install.sh --execute --withgpu

# Optionally select a CUDA runtime
bash Install/install.sh --execute --withgpu --cuda-runtime 12.8
```

Standard calculation inputs use the CPU by default. To use the installed GPU environment for NOCI matrix assembly (stage 05), set `runtime.use_gpu: true` in the stage-00 manifest. Select devices with `runtime.gpu_device` or `runtime.gpu_devices`. The Julia thread count must be a multiple of the number of selected GPUs. CAS, orbital, and integral preparation are performed on the CPU.

## Output

NOCI.jl records the overlap and Hamiltonian matrices, energies before and after correlation-energy shifts, state couplings, integral provenance, and calculation snapshots. Optional restart checkpoints support continuation of matrix assembly.

## Citation

The **CommonOrb paper must be cited for every publication reporting calculations performed with NOCI.jl**, regardless of whether AHPC-NOCI or G-NOCI is used. Both implementations use the reduced common molecular orbital basis that underlies the NOCI-F framework.

> R. K. Kathir, C. de Graaf, R. Broer, and R. W. A. Havenith, “Reduced Common Molecular Orbital Basis for Nonorthogonal Configuration Interaction,” *Journal of Chemical Theory and Computation* **16**, 2941–2951 (2020). [https://doi.org/10.1021/acs.jctc.9b01144](https://doi.org/10.1021/acs.jctc.9b01144)

For calculations using **G-NOCI**, also cite the original GronOR paper, which describes the NOCI-F method implemented in G-NOCI.

> T. P. Straatsma, R. Broer, A. Sanchez-Mansilla, C. Sousa, and C. de Graaf, “GronOR: Scalable and Accelerated Nonorthogonal Configuration Interaction for Molecular Fragment Wave Functions,” *Journal of Chemical Theory and Computation* **18**, 3549–3565 (2022). [https://doi.org/10.1021/acs.jctc.2c00266](https://doi.org/10.1021/acs.jctc.2c00266)

The original GronOR project is available at [gronor/gronor](https://gitlab.com/gronor/gronor).
