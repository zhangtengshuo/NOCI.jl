# NOCI.jl

[中文](README.zh.md)

NOCI.jl implements determinant-pair NOCI (DP-NOCI) and active-Hamiltonian precontraction NOCI (AHPC-NOCI) in Julia.
It evaluates raw and overlap-normalized $S/H_1/H_2/H$ matrices and solves the symmetric generalized NOCI eigenproblem from determinant, orbital and integral data.
This calendar release supports CPU execution and optional GPU acceleration; its version is defined in `Project.toml` and reported consistently by the CLI, scientific output and JSON.

## Install and prepare

Read the [installation guide](Install/README.md), then use the [schema-4 standard workflow](examples/ahpc_noci_standard/TUTORIAL.md).
The default installation includes OpenMolcas4NOCI 26.06.2 as the CAS/integral backend; it is versioned independently and checked against `OpenMolcasCompatibility.toml`.
Existing compatible integral data can be used without invoking the CAS/integral backend.

```bash
bash Install/install.sh --plan
# The next command performs installation and small installation checks.
bash Install/install.sh --execute
```

The recommended C1 workflow shares intrinsic monomer CAS tasks, rotates each instance, constructs its own CommonOrb basis and imports RICD/CD factors for AHPC-NOCI.
CASPT2 is enabled by default for selected monomer roots and can be disabled explicitly.
PySCF DFCD and mixed backends have a [reserved contract](examples/ahpc_noci_standard/BACKEND_CONTRACT.md), without an executable numerical adapter in this release.
Runtime does not install, download or resolve packages.

## Methods and data

`DPNOCI` evaluates screened determinant pairs with physical four-index ERIs from packed `.two` files; an explicitly configured embedded SEWARD source is first materialized into memory-preflighted four-index ERIs.
Direct factor bundles are rejected for DP-NOCI.
`AHPCNOCI` combines active-Hamiltonian precontraction, canonical spin-string-pair reuse, sparse contraction and occupied-spin complementary cofactors.
Its one production path accepts packed four-index ERIs or unit-metric Gram factors from molecular CD and metric-whitened RICD.
Support lookup and polynomial/certificate-guarded LU cofactor choices remain typed algorithm options; precontraction is an AHPC feature, not a third method.
Large conventional four-index workloads need host-memory preflight; the recommended factor workflow avoids forming a full four-index matrix.

OpenMolcas SEWARD portable CHH5 data are validated and transformed directly into the actual common orbital basis, including frozen-core corrections, without a library call in the HDF5 importer.
The original embedded RunFile/SEWARD Julia-MOTRA path remains an expert input route and requires a declared compatible backend when its shared library is used.
Use `integrals.backend: factorized`; legacy `cholesky` input spelling and version-1/version-2 bundles remain readable.
RICD atomic auxiliary-basis thresholds and molecular-CD thresholds remain distinct in provenance.
The importer accepts CHH5 1.0 and explicit-provenance 1.1; unknown legacy factor origins remain unknown rather than being inferred.
The standard case writes `factors.jld2`; standalone conversion tools retain their documented prefix-based naming.

## Commands

```bash
noci --version
noci run INPUT.yaml OUTPUT_DIRECTORY OUTPUT_ROOT
noci override SNAPSHOT.jld2 OVERRIDE.yaml
noci-prepare init NEW_CASE_DIRECTORY
noci-prepare check MANIFEST.yaml
noci-prepare plan MANIFEST.yaml
```

Production outputs retain unshifted and correlation-shifted matrices/energies, scalar state couplings, provenance, snapshots and optional restart checkpoints.
See the [preparation guide](docs/tutorials/AHPC_NOCI_Preparation_and_Run.md) and [standard tutorial](examples/ahpc_noci_standard/TUTORIAL.md) for real calculation commands and input assumptions.
Development tests, benchmark outputs, machine settings and release-management material are not distributed with the product.

## Citation

If you use NOCI.jl — through either the AHPC-NOCI or the DP-NOCI module — you must cite the paper that introduced the reduced common molecular orbital basis (`COMMONORB`), the foundation of the NOCI-F method on which both algorithms rest:

> R. K. Kathir, C. de Graaf, R. Broer, and R. W. A. Havenith, "Reduced Common Molecular Orbital Basis for Nonorthogonal Configuration Interaction", *J. Chem. Theory Comput.* **16**, 2941–2951 (2020), doi:[10.1021/acs.jctc.9b01144](https://doi.org/10.1021/acs.jctc.9b01144)

## Credits and provenance

The determinant-pair method (DP-NOCI) originates from the GronOR program:

> T. P. Straatsma, R. Broer, A. Sanchez-Mansilla, C. Sousa, and C. de Graaf, "GronOR: Scalable and Accelerated Nonorthogonal Configuration Interaction for Molecular Fragment Wave Functions", *J. Chem. Theory Comput.* **18**, 3549–3565 (2022), doi:[10.1021/acs.jctc.2c00266](https://doi.org/10.1021/acs.jctc.2c00266)

GronOR ([gitlab.com/gronor/gronor](https://gitlab.com/gronor/gronor)) is copyright of the University of Groningen and is distributed under the Apache License 2.0, the same license as this package. NOCI.jl rewrites the method in Julia with extensive single-node performance optimizations, keeping most functionality and settings consistent with GronOR; the provenance is also declared in the header of every `src/algorithms/dp_noci/` file.

The fragment correlation-energy (CASPT2) shift with Gallup-Norbeck weighting, implemented in `src/core/correlation_shifts.jl`, likewise originates from GronOR (routine `gronor_correlation_energy`, Kathir, Havenith, and de Graaf, 2019, published in the paper above) and is rewritten in Julia with both the GronOR Fortran and the original paper weight conventions.

## Optional GPU installation

The default installation includes CPU NOCI and OpenMolcas4NOCI without installing CUDA.
Add `--withgpu` to install the separate locked GPU environment with CUDA.jl 5.11.0 and the preparation dependencies, including YAML.

```bash
bash Install/install.sh --execute --withgpu
bash Install/install.sh --execute --withgpu --cuda-runtime 12.8
```

Without `--execute`, the installer only prints a plan.
`--cuda-runtime` is optional and requires `--withgpu`; choose a runtime supported by the NVIDIA device and driver.
The CPU and GPU environments use their own declared dependencies and do not load packages from the account's default Julia environment.
Machine-specific CUDA preferences are written only to the installation.
The installation records the selected environment and checks the GPU extension and a small Float64 device operation before activation.

Standard calculation inputs still default to CPU.
Set `runtime.use_gpu: true` in the stage-00 manifest to use the installed GPU environment for stage 05; `runtime.gpu_device` or `runtime.gpu_devices` select devices.
The Julia thread count must be a multiple of the selected GPU count.
CAS, orbital and integral preparation use CPU, and stage-05 workers retain the selected Julia environment.
