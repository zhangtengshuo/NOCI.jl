# Install NOCI.jl

[中文](README.zh.md)

This release targets Linux x86_64 with glibc 2.28 or newer and uses Julia 1.12.6.
The default installation includes CPU NOCI and OpenMolcas4NOCI 26.06.2; add `--withgpu` for CUDA support.
CPU installation does not install or load CUDA.

## Obtain the source and review the plan

Extract the selected source archive and run its own installer from the source root.
Compare a supplied archive with its published SHA-256 before extraction.
Keep `Install/install.sh`, `Install/release.toml` and the support files from the same source revision.
Required utilities are Bash, curl, tar, gzip, sha256sum, realpath and flock; preparation also uses Python 3.
`Project.toml` defines the NOCI version and `OpenMolcasCompatibility.toml` defines the accepted backend identity.

```bash
bash Install/install.sh --plan
bash Install/install.sh --plan --withgpu
```

A plan prints paths and components without writing files or installing software.
CPU is always included and has no separate selection option.

## Install

```bash
bash Install/install.sh --execute --prefix "$HOME/bin/NOCI.jl"
# Include the optional GPU environment:
bash Install/install.sh --execute --prefix "$HOME/bin/NOCI.jl" --withgpu
```

Choose one of these commands for an installation prefix.
Use `--julia /path/to/julia` to select an existing Julia 1.12.6; an explicitly selected incompatible version is rejected.
Otherwise, the installer uses a compatible Julia from PATH or downloads its pinned official archive and verifies its SHA-256.
`--source-dir /path/to/source` selects the product source and `--manifest /path/to/release.toml` selects its matching installation configuration.
`--no-activate` keeps the previously selected version active after successful installation.
An existing version directory is preserved and rejected; use another prefix for a separate installation.

The installer copies product files, instantiates exact Julia locks, and installs checksum-pinned native packages with a standalone micromamba executable.
It checks the direct dependencies of the selected profiles, including YAML for standalone preparation commands, before copying installation files.
It uses a dedicated depot and an explicit project with no fallback to the account's default Julia environment.
Project and Manifest files retain their identities during instantiation; no dependency updates or resolution are performed.
The OpenMolcas program and its exact native dependency prefix are separate, and completed compatible native prefixes can be reused.
Backend identity, shared-library binding and small serial H2 CD/RICD exports are checked before activation.
An explicit GPU installation also checks the CUDA extension and a small Float64 device operation.
A completed installation record is written only after these checks succeed.

## Installed layout and commands

```text
~/bin/NOCI.jl/
  v26.10.0/                   Product and preparation commands
    optional/gpu/             Optional GPU project and lock
    bin/noci-environment.sh   Generated version environment
  noci -> v26.10.0            Active version
  bin/noci                    Version-selection dispatcher
  bin/noci-prepare            Preparation dispatcher
  OpenMolcas4NOCI/
    v26.06.2-<identity>/       Program prefix
    dependencies/<identity>/  Native dependency prefix
  Julia/
    julia-1.12.6/             Downloaded when needed
    depot/                    Managed packages and caches
  installation/
    records/v26.10.0.toml     Completed installation identity
    downloads/                Verified asset cache
```

```bash
"$HOME/bin/NOCI.jl/bin/noci" --list
"$HOME/bin/NOCI.jl/bin/noci" --use v26.10.0
"$HOME/bin/NOCI.jl/bin/noci" --version
source "$HOME/bin/NOCI.jl/v26.10.0/bin/noci-environment.sh"
noci-prepare help
```

The generated environment selects the installation's Julia, project, depot and preparation commands.
Backend compatibility is checked when backend functionality is used; calculations with existing compatible integrals do not invoke the backend.
Scientific runtime does not install packages or resolve dependencies.
Continue with the [preparation guide](../docs/tutorials/AHPC_NOCI_Preparation_and_Run.md).

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
