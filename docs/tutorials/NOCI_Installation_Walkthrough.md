# Install the CPU release

[中文](NOCI_Installation_Walkthrough.zh.md)

This release targets Linux x86_64 with glibc 2.28 or newer and pins Julia 1.12.6.
GPU execution and installation are unavailable; the default components are CPU NOCI and OpenMolcas4NOCI 26.06.2.
Use `--with cpu` when compatible integral and wavefunction inputs are already available.
The installation commands below perform downloads, write environments and run small installation checks only when `--execute` is supplied.

## 1. Obtain the source

Download the selected tagged NOCI source or release archive and extract it into a user-owned directory.
For a supplied release archive, compare its SHA-256 with the published checksum before extracting.
Use the archive's own `Install/install.sh`, `Install/release.toml` and support files together; do not mix installer versions.
The installer reads the source directory containing it by default and needs no guessed NOCI repository URL.
Required system commands are Bash, curl, tar, sha256sum and flock; preparation also uses Python 3 for existing utility operations.
The package version is authoritative in `Project.toml`; backend compatibility is in `OpenMolcasCompatibility.toml`.

## 2. Review the plan

From the extracted source root:

```bash
bash Install/install.sh --plan
bash Install/install.sh --plan --with cpu
```

These commands print paths and components without downloads, file writes, installation or tests.
`--with openmolcas` and `--all` select CPU plus OpenMolcas; GPU component requests fail.

## 3. Install

```bash
bash Install/install.sh --execute --prefix "$HOME/bin/NOCI.jl"
```

Use `--julia /path/to/julia` to select an existing Julia 1.12.6 or `--source-dir /path/to/extracted/source` to select a separate source.
Without a suitable selected Julia, the installer downloads its pinned official archive and checks its SHA-256.
`--no-activate` keeps the previously selected version active after successful installation.
Installation rejects an existing version directory; inspect incomplete attempts rather than overwriting them.

The installer checks NOCI source/configuration versions and CPU dependencies, copies only product files, instantiates the exact Julia lock and installs checksum-pinned native assets.
OpenMolcas program and native dependency prefixes are separate; a compatible unchanged dependency identity can be reused by multiple versions.
The standalone micromamba executable installs native packages from exact locks, without updating the selected scientific dependency versions.
Installed program metadata, library bindings, source revision and accepted backend capabilities are checked before activation.
Small H2 CD and RICD/CHH5 checks run during an authorized installation; they do not certify a full CAS/CommonOrb/NOCI case.
Only a successful installation writes its completed record and optionally switches the active version.
Failure leaves the prior selection active and preserves incomplete directories for inspection.

## 4. Installed layout

```text
~/bin/NOCI.jl/
  v26.10.0/                   CPU product and version-owned preparation commands
  noci -> v26.10.0             Active version
  bin/noci                    Version-selection dispatcher
  OpenMolcas4NOCI/
    v26.06.2-<identity>/      Immutable program prefix
    dependencies/<identity>/ Shared native dependency prefix
  Julia/
    julia-1.12.6/             Downloaded when needed
    depot/                   Shared Julia package/cache storage
  installation/
    records/v26.10.0.toml     Completed installation identity
    downloads/               Verified asset cache
```

CPU-only installations omit the OpenMolcas component and do not require a backend to run already prepared compatible data.
Scientific runtime never installs packages or resolves/downloads dependencies.

## 5. Select the exact version

```bash
"$HOME/bin/NOCI.jl/bin/noci" --list
"$HOME/bin/NOCI.jl/bin/noci" --use v26.10.0
"$HOME/bin/NOCI.jl/bin/noci" --version
NOCI_INSTALL="$HOME/bin/NOCI.jl/v26.10.0"
source "$NOCI_INSTALL/bin/noci-environment.sh"
command -v noci-prepare
```

The version environment puts its own bin on PATH; preparation uses that version's implementation and recorded backend rather than an independent root-level dispatcher.
Normal preparation checks compare the installed backend identity with the declared compatibility policy; unknown versions fail even when their version number is larger.
Use the [schema-4 preparation guide](AHPC_NOCI_Preparation_and_Run.md) next.

## 6. Diagnose installation failures

Check available storage, network access to the pinned assets, Julia version and required utilities first.
A checksum failure is not bypassed; obtain the correct pinned asset.
A backend/library mismatch requires selecting a consistent recorded program/root/dependency environment.
Do not use a generic upstream OpenMolcas installation as a replacement for the declared OpenMolcas4NOCI build.
For an incomplete version, keep evidence and choose an independent prefix for a new attempt.
No command in this guide claims that the current release has already passed target-machine acceptance.
