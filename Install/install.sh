#!/usr/bin/env bash
# File: Install/install.sh
# Purpose: Plan or execute the managed NOCI.jl installation.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
prefix=$HOME/bin/NOCI.jl
julia_bin=
source_dir=$(cd -- "$script_dir/.." && pwd -P)
manifest=$script_dir/release.toml
execute=false
activate=true
features=cpu,openmolcas
with_gpu=false
gpu_runtime=auto
while (( $# )); do
    case "$1" in
        --execute) execute=true; shift ;;
        --plan) execute=false; shift ;;
        --withgpu) with_gpu=true; shift ;;
        --no-activate) activate=false; shift ;;
        --prefix|--julia|--source-dir|--manifest|--cuda-runtime)
            (( $# >= 2 )) || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --prefix) prefix=$2 ;;
                --julia) julia_bin=$2 ;;
                --source-dir) source_dir=$2 ;;
                --manifest) manifest=$2 ;;
                --cuda-runtime) gpu_runtime=$2 ;;
            esac
            shift 2 ;;
        --help|-h)
            cat <<'HELP'
NOCI.jl installer; CUDA is installed only when explicitly selected. Default: print the plan without changing files.
  --execute               Install the selected, checksum-pinned release.
  --plan                  Print paths and components only.
  --prefix DIR            Default: ~/bin/NOCI.jl
  --source-dir DIR        Default: the extracted source containing this installer.
  --julia FILE            Use Julia 1.12.6; otherwise find or download its pinned build.
  --manifest FILE         Default: Install/release.toml
  Default components      CPU NOCI and OpenMolcas4NOCI.
  --withgpu               Also instantiate the independent locked Julia CUDA environment.
  --cuda-runtime VERSION  Optional CUDA runtime version, e.g. 12.8; requires GPU selection.
  --no-activate           Keep the previously selected installation active.
CPU installation never instantiates or loads CUDA. GPU execution also requires runtime.use_gpu=true.
HELP
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ "$with_gpu" == true ]]; then features=$features,gpu; fi
[[ "$gpu_runtime" == auto || "$gpu_runtime" =~ ^[0-9]+\.[0-9]+$ ]] || { echo 'CUDA runtime must be auto or MAJOR.MINOR.' >&2; exit 2; }
[[ "$gpu_runtime" == auto || "$with_gpu" == true ]] || { echo '--cuda-runtime requires --withgpu.' >&2; exit 2; }
[[ -f "$manifest" && -f "$source_dir/Project.toml" && -f "$script_dir/support/install.jl" ]] || { echo 'Source, configuration or installer support is missing.' >&2; exit 2; }
command -v realpath >/dev/null || { echo 'Missing tool: realpath' >&2; exit 2; }
prefix=$(realpath -m -- "$prefix")
source_dir=$(cd -- "$source_dir" && pwd -P)
manifest=$(realpath -e -- "$manifest")
if [[ "$execute" != true ]]; then
    printf 'Plan: prefix=%s; components=%s; source=%s; config=%s; activation=%s; cuda_runtime=%s\n' "$prefix" "$features" "$source_dir" "$manifest" "$activate" "$gpu_runtime"
    printf '%s\n' 'No downloads, installation, tests or file writes have occurred.'
    exit 0
fi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'This release supports Linux x86_64.' >&2; exit 2; }
for utility in curl tar gzip sha256sum flock realpath; do command -v "$utility" >/dev/null || { echo "Missing tool: $utility" >&2; exit 2; }; done
export JULIA_DEPOT_PATH=$prefix/Julia/depot
export JULIA_LOAD_PATH='@:@stdlib'
unset JULIA_PROJECT
if [[ -n "$julia_bin" ]]; then
    [[ -x "$julia_bin" ]] || { echo 'Julia is not executable.' >&2; exit 2; }
    julia_bin=$(realpath -- "$julia_bin")
    "$julia_bin" --startup-file=no -e 'VERSION == v"1.12.6" || error("This locked release requires Julia 1.12.6")'
else
    candidate=$(command -v julia || true)
    if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" --startup-file=no -e 'exit(VERSION == v"1.12.6" ? 0 : 1)' >/dev/null 2>&1; then
        julia_bin=$(realpath -- "$candidate")
    fi
fi
mkdir -p -- "$prefix/installation"
exec 9>"$prefix/installation/install.lock"
flock -n 9 || { echo 'Another installer is using this prefix.' >&2; exit 2; }
if [[ -z "$julia_bin" ]]; then
    julia_dir=$prefix/Julia/julia-1.12.6
    if [[ ! -x "$julia_dir/bin/julia" ]]; then
        temporary=$(mktemp -d "$prefix/installation/julia-bootstrap.XXXXXX")
        trap 'rm -rf -- "$temporary"' EXIT
        curl --fail --location --proto '=https' --proto-redir '=https' --retry 3 \
            'https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.6-linux-x86_64.tar.gz' -o "$temporary/julia.tar.gz"
        printf '%s  %s\n' bbabf3bef19421a9dbd24a767d807606ab85e444323b5a1c73ffe293fa3d079a "$temporary/julia.tar.gz" | sha256sum --check --status
        [[ ! -e "$julia_dir" ]] || { echo 'Refusing to overwrite a partial Julia directory.' >&2; exit 2; }
        mkdir -p -- "$prefix/Julia"
        tar -xzf "$temporary/julia.tar.gz" -C "$temporary" --no-same-owner
        mv -- "$temporary/julia-1.12.6" "$julia_dir"
        rm -rf -- "$temporary"
        trap - EXIT
    fi
    julia_bin=$julia_dir/bin/julia
fi
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
exec "$julia_bin" --startup-file=no --project="$source_dir" "$script_dir/support/install.jl" "$prefix" "$source_dir" "$manifest" "$features" "$activate" "$script_dir" "$gpu_runtime"
