#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
prefix=$HOME/bin/NOCI.jl
julia_bin=
source_dir=$(cd -- "$script_dir/.." && pwd -P)
manifest=$script_dir/release.toml
execute=false
activate=true
features=cpu,openmolcas
while (( $# )); do
    case "$1" in
        --execute) execute=true; shift ;;
        --plan) execute=false; shift ;;
        --no-activate) activate=false; shift ;;
        --prefix|--julia|--source-dir|--manifest|--with)
            (( $# >= 2 )) || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --prefix) prefix=$2 ;;
                --julia) julia_bin=$2 ;;
                --source-dir) source_dir=$2 ;;
                --manifest) manifest=$2 ;;
                --with)
                    case "$2" in
                        cpu) features=cpu ;;
                        openmolcas|cpu,openmolcas) features=cpu,openmolcas ;;
                        *) echo 'This release supports --with cpu or --with openmolcas only.' >&2; exit 2 ;;
                    esac ;;
            esac
            shift 2 ;;
        --all) features=cpu,openmolcas; shift ;;
        --help|-h)
            cat <<'HELP'
NOCI.jl CPU installer. Default: print the plan without changing files.
  --execute               Install the selected, checksum-pinned CPU release.
  --plan                  Print paths and components only.
  --prefix DIR            Default: ~/bin/NOCI.jl
  --source-dir DIR        Default: the extracted source containing this installer.
  --julia FILE            Use Julia 1.12.6; otherwise find or download its pinned build.
  --manifest FILE         Default: Install/release.toml
  --with cpu              Numerical package for already prepared integral data.
  --with openmolcas       CPU plus OpenMolcas4NOCI (default).
  --all                   CPU plus OpenMolcas4NOCI.
  --no-activate           Keep the previously selected installation active.
GPU installation is unavailable in this release.
HELP
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -f "$manifest" && -f "$source_dir/Project.toml" && -f "$script_dir/support/install.jl" ]] || { echo 'Source, configuration or installer support is missing.' >&2; exit 2; }
if [[ "$execute" != true ]]; then
    printf 'Plan: prefix=%s; components=%s; source=%s; config=%s; activation=%s\n' "$prefix" "$features" "$source_dir" "$manifest" "$activate"
    printf '%s\n' 'No downloads, installation, tests or file writes have occurred.'
    exit 0
fi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'This release supports Linux x86_64.' >&2; exit 2; }
for utility in curl tar sha256sum flock; do command -v "$utility" >/dev/null || { echo "Missing tool: $utility" >&2; exit 2; }; done
if [[ -z "$julia_bin" ]]; then julia_bin=$(command -v julia || true); fi
if [[ -n "$julia_bin" ]]; then
    [[ -x "$julia_bin" ]] || { echo 'Julia is not executable.' >&2; exit 2; }
    julia_bin=$(realpath -- "$julia_bin")
    "$julia_bin" --startup-file=no -e 'VERSION == v"1.12.6" || error("This locked release requires Julia 1.12.6")'
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
            https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.6-linux-x86_64.tar.gz -o "$temporary/julia.tar.gz"
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
export JULIA_DEPOT_PATH=$prefix/Julia/depot
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
exec "$julia_bin" --startup-file=no "$script_dir/support/install.jl" "$prefix" "$source_dir" "$manifest" "$features" "$activate" "$script_dir"
