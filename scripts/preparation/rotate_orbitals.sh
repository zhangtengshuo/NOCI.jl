#!/usr/bin/env bash
set -euo pipefail
export JULIA_LOAD_PATH='@:@stdlib'

(( $# >= 4 )) || { echo "usage: rotate_orbitals.sh MATRIX.txt RUNFILE OUTPUT_DIRECTORY STATE.orb [...]" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
julia_bin=${JULIA_BIN:-$(command -v julia || true)}
[[ -n "$julia_bin" ]] || { echo "Julia was not found; set JULIA_BIN." >&2; exit 2; }

matrix=$1
runfile=$2
output=$3
shift 3
mkdir -p -- "$output"
"$julia_bin" --startup-file=no --project="$root" "$root/scripts/preparation/lib/RotateOrb.jl" "$matrix" "$@" --runfile "$runfile" --out-dir "$output"

