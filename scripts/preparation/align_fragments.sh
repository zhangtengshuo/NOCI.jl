#!/usr/bin/env bash
set -euo pipefail
export JULIA_LOAD_PATH='@:@stdlib'

(( $# >= 3 )) || { echo "usage: align_fragments.sh AGGREGATE.xyz OUTPUT_DIRECTORY FRAGMENT.xyz [...]" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
julia_bin=${JULIA_BIN:-$(command -v julia || true)}
[[ -n "$julia_bin" ]] || { echo "Julia was not found; set JULIA_BIN." >&2; exit 2; }

aggregate=$1
output=$2
shift 2
mkdir -p -- "$output"
"$julia_bin" --startup-file=no --project="$root" "$root/scripts/preparation/lib/AutoRotate.jl" "$aggregate" "$@" --out-dir "$output"

