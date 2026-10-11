#!/usr/bin/env bash
set -euo pipefail

(( $# == 3 )) || { echo "usage: build_common_orbitals.sh WORK_DIRECTORY COMMONORB.input FILE_MAP.tsv" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
julia_bin=${JULIA_BIN:-$(command -v julia || true)}
[[ -n "$julia_bin" ]] || { echo "Julia was not found; set JULIA_BIN." >&2; exit 2; }

work=$(mkdir -p -- "$1" && cd -- "$1" && pwd -P)
control=$(cd -- "$(dirname -- "$2")" && printf '%s/%s\n' "$PWD" "$(basename -- "$2")")
file_map=$(cd -- "$(dirname -- "$3")" && printf '%s/%s\n' "$PWD" "$(basename -- "$3")")

cd -- "$work"
"$julia_bin" --startup-file=no --project="$root" "$root/scripts/preparation/lib/CommonOrbBase.jl" --file-map "$file_map" < "$control" > commonorb.out

