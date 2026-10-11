#!/usr/bin/env bash
set -euo pipefail

(( $# == 2 )) || { echo "usage: make_one_electron.sh WORK_DIRECTORY COMMONORB.input" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
julia_bin=${JULIA_BIN:-$(command -v julia || true)}
[[ -n "$julia_bin" ]] || { echo "Julia was not found; set JULIA_BIN." >&2; exit 2; }

work=$(cd -- "$1" && pwd -P)
control=$(cd -- "$(dirname -- "$2")" && printf '%s/%s\n' "$PWD" "$(basename -- "$2")")
cd -- "$work"
"$julia_bin" --startup-file=no --project="$root" "$root/scripts/preparation/lib/run_one_electron_integrals.jl" < "$control" > one_electron_conversion.out

