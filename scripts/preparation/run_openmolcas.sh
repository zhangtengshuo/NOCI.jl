#!/usr/bin/env bash
set -euo pipefail

(( $# >= 1 && $# <= 2 )) || { echo "usage: run_openmolcas.sh INPUT.in [SCRATCH_DIRECTORY]" >&2; exit 2; }
input=$1
input=$(cd -- "$(dirname -- "$input")" && printf '%s/%s\n' "$PWD" "$(basename -- "$input")")
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
julia_bin=${JULIA_BIN:-$(command -v julia || true)}
[[ -n "$julia_bin" ]] || { echo "Julia was not found; set JULIA_BIN." >&2; exit 2; }

scratch=
if (( $# == 2 )); then
    mkdir -p -- "$2"
    scratch=$(cd -- "$2" && pwd -P)
fi
arguments=("$input")
[[ -z "$scratch" ]] || arguments+=("$scratch")
exec "$julia_bin" --startup-file=no --project="$root" "$root/tools/openmolcas4noci.jl" "${arguments[@]}"
