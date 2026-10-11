#!/usr/bin/env bash
set -euo pipefail

(( $# == 2 )) || { echo "usage: make_factor_bundle.sh SEWARD_SOURCE.yaml OUTPUT.jld2" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
julia_bin=${JULIA_BIN:-$(command -v julia || true)}
[[ -n "$julia_bin" ]] || { echo "Julia was not found; set JULIA_BIN." >&2; exit 2; }

"$julia_bin" --startup-file=no --project="$root" "$root/tools/seward_to_factor_bundle.jl" "$1" "$2"

