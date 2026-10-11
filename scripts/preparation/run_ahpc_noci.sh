#!/usr/bin/env bash
set -euo pipefail

(( $# >= 3 && $# <= 4 )) || { echo "usage: run_ahpc_noci.sh INPUT.yaml OUTPUT_DIRECTORY OUTPUT_ROOT [GEOMETRY.xyz]" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
exec "${JULIA_BIN:-julia}" --startup-file=no --project="${NOCI_JULIA_PROJECT:-$root}" "$root/bin/noci" run "$@"

