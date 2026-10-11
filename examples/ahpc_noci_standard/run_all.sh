#!/usr/bin/env bash
set -euo pipefail
case_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
manifest=${1:-ethene_dimer.yaml}
if [[ "$manifest" != /* ]]; then manifest="$case_root/00_system_define/$manifest"; fi
command -v noci-prepare >/dev/null || { echo "Set PATH to the selected NOCI installation bin directory." >&2; exit 2; }
# This command runs real calculations. Use noci-prepare plan for a read-only review.
noci-prepare check "$manifest"
for stage in states placement commonorb integrals run; do
    noci-prepare stage "$stage" "$manifest" all
done
