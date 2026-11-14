#!/usr/bin/env bash
# File: scripts/preparation/initialize_case.sh
# Purpose: Create the standard case directory layout and initial inputs.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

(( $# == 1 )) || { echo "usage: initialize_case.sh NEW_CASE_DIRECTORY" >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
target=$1
[[ ! -e "$target" ]] || { echo "Refusing to overwrite existing path: $target" >&2; exit 2; }
cp -R -- "$root/examples/ahpc_noci_standard" "$target"
echo "Created AHPC-NOCI case: $target"
