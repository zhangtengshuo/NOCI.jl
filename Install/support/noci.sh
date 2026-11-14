#!/usr/bin/env bash
# File: Install/support/noci.sh
# Purpose: Run the NOCI command from the managed release installation.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
script=$(readlink -f -- "${BASH_SOURCE[0]}")
root=$(cd -- "$(dirname -- "$script")/.." && pwd -P)
if [[ ${1:-} == --list ]]; then
    for record in "$root"/installation/records/v*.toml; do
        [[ -f "$record" ]] && basename "$record" .toml
    done
    exit 0
fi
if [[ ${1:-} == --use ]]; then
    [[ $# == 2 && $2 =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'Usage: noci --use v26.10.0' >&2; exit 2; }
    [[ -f "$root/installation/records/$2.toml" && -f "$root/$2/bin/noci-environment.sh" ]] || { echo 'Version is not completely installed.' >&2; exit 2; }
    exec 9>"$root/installation/install.lock"
    flock -n 9 || { echo 'Installation/version selection is busy.' >&2; exit 2; }
    [[ ! -e "$root/noci" || -L "$root/noci" ]] || { echo 'Refusing to replace a real noci directory/file.' >&2; exit 2; }
    temporary=$(mktemp -d "$root/.noci-selection.XXXXXX")
    trap 'rm -rf -- "$temporary"' EXIT
    ln -s -- "$2" "$temporary/noci"
    mv -Tf -- "$temporary/noci" "$root/noci"
    exit 0
fi
version=$(readlink -- "$root/noci")
[[ $version =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ && -f "$root/installation/records/$version.toml" ]] || { echo 'Select a completed installation with noci --use v26.10.0.' >&2; exit 2; }
source "$root/$version/bin/noci-environment.sh"
export JULIA_LOAD_PATH='@:@stdlib'
exec "$JULIA_BIN" --startup-file=no --project="$NOCI_JULIA_PROJECT" "$NOCI_ROOT/bin/noci" "$@"
