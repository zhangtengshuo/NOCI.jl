#!/usr/bin/env bash
set -euo pipefail
script=$(readlink -f -- "${BASH_SOURCE[0]}")
root=$(cd -- "$(dirname -- "$script")/.." && pwd -P)
version=$(readlink -- "$root/noci")
[[ $version =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ && -f "$root/installation/records/$version.toml" ]] || { echo 'Select a completed NOCI installation first.' >&2; exit 2; }
source "$root/$version/bin/noci-environment.sh"
export JULIA_LOAD_PATH='@:@stdlib'
exec "$NOCI_ROOT/bin/noci-prepare" "$@"
