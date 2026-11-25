#!/bin/bash
set -euo pipefail
project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if ! command -v python3 >/dev/null || ! python3 -IB -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  printf '%s\n' 'Building RLinf requires a system Python 3.10 or newer (standard library only).' >&2
  exit 2
fi
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
exec python3 -EB "$project/delivery/release_build.py" "$@"
