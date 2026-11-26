#!/bin/bash
set -euo pipefail

usage() {
  printf '%s\n' 'Usage: bash install.sh [--offline]' \
    'Installs the embedded FrankaSim and ManiSkill task runtime locally. Requires Bash, flock, tar, gzip and standard GNU utilities.'
}

case ${1:-} in
  --help|-h) usage; exit 0;;
  ""|--offline) if (( $# > 1 )); then usage >&2; exit 2; fi;;
  *) usage >&2; exit 2;;
esac

umask 077
shopt -s nullglob
export PATH=/usr/local/bin:/usr/bin:/bin
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher="$root/run"
if ! embedded_sha=$("$launcher" --offline-runtime-sha256) || [[ ! $embedded_sha =~ ^[a-f0-9]{64}$ ]]; then
  printf 'Could not read the embedded offline runtime checksum.\n' >&2; exit 1
fi
setup_dir=$("$launcher" --runtime-dir)/setup
envs="$root/.envs"
mkdir -p -- "$envs"
exec 9>"$envs/.install.lock"
flock 9
if [[ -e $root/.venv && ! -L $root/.venv ]]; then
  printf 'Existing .venv is not managed by this installer.\n' >&2; exit 1
fi

install_runtime() (
  stage=$(mktemp -d "$envs/.offline-XXXXXXXX")
  trap 'rm -rf -- "$stage"' EXIT
  mkdir -- "$stage/profile"
  mkfifo -- "$stage/bytes"
  sha256sum < "$stage/bytes" > "$stage/digest" &
  digest_pid=$!
  # Hash the actual compressed stream while GNU tar extracts into a private directory.
  if ! "$launcher" --offline-runtime-archive |
      tee -- "$stage/bytes" |
      env -i PATH="$PATH" tar --extract --gzip --file=- --directory="$stage/profile" \
        --no-same-owner --no-same-permissions; then
    wait "$digest_pid" || true
    printf 'Could not extract the embedded offline runtime.\n' >&2; exit 1
  fi
  wait "$digest_pid"
  read -r actual ignored < "$stage/digest"
  if [[ $actual != "$embedded_sha" ]]; then
    printf 'Embedded offline runtime checksum does not match.\n' >&2; exit 1
  fi
  if [[ -e $stage/profile/assets || -L $stage/profile/assets ]]; then
    if [[ ! -d $stage/profile/assets || -L $stage/profile/assets ]]; then
      printf 'Embedded offline assets are not a directory.\n' >&2; exit 1
    fi
    touch -- "$stage/profile/ready-assets"
  fi
  runtime="$stage/profile/runtime"
  home="$stage/profile/home"
  mkdir -p -- "$stage/profile/work" "$home/.cache"
  env -i PATH="$PATH" HOME="$home" XDG_CACHE_HOME="$home/.cache" \
    MS_ASSET_DIR="$home/.maniskill" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PIP_NO_INDEX=1 UV_OFFLINE=1 WANDB_MODE=offline NO_ALBUMENTATIONS_UPDATE=1 \
    LD_LIBRARY_PATH="$runtime/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib" \
    "$runtime/bin/python" -IB -c '
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

setup = sys.argv.pop(1)
sys.path.insert(0, setup)
spec = importlib.util.find_spec("_train_check_environment")
sys.path.pop(0)
if (spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)
        or Path(spec.origin).parent != Path(setup)):
    raise SystemExit("Missing compiled offline runtime verifier.")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
' "$setup_dir" "$runtime" offline openvla frankasim
  printf '%s\n' "$embedded_sha" > "$stage/profile/ready"
  mv -T -- "$stage/profile" "$profile"
)

is_ready() {
  local assets=("$1"/home/.sapien/physx/*/libPhysXGpu_64.so)
  [[ ( ! -f $1/ready-assets || ( -d $1/assets && ! -L $1/assets ) ) &&
    -f $1/ready && -x $1/runtime/bin/python &&
    $(<"$1/ready") == "$embedded_sha" &&
    ${#assets[@]} -eq 1 && -s ${assets[0]} ]]
}

# Select an installed profile by the archive digest, including a prior repair.
profile="$envs/offline-$embedded_sha"
if ! is_ready "$profile"; then
  for candidate in "$profile"-*; do
    if is_ready "$candidate"; then profile=$candidate; break; fi
  done
fi
if ! is_ready "$profile"; then
  if [[ -e $profile || -L $profile ]]; then
    profile=$(mktemp -d "$profile-XXXXXXXX")
    rmdir -- "$profile"
  fi
  install_runtime
fi
pending=$(mktemp -d "$envs/.select-XXXXXXXX")
trap 'rm -rf -- "$pending"' EXIT
ln -s -- ".envs/${profile##*/}/runtime" "$pending/active"
mv -Tf -- "$pending/active" "$root/.venv"
printf 'Active offline environment: %s\n' "$profile/runtime"
