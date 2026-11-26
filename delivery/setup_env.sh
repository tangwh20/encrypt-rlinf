#!/bin/bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash install.sh embodied --model MODEL --env ENV [options]
       bash install.sh --prepared-runtime DIR
Example: bash install.sh embodied --model openvla --env frankasim
Runs the bundled upstream installer in a local CPython 3.11 environment. This downloads the selected libraries and assets.
Requires Linux x86-64, Bash, flock, and standard GNU utilities. An online upstream install also needs its model and simulator system dependencies.
Use --use-mirror, --no-flash-attn or --no-apex to pass the corresponding option to the bundled installer.
Use --task frankasim_ppo_mlp to install the pinned FrankaSim PPO runtime without the OpenVLA policy-training stack.
Use --prepared-runtime DIR alone to import a standalone CPython 3.11 prefix for any task.
With embodied, --prepared-runtime DIR also checks the selected simulator.
Both forms keep the packages provided in DIR.
HELP
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
model= env_name= prepared= task= embodied=false
options=()
if [[ ${1:-} == --prepared-runtime ]]; then
  if [[ $# -ne 2 || -z $2 || $2 == --* ]]; then usage >&2; exit 2; fi
  prepared=$2
  model=generic
  env_name=runtime
  shift 2
elif [[ ${1:-} == embodied ]]; then
  embodied=true
  shift
else
  usage >&2
  exit 2
fi
while (($#)); do
  case $1 in
    --model|--env|--prepared-runtime|--task)
      if [[ $# -lt 2 || -z $2 || $2 == --* ]]; then usage >&2; exit 2; fi
      case $1 in --model) model=$2;; --env) env_name=$2;; --prepared-runtime) prepared=$2;; --task) task=$2;; esac
      shift 2;;
    --use-mirror|--no-flash-attn|--no-apex|--no-natten) options+=("$1"); shift;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2;;
  esac
done
if [[ ! $model =~ ^[A-Za-z0-9_-]+$ || ! $env_name =~ ^[A-Za-z0-9_-]+$ ]]; then usage >&2; exit 2; fi
if [[ -n $task && ( $model/$env_name/$task != openvla/frankasim/frankasim_ppo_mlp || -n $prepared || ${#options[@]} -gt 0 ) ]]; then
  printf 'The pinned task recipe is embodied --model openvla --env frankasim --task frankasim_ppo_mlp.\n' >&2; exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
setup_dir=$("$root/run" --runtime-dir)/setup
# Check embodied names against the inventory from the same checkout as the executable.
if [[ $embodied == true ]]; then
  for key in "SUPPORTED_MODELS:$model" "SUPPORTED_ENVS:$env_name"; do
    names=$(sed -n "s/^${key%%:*}=(\(.*\))$/\1/p" "$setup_dir/requirements/install.sh")
    if [[ " $names " != *" \"${key#*:}\" "* ]]; then printf 'Unknown %s\n' "$key" >&2; exit 2; fi
  done
fi
if [[ -n $prepared ]]; then
  prepared=$(cd -- "$prepared" && pwd -P)
  if [[ ! -x $prepared/bin/python || -e $prepared/pyvenv.cfg || ${#options[@]} -gt 0 ]]; then
    printf 'Expected a standalone prepared prefix, with no upstream install flags.\n' >&2; exit 2
  fi
fi
envs="$root/.envs"
mkdir -p "$envs"
exec 9>"$envs/.install.lock"
flock 9
if [[ -e $root/.venv && ! -L $root/.venv ]]; then printf 'Existing .venv is not managed by this installer.\n' >&2; exit 1; fi
kind=upstream
[[ -z $task ]] || kind=$task
[[ -z $prepared ]] || kind=prepared
data_recipe=
# Use the embedded payload digest, so relocation keeps the same profile and a
# different executable cannot silently reuse the old dependency recipe.
payload_dir=${setup_dir%/setup}
identity_parts=("$kind" "${payload_dir##*/}" "${options[@]}")
if [[ -n $task ]]; then
  for path in "$setup_dir/recipes/$task/"*; do
    identity_parts+=("$(sha256sum < "$path" | cut -d' ' -f1)")
  done
fi
if [[ $kind == upstream && $model/$env_name == openvla/frankasim ]]; then
  data_recipe="$setup_dir/recipes/openvla-frankasim-data.txt"
  identity_parts+=("$(sha256sum < "$data_recipe" | cut -d' ' -f1)")
fi
identity=$(printf '%s\0' "${identity_parts[@]}" | sha256sum | cut -c1-12)
profile="$envs/$model-$env_name-$identity"
if [[ -n $prepared ]]; then
  # Each explicit import is a new snapshot, including when the source path was updated.
  profile=$(mktemp -d "$envs/$model-$env_name-prepared-XXXXXXXX")
fi
# Keep an incomplete attempt resumable and keep the active environment usable on failure.
mkdir -p "$profile/home" "$profile/work" "$envs/cache"
venv="$profile/runtime"
if [[ ! -f $profile/ready || ! -x $venv/bin/python ]]; then
  if [[ -n $prepared ]]; then
    case "$venv/" in "$prepared/"*) printf 'Cannot copy an environment into itself.\n' >&2; exit 2;; esac
    mkdir -p "$venv"
    cp -a --reflink=auto -- "$prepared/." "$venv/"
  else
    seed="$setup_dir/seed"
    tools="$profile/tools"
    # Keep downloads, caches and any dependency checkouts inside the distribution.
    export HOME="$profile/home" XDG_CACHE_HOME="$envs/cache" UV_CACHE_DIR="$envs/cache/uv"
    export XDG_CONFIG_HOME="$HOME/.config" XDG_DATA_HOME="$HOME/.local/share"
    export HF_HOME="$envs/cache/huggingface" PIP_CACHE_DIR="$envs/cache/pip"
    export TMPDIR="$profile/work" UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy PYTHONNOUSERSITE=1
    unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX UV_PROJECT_ENVIRONMENT UV_PYTHON
    unset UV_CONFIG_FILE PIP_TARGET PIP_PREFIX PIP_USER HF_HUB_CACHE HF_DATASETS_CACHE
    unset SERL_PATH LIBERO_REPO_PATH ROBOTWIN_PATH LD_PRELOAD LD_AUDIT
    export PATH="$tools/bin:$seed/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
    export LD_LIBRARY_PATH="$seed/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib"
    if [[ ! -x $tools/bin/uv ]]; then
      "$seed/bin/python" -Im venv --copies "$tools"
      "$tools/bin/python" -Im pip install --disable-pip-version-check 'uv==0.9.18'
    fi
    if [[ -n $task ]]; then
      mkdir -p "$profile/portable"
      cp -a -- "$setup_dir/recipes/$task/." "$profile/portable/"
      # build_runtime.sh publishes this prefix only after its own checks succeed.
      if [[ ! -d $venv ]]; then
        UV="$tools/bin/uv" TRAIN_UV_CACHE_DIR="$UV_CACHE_DIR" TRAIN_CPYTHON_PREFIX="$seed" bash "$profile/portable/build_runtime.sh"
      fi
    elif [[ -f $venv/pyvenv.cfg || ! -x $venv/bin/python ]]; then
      if [[ ! -f $venv/bin/activate || ! -x $venv/bin/python ]]; then
        uv venv --relocatable --python "$seed/bin/python" "$venv"
      fi
      recipe="$profile/recipe"
      mkdir -p "$recipe"
      cp -a -- "$setup_dir/requirements" "$setup_dir/pyproject.toml" "$setup_dir/LICENSE" "$setup_dir/README.md" "$recipe/"
      # Some upstream shell paths are unquoted when invoked from a directory with spaces.
      sed -i 's|-r \$SCRIPT_DIR/embodied/envs/common.txt|-r "$SCRIPT_DIR/embodied/envs/common.txt"|g; s|bash \$SCRIPT_DIR/embodied/download_assets.sh|bash "$SCRIPT_DIR/embodied/download_assets.sh"|g' "$recipe/requirements/install.sh"
      (cd -- "$recipe" && bash requirements/install.sh embodied --model "$model" --env "$env_name" --venv "$venv" --python 3.11.14 --no-root "${options[@]}")
      if [[ -n $data_recipe ]]; then
        printf 'Installing the data packages compatible with FrankaSim NumPy 1.26.4.\n'
        uv pip install --python "$venv/bin/python" -r "$data_recipe"
        "$venv/bin/python" -IB -c 'import cv2, pandas, pyarrow'
      fi
      "$seed/bin/python" -IB "$setup_dir/finish_venv.py" "$venv" "$seed"
    else
      # A prior run finished converting the venv before the final import check.
      "$venv/bin/python" -IB "$setup_dir/finalize_runtime.py"
    fi
  fi
  LD_LIBRARY_PATH="$venv/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib" \
    "$venv/bin/python" -IB "$setup_dir/check_environment.py" "$venv" "$kind" "$model" "$env_name"
  printf '%s\n' "$kind $model $env_name" > "$profile/ready"
fi
pending=$(mktemp -d "$envs/.select-XXXXXXXX")
trap 'rmdir -- "$pending"' EXIT
ln -s -- ".envs/${profile##*/}/runtime" "$pending/active"
mv -Tf -- "$pending/active" "$root/.venv"
printf 'Active %s environment: %s\n' "$kind" "$profile/runtime"
