#!/usr/bin/env bash
set -euo pipefail

portable_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(dirname -- "$portable_dir")
runtime="$root/runtime"
wheelhouse=${RLINF_WHEELHOUSE:-}
uv=${UV:-uv}
serl_revision=d5283dc984de78d223d29dc3b98016be71d50ffc

if [[ -e "$runtime" || -L "$runtime" ]]; then
  printf '%s\n' "Runtime already exists at $runtime" >&2
  exit 1
fi

stage=$(mktemp -d "$root/.runtime-build.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
mkdir -p "$stage/tmp"
export TMPDIR="$stage/tmp"
export UV_CACHE_DIR="${TRAIN_UV_CACHE_DIR:-${RLINF_UV_CACHE_DIR:-$stage/cache}}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH

fetch() {
  local path=$1 sha256=$2 url=$3
  if [[ ! -f "$path" ]]; then
    curl -q -fsSL --connect-timeout 15 --retry 3 --retry-all-errors "$url" -o "$path"
  fi
  printf '%s  %s\n' "$sha256" "$path" | sha256sum --check --status
}

python_prefix=${TRAIN_CPYTHON_PREFIX:-${RLINF_CPYTHON_PREFIX:-}}
if [[ -n $python_prefix ]]; then
  python_seed=$(cd -- "$python_prefix" && pwd -P)
else
  cpython=${RLINF_CPYTHON_ARCHIVE:-"$stage/python.tar.gz"}
  fetch "$cpython" 1fb6a89daa750f7eff00eb65d901c3c236713812eb9c8c7e07ecdfcd78203c8e \
    'https://github.com/astral-sh/python-build-standalone/releases/download/20260211/cpython-3.11.14%2B20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz'
  tar -xzf "$cpython" -C "$stage"
  python_seed="$stage/python"
fi

mkdir -p "$stage/runtime"
cp -a -- "$python_seed/." "$stage/runtime/"
python="$stage/runtime/bin/python"
"$python" -IB -c 'import sys; assert sys.version_info[:3] == (3, 11, 14)'
if ! "$python" -IB -m pip --version >/dev/null 2>&1; then
  rm -f -- "$stage/runtime/lib/python3.11/EXTERNALLY-MANAGED"
  "$python" -IB -m ensurepip --default-pip
fi
if command -v "$uv" >/dev/null; then
  install_args=("$uv" --no-config pip install --system --break-system-packages --python "$python"
                --link-mode copy --no-deps)
  index_option=--default-index
else
  install_args=(env PIP_CONFIG_FILE=/dev/null "$python" -IB -m pip --isolated install
                --disable-pip-version-check --no-cache-dir --break-system-packages --no-deps)
  index_option=--index-url
fi
if [[ -n "$wheelhouse" ]]; then
  "${install_args[@]}" --only-binary :all: --no-index --find-links "$wheelhouse" \
    -r "$portable_dir/runtime-requirements.txt"
else
  sed -E '/^(torch|torchvision)==/d' "$portable_dir/runtime-requirements.txt" \
    > "$stage/pypi.txt"
  sed -nE '/^(torch|torchvision)==/p' "$portable_dir/runtime-requirements.txt" \
    > "$stage/pytorch.txt"
  "${install_args[@]}" "$index_option" https://pypi.org/simple -r "$stage/pypi.txt"
  "${install_args[@]}" "$index_option" https://download.pytorch.org/whl/cu128 \
    -r "$stage/pytorch.txt"
fi

install_library() {
  local deb=$1 package=$2 library_glob=$3
  dpkg-deb -x "$deb" "$stage/$package"
  local lib_dir="$stage/$package/usr/lib/x86_64-linux-gnu"
  local libraries=()
  mapfile -d '' -t libraries < <(
    find "$lib_dir" -maxdepth 1 -name "$library_glob" -print0
  )
  if (( ${#libraries[@]} == 0 )); then
    printf '%s\n' "Missing $library_glob in $package" >&2
    exit 1
  fi
  cp -a -- "${libraries[@]}" "$stage/runtime/lib/"
  install -Dm644 "$stage/$package/usr/share/doc/$package/copyright" \
    "$stage/runtime/share/doc/$package/copyright"
}

vulkan_deb=${RLINF_VULKAN_DEB:-"$stage/libvulkan1.deb"}
egl_deb=${RLINF_EGL_DEB:-"$stage/libegl1.deb"}
glvnd_deb=${RLINF_GLVND_DEB:-"$stage/libglvnd0.deb"}
opengl_deb=${RLINF_OPENGL_DEB:-"$stage/libopengl0.deb"}
fetch "$vulkan_deb" 192adcff489996b3398e7e7c0012b98e9586b46fe9a9eb13fb02c0feba88548b \
  'https://archive.ubuntu.com/ubuntu/pool/main/v/vulkan-loader/libvulkan1_1.3.204.1-2_amd64.deb'
fetch "$egl_deb" 4a35c0e925e15a076e7ce11d7c76f8ecb16615fc347c2c09afa4c205a7ca8ec4 \
  'https://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libegl1_1.4.0-1_amd64.deb'
fetch "$glvnd_deb" 6956d8be93b160b439a043141a061adfd54561d877423a956ace84d2fb3bfb4e \
  'https://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libglvnd0_1.4.0-1_amd64.deb'
fetch "$opengl_deb" 486473b82c475d8137d2a061adb22fbcd1d681936e9d85a3ca7f8b4b83e2ccde \
  'https://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libopengl0_1.4.0-1_amd64.deb'
install_library "$vulkan_deb" libvulkan1 'libvulkan.so.1*'
install_library "$egl_deb" libegl1 'libEGL.so.1*'
install_library "$glvnd_deb" libglvnd0 'libGLdispatch.so.0*'
install_library "$opengl_deb" libopengl0 'libOpenGL.so.0*'

serl=${RLINF_FRANKASIM_ARCHIVE:-"$stage/serl.tar.gz"}
fetch "$serl" 4b72d15cec86fd3a5f2f39495212a3428e6e50f2b14195178d270cf52c820491 \
  "https://codeload.github.com/RLinf/serl/tar.gz/$serl_revision"
serl_dir="serl-$serl_revision"
tar -xzf "$serl" -C "$stage" "$serl_dir/franka_sim/franka_sim" "$serl_dir/LICENSE"
site_packages="$stage/runtime/lib/python3.11/site-packages"
cp -a -- "$stage/$serl_dir/franka_sim/franka_sim" "$site_packages/"
install -Dm644 "$stage/$serl_dir/LICENSE" "$stage/runtime/share/doc/franka_sim/LICENSE"

if [[ ! -x $stage/runtime/toolchain/bin/gcc ]]; then
  if ! "$python" -IB "$portable_dir/build_toolchain.py" "$stage/runtime/toolchain" >"$stage/toolchain.log" 2>&1; then
    # Torch and Triton can use the system compiler on Linux hosts without the
    # Ubuntu GCC 11 packages used to assemble the relocatable compiler.
    if ! command -v gcc >/dev/null || ! printf 'int main() { return 0; }\n' | g++ -x c++ - -o "$stage/compiler-test"; then
      cat "$stage/toolchain.log" >&2
      printf '%s\n' 'Install a working system GCC/G++ toolchain and its development libraries, or provide a standalone prefix with a relocatable compiler.' >&2
      exit 1
    fi
    printf '%s\n' 'Using the system GCC/G++ toolchain for runtime compilation.'
  fi
fi
"$python" -IB "$portable_dir/finalize_runtime.py"
"$python" -IB -m pip check
mv -T -- "$stage/runtime" "$runtime"
