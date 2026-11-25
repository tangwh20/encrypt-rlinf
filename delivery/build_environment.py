"""Prepare the standalone Python seed and the build-only Cython environment."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time


CPYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    "20260211/cpython-3.11.14%2B20260211-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
)
CPYTHON_ASSET = "https://api.github.com/repos/astral-sh/python-build-standalone/releases/assets/354545489"
CPYTHON_SHA256 = "1fb6a89daa750f7eff00eb65d901c3c236713812eb9c8c7e07ecdfcd78203c8e"
CYTHON_VERSION = "3.2.8"
UBUNTU_POOL = "https://archive.ubuntu.com/ubuntu/pool/main"
GRAPHICS_PACKAGES = (
    ("RLINF_VULKAN_DEB", "v/vulkan-loader/libvulkan1_1.3.204.1-2_amd64.deb",
     "192adcff489996b3398e7e7c0012b98e9586b46fe9a9eb13fb02c0feba88548b", "libvulkan.so.1*"),
    ("RLINF_EGL_DEB", "libg/libglvnd/libegl1_1.4.0-1_amd64.deb",
     "4a35c0e925e15a076e7ce11d7c76f8ecb16615fc347c2c09afa4c205a7ca8ec4", "libEGL.so.1*"),
    ("RLINF_GLVND_DEB", "libg/libglvnd/libglvnd0_1.4.0-1_amd64.deb",
     "6956d8be93b160b439a043141a061adfd54561d877423a956ace84d2fb3bfb4e", "libGLdispatch.so.0*"),
    ("RLINF_OPENGL_DEB", "libg/libglvnd/libopengl0_1.4.0-1_amd64.deb",
     "486473b82c475d8137d2a061adb22fbcd1d681936e9d85a3ca7f8b4b83e2ccde", "libOpenGL.so.0*"),
)


def package_file(env_name, path, url, expected_hash, fallback=None):
    local_path = os.environ.get(env_name)
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"{env_name} does not point to a file: {path}")
    else:
        command = ["curl", "-fsSL", "--connect-timeout", "15", "--retry", "2",
                   "--output", str(path)]
        try:
            subprocess.run([*command, url], check=True)
        except subprocess.CalledProcessError:
            if fallback is None:
                raise ValueError(f"Download failed: {url}. Supply {env_name} with a local copy.") from None
            try:
                # Refresh the temporary redirect if an HTTP cache retained an expired one.
                subprocess.run([*command, "-H", "Accept: application/octet-stream",
                                f"{fallback}?download={time.time_ns()}"], check=True)
            except subprocess.CalledProcessError:
                raise ValueError(f"Download failed: {url}. Supply {env_name} with a local copy.") from None

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path} ({env_name})")
    return path


def install_graphics(work, seed):
    extracted = work / "graphics"
    for variable, name, digest, pattern in GRAPHICS_PACKAGES:
        package = package_file(variable, work / Path(name).name, f"{UBUNTU_POOL}/{name}", digest)
        subprocess.run(["dpkg-deb", "-x", str(package), str(extracted)], check=True)
        lib_dir = extracted / "usr/lib/x86_64-linux-gnu"
        libraries = sorted(lib_dir.glob(pattern))
        if not libraries:
            raise ValueError(f"No {pattern} files found in {package}")
        for library in libraries:
            shutil.copy2(library, seed / "lib" / library.name, follow_symlinks=False)


def prepare(work, portable, env):
    archive = package_file("RLINF_CPYTHON_ARCHIVE", work / "python.tar.gz", CPYTHON_URL,
                           CPYTHON_SHA256, fallback=CPYTHON_ASSET)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(work)], check=True)
    seed = work / "python"
    python = seed / "bin/python"
    tools = work / "build-tools"
    subprocess.run([str(python), "-IB", "-m", "venv", str(tools)], env=env, check=True)
    tool_python = tools / "bin/python"
    pip = [
        str(tool_python), "-IB", "-m", "pip", "install", "--disable-pip-version-check",
        "--no-cache-dir", "--only-binary", ":all:", "--no-deps", f"Cython=={CYTHON_VERSION}",
    ]
    if wheelhouse := os.environ.get("RLINF_WHEELHOUSE"):
        wheels = Path(wheelhouse).expanduser().resolve()
        if not wheels.is_dir():
            raise ValueError(f"RLINF_WHEELHOUSE does not point to a directory: {wheels}")
        pip.extend(["--no-index", "--find-links", str(wheels)])
    try:
        subprocess.run(pip, env=env, check=True)
    except subprocess.CalledProcessError:
        raise ValueError(f"Could not install the Cython {CYTHON_VERSION} wheel. "
                         "Check the pip index or supply RLINF_WHEELHOUSE.") from None

    install_graphics(work, seed)
    subprocess.run([str(python), "-IB", str(portable / "finalize_runtime.py")],
                   env=env, check=True)
    return seed, tool_python
