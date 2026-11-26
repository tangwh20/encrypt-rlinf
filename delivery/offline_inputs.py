"""Prepare project-local inputs for the self-contained release build."""

import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from offline_runtime import PHYSX_LIBRARY, validate_offline_prefix


PHYSX_SHA256 = "4c582a16509a71faf5592fe9708586dfcc7ab61ae932eabc1ddd81f290818706"
PHYSX_SIZE = 236705440
PHYSX_URL = (
    "https://github.com/sapien-sim/physx-precompiled/releases/download/"
    f"{PHYSX_LIBRARY.parent.name}/linux-so.zip"
)
BUFFER_SIZE = 1 << 20


def check_physx(path):
    if not path.is_file() or path.resolve() != path:
        raise ValueError(f"Missing pinned SAPIEN PhysX GPU library: {path}")
    if path.stat().st_size != PHYSX_SIZE:
        raise ValueError(f"Invalid size for the pinned SAPIEN PhysX GPU library: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if (header := stream.read(4)) != b"\x7fELF":
            raise ValueError(f"Expected an ELF for the pinned SAPIEN PhysX GPU library: {path}")
        digest.update(header)
        while chunk := stream.read(BUFFER_SIZE):
            digest.update(chunk)
    if digest.hexdigest() != PHYSX_SHA256:
        raise ValueError(f"SAPIEN PhysX GPU library SHA-256 mismatch: {path}")


def copy_physx(source, destination):
    if not source.is_file():
        raise ValueError(f"Expected a local PhysX shared object or ZIP archive: {source}")
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            entries = [item for item in archive.infolist()
                       if item.filename.removeprefix("./") == PHYSX_LIBRARY.name]
            if len(entries) != 1 or entries[0].file_size != PHYSX_SIZE:
                raise ValueError(f"Missing unique pinned PhysX GPU library in {source}")
            with archive.open(entries[0]) as incoming, destination.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, BUFFER_SIZE)
    else:
        if source.stat().st_size != PHYSX_SIZE:
            raise ValueError(f"Expected a local PhysX shared object or ZIP archive: {source}")
        shutil.copyfile(source, destination)
    check_physx(destination)


def download_physx(destination):
    if shutil.which("curl") is None:
        raise ValueError("Install curl or pass a local PhysX asset with --physx-asset.")
    command = [
        "curl", "-q", "-fSL", "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", "15", "--max-time", "600", "--retry", "3", "--retry-all-errors",
        "--output", str(destination), PHYSX_URL,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        raise ValueError(f"Could not download the pinned PhysX GPU asset from {PHYSX_URL}. "
                         "Pass a local copy with --physx-asset.") from None


def cache_physx(directory, destination, source):
    if destination.parent.resolve() != destination.parent:
        raise ValueError(f"Expected a project-local PhysX cache: {destination.parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".physx-", dir=directory) as tmp:
        scratch = Path(tmp)
        if source is None:
            source = scratch / "linux-so.zip"
            print(f"Downloading the pinned SAPIEN PhysX GPU asset for {destination}...", flush=True)
            download_physx(source)
        pending = scratch / PHYSX_LIBRARY.name
        copy_physx(source, pending)
        pending.chmod(0o644)
        pending.replace(destination)


def prepare_runtime(project, prefix):
    if prefix is not None:
        return validate_offline_prefix(prefix)
    preset = project / "runtime"
    if preset.is_symlink() and not preset.resolve().is_relative_to(project):
        raise ValueError(f"The project runtime points outside {project}. "
                         "Pass --offline-runtime explicitly to use that prefix.")
    if not preset.exists() and not preset.is_symlink():
        recipe = project / "portable/build_runtime.sh"
        print(f"Preparing a task runtime at {preset}...", flush=True)
        try:
            subprocess.run(["bash", str(recipe)], check=True, cwd=project)
        except subprocess.CalledProcessError:
            raise ValueError(f"Could not prepare {preset} with {recipe}. "
                             "Check the builder output or pass --offline-runtime.") from None
    return validate_offline_prefix(preset)


def prepare_offline_inputs(project, *, prefix=None, asset_home=None, physx_asset=None):
    if asset_home is not None and physx_asset is not None:
        raise ValueError("Choose either an existing --offline-home or a local --physx-asset.")
    project = project.resolve()
    if physx_asset is not None:
        physx_asset = physx_asset.resolve()
        if not physx_asset.is_file():
            raise ValueError(f"Expected a local PhysX shared object or ZIP archive: {physx_asset}")
    if asset_home is not None:
        asset_home = asset_home.resolve()
        check_physx(asset_home / PHYSX_LIBRARY)
        if prefix is not None:
            return prepare_runtime(project, prefix), asset_home

    directory = project / ".offline-build"
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink():
        raise ValueError(f"Expected a project-local offline build directory: {directory}")
    lock = os.open(directory / "prepare.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if asset_home is not None:
            return prepare_runtime(project, prefix), asset_home
        asset_home = directory / "home"
        cached = asset_home / PHYSX_LIBRARY
        if physx_asset is not None:
            cache_physx(directory, cached, physx_asset)
        runtime = prepare_runtime(project, prefix)
        if physx_asset is None:
            if cached.exists() or cached.is_symlink():
                try:
                    check_physx(cached)
                except ValueError as exc:
                    raise ValueError(f"{exc}. Replace the cache with --physx-asset.") from None
            else:
                sibling = runtime.parent / "home" / PHYSX_LIBRARY
                if sibling.exists() and sibling.resolve().is_relative_to(project):
                    physx_asset = sibling.resolve()
                cache_physx(directory, cached, physx_asset)
        return runtime, asset_home
    finally:
        os.close(lock)
