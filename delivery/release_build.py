"""Compile RLinf into a relocatable ELF and ship a separate task environment installer."""

import argparse
import concurrent.futures
from collections import deque
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zlib

from build_environment import GRAPHICS_PACKAGES, prepare
from launcher import public_scripts
from source_inventory import inventory

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
HEADER = struct.Struct("<cHIQ")
BLOCK = 2 << 20
TASK_DIRECTORY = Path("examples/embodiment/config")
PREFERRED_EXAMPLES = ("frankasim_ppo_mlp.yaml", "maniskill_ppo_openvlaoft.yaml")
EXAMPLE_COMMENTS = (
    b"# The defaults list selects bundled configuration groups for the environment and model.\n"
    b"# Set runner.max_steps to a positive number to limit the number of training steps.\n"
    b"# Preview this configuration with ./rlinf-train run_embodiment example.yaml --cfg job\n"
)


def members(directory, prefix, exclude_dirs=()):
    """Record directories before their children; symlinks must stay inside their tree."""
    yield b"D", prefix, directory
    for parent, dirs, files in os.walk(directory):
        dirs[:] = sorted(name for name in dirs if name not in exclude_dirs)
        for name in sorted([*dirs, *files]):
            path = Path(parent) / name
            rel = prefix / path.relative_to(directory)
            if path.is_symlink():
                target = os.readlink(path)
                if os.path.isabs(target) or not path.resolve(strict=True).is_relative_to(directory):
                    raise ValueError(f"Nonportable symlink: {rel} -> {target}")
                yield b"L", rel, path
            elif path.is_dir():
                yield b"D", rel, path
            elif path.is_file():
                yield b"F", rel, path
            else:
                raise ValueError(f"Unsupported runtime member: {rel}")


def record_bytes(records):
    for kind, rel, path in records:
        name = os.fsencode(rel)
        mode = stat.S_IMODE(path.lstat().st_mode)
        target = os.fsencode(os.readlink(path)) if kind == b"L" else b""
        size = path.stat().st_size if kind == b"F" else len(target)
        yield HEADER.pack(kind, mode, len(name), size) + name
        if kind == b"L":
            yield target
        if kind == b"F":
            with path.open("rb") as stream:
                while chunk := stream.read(BLOCK):
                    yield chunk
    yield b"\0"


def blocks(chunks):
    data = bytearray()
    for chunk in chunks:
        data.extend(chunk)
        if len(data) >= BLOCK:
            yield bytes(data)
            data.clear()
    if data:
        yield bytes(data)


def deflate(block):
    compressor = zlib.compressobj(level=1, wbits=31)
    return compressor.compress(block) + compressor.flush()


def write_payload(exe, records, jobs):
    digest = hashlib.sha256()
    with exe.open("ab") as stream, concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        offset = stream.tell()
        pending = deque()

        def flush_one():
            data = pending.popleft().result()
            stream.write(data)
            digest.update(data)

        for block in blocks(record_bytes(records)):
            pending.append(pool.submit(deflate, block))
            if len(pending) >= 2 * jobs:
                flush_one()
        while pending:
            flush_one()
        stream.write(struct.pack("<8sQ32s", b"RLFRUN01", offset, digest.digest()))


def installer_recipes(portable):
    recipes = Path("setup/recipes")
    task_recipe = recipes / "frankasim_ppo_mlp"
    yield b"D", recipes, portable
    yield b"D", task_recipe, portable
    name = "openvla-frankasim-data.txt"
    yield b"F", recipes / name, portable / name
    for name in ("build_runtime.sh", "runtime-requirements.txt", "build_toolchain.py", "finalize_runtime.py"):
        yield b"F", task_recipe / name, portable / name


def choose_source(explicit):
    if explicit is not None:
        return explicit.resolve()
    directories = sorted(path for path in PROJECT.iterdir()
                         if not path.name.startswith(".") and path.is_dir())
    candidates = [path for path in directories
                  if (path / "rlinf/config.py").is_file()
                  and (path / "requirements/install.sh").is_file()]
    if not candidates:
        # Name-based fallback gives an incomplete checkout a specific missing-file diagnostic.
        candidates = [path for path in directories
                      if path.name.lower() in {"rlinf", "rlinf-striding"}]
    if not candidates:
        raise ValueError("Clone RLinf into a directory next to compile.sh, or pass --source DIR.")
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Found multiple possible RLinf checkouts ({names}). Select one with --source DIR.")
    return candidates[0].resolve()


def choose_example(repo):
    directory = repo / TASK_DIRECTORY
    for name in PREFERRED_EXAMPLES:
        task = directory / name
        if task.is_file():
            return task
    for task in sorted(directory.glob("*.yaml")):
        if task.is_file():
            return task
    raise ValueError(f"No task YAML found in {directory}. Pass a task YAML as the second argument.")


def check_inputs(repo, task, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}. Choose a new output directory.")
    for tree in (repo, TOOLS, PROJECT / "portable"):
        if output.is_relative_to(tree):
            raise ValueError(f"Choose an output directory outside {tree}")
    required = ("pyproject.toml", "LICENSE", "README.md", "requirements/install.sh",
                "rlinf/config.py")
    missing = [name for name in required if not (repo / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete RLinf checkout {repo}. Missing: {', '.join(missing)}")
    if not task.is_file() or task.suffix != ".yaml":
        raise ValueError(f"Expected a task YAML: {task}. Pass a task YAML as the second argument.")
    _, public = inventory(repo)
    for data in public:
        if not (repo / data).is_file():
            raise ValueError(f"Missing RLinf runtime resource: {repo / data}")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("Building this release requires a Linux x86-64 host.")
    tools = ["gcc", "tar", "dpkg-deb"]
    if not all(os.environ.get(name) for name in
               ("RLINF_CPYTHON_ARCHIVE", *(package[0] for package in GRAPHICS_PACKAGES))):
        tools.append("curl")
    if missing_tools := [name for name in tools if shutil.which(name) is None]:
        raise ValueError(f"Missing host build tools: {', '.join(missing_tools)}")


def release(output, repo, task, jobs):
    default_example = task is None
    if default_example:
        task = choose_example(repo)
    check_inputs(repo, task, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rlinf-compile-", dir=output.parent) as tmp:
        work = Path(tmp)
        native = work / "native"
        published = work / "release"
        published.mkdir()
        exe = published / "rlinf-train"
        script_list = json.dumps("\n".join(public_scripts()))
        try:
            subprocess.run(["gcc", "-std=c11", "-O2", "-s", "-static", "-Wall", "-Wextra",
                            f"-DRLINF_SCRIPT_LIST={script_list}", str(TOOLS / "release_main.c"),
                            "-lz", "-o", str(exe)], check=True)
        except subprocess.CalledProcessError:
            raise ValueError("Could not link the release launcher. Check the GCC errors above "
                             "and install static libc and zlib development libraries.") from None
        env = dict(os.environ, TMPDIR=str(work), PYTHONDONTWRITEBYTECODE="1")
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(key, None)
        print(f"Preparing CPython 3.11 and Cython build tools for {repo}...", flush=True)
        seed, tool_python = prepare(work, PROJECT / "portable", env)
        subprocess.run([str(tool_python), "-EB", str(TOOLS / "native_build.py"),
                        "--source", str(repo), "--output", str(native), "--jobs", str(jobs)],
                       env=env, check=True)

        def inputs():
            yield b"F", Path("start.py"), TOOLS / "release_start.py"
            yield from members(native, Path("app"))
            yield b"D", Path("setup"), repo
            yield from members(repo / "requirements", Path("setup/requirements"))
            yield from installer_recipes(PROJECT / "portable")
            for name in ("pyproject.toml", "LICENSE", "README.md"):
                yield b"F", Path("setup") / name, repo / name
            for name in ("finish_venv.py", "finalize_runtime.py"):
                yield b"F", Path("setup") / name, PROJECT / "portable" / name
            yield b"F", Path("setup/check_environment.py"), TOOLS / "check_environment.py"
            # The installer can bootstrap uv and create a venv without a system Python.
            seed_path = Path("setup/seed")
            for subdir in (Path("."), Path("bin"), Path("lib")):
                yield b"D", seed_path / subdir, seed / subdir
            for name in ("bin/python3.11", "lib/libpython3.11.so.1.0"):
                yield b"F", seed_path / name, seed / name
            for name in ("bin/python", "bin/python3", "lib/libpython3.11.so", "lib/libpython3.so"):
                path = seed / name
                yield (b"L" if path.is_symlink() else b"F"), seed_path / name, path
            for library in ("libEGL.so.", "libGLdispatch.so.", "libOpenGL.so.", "libvulkan.so."):
                for path in sorted((seed / "lib").glob(library + "*")):
                    yield (b"L" if path.is_symlink() else b"F"), seed_path / "lib" / path.name, path
            yield from members(seed / "lib/python3.11", seed_path / "lib/python3.11",
                               exclude_dirs=("site-packages", "__pycache__"))
            yield from members(seed / "include", seed_path / "include")
        print("Packing compiled modules, built-in config groups and upstream dependency recipes...", flush=True)
        write_payload(exe, inputs(), jobs)
        example = published / "example.yaml"
        if default_example:
            example.write_bytes(EXAMPLE_COMMENTS + task.read_bytes())
        else:
            shutil.copyfile(task, example)
        shutil.copy2(TOOLS / "RELEASE.md", published / "README.md")
        shutil.copy2(TOOLS / "setup_env.sh", published / "install.sh")
        fd = os.open(output.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            published.rename(output)
        finally:
            os.close(fd)
    executable = output / "rlinf-train"
    print(f"Built {executable} ({executable.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compile.sh", description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=PROJECT / "release")
    parser.add_argument("task", type=Path, nargs="?", help="task YAML to publish as example.yaml")
    parser.add_argument("--source", type=Path, help="RLinf checkout (default: autodetect beside compile.sh)")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 64:
        parser.error("--jobs must be between 1 and 64")
    output = args.output.parent.resolve() / args.output.name
    try:
        source = choose_source(args.source)
        task = args.task.resolve() if args.task else None
        release(output, source, task, args.jobs)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Build failed: {exc}\n")
