"""Compile RLinf into a relocatable ELF with a self-contained task installer."""

import argparse
import concurrent.futures
from collections import deque
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zlib

from build_environment import GRAPHICS_PACKAGES, prepare
from launcher import public_commands
from offline_assets import validate_asset_directory
from offline_inputs import prepare_offline_inputs
from offline_runtime import audit_offline_runtime, pack_offline_runtime
from protection_artifact import ArtifactInventory
from sealed_modules import native_member, prepare_key, seal_native_modules
from protection_report import (
    analyze_inputs, attach_release, check_audit_destination, check_destination, report_draft,
)
from public_baseline import DEFAULT_PUBLIC_COMMIT
from source_audit import LAUNCHER_NAME, SIDECARS, audit_release, public_input_digests, release_executable
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
    b"# Preview this configuration with ./run embodiment example.yaml --cfg job\n"
)

LEGACY_EXAMPLE_COMMENTS = EXAMPLE_COMMENTS.replace(
    b"./run embodiment ", b"./rlinf-train run_embodiment ",
)
EXAMPLE_LOGGER_PROJECT = re.compile(
    rb"(^runner:[^\r\n]*\r?\n"
    rb"(?:[ \t][^\r\n]*\r?\n|(?:#[^\r\n]*)?\r?\n)*?"
    rb"^  logger:[^\r\n]*\r?\n"
    rb"(?: {4,}[^\r\n]*\r?\n|(?:#[^\r\n]*)?\r?\n)*?"
    rb"^    project_name:[ \t]*)(rlinf|training)(?=(?:[ \t]+#|[ \t]*\r?$))",
    re.MULTILINE,
)


def example_logger_project(data, original, replacement):
    return EXAMPLE_LOGGER_PROJECT.sub(
        lambda match: match[1] + (replacement if match[2] == original else match[2]),
        data, count=1,
    )


def publish_default_example(data):
    return EXAMPLE_COMMENTS + example_logger_project(data, b"rlinf", b"training")


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
        size = path.stat().st_size if kind in (b"F", b"E") else len(target)
        yield HEADER.pack(kind, mode, len(name), size) + name
        if kind == b"L":
            yield target
        if kind in (b"F", b"E"):
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


def embed_offline_runtime(exe, archive):
    digest = hashlib.sha256()
    with archive.open("rb") as source, exe.open("ab") as stream:
        offset = stream.tell()
        while chunk := source.read(BLOCK):
            digest.update(chunk)
            stream.write(chunk)
        length = stream.tell() - offset
        stream.write(struct.pack("<8sQQ32s", b"RLOFFL01", offset, length, digest.digest()))


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


def check_task_yaml(task):
    if not task.is_file() or task.suffix != ".yaml":
        raise ValueError(f"Expected a task YAML: {task}. Pass a task YAML as the second argument.")


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
    check_task_yaml(task)
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


def release(output, repo, task, jobs, offline_runtime=None, offline_home=None, audit_report=None,
            *, online_installer=False, physx_asset=None, offline_assets=None,
            protection_dir=None, public_repo=None, public_ref=DEFAULT_PUBLIC_COMMIT):
    default_example = task is None
    if default_example:
        task = choose_example(repo)
    check_inputs(repo, task, output)
    if online_installer and (offline_runtime or offline_home or physx_asset or offline_assets):
        raise ValueError("--online-installer cannot be used with offline inputs")
    if offline_assets is not None:
        offline_assets = validate_asset_directory(
            offline_assets, (PROJECT, repo, Path.home().resolve(), output)
        )
    protected = (repo, TOOLS, PROJECT / "portable", PROJECT / "runtime", PROJECT / ".offline-build",
                 PROJECT / "release", offline_runtime, offline_assets, public_repo)
    if audit_report is not None:
        check_audit_destination(audit_report, output, protected)
    if protection_dir is not None:
        if online_installer:
            raise ValueError("Protection analysis requires the complete offline delivery profile.")
        check_destination(protection_dir, output, protected)
        if audit_report is not None and (audit_report.is_relative_to(protection_dir)
                                         or protection_dir.is_relative_to(audit_report)):
            raise ValueError("Keep the source-audit file and the protection report directory separate.")
    protection = analyze_inputs(repo, public_repo or repo, public_ref, task, offline_assets)
    selected = sorted(name for name, row in protection["modules"].items() if row["priority"] == "high")
    print(f"Selected {len(selected)} priority modules for native hardening and AES-256-GCM.", flush=True)
    if not online_installer:
        offline_runtime, offline_home = prepare_offline_inputs(
            PROJECT, prefix=offline_runtime, asset_home=offline_home, physx_asset=physx_asset
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rlinf-compile-", dir=output.parent) as tmp:
        work = Path(tmp)
        native = work / "native"
        published = work / "release"
        published.mkdir()
        exe = published / LAUNCHER_NAME
        script_list = json.dumps("\n".join(public_commands()))
        header = work / "native-key.h"
        key = prepare_key(header) if selected else None
        # Keep an ordinary static zlib-only launcher for builds with no selected modules.
        flags = ["-include", str(header)] if key else []
        libraries = ["-lcrypto", "-ldl", "-pthread"] if key else []
        try:
            subprocess.run(["gcc", "-std=c11", "-O2", "-s", "-static", "-Wall", "-Wextra",
                            f"-DRLINF_SCRIPT_LIST={script_list}", *flags, str(TOOLS / "release_main.c"),
                            "-lz", *libraries, "-o", str(exe)], check=True)
        except subprocess.CalledProcessError:
            raise ValueError("Could not link the release launcher. Check the GCC errors above "
                             "and install static libc, zlib and OpenSSL development libraries.") from None
        env = dict(os.environ, TMPDIR=str(work), PYTHONDONTWRITEBYTECODE="1")
        for variable in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(variable, None)
        print(f"Preparing CPython 3.11 and Cython build tools for {repo}...", flush=True)
        seed, tool_python = prepare(work, PROJECT / "portable", env)
        compiler_manifest = work / "source-manifest.json"
        native_command = [str(tool_python), "-EB", str(TOOLS / "native_build.py"),
                          "--source", str(repo), "--output", str(native), "--jobs", str(jobs)]
        harden_list = work / "priority.json"
        harden_receipt = work / "native-hardening.json"
        harden_list.write_text(json.dumps(selected), encoding="utf-8")
        native_command.extend(["--source-manifest", str(compiler_manifest), "--harden-list", str(harden_list),
                               "--harden-receipt", str(harden_receipt)])
        subprocess.run(native_command, env=env, check=True)
        # These are private deployment modules. Compile them with the target ABI
        # just like the training modules, without publishing the Python shims.
        glue = work / "glue"
        glue.mkdir()
        helpers = {"_train_release_start": "release_start.py"}
        if offline_runtime is not None:
            helpers["_train_check_environment"] = "check_environment.py"
        for module, filename in helpers.items():
            shutil.copyfile(TOOLS / filename, glue / (module + ".py"))
        subprocess.run([str(tool_python), "-EB", str(TOOLS / "native_build.py"),
                        "--deployment-helpers", str(glue), "--output", str(work / "helpers"),
                        "--jobs", str(min(jobs, len(helpers)))], env=env, check=True)
        (native / "launcher.py").unlink()

        envelopes = seal_native_modules(selected, native, work / "sealed", key) if selected else {}

        def inputs():
            startup, = (work / "helpers").glob("_train_release_start.*.so")
            yield b"F", Path(startup.name), startup
            for kind, name, path in members(native, Path("app")):
                if name.as_posix() in envelopes:
                    yield b"E", name, envelopes[name.as_posix()]
                else:
                    yield kind, name, path
            yield b"D", Path("setup"), repo
            if offline_runtime is None:
                yield from members(repo / "requirements", Path("setup/requirements"))
                yield from installer_recipes(PROJECT / "portable")
                for name in ("pyproject.toml", "LICENSE", "README.md"):
                    yield b"F", Path("setup") / name, repo / name
                for name in ("finish_venv.py", "finalize_runtime.py"):
                    yield b"F", Path("setup") / name, PROJECT / "portable" / name
                yield b"F", Path("setup/check_environment.py"), TOOLS / "check_environment.py"
            else:
                product, = (work / "helpers").glob("_train_check_environment.*.so")
                yield b"F", Path("setup") / product.name, product
                return
            # Only the network installer needs a separate seed interpreter.
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
        print("Packing compiled modules and registered configurations...", flush=True)
        if offline_runtime is not None:
            archive = pack_offline_runtime(offline_runtime, offline_home, work / "offline-runtime.tar.gz",
                                           assets=offline_assets)
            embed_offline_runtime(exe, archive)
            archive.unlink()
        write_payload(exe, inputs(), jobs)
        example = published / "example.yaml"
        if default_example:
            example.write_bytes(publish_default_example(task.read_bytes()))
        else:
            shutil.copyfile(task, example)
        shutil.copy2(TOOLS / ("OFFLINE.md" if offline_runtime else "RELEASE.md"), published / "README.md")
        shutil.copy2(TOOLS / ("install_offline.sh" if offline_runtime else "setup_env.sh"), published / "install.sh")
        files = {entry.name for entry in published.iterdir()}
        if files != {LAUNCHER_NAME, *SIDECARS}:
            raise ValueError(f"Unexpected release contents: {sorted(files)}")
        print("Auditing the packaged application and release assets...", flush=True)
        observed = None if online_installer else ArtifactInventory()
        report = audit_release(exe, repo, trusted_inputs=public_input_digests(inputs()),
                               offline_audit=audit_offline_runtime, sidecars=published,
                               report_path=audit_report, member_observer=observed.observe if observed else None)
        compiler_inputs = json.loads(compiler_manifest.read_text(encoding="utf-8"))
        verify_hardening(protection, selected, report, json.loads(harden_receipt.read_text()), compiler_inputs["staged"])
        if observed is None:
            if compiler_inputs["original"] != {name: row["source_sha256"] for name, row in protection["modules"].items()}:
                raise ValueError("Working tree changed between selection and compilation.")
        else:
            bind_protection(protection, published, observed, report,
                            compiler_inputs,
                            default_example=default_example)
        print(f"Source audit checked {report['members']['types']['F']:,} payload files.", flush=True)
        fd = os.open(output.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            context = report_draft(protection_dir, protection) if protection_dir is not None else nullcontext()
            with context as draft:
                published.rename(output)
                if draft is not None:
                    try:
                        draft.rename(protection_dir)
                    except OSError:
                        output.rename(published)
                        raise
        finally:
            os.close(fd)
    executable = output / LAUNCHER_NAME
    print(f"Built {executable} ({executable.stat().st_size:,} bytes)")


def verify_hardening(protection, selected, audit, receipt, staged):
    compiled = receipt["modules"]
    encrypted = audit["members"]["encrypted_native_modules"]
    if set(selected) != set(compiled) or {native_member(name) for name in selected} != set(encrypted):
        raise ValueError("Planned, hardened and encrypted module selections differ.")
    for name in selected:
        if compiled[name]["staged_sha256"] != staged[name]:
            raise ValueError(f"Hardened input differs from compiler snapshot: {name}")
        if compiled[name]["sha256"] != encrypted[native_member(name)]["sha256"]:
            raise ValueError(f"Encrypted member differs from hardened compiler output: {name}")
    protection["automatic_hardening"] = {
        "status": "verified", "algorithm": "AES-256-GCM", "native_profile": receipt["profile"],
        "modules": {name: dict(compiled[name], ciphertext_sha256=encrypted[native_member(name)]["ciphertext_sha256"])
                    for name in selected},
    }


def bind_protection(protection, directory, observed, audit, manifest=None, *, default_example=None):
    modules = {Path(name) for name in protection["modules"]}
    resources = {name: row["source_sha256"] for name, row in protection["readable_resources"].items()}
    artifact = observed.bind(directory, audit, modules, resources)
    if manifest is None:
        encrypted = audit.get("members", {}).get("encrypted_native_modules", {})
        protection["automatic_hardening"] = {
            "status": "observed_archive_only", "modules": {
                name: encrypted[native_member(name)] for name in protection["modules"] if native_member(name) in encrypted},
        }
    if manifest is not None:
        actual_assets = artifact["offline_archive"]["content_audit"]["assets"]
        if actual_assets != protection["planned_data_assets"]:
            raise ValueError("Packaged offline assets differ from the planned files, sizes or SHA-256 digests.")
    text = (directory / "example.yaml").read_bytes()
    delivered_sha256 = hashlib.sha256(text).hexdigest()
    if delivered_sha256 != artifact["sidecars"]["example.yaml"]["sha256"]:
        raise ValueError("Published example.yaml differs from the audited sidecar.")
    candidates = {"as_delivered": delivered_sha256}
    if text.startswith(EXAMPLE_COMMENTS):
        body = text[len(EXAMPLE_COMMENTS):]
        if default_example is not True or publish_default_example(body) == text:
            candidates["without_publisher_header"] = hashlib.sha256(body).hexdigest()
        restored = example_logger_project(body, b"training", b"rlinf")
        if restored != body and publish_default_example(restored) == text:
            candidates["without_publisher_header_and_logger_label"] = hashlib.sha256(restored).hexdigest()
    elif default_example is None and text.startswith(LEGACY_EXAMPLE_COMMENTS):
        candidates["without_publisher_header"] = hashlib.sha256(text[len(LEGACY_EXAMPLE_COMMENTS):]).hexdigest()
    source_sha = protection["example_source"]["source_sha256"]
    if default_example is not None:
        candidates = {form: digest for form, digest in candidates.items()
                      if (form != "as_delivered") == default_example and digest == source_sha}
        if not candidates:
            raise ValueError("Published example differs from the planned YAML input.")
    forms = [form for form, digest in candidates.items() if digest == source_sha]
    attach_release(protection, artifact, manifest)
    protection["delivered_example"] = {
        "sha256": delivered_sha256,
        "matching_checkout_yaml": sorted(name for name, row in protection["readable_resources"].items()
                                        if name.endswith((".yaml", ".yml")) and row["source_sha256"] in candidates.values()),
        "matches_planned_example": bool(forms), "matching_forms": forms,
    }


def analyze_only(repo, task, public_repo, public_ref, destination, release_dir, assets, output):
    check_destination(destination, release_dir or output, (repo, TOOLS, PROJECT / "portable", PROJECT / "runtime",
                                                        PROJECT / ".offline-build", PROJECT / "release", assets, public_repo))
    if assets is not None:
        assets = validate_asset_directory(assets, (PROJECT, repo, Path.home().resolve(), output))
    task = task or choose_example(repo)
    check_task_yaml(task)
    plan = analyze_inputs(repo, public_repo or repo, public_ref, task, assets)
    if release_dir is not None:
        observed = ArtifactInventory()
        audit = audit_release(release_executable(release_dir), repo, offline_audit=audit_offline_runtime,
                              sidecars=release_dir, member_observer=observed.observe)
        bind_protection(plan, release_dir, observed, audit)
    with report_draft(destination, plan) as draft:
        draft.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compile.sh", description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=PROJECT / "release")
    parser.add_argument("task", type=Path, nargs="?", help="task YAML to publish as example.yaml")
    parser.add_argument("--source", type=Path, help="RLinf checkout (default: autodetect beside compile.sh)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--online-installer", action="store_true",
                        help="publish the installer that downloads dependencies on the target machine")
    parser.add_argument("--offline-runtime", type=Path,
                        help="runtime prefix to embed (default: reuse or build the project runtime)")
    physx = parser.add_mutually_exclusive_group()
    physx.add_argument("--offline-home", type=Path,
                       help="existing directory containing the pinned SAPIEN PhysX asset")
    physx.add_argument("--physx-asset", type=Path,
                       help="local upstream PhysX ZIP or libPhysXGpu_64.so to verify and cache")
    parser.add_argument("--offline-assets", type=Path,
                        help="bundle an explicitly selected data-only directory in the offline profile")
    parser.add_argument("--audit-report", type=Path,
                        help="write the source audit JSON outside the published release directory")
    parser.add_argument("--analyze-only", action="store_true",
                        help="compare the whole offline profile without compiling or installing dependencies")
    parser.add_argument("--inspect-release", type=Path, metavar="DIR",
                        help="audit an existing pristine offline release with --analyze-only")
    parser.add_argument("--public-repo", type=Path, metavar="DIR",
                        help="local public checkout or bare mirror; defaults to the common ancestor with the source checkout")
    parser.add_argument("--public-ref", metavar="REV",
                        help=f"public commit/ref or auto (default without a public mirror: upstream {DEFAULT_PUBLIC_COMMIT[:12]})")
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--protection-report", type=Path, metavar="NEW_DIR",
                         help="publisher-only JSON and Markdown (offline default: OUTPUT.protection)")
    options.add_argument("--skip-protection-report", action="store_true",
                         help="omit publisher report files; automatic classification, protection and auditing still run")
    args = parser.parse_args()
    if args.inspect_release and not args.analyze_only:
        parser.error("--inspect-release requires --analyze-only")
    if args.online_installer and (args.protection_report or args.analyze_only):
        parser.error("Publishing a protection report or using --analyze-only requires the offline profile")
    if args.skip_protection_report and args.analyze_only:
        parser.error("--skip-protection-report cannot be combined with analysis options")
    if args.analyze_only and (args.audit_report or args.offline_runtime or args.offline_home or args.physx_asset):
        parser.error("--analyze-only uses the checkout and optional --inspect-release, not build inputs or --audit-report")
    if args.inspect_release and args.offline_assets:
        parser.error("--inspect-release reads data assets from the actual embedded offline runtime")
    if not 1 <= args.jobs <= 64:
        parser.error("--jobs must be between 1 and 64")
    if args.online_installer and (args.offline_runtime or args.offline_home or args.physx_asset
                                  or args.offline_assets):
        parser.error("--online-installer cannot be used with offline inputs")
    output = args.output.parent.resolve() / args.output.name
    try:
        source = choose_source(args.source)
        task = args.task.resolve() if args.task else None
        report_path = args.audit_report.parent.resolve() / args.audit_report.name if args.audit_report else None
        reference = args.public_ref or ("auto" if args.public_repo else DEFAULT_PUBLIC_COMMIT)
        inspected = args.inspect_release.resolve() if args.inspect_release else None
        protection_dir = None
        if not args.online_installer and not args.skip_protection_report:
            target = inspected or output
            suffix = ".protection-plan" if args.analyze_only and inspected is None else ".protection"
            dest = args.protection_report or target.with_name(target.name + suffix)
            protection_dir = dest.parent.resolve() / dest.name
        if args.analyze_only:
            analyze_only(source, task, args.public_repo, reference,
                         protection_dir, inspected, args.offline_assets, output)
        else:
            release(output, source, task, args.jobs, args.offline_runtime, args.offline_home, report_path,
                    online_installer=args.online_installer, physx_asset=args.physx_asset,
                    offline_assets=args.offline_assets, protection_dir=protection_dir,
                    public_repo=args.public_repo, public_ref=reference)
        if protection_dir is not None:
            print(f"Publisher protection report: {protection_dir / 'priorities.md'}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Build failed: {exc}\n")
