"""Build a source-free Linux RLinf directory using Cython and GCC."""

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from contextlib import nullcontext
from pathlib import Path

import Cython
from harden_native import GCC_FLAGS, load_selection, sha256, trim_code_objects, write_receipt
from source_inventory import copy_inputs, inventory, write_notices
from protection_inputs import DELIVERY_TOOLS, prepare_application_sources
from Cython.Compiler import Main, Options

SHIM = """from _rlinf_launcher import main

if __name__ == "__main__":
    main()
"""
PUBLIC_HELP_DOCS = frozenset(
    Path(name)
    for name in (
        "examples/offline_rl/advantage_labeling/steam/process/merge_steam_ensemble.py",
        "examples/offline_rl/advantage_labeling/steam/process/visualize_advantage.py",
        "rlinf/utils/ckpt_convertor/openpi/convert.py",
        "rlinf/utils/ckpt_convertor/openpi/jax_to_openpi.py",
        "rlinf/utils/ckpt_convertor/openpi/openpi_pytorch_to_openpi.py",
        "rlinf/utils/ckpt_convertor/openpi/openpi_to_openpi_pytorch.py",
        "rlinf/utils/ckpt_convertor/openpi/pt_to_safetensors.py",
        "rlinf/utils/ckpt_convertor/openpi/sft2deploy.py",
        "toolkits/cosmos3/convert_checkpoint.py",
        "toolkits/lerobot/merge_lerobot_datasets.py",
        "toolkits/realworld_check/check_robot_parts.py",
        "toolkits/realworld_check/collect_monitor.py",
        "toolkits/realworld_check/test_piper_controller.py",
        "toolkits/realworld_check/test_so101_env.py",
    )
)


def extension_name(module: Path) -> str:
    return ".".join(
        module.parent.parts
        if module.stem == "__init__"
        else module.with_suffix("").parts
    )


def native_path(module: Path) -> Path:
    return module.with_suffix(sysconfig.get_config_var("EXT_SUFFIX"))


def compile_module(
    module: Path, c_root: Path, output: Path, gcc: str, harden: bool = False
) -> dict | None:
    c_file = c_root / module.with_suffix(".c")
    c_file.parent.mkdir(parents=True, exist_ok=True)
    Options.docstrings = module in PUBLIC_HELP_DOCS
    options = Main.CompilationOptions(
        Main.default_options,
        output_file=str(c_file),
        language_level=3,
        compiler_directives={
            "annotation_typing": False,
            "binding": True,
            "emit_code_comments": False,
        },
    )
    counts = {"code_objects": 0, "removed_local_slots": 0, "removed_source_positions": 0}
    staged_digest = sha256(module) if harden else None
    with trim_code_objects(counts) if harden else nullcontext():
        result = Main.compile(
            str(module), options=options, full_module_name=extension_name(module)
        )
    if result.num_errors:
        raise ValueError(f"Cython reported {result.num_errors} error(s)")
    destination = output / native_path(module)
    python_headers = sysconfig.get_path("include")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            gcc,
            "-shared",
            "-fPIC",
            "-fvisibility=hidden",
            f"-ffile-prefix-map={c_root}=.",
            f"-ffile-prefix-map={python_headers}=python-headers",
            *(GCC_FLAGS if harden else ("-O1",)),
            "-s",
            f"-I{python_headers}",
            str(c_file),
            "-o",
            str(destination),
        ],
        check=True,
    )
    if harden:
        return {
            "extension": native_path(module).as_posix(),
            "staged_sha256": staged_digest,
            "sha256": sha256(destination),
            **counts,
        }
    return None


def compile_all(
    modules: set[Path], staged: Path, output: Path, gcc: str, jobs: int,
    *, hardened: frozenset[Path] = frozenset(),
) -> dict[str, dict]:
    print(
        f"Compiling {len(modules)} modules with Cython {Cython.__version__} using {jobs} jobs",
        flush=True,
    )
    old_cwd = Path.cwd()
    c_root = staged.parent / "generated-c"
    failures = []
    receipts = {}
    try:
        os.chdir(staged)
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as workers:
            futures = {
                workers.submit(
                    compile_module, module, c_root, output, gcc, module in hardened
                ): module
                for module in sorted(modules)
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                module = futures[future]
                try:
                    if (receipt := future.result()) is not None:
                        receipts[module.as_posix()] = receipt
                except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                    failures.append(module)
                    print(f"FAILED {module}: {exc}", file=sys.stderr, flush=True)
                if index % 50 == 0 or index == len(modules):
                    print(f"Processed {index}/{len(modules)} modules", flush=True)
    finally:
        os.chdir(old_cwd)
    if failures:
        names = ", ".join(map(str, sorted(failures)))
        raise ValueError(
            f"Native compilation failed for {len(failures)} modules: {names}"
        )
    return receipts


def validate_output(output: Path, files: set[Path], modules: set[Path]) -> None:
    entries = set(output.rglob("*"))
    if output.is_symlink() or any(path.is_symlink() for path in entries):
        raise ValueError("Native output contains a symlink")
    actual = {path.relative_to(output) for path in entries if path.is_file()}
    if actual != files:
        missing = ", ".join(map(str, sorted(files - actual)[:10])) or "none"
        extra = ", ".join(map(str, sorted(actual - files)[:10])) or "none"
        raise ValueError(
            f"Unexpected native output (missing: {missing}, extra: {extra})"
        )
    if (output / "launcher.py").read_text(encoding="utf-8") != SHIM:
        raise ValueError("Unexpected content in the Python launcher shim")
    binaries = {native_path(module) for module in modules}
    for binary in sorted(binaries):
        with (output / binary).open("rb") as extension:
            if extension.read(4) != b"\x7fELF":
                raise ValueError(f"Invalid native binary: {binary}")


def build(
    source: Path, output: Path, jobs: int, compiler: str, include: list[str],
    *, source_manifest: Path | None = None,
    harden_list: Path | None = None, harden_receipt: Path | None = None,
) -> None:
    if sys.platform != "linux" or sys.implementation.name != "cpython":
        raise ValueError("Build with the target Linux CPython interpreter")
    if not (Path(sysconfig.get_path("include")) / "Python.h").is_file():
        raise FileNotFoundError(
            "Install the development headers for the target Python interpreter"
        )
    if output.is_relative_to(source):
        raise ValueError("Build output must be outside the source checkout")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    gcc = shutil.which(compiler)
    if gcc is None:
        raise FileNotFoundError(f"C compiler not found: {compiler}")
    gcc = str(Path(gcc).absolute())
    modules, public = inventory(source)
    if include:
        selected = {Path(name) for name in include}
        if missing := selected - modules:
            raise ValueError(
                f"Unknown diagnostic modules: {', '.join(map(str, sorted(missing)))}"
            )
        print("Diagnostic partial build; dependencies may be absent", flush=True)
        modules = selected
    hardened = frozenset(
        load_selection(harden_list, modules) if harden_list is not None else ()
    )
    if harden_receipt is not None and harden_list is None:
        raise ValueError("A hardening receipt requires --harden-list")
    if harden_list is not None:
        if harden_receipt is None:
            harden_receipt = output.with_name(f"{output.name}.hardening.json")
        harden_receipt = harden_receipt.parent.resolve() / harden_receipt.name
        if harden_receipt.is_relative_to(output) or harden_receipt.is_relative_to(source):
            raise ValueError("Hardening receipt must be outside the source and native output")
        if source_manifest is not None and harden_receipt == source_manifest.resolve():
            raise ValueError("Hardening receipt and source manifest need distinct paths")
        if harden_receipt.exists() or harden_receipt.is_symlink():
            raise FileExistsError(f"Hardening receipt already exists: {harden_receipt}")

    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(f".{output.name}.build-lock")
    lock.mkdir()
    try:
        with (
            tempfile.TemporaryDirectory(prefix="rlinf-native-source-") as private_dir,
            tempfile.TemporaryDirectory(
                prefix=".rlinf-native-", dir=output.parent
            ) as output_dir,
        ):
            staged = Path(private_dir) / "source"
            generated = Path(output_dir) / "payload"
            manifest_file = Path(private_dir) / "source-manifest.json"
            compiled = prepare_application_sources(
                source, staged, modules, partial=bool(include),
                source_manifest=manifest_file if source_manifest is not None else None,
            )
            receipts = compile_all(compiled, staged, generated, gcc, jobs, hardened=hardened)
            copy_inputs(source, generated, public)
            write_notices(source, generated, modules)
            (generated / "launcher.py").write_text(SHIM, encoding="utf-8")
            expected = (
                {native_path(module) for module in compiled}
                | public
                | {
                    Path("launcher.py"),
                    Path("NOTICE"),
                }
            )
            validate_output(generated, expected, compiled)
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"Output already exists: {output}")
            if source_manifest is not None:
                shutil.copyfile(manifest_file, source_manifest)
            if harden_receipt is None:
                generated.rename(output)
            else:
                write_receipt(harden_receipt, receipts)
                try:
                    generated.rename(output)
                except OSError:
                    harden_receipt.unlink()
                    raise
    finally:
        lock.rmdir()
    print(f"Published {output} with {len(compiled)} native extensions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DELIVERY_TOOLS.parent / "RLinf-Striding"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment-helpers", type=Path, metavar="DIR",
                        help="Compile staged deployment modules for the release builder")
    parser.add_argument("--source-manifest", type=Path, metavar="PATH",
                        help="Write original and compiler-input SHA-256 maps to a publisher file")
    parser.add_argument("--harden-list", type=Path, metavar="PATH",
                        help="Apply the hardening profile to a JSON array of source paths")
    parser.add_argument("--harden-receipt", type=Path, metavar="PATH",
                        help="Write the publisher receipt (default: OUTPUT.hardening.json)")
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--cc", default="gcc", help="C compiler executable")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="FILE",
        help="Build only this source module plus the launcher for diagnostics. Repeat as needed.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    try:
        if args.deployment_helpers:
            if args.harden_list is not None or args.harden_receipt is not None:
                raise ValueError("Hardening selection applies to application modules only")
            staged = args.deployment_helpers.resolve()
            modules = {
                path.relative_to(staged)
                for path in staged.glob("*.py")
                if path.name.startswith(("_train_", "_rlinf_"))
            }
            if not modules:
                raise ValueError(f"No deployment modules in {staged}")
            compile_all(modules, staged, args.output.resolve(), args.cc, args.jobs)
            return
        build(
            args.source.resolve(),
            args.output.parent.resolve() / args.output.name,
            args.jobs,
            args.cc,
            args.include,
            source_manifest=args.source_manifest,
            harden_list=args.harden_list,
            harden_receipt=args.harden_receipt,
        )
    except (OSError, SyntaxError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Native build failed: {exc}\n")


if __name__ == "__main__":
    main()
