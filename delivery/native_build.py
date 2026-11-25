"""Build a source-free Linux RLinf directory using Cython and GCC."""

import argparse
import ast
import concurrent.futures
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import Cython
from source_inventory import copy_inputs, inventory, write_notices
from compat import prepare_compatibility
from judge_template import JUDGE_MODULE, strip_judge_template_comments
from Cython.Compiler import Main, Options

DELIVERY_TOOLS = Path(__file__).resolve().parent
NATIVE_MODULE = Path("_rlinf_launcher.py")
SHIM = """from _rlinf_launcher import main

if __name__ == "__main__":
    main()
"""
COMPAT_MATCH_MODULES = frozenset(
    Path(name)
    for name in (
        "rlinf/data/schema/reasoning_requests.py",
        "rlinf/envs/sim/diffusion/rewards/embodied/models/vidar_idm/idm.py",
        "rlinf/hybrid_engines/fsdp/strategy/base.py",
        "rlinf/models/embodiment/openpi/dataconfig/polaris_dataconfig.py",
        "rlinf/scheduler/dynamic_scheduler/manager.py",
    )
)

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


def lower_value_matches(path: Path) -> None:
    """Preserve upstream value-only matches in a form understood by Cython."""
    tree = ast.parse(path.read_bytes(), filename=str(path))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Match)]
    if len(matches) != 1:
        raise ValueError(f"Review native match compatibility for {path}")
    match = matches[0]
    temporary = "_rlinf_match_value"
    if any(
        isinstance(node, ast.Name) and node.id == temporary for node in ast.walk(tree)
    ):
        raise ValueError(f"Native match variable already used in {path}")

    def comparisons(pattern: ast.pattern) -> list[ast.expr]:
        if isinstance(pattern, ast.MatchValue):
            return [
                ast.Compare(
                    left=ast.Name(id=temporary, ctx=ast.Load()),
                    ops=[ast.Eq()],
                    comparators=[pattern.value],
                )
            ]
        if isinstance(pattern, ast.MatchOr):
            return [test for part in pattern.patterns for test in comparisons(part)]
        raise ValueError(f"Review native match value compatibility for {path}")

    def lower() -> list[ast.stmt]:
        arms: list[ast.stmt] = []
        for case in reversed(match.cases):
            if case.guard is not None:
                raise ValueError(f"Review native match guard compatibility for {path}")
            if (
                isinstance(case.pattern, ast.MatchAs)
                and case.pattern.name is None
                and not arms
            ):
                arms = case.body
                continue
            tests = comparisons(case.pattern)
            test = (
                tests[0] if len(tests) == 1 else ast.BoolOp(op=ast.Or(), values=tests)
            )
            arms = [ast.If(test=test, body=case.body, orelse=arms)]
        assignment = ast.Assign(
            targets=[ast.Name(id=temporary, ctx=ast.Store())],
            value=match.subject,
        )
        return [assignment, *arms]

    class MatchLowering(ast.NodeTransformer):
        def visit_Match(self, node: ast.Match) -> list[ast.stmt]:
            return lower()

    converted = MatchLowering().visit(tree)
    path.write_text(
        ast.unparse(ast.fix_missing_locations(converted)) + "\n", encoding="utf-8"
    )


def split_dynamic_slice_unpacking(path: Path) -> None:
    """Work around a Cython compiler crash on two image-shape unpacking sites."""
    text = path.read_text(encoding="utf-8")
    before = "            cur_h, cur_w = im.shape[0 + batched : 2 + batched]\n"
    after = (
        "            image_dimensions = im.shape[0 + batched : 2 + batched]\n"
        "            cur_h, cur_w = image_dimensions\n"
    )
    if text.count(before) != 1 or "image_dimensions" in text:
        raise ValueError(f"Review native image-shape compatibility for {path}")
    path.write_text(text.replace(before, after), encoding="utf-8")


def normalize_optional_expert_indices(path: Path) -> None:
    """Keep the unequal-length tuple alternatives as Python tuples."""
    text = path.read_text(encoding="utf-8")
    before = "        expert_ids = (0, 1) if freeze_action_expert else (0,)\n"
    after = "        expert_ids = tuple(range(2 if freeze_action_expert else 1))\n"
    if text.count(before) != 1:
        raise ValueError(f"Review native optional expert compatibility for {path}")
    path.write_text(text.replace(before, after), encoding="utf-8")


def extension_name(module: Path) -> str:
    return ".".join(
        module.parent.parts
        if module.stem == "__init__"
        else module.with_suffix("").parts
    )


def native_path(module: Path) -> Path:
    return module.with_suffix(sysconfig.get_config_var("EXT_SUFFIX"))


def compile_module(module: Path, c_root: Path, output: Path, gcc: str) -> Path:
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
            "-O1",
            "-s",
            f"-I{python_headers}",
            str(c_file),
            "-o",
            str(destination),
        ],
        check=True,
    )
    return module


def compile_all(
    modules: set[Path], staged: Path, output: Path, gcc: str, jobs: int
) -> None:
    print(
        f"Compiling {len(modules)} modules with Cython {Cython.__version__} using {jobs} jobs",
        flush=True,
    )
    old_cwd = Path.cwd()
    c_root = staged.parent / "generated-c"
    failures = []
    try:
        os.chdir(staged)
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as workers:
            futures = {
                workers.submit(compile_module, module, c_root, output, gcc): module
                for module in sorted(modules)
            }
            for index, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                module = futures[future]
                try:
                    future.result()
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
    source: Path, output: Path, jobs: int, compiler: str, include: list[str]
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
            copy_inputs(source, staged, modules)
            shutil.copyfile(DELIVERY_TOOLS / "launcher.py", staged / NATIVE_MODULE)
            if not include:
                prepare_compatibility(staged)
            if JUDGE_MODULE in modules:
                strip_judge_template_comments(staged / JUDGE_MODULE)
            for module in modules & COMPAT_MATCH_MODULES:
                lower_value_matches(staged / module)
            for module in (
                Path("rlinf/envs/utils.py"),
                Path("rlinf/envs/sim/robocasa/utils.py"),
            ):
                if module in modules:
                    split_dynamic_slice_unpacking(staged / module)
            openpi = Path("rlinf/models/embodiment/openpi/pi0.py")
            if openpi in modules:
                normalize_optional_expert_indices(staged / openpi)
            compiled = modules | {NATIVE_MODULE}
            compile_all(compiled, staged, generated, gcc, jobs)
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
            generated.rename(output)
    finally:
        lock.rmdir()
    print(f"Published {output} with {len(compiled)} native extensions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DELIVERY_TOOLS.parent / "RLinf-Striding"
    )
    parser.add_argument("--output", type=Path, required=True)
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
        build(
            args.source.resolve(),
            args.output.parent.resolve() / args.output.name,
            args.jobs,
            args.cc,
            args.include,
        )
    except (OSError, SyntaxError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Native build failed: {exc}\n")


if __name__ == "__main__":
    main()
