"""Prepare the exact application sources passed to the native compiler."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

from compat import prepare_compatibility
from judge_template import JUDGE_MODULE, SANDBOX_MODULE, stage_native_judge
from source_inventory import copy_inputs

DELIVERY_TOOLS = Path(__file__).resolve().parent
NATIVE_MODULE = Path("_rlinf_launcher.py")
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


def prepare_application_sources(
    source: Path, staged: Path, modules: set[Path], *, partial: bool = False,
    source_manifest: Path | None = None,
) -> set[Path]:
    """Stage selected upstream modules and build helpers in a fresh directory."""
    copy_inputs(source, staged, modules)
    if source_manifest is not None:
        original = staged_source_hashes(staged, modules)
    shutil.copyfile(DELIVERY_TOOLS / "launcher.py", staged / NATIVE_MODULE)
    if not partial:
        prepare_compatibility(staged)
    compiled = modules | {NATIVE_MODULE}
    if JUDGE_MODULE in modules:
        stage_native_judge(staged / JUDGE_MODULE, staged / SANDBOX_MODULE)
        compiled.add(SANDBOX_MODULE)
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
    if source_manifest is not None:
        manifest = {"original": original, "staged": staged_source_hashes(staged, modules)}
        source_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return compiled


def staged_source_hashes(staged: Path, modules: set[Path]) -> dict[str, str]:
    """Hash selected compiler inputs by upstream-relative POSIX path."""
    return {
        module.as_posix(): hashlib.sha256((staged / module).read_bytes()).hexdigest()
        for module in sorted(modules)
    }
