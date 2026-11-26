"""Inventory protected modules, runtime data and upstream attribution."""

from pathlib import Path
import re
import shutil
import tokenize


EXTENSION_SUFFIX = ".cpython-311-x86_64-linux-gnu.so"
SOURCE_TREES = ("rlinf", "examples", "evaluations", "toolkits")
ATTRIBUTION = re.compile(
    r"^Licensed under the Apache License, Version 2\.0[^\n]*\n"
    r"[\s\S]*?^limitations under the License\.$"
    r"|^Permission is hereby granted[^\n]*\n[\s\S]*?^SOFTWARE\.?$"
    r"|^(?:Copyright |SPDX-License-Identifier: |The original code is licensed under |"
    r"Adapted from https?://|Copied and adapted from https?://|"
    r"This file contains code copied from |Reference: |"
    r"https://github\.com/ray-project/ray/)[^\n]*$",
    re.MULTILINE,
)
CALVIN_CONFIGS = frozenset(
    Path("rlinf/envs/sim/calvin/calvin_cfg/.hydra") / filename
    for filename in (
        "calvin_scene_A.yaml",
        "calvin_scene_B.yaml",
        "calvin_scene_C.yaml",
        "calvin_scene_D.yaml",
        "merged_config.yaml",
    )
)
SIMULATOR_DATA = frozenset(
    Path(name)
    for name in (
        "rlinf/envs/sim/behavior/behavior_task.jsonl",
        "rlinf/envs/sim/metaworld/metaworld_config.json",
        "rlinf/envs/sim/robotwin/seeds/eval_seeds.json",
        "rlinf/envs/sim/robotwin/seeds/train_seeds.json",
    )
)


def inventory(source: Path) -> tuple[set[Path], set[Path]]:
    """Find modules to protect and editable configurations and runtime data."""
    modules: set[Path] = set()
    public = {Path("LICENSE")}
    for name in SOURCE_TREES:
        tree = source / name
        if not tree.is_dir() or tree.is_symlink():
            raise ValueError(f"Expected a local source directory: {tree}")
        for candidate in tree.rglob("*"):
            relative = candidate.relative_to(source)
            if relative not in CALVIN_CONFIGS and any(
                part.startswith(".") or part == "__pycache__" for part in relative.parts
            ):
                continue
            if candidate.is_symlink():
                raise ValueError(f"Unexpected symlink in source tree: {relative}")
            if not candidate.is_file():
                continue
            if candidate.suffix == ".py":
                modules.add(relative)
            elif candidate.suffix in (".yaml", ".yml"):
                public.add(relative)
        if not any(path.parts[0] == name for path in modules):
            raise ValueError(f"No Python modules found in {tree}")
    if not any(path.suffix in (".yaml", ".yml") for path in public):
        raise ValueError(f"No editable configuration found in {source}")
    calvin_module = Path("rlinf/envs/sim/calvin/__init__.py")
    if calvin_module in modules and not CALVIN_CONFIGS <= public:
        missing = ", ".join(map(str, sorted(CALVIN_CONFIGS - public)))
        raise ValueError(f"Missing CALVIN configuration: {missing}")
    if (source / "NOTICE").is_file():
        public.add(Path("NOTICE"))
    for resource in SIMULATOR_DATA:
        component = source / Path(*resource.parts[:4])
        if component.is_dir():
            if not (source / resource).is_file():
                raise ValueError(f"Missing simulator data: {source / resource}")
            public.add(resource)
    return modules, public


def copy_inputs(source: Path, target: Path, files: set[Path]) -> None:
    for relative in sorted(files):
        original = source / relative
        if not original.is_file() or original.is_symlink():
            raise ValueError(f"Expected a regular file: {original}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)


def write_notices(
    source: Path, output: Path, modules: set[Path]
) -> None:
    """Preserve attribution from Python headers removed by obfuscation."""
    notices = set()
    for module in sorted(modules):
        with tokenize.open(source / module) as stream:
            leading_comments = []
            for line in stream:
                if line.startswith("#"):
                    leading_comments.append(line.removeprefix("#").removeprefix(" ").rstrip())
                elif not line.strip():
                    leading_comments.append("")
                else:
                    break
        header = "\n".join(leading_comments).strip()
        statements = ATTRIBUTION.findall(header)
        if statements:
            notices.add("\n".join(statements))
    notice = output / "NOTICE"
    upstream = notice.read_text(encoding="utf-8").rstrip() + "\n\n" if notice.exists() else ""
    preamble = (
        "This distribution contains RLinf modules processed with Cython and GCC. "
        "Delivery-specific compatibility changes were applied to the staged code.\n\n"
        "Original module attribution and license statements follow.\n\n"
    )
    notice.write_text(upstream + preamble + "\n\n".join(sorted(notices)) + "\n", encoding="utf-8")
