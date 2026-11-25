"""Convert an installed Python 3.11 venv into a relocatable standalone prefix."""

import argparse
import ast
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit


def local_path(path: Path, venv: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(venv) or not path.exists():
        raise ValueError(f"Editable path must already exist inside {venv}: {path}")
    return path


def rewrite_finder(path: Path, venv: Path) -> str:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    replacements = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if not node.value.startswith("/"):
            continue
        mapped = local_path(Path(node.value), venv)
        relative = os.path.relpath(mapped, path.parent)
        expression = f"str((__import__('pathlib').Path(__file__).parent / {relative!r}).resolve())"
        replacements[node.lineno, node.col_offset] = node.end_col_offset, expression
        if node.lineno != node.end_lineno:
            raise ValueError(f"Unsupported multiline editable mapping in {path}")
    if not replacements:
        raise ValueError(f"No local path mappings found in editable finder {path}")
    lines = source.splitlines(keepends=True)
    for (line, column), (end, expression) in sorted(replacements.items(), reverse=True):
        data = lines[line - 1].encode()
        lines[line - 1] = (data[:column] + expression.encode() + data[end:]).decode()
    return "".join(lines)


def editable_changes(venv: Path, site: Path) -> dict[Path, str | None]:
    changes = {}
    for link in site.glob("*.egg-link"):
        raise ValueError(f"Unsupported legacy editable install: {link}")
    for finder in site.glob("__editable__*_finder.py"):
        changes[finder] = rewrite_finder(finder, venv)
    for pth in site.glob("*.pth"):
        lines = []
        for line in pth.read_text().splitlines():
            if not line or line.startswith("#") or line.startswith(("import ", "import\t")):
                if "__editable__" in line and not any(
                    finder.stem in line for finder in changes
                ):
                    raise ValueError(f"Unsupported editable import hook in {pth}: {line}")
                lines.append(line)
                continue
            target = local_path(site / line, venv)
            lines.append(os.path.relpath(target, site))
        new = "\n".join(lines) + "\n"
        if new != pth.read_text():
            changes[pth] = new
    for url in site.glob("*.dist-info/direct_url.json"):
        document = json.loads(url.read_text())
        if document.get("dir_info", {}).get("editable"):
            parsed = urlsplit(document["url"])
            if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
                raise ValueError(f"Unsupported editable URL in {url}: {document['url']}")
            local_path(Path(unquote(parsed.path)), venv)
            changes[url] = None
    return changes


def copy_base(seed: Path, venv: Path) -> None:
    for item in ("include", "lib", "share", "toolchain"):
        source = seed / item
        if not source.exists():
            continue
        ignore = shutil.ignore_patterns("site-packages", "__pycache__", "*.pyc")
        shutil.copytree(source, venv / item, symlinks=True, dirs_exist_ok=True, ignore=ignore)
    for name in ("python", "python3", "python3.11", "python3-config", "python3.11-config"):
        source = seed / "bin" / name
        if not source.exists():
            continue
        target = venv / "bin" / name
        target.unlink(missing_ok=True)
        if source.is_symlink():
            target.symlink_to(source.readlink())
        else:
            shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("venv", type=Path, help="completed project-local uv venv")
    parser.add_argument("seed", type=Path, help="standalone CPython 3.11 prefix")
    args = parser.parse_args()
    venv, seed = args.venv.resolve(), args.seed.resolve()
    if venv.is_relative_to(seed) or seed.is_relative_to(venv):
        raise ValueError("The venv and standalone seed must be separate prefixes")
    if not (venv / "pyvenv.cfg").is_file():
        raise FileNotFoundError(venv / "pyvenv.cfg")
    if not (seed / "include/python3.11/Python.h").is_file():
        raise FileNotFoundError(seed / "include/python3.11/Python.h")
    python = seed / "bin/python"
    subprocess.run(
        [python, "-IB", "-c", "import sys; assert sys.version_info[:2] == (3, 11)"],
        check=True,
    )
    site = venv / "lib/python3.11/site-packages"
    if not site.is_dir():
        raise FileNotFoundError(site)
    changes = editable_changes(venv, site)
    for path, content in changes.items():
        if content is None:
            path.unlink()
        else:
            path.write_text(content)
        for pyc in path.parent.glob(f"__pycache__/{path.stem}.*.pyc"):
            pyc.unlink()
    copy_base(seed, venv)
    (venv / "pyvenv.cfg").unlink()
    subprocess.run(
        [venv / "bin/python", "-IB", Path(__file__).with_name("finalize_runtime.py")],
        check=True,
    )


if __name__ == "__main__":
    main()
