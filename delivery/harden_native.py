"""Apply a pinned Cython metadata profile to selected native extensions."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import Cython
from Cython.Compiler.ExprNodes import CodeObjectNode


PROFILE = "cython-metadata-v1"
CYTHON_VERSION = "3.2.8"
GCC_FLAGS = (
    "-O2",
    "-fno-ident",
    "-DCYTHON_CLINE_IN_TRACEBACK=0",
    "-DCYTHON_CLINE_IN_TRACEBACK_RUNTIME=0",
)


def load_selection(path: Path, inventory: set[Path]) -> set[Path]:
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"Invalid hardening list {path}: {exc}") from exc
    if not isinstance(entries, list):
        raise ValueError("The hardening list must be a JSON array")
    available = {module.as_posix(): module for module in inventory}
    selected = set()
    for entry in entries:
        if not isinstance(entry, str) or entry not in available:
            raise ValueError(f"Unknown or noncanonical hardening path: {entry!r}")
        module = available[entry]
        if module in selected:
            raise ValueError(f"Duplicate hardening path: {entry}")
        selected.add(module)
    if selected and Cython.__version__ != CYTHON_VERSION:
        raise ValueError(f"{PROFILE} requires Cython {CYTHON_VERSION}")
    return selected


@contextmanager
def trim_code_objects(counts: dict[str, int]):
    """Omit implementation metadata from Cython's synthetic Python code objects."""
    generate = CodeObjectNode.generate_codeobj

    def generate_trimmed(code_object, *args, **kwargs):
        function = code_object.def_node
        node = getattr(function, "_cfuncdef_node", function)
        positions = node.node_positions
        varnames = code_object.varnames
        # CPython inspects argument slots in exactly this order when building
        # signatures. Generator expressions expose no Python argument slots.
        arguments = 0 if function.is_generator_expression else (
            len(function.args)
            + int(function.star_arg is not None)
            + int(function.starstar_arg is not None)
        )
        if arguments > len(varnames):
            raise ValueError(f"Cannot preserve argument metadata for {function.name}")
        counts["removed_local_slots"] += len(varnames) - arguments
        counts["removed_source_positions"] += len(positions or ())
        counts["code_objects"] += 1
        code_object.varnames = varnames[:arguments]
        node.node_positions = None
        try:
            return generate(code_object, *args, **kwargs)
        finally:
            code_object.varnames = varnames
            node.node_positions = positions

    # Each worker compiles one source at a time. Restore the class method
    # before the same worker handles another module.
    CodeObjectNode.generate_codeobj = generate_trimmed
    try:
        yield
    finally:
        CodeObjectNode.generate_codeobj = generate


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_receipt(path: Path, modules: dict[str, dict]) -> None:
    receipt = {
        "schema_version": 1,
        "profile": PROFILE,
        "cython_version": Cython.__version__,
        "gcc_flags": list(GCC_FLAGS),
        "modules": dict(sorted(modules.items())),
    }
    content = json.dumps(receipt, indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
    except OSError:
        path.unlink()
        raise
