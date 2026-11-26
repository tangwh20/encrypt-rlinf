"""Verify selection, metadata reduction, and callable behavior on real ELF builds."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harden_native import load_selection, sha256, write_receipt
from native_build import compile_all


FIXTURE = """
def callback(a, /, b=2, *extra, scale=1, **kwargs):
    implementation_tally = a + b + sum(extra) + sum(kwargs.values())
    return implementation_tally * scale

def closure(delta):
    def apply(value):
        return value + delta
    return apply

def stream(limit):
    yield from (value + 1 for value in range(limit))

async def compute(value):
    return value + 2

def fails():
    raise ValueError("expected")
"""

VERIFY = """
import asyncio
import inspect
import pickle
import traceback

import ordinary
import selected

for module in (ordinary, selected):
    callback = module.callback
    assert str(inspect.signature(callback)) == "(a, /, b=2, *extra, scale=1, **kwargs)"
    assert callback.__module__ == module.__name__
    assert callback.__qualname__ == "callback"
    assert callback(1, 2, 3, scale=2, more=4) == 20
    assert pickle.loads(pickle.dumps(callback)) is callback
    assert module.closure(2)(5) == 7
    assert list(module.stream(3)) == [1, 2, 3]
    assert asyncio.run(module.compute(3)) == 5
    try:
        module.fails()
    except ValueError as exc:
        final = traceback.extract_tb(exc.__traceback__)[-1]
        assert final.filename == module.__name__ + ".py", final
        assert final.name == module.__name__ + ".fails", final
        assert final.lineno > 0
    else:
        raise AssertionError("expected a ValueError")
assert ordinary.callback.__code__.co_firstlineno == selected.callback.__code__.co_firstlineno
assert "implementation_tally" in ordinary.callback.__code__.co_varnames
assert selected.callback.__code__.co_varnames == ("a", "b", "scale", "extra", "kwargs")
assert ordinary.callback.__code__.co_linetable
assert not selected.callback.__code__.co_linetable
"""


class HardeningTest(unittest.TestCase):
    def test_selection_must_exactly_match_available_paths(self):
        module = Path("rlinf/example.py")
        with tempfile.TemporaryDirectory() as tmp:
            selection = Path(tmp) / "selection.json"
            for values, expected in (([], set()), ([module.as_posix()], {module})):
                selection.write_text(json.dumps(values))
                self.assertEqual(load_selection(selection, {module}), expected)
            for invalid in (None, {}, "rlinf/example.py", [3], ["/rlinf/example.py"],
                            ["rlinf/../rlinf/example.py"], ["rlinf//example.py"],
                            ["rlinf/example.py", "rlinf/example.py"], ["missing.py"]):
                selection.write_text(json.dumps(invalid))
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_selection(selection, {module})

    def test_only_selected_extension_loses_implementation_metadata(self):
        gcc = shutil.which("gcc")
        if gcc is None:
            self.skipTest("GCC is required to build the actual extensions")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = root / "source"
            output = root / "output"
            staged.mkdir()
            hardened = Path("selected.py")
            modules = {Path("ordinary.py"), hardened}
            for module in modules:
                (staged / module).write_text(FIXTURE)
            result = compile_all(modules, staged, output, gcc, 1, hardened=frozenset({hardened}))
            self.assertEqual(set(result), {"selected.py"})
            record = result["selected.py"]
            self.assertEqual(sha256(staged / hardened), record["staged_sha256"])
            self.assertEqual(sha256(output / record["extension"]), record["sha256"])
            self.assertGreater(record["removed_source_positions"], 0)
            self.assertGreater(record["removed_local_slots"], 0)
            receipt = root / "receipt.json"
            write_receipt(receipt, result)
            self.assertEqual(json.loads(receipt.read_bytes())["modules"], result)
            self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_receipt(receipt, {})
            subprocess.run([sys.executable, "-EB", "-c", VERIFY], cwd=output,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True)


if __name__ == "__main__":
    unittest.main()
