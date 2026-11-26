"""Exercise sealed native records with the static launcher and publisher audit."""

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from release_build import verify_hardening, write_payload
from sealed_modules import aes, launcher_key, native_member, prepare_key, seal_native_modules
from source_audit import audit_release
from source_inventory import SOURCE_TREES


TOOLS = Path(__file__).resolve().parent
MODULE = "rlinf/example.py"
MEMBER = native_member(MODULE)
OTHER = native_member("rlinf/renamed.py")


class SealedModulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = tempfile.TemporaryDirectory()
        cls.addClassCleanup(scratch.cleanup)
        cls.root = Path(scratch.name)
        cls.source = cls.root / "source"
        for name in SOURCE_TREES:
            (cls.source / name).mkdir(parents=True)
            (cls.source / name / "__init__.py").touch()
        (cls.source / MODULE).write_text("ANSWER = 42\n", encoding="utf-8")
        (cls.source / "LICENSE").write_text("fixture license\n", encoding="utf-8")
        native = cls.root / "native"
        cls.plain = native / MEMBER.removeprefix("app/")
        cls.plain.parent.mkdir(parents=True)
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-s", "-x", "c", "-o", str(cls.plain), "-"],
            input="int exported_fixture_answer(void) { return 42; }\n",
            text=True,
            check=True,
        )
        cls.public = cls.source / "examples/task.yaml"
        cls.public.write_text("answer: 42\n", encoding="utf-8")
        cls.launcher, cls.key = cls.compile_launcher("original")
        cls.wrong_launcher, _ = cls.compile_launcher("wrong")
        cls.envelope = seal_native_modules([MODULE], native, cls.root / "sealed", cls.key)[MEMBER]

    @classmethod
    def compile_launcher(cls, label):
        header = cls.root / f"{label}.h"
        key = prepare_key(header)
        launcher = cls.root / label
        subprocess.run(
            ["gcc", "-std=c11", "-s", "-static", "-Wall", "-Wextra",
             '-DRLINF_SCRIPT_LIST="fixture"', "-include", str(header),
             str(TOOLS / "release_main.c"), "-lz", "-lcrypto", "-o", str(launcher)],
            check=True,
        )
        return launcher, key

    def release(self, label, *, launcher=None, envelope=None, member=MEMBER):
        directory = self.root / label
        directory.mkdir()
        executable = directory / "run"
        shutil.copy2(launcher or self.launcher, executable)
        write_payload(executable, [
            (b"D", Path("app"), directory),
            (b"D", Path("app/rlinf"), directory),
            (b"F", Path("app/task.yaml"), self.public),
            (b"E", Path(member), envelope or self.envelope),
        ], 1)
        return executable

    def runtime(self, executable):
        return subprocess.run([str(executable), "--runtime-dir"], capture_output=True,
                              text=True, timeout=30)

    def test_static_launcher_and_audit_read_authenticated_bytes(self):
        self.assertEqual(launcher_key(self.launcher.read_bytes()), self.key)
        executable = self.release("correct")
        help_output = subprocess.check_output([str(executable), "--help"], text=True, timeout=30)
        self.assertIn("Usage: ./run COMMAND", help_output)
        self.assertNotIn("rlinf", help_output.lower())
        process = self.runtime(executable)
        self.assertEqual(process.returncode, 0, process.stderr)
        cache = Path(process.stdout.strip())
        self.assertEqual(cache.parent, executable.parent / ".train-app")
        self.assertEqual((cache / MEMBER).read_bytes(), self.plain.read_bytes())
        self.assertEqual((cache / "app/task.yaml").read_bytes(), self.public.read_bytes())
        self.assertEqual(self.runtime(executable).stdout, process.stdout)
        observed = {}
        audit = audit_release(executable, self.source, member_observer=observed.__setitem__)
        self.assertEqual(observed[MEMBER], self.plain.read_bytes())
        self.assertEqual(observed["app/task.yaml"], self.public.read_bytes())
        record = audit["members"]["encrypted_native_modules"][MEMBER]
        self.assertEqual(record["sha256"], hashlib.sha256(self.plain.read_bytes()).hexdigest())
        self.assertEqual(record["ciphertext_sha256"],
                         hashlib.sha256(self.envelope.read_bytes()).hexdigest())

        report = {}
        staged = {MODULE: hashlib.sha256((self.source / MODULE).read_bytes()).hexdigest()}
        receipt = {"profile": "fixture", "modules": {MODULE: {"sha256": record["sha256"], "staged_sha256": staged[MODULE]}}}
        verify_hardening(report, [MODULE], audit, receipt, staged)
        self.assertEqual(report["automatic_hardening"]["status"], "verified")
        with self.assertRaisesRegex(ValueError, "selections differ"):
            verify_hardening({}, [MODULE], audit, {"profile": "fixture", "modules": {}}, staged)
        with self.assertRaisesRegex(ValueError, "differs from compiler snapshot"):
            verify_hardening({}, [MODULE], audit, receipt, {MODULE: "0" * 64})
        with self.assertRaisesRegex(ValueError, "differs from hardened compiler output"):
            verify_hardening({}, [MODULE], audit,
                             {"profile": "fixture", "modules": {MODULE: {"sha256": "0" * 64, "staged_sha256": staged[MODULE]}}}, staged)

    def test_same_size_tampering_wrong_key_and_wrong_path_leave_no_cache(self):
        data = self.envelope.read_bytes()
        changed = self.root / "tampered-envelope"
        midpoint = len(data) // 2
        changed.write_bytes(data[:midpoint] + bytes([data[midpoint] ^ 1]) + data[midpoint + 1:])
        self.assertEqual(changed.stat().st_size, self.envelope.stat().st_size)

        for label, options in (
            ("tampered", {"envelope": changed}),
            ("wrong-key", {"launcher": self.wrong_launcher}),
            ("wrong-path", {"member": OTHER}),
        ):
            with self.subTest(label=label):
                executable = self.release(label, **options)
                process = self.runtime(executable)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("encrypted module authentication failed", process.stderr)
                self.assertEqual(process.stdout, "")
                self.assertEqual({path.name for path in (executable.parent / ".train-app").iterdir()},
                                 {".lock"})
                with self.assertRaisesRegex(ValueError, "Authentication failed"):
                    audit_release(executable, self.source)

    def test_authenticated_non_elf_is_rejected(self):
        envelope = self.root / "non-elf-envelope"
        nonce = bytes(12)
        envelope.write_bytes(nonce + aes(self.key).encrypt(nonce, b"not an ELF", MEMBER.encode()))
        executable = self.release("non-elf", envelope=envelope)
        process = self.runtime(executable)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("expected a native module", process.stderr)
        self.assertEqual({path.name for path in (executable.parent / ".train-app").iterdir()},
                         {".lock"})
        with self.assertRaisesRegex(ValueError, "Invalid authenticated native module"):
            audit_release(executable, self.source)


if __name__ == "__main__":
    unittest.main()
