"""Exercise release binding without preparing a production offline environment."""

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from offline_runtime import PHYSX_LIBRARY, audit_offline_runtime, pack_offline_runtime
from protection_artifact import ArtifactInventory, EXTENSION_SUFFIX
from release_build import embed_offline_runtime, members, write_payload
from source_audit import LAUNCHER_NAME, LEGACY_LAUNCHER_NAME, SIDECARS, audit_release
from source_inventory import inventory


class ArtifactBindingTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.source = self.root / "source"
        self.release = self.root / "release"
        self.release.mkdir()
        self.archive = self.root / "tree"
        (self.archive / "app").mkdir(parents=True)
        for tree in ("rlinf", "examples", "evaluations", "toolkits"):
            (self.source / tree).mkdir(parents=True)
            (self.source / tree / "__init__.py").touch()
        (self.source / "examples/task.yaml").write_bytes(b"task: demo\n")
        (self.source / "LICENSE").write_bytes(b"example license")
        self.modules, resources = inventory(self.source)
        self.resources = {}
        for res in resources:
            data = (self.source / res).read_bytes()
            dest = self.archive / "app" / res
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            self.resources[str(res)] = hashlib.sha256(data).hexdigest()
        (self.archive / "app/NOTICE").write_bytes(b"generated notices")
        self.elf_header = bytearray(64)
        self.elf_header[:7] = b"\x7fELF\x02\x01\x01"
        self.elf_header[16:20] = b"\x03\0\x3e\0"
        self.elf_header = bytes(self.elf_header)
        extensions = [Path("app") / path.with_suffix(EXTENSION_SUFFIX) for path in self.modules]
        extensions.extend(Path(n + EXTENSION_SUFFIX) for n in
                          ("_train_release_start", "app/_rlinf_launcher", "setup/_train_check_environment"))
        for member in extensions:
            path = self.archive / member
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.elf_header)

        self.exe = self.release / LAUNCHER_NAME
        self.exe.write_bytes(self.elf_header)
        self.exe.chmod(0o755)
        for name in SIDECARS:
            (self.release / name).write_text("sample")
        (self.release / "install.sh").chmod(0o755)

    def audit(self, *, offline=True, startup="_train_release_start"):
        if offline:
            runtime = self.root / "runtime"
            (runtime / "bin").mkdir(parents=True)
            (runtime / "bin/python").write_bytes(b"fixture")
            (runtime / "lib/python3.11/site-packages").mkdir(parents=True)
            home = self.root / "home"
            lib = home / PHYSX_LIBRARY
            lib.parent.mkdir(parents=True)
            lib.write_bytes(self.elf_header)
            assets = self.root / "assets"
            assets.mkdir()
            (assets / "demo.json").write_bytes(b'{"example": 1}\n')
            archive = pack_offline_runtime(runtime, home, self.root / "runtime.tar.gz", assets=assets)
            embed_offline_runtime(self.exe, archive)
        records = [(b"F", Path(startup + EXTENSION_SUFFIX),
                    self.archive / (startup + EXTENSION_SUFFIX))]
        for name in ("app", "setup"):
            records.extend(members(self.archive / name, Path(name)))
        write_payload(self.exe, records, 1)
        collected = ArtifactInventory()
        audit = audit_release(self.exe, self.source, offline_audit=audit_offline_runtime,
                              sidecars=self.release, member_observer=collected.observe)
        return collected, audit

    def test_records_exact_module_resource_and_release_hashes(self):
        collected, audit = self.audit()
        result = collected.bind(self.release, audit, self.modules, self.resources)
        sha = hashlib.sha256(self.elf_header).hexdigest()
        self.assertEqual(result["module_binaries"], {str(p): sha for p in self.modules})
        self.assertEqual(result["resources"]["LICENSE"]["sha256"], self.resources["LICENSE"])
        self.assertEqual(result["executable"]["sha256"], hashlib.sha256(self.exe.read_bytes()).hexdigest())
        self.assertEqual(set(result["sidecars"]), set(SIDECARS))
        self.assertIn("demo.json", result["offline_archive"]["content_audit"]["assets"])
        self.assertEqual(len(result["publisher_helpers"]), 3)
        self.assertIn("sha256", result["generated_notice"])

    def test_reaudits_legacy_launcher_and_helpers(self):
        legacy = self.release / LEGACY_LAUNCHER_NAME
        self.exe.rename(legacy)
        self.exe = legacy
        for parent, name in (("", "release_start"), ("setup", "check_environment")):
            directory = self.archive / parent
            (directory / f"_train_{name}{EXTENSION_SUFFIX}").rename(
                directory / f"_rlinf_{name}{EXTENSION_SUFFIX}"
            )
        collected, audit = self.audit(startup="_rlinf_release_start")
        result = collected.bind(self.release, audit, self.modules, self.resources)
        self.assertIn("_rlinf_release_start" + EXTENSION_SUFFIX, result["publisher_helpers"])
        self.assertIn("setup/_rlinf_check_environment" + EXTENSION_SUFFIX, result["publisher_helpers"])

    def test_rejects_both_launcher_names(self):
        collected, audit = self.audit()
        (self.release / LEGACY_LAUNCHER_NAME).symlink_to("missing")
        with self.assertRaisesRegex(ValueError, "exactly one release launcher"):
            audit_release(self.exe, self.source, sidecars=self.release)
        with self.assertRaisesRegex(ValueError, "exactly one release launcher"):
            collected.bind(self.release, audit, self.modules, self.resources)

    def test_rejects_changed_inputs_unexpected_members_and_swapped_executable(self):
        collected, audit = self.audit()
        bad = dict(self.resources, LICENSE="incorrect")
        with self.assertRaisesRegex(ValueError, "Packaged resource differs"):
            collected.bind(self.release, audit, self.modules, bad)
        with self.assertRaisesRegex(ValueError, "Application files mismatch"):
            collected.bind(self.release, audit, set(), self.resources)
        replacement = self.root / "replacement"
        shutil.copy2(self.exe, replacement)
        replacement.replace(self.exe)
        with self.assertRaisesRegex(ValueError, "does not match audited"):
            collected.bind(self.release, audit, self.modules, self.resources)

    def test_rejects_application_symlinks(self):
        (self.archive / "app/linked-license").symlink_to("LICENSE")
        collected, audit = self.audit()
        with self.assertRaisesRegex(ValueError, "symlinks"):
            collected.bind(self.release, audit, self.modules, self.resources)

    def test_rejects_extra_directory(self):
        (self.archive / "app/unused").mkdir()
        collected, audit = self.audit()
        with self.assertRaisesRegex(ValueError, "Application directories mismatch"):
            collected.bind(self.release, audit, self.modules, self.resources)

    def test_rejects_replaced_sidecar(self):
        collected, audit = self.audit()
        (self.release / "example.yaml").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "sidecars differ"):
            collected.bind(self.release, audit, self.modules, self.resources)

    def test_rejects_online_payload(self):
        collected, audit = self.audit(offline=False)
        with self.assertRaisesRegex(ValueError, "fully audited offline release"):
            collected.bind(self.release, audit, self.modules, self.resources)


if __name__ == "__main__":
    unittest.main()
