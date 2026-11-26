"""Small, standard-library regressions for the packaged source audit."""

import bz2
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile
import zlib

from offline_runtime import PHYSX_LIBRARY, audit_offline_runtime, pack_offline_runtime
from release_build import embed_offline_runtime, write_payload
from source_audit import audit_release


TEMPLATE = b"""import sys
# Comment that may be stripped by a release build.
def run_submission(parameters):
    supplied_input = parameters["submitted_program"]
    output_reply = supplied_input.replace("fixture", "expected")
    final_feedback = output_reply.casefold().splitlines()
    submission_score = len(final_feedback) * 13
    return submission_score
"""
WORKER = b"""def worker(config):
    configured_value = config.delivery_test_private_formula
    output_value = configured_value * 13
    return output_value
"""


class SourceAuditTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        for tree in ("rlinf", "examples", "evaluations", "toolkits"):
            (self.source / tree).mkdir(parents=True)
            (self.source / tree / "__init__.py").touch()
        (self.source / "examples/example.yaml").write_text("task: demo\n")
        (self.source / "rlinf/worker.py").write_bytes(WORKER)
        (self.source / "rlinf/http_code_judge_tool.py").write_text(
            f"PROGRAM = {TEMPLATE.decode()!r}\n"
        )

    def package(self, *members, archive=None):
        executable = self.root / "rlinf-train"
        # The scanner only needs an ELF64 header with an empty section table.
        executable.write_bytes(b"\x7fELF\x02\x01" + bytes(58))
        if archive is not None:
            embed_offline_runtime(executable, archive)
        records = []
        for index, (name, content) in enumerate(members):
            path = self.root / f"member-{index}"
            path.write_bytes(content)
            records.append((b"F", Path(name), path))
        write_payload(executable, records, 1)
        return executable

    def test_accepts_public_config_and_a_single_common_line(self):
        executable = self.package(("app/example.yaml", b"task: demo\n"),
                                  ("app/note.txt", b"return output_value\n"))
        self.assertEqual(audit_release(executable, self.source)["findings"], [])

    def test_audit_report_replaces_existing_file_with_owner_only_permissions(self):
        executable = self.package(("app/example.yaml", b"task: demo\n"))
        report = self.root / "audit.json"
        report.write_text("old report")
        report.chmod(0o644)
        audit_release(executable, self.source, report_path=report)
        self.assertEqual(json.loads(report.read_text())["findings"], [])
        self.assertEqual(stat.S_IMODE(report.stat().st_mode), 0o600)

    def test_rejects_comment_stripped_template_in_compressed_string_pool(self):
        stripped = TEMPLATE.replace(b"# Comment that may be stripped by a release build.\n", b"")
        for codec in (zlib, bz2):
            with self.subTest(codec=codec.__name__):
                executable = self.package(("app/constants.bin", b"native table\0" + codec.compress(stripped)))
                with self.assertRaisesRegex(ValueError, "embedded protected program"):
                    audit_release(executable, self.source)

    def test_rejects_renamed_source_inside_zip(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("picture.png", WORKER)
        executable = self.package(("app/resource", buffer.getvalue()))
        with self.assertRaisesRegex(ValueError, "complete protected source"):
            audit_release(executable, self.source)

    def test_rejects_template_beyond_text_limit_in_streamed_offline_member(self):
        runtime = self.root / "runtime"
        (runtime / "bin").mkdir(parents=True)
        (runtime / "bin/python").write_bytes(b"fixture")
        (runtime / "lib/python3.11/site-packages").mkdir(parents=True)
        with zipfile.ZipFile(runtime / "lib/vendor.jar", "w") as vendor:
            vendor.writestr("com/vendor/examples/sample.json", '{"example": true}')
            vendor.writestr("third_party/tests/test_sample.py", "assert True\n")
        # Put the literal beyond the text scanner's limit and across a read boundary.
        with (runtime / "lib/resource.bin").open("wb") as stream:
            stream.seek((9 << 20) - 8)
            stream.write(TEMPLATE)
        asset_home = self.root / "home"
        physx = asset_home / PHYSX_LIBRARY
        physx.parent.mkdir(parents=True)
        physx.write_bytes(b"\x7fELFfixture")
        archive = pack_offline_runtime(runtime, asset_home, self.root / "runtime.tar.gz")
        executable = self.package(("app/example.yaml", b"task: demo\n"), archive=archive)
        with self.assertRaisesRegex(ValueError, "streamed embedded RStar2 program"):
            audit_release(executable, self.source, offline_audit=audit_offline_runtime)


if __name__ == "__main__":
    unittest.main()
