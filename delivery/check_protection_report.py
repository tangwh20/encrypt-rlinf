"""Publisher report integration regressions with a local Git baseline and a tiny checkout."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from protection_artifact import ArtifactInventory
from protection_report import attach_release, check_audit_destination, check_destination, markdown
from release_build import (
    EXAMPLE_COMMENTS, LEGACY_EXAMPLE_COMMENTS, PROJECT, bind_protection, publish_default_example,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


CONFIG = """@torch.jit.script
def gelu_impl(x): return x
@torch.jit.script
def erf_gelu(x): return x
@jit_fuser
def squared_relu(x): return x
"""
FSDP = "rlinf/hybrid_engines/fsdp/"
MODULE = "rlinf/private_score.py"
RAW_MARKERS = ("PRIVATE_SOURCE_NOTE_e952", "PRIVATE_FORMULA_193c", "PRIVATE_YAML_VALUE_2a8b")
TASK = "examples/embodiment/config/frankasim_ppo_mlp.yaml"
PLAIN_YAML = ("fixture_parameter: " + RAW_MARKERS[2] + "\n").encode()
ASSET_NAME = "records/samples.csv"
ASSET_DATA = b"id,value\n1,alpha\n"


def source_plan(root):
    """Run the real preview CLI on a private working tree with a committed public reference."""
    repo = root / "checkout"
    for tree in ("rlinf", "examples", "evaluations", "toolkits"):
        write(repo, tree + "/__init__.py", "")
    write(repo, "LICENSE", "fixture license\n")
    write(repo, "rlinf/config.py", CONFIG)
    # These small modules exercise the actual FSDP compatibility anchors during staging.
    branch = ("        from torch.optim.lr_scheduler import LambdaLR\n\n"
              "        return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)\n")
    write(repo, FSDP + "utils.py", "from torch.optim import Optimizer\ndef get_lr_scheduler(\n"
          "    optimizer, lr_lambda, last_epoch\n):\n" +
          "".join(f"    if last_epoch == {i}:\n" + branch for i in range(3)))
    write(repo, FSDP + "strategy/base.py", """from torch.distributed.checkpoint.state_dict import (
    load,
)
from rlinf.hybrid_engines.fsdp.utils import (
    FSDPVersion,
)
class FSDPStrategyBase(ABC):
    def restore(self, dcp_load_path, training_state):
        match dcp_load_path:
            case "skip":
                return
        load(
                    {"fsdp_checkpoint": training_state},
                    checkpoint_id=dcp_load_path,
        )
""")
    write(repo, TASK, "fixture_parameter: public\n")
    write(repo, MODULE, "def score(value): return value\n")
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                  "-c", "commit.gpgsign=false", "commit", "-qm", "public baseline"]):
        subprocess.run(["git", "-C", repo, *args], check=True)
    write(repo, MODULE, f'# {RAW_MARKERS[0]}\ndef score(value):\n    return value + "{RAW_MARKERS[1]}"\n')
    (repo / TASK).write_bytes(PLAIN_YAML)
    write(repo, "examples/prefixed.yaml", (EXAMPLE_COMMENTS + PLAIN_YAML).decode())
    assets = root / "offline-data"
    write(assets, ASSET_NAME, ASSET_DATA.decode())
    destination = root / "publisher"
    subprocess.run([PROJECT / "compile.sh", root / "new-release", "--source", repo,
                    "--public-repo", repo, "--public-ref", "HEAD", "--analyze-only",
                    "--offline-assets", assets, "--protection-report", destination],
                   check=True, stdout=subprocess.PIPE)
    return destination


class ProtectionReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.root = Path(directory.name)
        cls.destination = source_plan(cls.root)
        cls.plan = json.loads((cls.destination / "report.json").read_text())

    def test_preview_and_no_raw_source_or_yaml(self):
        self.assertEqual(self.plan["mode"], "source_plan")
        self.assertEqual(self.plan["prepared_hash_basis"]["mode"], "publisher_replay")
        self.assertEqual(self.plan["modules"][MODULE]["status"], "modified")
        self.assertEqual(self.plan["readable_resources"][TASK]["source_sha256"], digest(PLAIN_YAML))
        self.assertEqual({p.name for p in self.destination.iterdir()}, {"report.json", "priorities.md"})
        for file in self.destination.iterdir():
            for marker in RAW_MARKERS:
                self.assertNotIn(marker, file.read_text(), file.name)
        self.assertFalse((self.root / "new-release").exists())

    def test_preview_rejects_invalid_task_path(self):
        repo = self.root / "checkout"
        destination = self.root / "invalid-task-report"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "directory.yaml"
            directory.mkdir()
            for task in (repo / "LICENSE", directory, directory / "missing.yaml"):
                with self.subTest(task=task):
                    result = subprocess.run(
                        [PROJECT / "compile.sh", self.root / "new-release", task, "--source", repo,
                         "--public-repo", repo, "--public-ref", "HEAD", "--analyze-only",
                         "--protection-report", destination],
                        capture_output=True, text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Expected a task YAML", result.stderr)
                    self.assertFalse(destination.exists())

    def test_diverged_public_tip_provides_provenance_but_not_fork_status(self):
        def git(repo, *args):
            return subprocess.run(["git", "-C", repo, "-c", "user.name=fixture", "-c",
                                   "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
                                   *args], check=True, capture_output=True, text=True).stdout.strip()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fork, mirror = root / "fork", root / "public"
            for clone in (fork, mirror):
                git(root, "clone", "--no-local", str(self.root / "checkout"), str(clone))
            base = git(fork, "rev-parse", "HEAD")
            reused = ("def accept_sample(value):\n    if value < 0:\n"
                      "        return False\n    return True\n")
            whole = "if filter_gate > 2:\n    threshold = filter_gate\n\n" + reused
            public_path = "rlinf/published_algorithm.py"
            mixed_path = "rlinf/mixed.py"
            write(mirror, public_path, whole)
            write(fork, MODULE, whole)
            write(fork, mixed_path, reused + "\ndef update_episode(self):\n"
                  "    if self.history:\n        self.history.pop()\n")
            (fork / "toolkits/__init__.py").unlink()
            write(fork, "toolkits/training.py", "def start(): return None\n")
            for clone in (fork, mirror):
                git(clone, "add", "-A")
                git(clone, "commit", "-qm", clone.name)
            tip = git(mirror, "rev-parse", "HEAD")

            def preview(name, *extra):
                dest = root / name
                subprocess.run([PROJECT / "compile.sh", "--analyze-only", "--source", fork,
                                "--public-repo", mirror, "--protection-report", dest, *extra],
                               check=True, capture_output=True)
                return json.loads((dest / "report.json").read_text())

            plan = preview("auto")
            self.assertEqual((plan["baseline"]["commit"], plan["baseline"]["public_tip"]), (base, tip))
            self.assertEqual(plan["removed_public_modules"], ["toolkits/__init__.py"])
            found = plan["modules"][MODULE]
            self.assertEqual((found["status"], found["priority"]), ("modified", "low"))
            self.assertEqual(found["public_content_matches"], [{"commit": tip, "paths": [public_path]}])
            self.assertFalse(any(symbol["priority"] == "high" for symbol in found["symbols"]))
            mixed = plan["modules"][mixed_path]
            self.assertEqual((mixed["status"], mixed["priority"]), ("added", "high"))
            symbols = {row["qualified_name"]: row for row in mixed["symbols"]}
            self.assertEqual(symbols["accept_sample"]["priority"], "low")
            self.assertEqual(symbols["accept_sample"]["public_matches"], [
                {"commit": tip, "file": public_path, "qualified_name": "accept_sample", "line": 4}])
            self.assertEqual(symbols["update_episode"]["priority"], "high")
            self.assertNotIn("public_matches", symbols["update_episode"])
            self.assertIn(tip, markdown(plan))

            fixed = preview("explicit", "--public-ref", base)
            self.assertIsNone(fixed["baseline"]["public_tip"])
            self.assertEqual(fixed["modules"][MODULE]["priority"], "high")
            self.assertNotIn("public_content_matches", fixed["modules"][MODULE])
            unmatched = next(symbol for symbol in fixed["modules"][mixed_path]["symbols"]
                             if symbol["qualified_name"] == "accept_sample")
            self.assertNotIn("public_matches", unmatched)

    def test_original_snapshot_required_and_compiler_hash_wins(self):
        plan = copy.deepcopy(self.plan)
        rows = plan["modules"]
        manifest = {"original": {k: row["source_sha256"] for k, row in rows.items()},
                    "staged": {k: row["prepared_sha256"] for k, row in rows.items()}}
        wrong = copy.deepcopy(manifest)
        wrong["original"][MODULE] = digest(b"checkout changed")
        with self.assertRaisesRegex(ValueError, "private input snapshot"):
            attach_release(plan, {}, wrong)
        # Model the actual tuple-target formatting difference between 3.10 and 3.11.
        target = FSDP + "strategy/base.py"
        local = digest(b"for (key, value) in pairs:\n    pass\n")
        compiled = digest(b"for key, value in pairs:\n    pass\n")
        rows[target].update(prepared_sha256=local, public_sha256=compiled, prepared_matches_public=False)
        manifest["staged"][target] = compiled
        # A replayed adjustment can also be absent from a compiler snapshot.
        manifest["staged"]["rlinf/config.py"] = manifest["original"]["rlinf/config.py"]
        attach_release(plan, {}, manifest)
        self.assertEqual(plan["source_binding"], "compilation_input_hashes_verified")
        self.assertEqual(plan["prepared_hash_basis"]["mode"], "actual_compiler_snapshot")
        self.assertEqual(rows[target]["prepared_sha256"], compiled)
        self.assertTrue(rows[target]["prepared_matches_public"])
        self.assertFalse(rows["rlinf/config.py"]["adapted_by_publisher"])
        self.assertEqual(plan["summary"]["public_files_identical_after_preparation"],
                         sum(row["prepared_matches_public"] for row in rows.values()))
        self.assertEqual(plan["summary"]["publisher_adapted_modules"],
                         sum(row["prepared_sha256"] != row["source_sha256"] for row in rows.values()))

    def test_example_binding_and_existing_release_claim(self):
        # The archive audit has its own fixture in check_protection_artifact.py.
        # Supply its receipt here to isolate how the publisher binds example.yaml.
        with tempfile.TemporaryDirectory() as tmp:
            delivery = Path(tmp)
            published = EXAMPLE_COMMENTS + PLAIN_YAML
            (delivery / "example.yaml").write_bytes(published)
            receipt = {"sidecars": {"example.yaml": {"sha256": digest(published)}},
                       "executable": {"sha256": digest(b"fixture executable")},
                       "offline_archive": {"content_audit": {"assets": {}}}}
            observed = Mock(spec=ArtifactInventory)
            observed.bind.return_value = receipt
            for default, name, forms in (
                    (True, TASK, ["without_publisher_header"]),
                    (False, "examples/prefixed.yaml", ["as_delivered"]),
                    (None, TASK, ["without_publisher_header"])):
                with self.subTest(default_example=default):
                    plan = copy.deepcopy(self.plan)
                    plan["example_source"]["source_sha256"] = plan["readable_resources"][name]["source_sha256"]
                    bind_protection(plan, delivery, observed, {}, default_example=default)
                    example = plan["delivered_example"]
                    self.assertEqual(example["matching_forms"], forms)
                    self.assertTrue(example["matches_planned_example"])
                    if default is not None:
                        self.assertEqual(example["matching_checkout_yaml"], [name])
                    self.assertEqual(example["sha256"], digest(published))
                    self.assertEqual(plan["mode"], "existing_release")
                    self.assertEqual(plan["source_binding"], "replayed_from_supplied_checkout_without_build_manifest")
                    self.assertEqual(plan["prepared_hash_basis"]["mode"], "publisher_replay")
                    self.assertIn("推断", markdown(plan))
            with self.assertRaisesRegex(ValueError, "planned YAML"):
                bind_protection(copy.deepcopy(self.plan), delivery, observed, {}, default_example=False)
            (delivery / "example.yaml").write_bytes(PLAIN_YAML)
            for default in (False, None):
                with self.subTest(stale_receipt=True, default_example=default):
                    plan = copy.deepcopy(self.plan)
                    with self.assertRaisesRegex(ValueError, "audited sidecar"):
                        bind_protection(plan, delivery, observed, {}, default_example=default)
                    self.assertEqual(plan["mode"], "source_plan")
                    self.assertNotIn("delivered_example", plan)

    def test_default_example_logger_label_and_legacy_binding(self):
        source = (b"runner:\r\n  logger:\r\n    project_name: rlinf # logger label\r\n"
                  b"other:\r\n  logger:\r\n    project_name: rlinf # unrelated\r\n")
        expected = source.replace(b"rlinf # logger label", b"training # logger label")
        self.assertEqual(publish_default_example(source), EXAMPLE_COMMENTS + expected)
        observed = Mock(spec=ArtifactInventory)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for published, default, form in (
                (EXAMPLE_COMMENTS + expected, True, "without_publisher_header_and_logger_label"),
                (EXAMPLE_COMMENTS + expected, None, "without_publisher_header_and_logger_label"),
                (LEGACY_EXAMPLE_COMMENTS + source, None, "without_publisher_header"),
                (source, False, "as_delivered"),
            ):
                with self.subTest(default=default, form=form):
                    (directory / "example.yaml").write_bytes(published)
                    observed.bind.return_value = {
                        "sidecars": {"example.yaml": {"sha256": digest(published)}},
                        "executable": {"sha256": digest(b"fixture executable")},
                    }
                    plan = copy.deepcopy(self.plan)
                    plan["example_source"]["source_sha256"] = digest(source)
                    bind_protection(plan, directory, observed, {}, default_example=default)
                    self.assertEqual(plan["delivered_example"]["matching_forms"], [form])

    def test_offline_asset_snapshot_is_bound_only_for_new_builds(self):
        expected = {"size": len(ASSET_DATA), "sha256": digest(ASSET_DATA)}
        self.assertEqual(self.plan["planned_data_assets"], {ASSET_NAME: expected})
        rows = self.plan["modules"]
        manifest = {"original": {name: row["source_sha256"] for name, row in rows.items()},
                    "staged": {name: row["prepared_sha256"] for name, row in rows.items()}}
        replacement = ASSET_DATA.replace(b"alpha", b"omega")
        changed = {"size": len(replacement), "sha256": digest(replacement)}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "example.yaml").write_bytes(PLAIN_YAML)
            observed = Mock(spec=ArtifactInventory)
            for assets in ({ASSET_NAME: changed}, {}, {ASSET_NAME: expected, "added.csv": expected}):
                with self.subTest(assets=assets):
                    observed.bind.return_value = {
                        "offline_archive": {"content_audit": {"assets": assets}},
                        "sidecars": {"example.yaml": {"sha256": digest(PLAIN_YAML)}},
                    }
                    plan = copy.deepcopy(self.plan)
                    with self.assertRaisesRegex(ValueError, "offline assets differ"):
                        bind_protection(plan, directory, observed, {}, manifest, default_example=False)
                    self.assertEqual(plan["mode"], "source_plan")
                    bind_protection(plan, directory, observed, {})
                    self.assertEqual(plan["artifact"]["offline_archive"]["content_audit"]["assets"], assets)
            observed.bind.return_value["offline_archive"]["content_audit"]["assets"] = {ASSET_NAME: expected}
            plan = copy.deepcopy(self.plan)
            bind_protection(plan, directory, observed, {}, manifest, default_example=False)
            self.assertEqual(plan["mode"], "verified_build")

    def test_destination_rejects_other_releases_and_installed_prefixes(self):
        release = self.root / "intended-release"
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "differently-named-build"
            write(other, "rlinf-train", "")
            write(other, "install.sh", "")
            current = Path(tmp) / "current-build"
            write(current, "run", "")
            write(current, "install.sh", "")
            runtime = Path(tmp) / "installed-prefix"
            write(runtime, "bin/python", "")
            (runtime / "lib/python3.11/site-packages").mkdir(parents=True)
            alias = Path(tmp) / "linked-prefix"
            alias.symlink_to(runtime, target_is_directory=True)
            for directory in (release / "report", other / "work/report", current / "work/report",
                              runtime / "lib/report", alias / "nested/report"):
                with self.subTest(destination=directory), self.assertRaises(ValueError):
                    check_destination(directory, release)
                with self.subTest(destination=directory), self.assertRaises(ValueError):
                    check_audit_destination(directory.with_suffix(".json"), release)
            check_destination(other.with_name("publisher-evidence"), release)
            report = write(Path(tmp), "audit.json", "existing report")
            check_audit_destination(report, release)
            with self.assertRaises(ValueError):
                check_audit_destination(report, release, (report.parent,))


if __name__ == "__main__":
    unittest.main()
