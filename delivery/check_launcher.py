"""Check public selectors and the evaluator used by real-world deployment."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from launcher import ENTRIES, SCRIPT_ROUTES, config_roots, public_commands, resolve_script, short_command
from release_start import select_config


class PublicCommandTest(unittest.TestCase):
    def test_published_commands_and_upstream_paths(self):
        commands = [line.split()[0] for line in public_commands()]
        self.assertEqual(len(commands), len(set(commands)))
        self.assertEqual(set(commands), {"deploy"} | {short_command(Path(p)) for p in SCRIPT_ROUTES})
        for path, expected in SCRIPT_ROUTES.items():
            script = Path(path)
            for selector in (short_command(script), path, "./" + path,
                             str(script.parent / script.stem.removeprefix("run_"))):
                with self.subTest(selector=selector):
                    self.assertEqual(resolve_script(selector), expected)

    def test_training_and_evaluation_namespaces(self):
        pairs = {"embodiment": "train", "run_embodiment.sh": "train", "vla_sft": "train-vla-sft",
                 "eval": "eval", "run_eval": "eval", "searchr1/eval": "eval-searchr1",
                 "wideseek_r1/eval": "eval-wideseek", "searchr1/train": "train-searchr1",
                 "deploy": "deploy", "eval_diffusion": "eval-diffusion"}
        for selector, mode in pairs.items():
            with self.subTest(selector=selector):
                self.assertEqual(resolve_script(selector), mode)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            resolve_script("run_train")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_script("missing/eval")

    def test_deployment_uses_evaluator_and_realworld_configs(self):
        self.assertEqual(ENTRIES["deploy"], ENTRIES["eval"])
        self.assertEqual(ENTRIES["eval-diffusion"], ENTRIES["eval"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            roots = config_roots(root / "app", "deploy")
            self.assertEqual(roots[0], root / "app/evaluations/realworld")
            self.assertEqual(config_roots(root / "app", "eval-diffusion")[0],
                             root / "app/examples/diffusion/config")
            roots[0].mkdir(parents=True)
            (roots[0] / "realworld_eval.yaml").touch()
            (root / "custom.yaml").write_text("defaults: [realworld_eval]\n")
            self.assertEqual(select_config(roots, root, "realworld_eval", "deploy", prefer_first=True),
                             (roots[0], "realworld_eval", roots[0]))
            self.assertEqual(select_config(roots, root, "custom.yaml", "deploy", prefer_first=True),
                             (root, "custom", roots[0]))


    @unittest.skipUnless(find_spec("hydra"), "requires the target Hydra runtime")
    def test_eval_search_paths_and_local_topology(self):
        with tempfile.TemporaryDirectory(prefix="eval with spaces ") as directory:
            root = Path(directory)
            shutil.copy(Path(__file__).with_name("launcher.py"), root / "_rlinf_launcher.py")
            benchmark = root / "evaluations" / "realworld"
            benchmark.mkdir(parents=True)
            shared = root / "examples" / "embodiment" / "config"
            custom = root / "my config groups"
            for group_dir, name in ((shared, "bundled"), (custom, "custom")):
                group = group_dir / "weight_syncer"
                group.mkdir(parents=True)
                (group / "patch.yaml").write_text(f"source: {name}\n")
            primary = benchmark / "rtc.yaml"
            primary_text = ("defaults: [weight_syncer/patch, _self_]\n"
                            "runner: {logger: {log_path: .}}\ncluster: {num_nodes: 2}\n")
            primary.write_text(primary_text)
            cmd = [sys.executable, "-B", "-c", "from _rlinf_launcher import main; main()",
                   "deploy", "--config-path", str(benchmark), "--config-name", "rtc"]
            env = dict(os.environ, PYTHONPATH=str(root), HYDRA_FULL_ERROR="1")
            def launch(*args):
                return subprocess.run([*cmd, *args], cwd=root, env=env, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
            preview = ("--cfg", "job", "runner.logger.log_path=local")
            composed = launch(*preview)
            self.assertEqual(composed.returncode, 0, composed.stdout)
            self.assertIn("source: bundled", composed.stdout)
            custom_uri = json.dumps("file://" + str(custom))
            for override in ([f"hydra.searchpath=[{custom_uri}]"], []):
                if not override:
                    primary.write_text(primary_text + f"hydra:\n  searchpath: [{custom_uri}]\n")
                composed = launch(*preview, *override)
                self.assertEqual(composed.returncode, 0, composed.stdout)
                self.assertIn("source: custom", composed.stdout)
            execution = launch()
            self.assertNotEqual(execution.returncode, 0, execution.stdout)
            self.assertIn("requires multiple nodes", execution.stdout)


if __name__ == "__main__":
    unittest.main()
