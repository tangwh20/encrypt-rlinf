"""Check the selected prefix before publishing it as the active training environment."""

from importlib import import_module
from pathlib import Path
import sys


prefix = Path(sys.argv[1]).resolve()
kind, model, environment = sys.argv[2:]
if sys.version_info[:2] != (3, 11) or Path(sys.base_prefix).resolve() != prefix:
    raise SystemExit(f"Expected standalone CPython 3.11 in {prefix}")
for path in sys.path:
    if path and not Path(path).resolve().is_relative_to(prefix):
        raise SystemExit(f"An installed import path points outside the local prefix: {path}")

modules = ["hydra", "ray", "torch"]
if environment == "frankasim":
    modules.extend(["franka_sim", "mujoco"])
if kind == "upstream" and model == "openvla":
    modules.append("prismatic.extern.hf.configuration_prismatic")
for name in modules:
    module = import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(prefix):
        raise SystemExit(f"{name} was imported from outside {prefix}: {module.__file__}")

from importlib.metadata import version
from packaging.version import Version
if Version(version("torch")) < Version("2.5.0"):
    raise SystemExit(f"Installed torch {version('torch')} conflicts with RLinf's torch>=2.5.0.")
