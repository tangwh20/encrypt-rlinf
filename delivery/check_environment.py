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

if kind == "offline":
    def disallow_network(event, args):
        if event in {"socket.getaddrinfo", "socket.connect", "socket.gethostbyname",
                     "socket.gethostbyaddr", "socket.getnameinfo"}:
            raise RuntimeError(f"Network access during offline environment verification: {event}")

    sys.addaudithook(disallow_network)

modules = ["hydra", "ray", "torch"]
if environment == "frankasim":
    modules.extend(["franka_sim", "mujoco"])
if kind == "offline":
    modules.extend(["sapien", "mani_skill", "cv2", "pandas", "pyarrow"])
if kind == "upstream" and model == "openvla":
    modules.append("prismatic.extern.hf.configuration_prismatic")
for name in modules:
    module = import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(prefix):
        raise SystemExit(f"{name} was imported from outside {prefix}: {module.__file__}")

if kind == "offline":
    from mani_skill.agents.robots.panda.panda import Panda
    from sapien import physx

    urdf = Path(Panda.urdf_path).resolve()
    if not urdf.is_relative_to(prefix) or not urdf.is_file():
        raise SystemExit(f"Missing bundled ManiSkill Panda robot asset: {urdf}")

    library = Path.home() / ".sapien" / "physx" / physx.version() / "libPhysXGpu_64.so"
    if not library.is_file() or library.stat().st_size == 0:
        raise SystemExit(f"Missing bundled PhysX GPU library: {library}")

from importlib.metadata import version
from packaging.version import Version
if Version(version("torch")) < Version("2.5.0"):
    raise SystemExit(f"Expected torch >= 2.5.0, found {version('torch')}.")
