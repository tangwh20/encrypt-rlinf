"""Generic CLI glue for the bundled interpreter and compiled RLinf launcher."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


def select_config(roots, release, selector, script):
    """Resolve a configuration within the namespace of the chosen script."""
    filename = Path(selector if selector.endswith(".yaml") else selector + ".yaml")
    external = release / filename
    if not filename.is_absolute() and "/" in selector and filename.is_file():
        external = filename.resolve()
    if external.is_file():
        fallback = next((root for root in roots if (root / filename.name).is_file()), roots[-1])
        return external.parent.resolve(), external.stem, fallback
    if not filename.is_absolute() and ".." not in filename.parts and not selector.startswith("./"):
        for root in roots:
            if (root / filename).is_file():
                return root, filename.with_suffix("").as_posix(), root
    raise SystemExit(f"No external YAML or registered config {selector!r} for {script!r}.")


def hydra_arguments(args, defaults):
    # Hydra parses overrides as one positional list. Keep options together,
    # including option values that may themselves contain '=' (e.g. paths).
    value_options = {"--cfg", "-c", "--package", "-p", "--config-path", "-cp",
                     "--config-name", "-cn", "--config-dir", "-cd", "--experimental-rerun"}
    options = []
    overrides = list(defaults)
    tokens = iter(args)
    for token in tokens:
        if token in value_options:
            options.append(token)
            value = next(tokens, None)
            if value is None:
                raise SystemExit(f"{token} requires a value.")
            options.append(value)
        elif not token.startswith("-") and ("=" in token or token.startswith("~")):
            overrides.append(token)
        else:
            options.append(token)
    return [*options, *overrides]


def main():
    installed = Path(__file__).resolve().parent
    release = Path(os.environ["RLINF_RELEASE_ROOT"])
    app = installed / "app"
    sys.path.insert(0, str(app))
    from _rlinf_launcher import config_roots, resolve_script

    script, selector, *args = sys.argv[1:]
    try:
        mode = resolve_script(script)
    except ValueError as exc:
        raise SystemExit(str(exc))
    config_dir, config_name, packaged = select_config(config_roots(app, mode), release, selector, script)
    if Path(sys.base_prefix).resolve() == installed / "setup" / "seed":
        print("Install the task environment with ./install.sh first. See README.md.", file=sys.stderr)
        return 2
    env = os.environ.copy()
    if mode in {"train", "train-async"} and args and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args[0]):
        env["ROBOT_PLATFORM"] = args.pop(0)
    if args[:1] == ["--"]:
        args.pop(0)
    profile = Path(sys.base_prefix).resolve().parent
    home = profile / "home"
    cache = release / ".rlinf-state" / "cache"
    home.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    env.update(
        HOME=str(home), XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        HF_HUB_CACHE=str(cache / "huggingface" / "hub"),
        HUGGINGFACE_HUB_CACHE=str(cache / "huggingface" / "hub"),
        HF_DATASETS_CACHE=str(cache / "huggingface" / "datasets"),
        HF_MODULES_CACHE=str(cache / "huggingface" / "modules"),
        TRANSFORMERS_CACHE=str(cache / "huggingface" / "hub"),
        TORCH_HOME=str(cache / "torch"), TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "torchinductor"),
        PYTHONPATH=str(installed / "app"),
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        RLINF_CODE_WORKING_DIR="0", RAY_ADDRESS="local", RAY_USAGE_STATS_ENABLED="0",
        HYDRA_FULL_ERROR="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
    )
    for key in ("PYTHONHOME", "VIRTUAL_ENV", "MEGATRON_PATH", "ROBOTWIN_PATH",
                "LIBERO_REPO_PATH", "DREAMZERO_PATH"):
        env.pop(key, None)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    tools = Path(sys.base_prefix) / "toolchain" / "bin"
    if (tools / "gcc").is_file():
        env.update(PATH=f"{tools}:{env['PATH']}", CC="gcc", CXX="g++")
    env["MS_ASSET_DIR"] = str(home / ".maniskill")
    log_root = release / "logs"
    log_root.mkdir(exist_ok=True)
    log = Path(tempfile.mkdtemp(prefix=f"{time.strftime('%Y%m%d-%H%M%S')}-{Path(config_name).name}-", dir=log_root))
    # Resolve the log path as a string via OmegaConf, including spaces in the deployment path.
    env["RLINF_RELEASE_LOG_DIR"] = str(log)
    overrides = ["runner.logger.log_path=${oc.env:RLINF_RELEASE_LOG_DIR}"]
    if env.get("NODES") not in (None, "", "-2", "1"):
        raise SystemExit("This standalone launcher runs a single local Ray node.")
    for variable, field in (("STEPS", "runner.max_steps"), ("SAVE_INTER", "runner.save_interval")):
        value = env.get(variable)
        if mode in {"train", "eval"} and value and value != "-2":
            overrides.append(f"{field}={int(value)}")
    if env.get("GPU") is not None:
        gpu = env["GPU"]
        if not re.fullmatch(r"0|[1-9][0-9]*", gpu):
            raise SystemExit("GPU must be a nonnegative physical GPU number")
        env["CUDA_VISIBLE_DEVICES"] = gpu
        if mode == "eval":
            overrides.extend(["~cluster.component_placement",
                              f"+cluster.component_placement={{env\\,rollout:{gpu}}}"])
        elif mode == "train":
            overrides.append(f"cluster.component_placement={{actor\\,env\\,rollout:{gpu}}}")
    cmd = [sys.executable, str(installed / "app" / "launcher.py"), mode,
           "--config-path", str(config_dir), "--config-name", config_name]
    if config_dir != packaged:
        # External YAML may inherit a sibling primary or group without declaring a search path.
        cmd.extend(["--config-dir", str(packaged)])
    # Hydra's own help composes only its internal config, without a runner or cluster.
    cmd.extend(hydra_arguments(args, [] if "--hydra-help" in args else overrides))
    # Ray socket addresses must fit the OS AF_UNIX limit, irrespective of release path length.
    with tempfile.TemporaryDirectory(prefix="rf.", dir=env.get("RLINF_RAY_TMP_BASE", "/tmp")) as tmp:
        env.update(RAY_TMPDIR=tmp, TMPDIR=tmp)
        # Ray's Unix worker launcher passes sys.executable to bash -c unquoted.
        # Only the alias lives in scratch. Its target is the installed local prefix.
        interpreter_root = Path(tmp) / "python-env"
        interpreter_root.symlink_to(sys.base_prefix, target_is_directory=True)
        cmd[0] = str(interpreter_root / "bin" / "python")
        if env.get("LIBERO_TYPE", "standard") == "standard":
            # LIBERO writes absolute paths computed from its installed package.
            # Give each invocation fresh paths after a delivery directory is moved.
            libero_config = Path(tmp) / "libero"
            libero_config.mkdir()
            env["LIBERO_CONFIG_PATH"] = str(libero_config)
        print(f"RLinf {script}: {config_name}.yaml  |  logs: {log}", file=sys.stderr, flush=True)
        with (log / "console.log").open("wb") as stream:
            with subprocess.Popen(cmd, cwd=release, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT) as child:
                for line in child.stdout:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                    stream.write(line)
                status = child.wait()
        if status and (Path(tmp) / "session_latest/logs").is_dir():
            shutil.copytree(Path(tmp) / "session_latest/logs", log / "ray-logs")
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    sys.exit(main())
