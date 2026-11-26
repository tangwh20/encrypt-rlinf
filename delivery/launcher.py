"""Launch the bundled RLinf entry points with editable Hydra configuration."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ENTRIES = {
    "train": "examples/embodiment/train_embodied_agent.py",
    "train-async": "examples/embodiment/train_async.py",
    "train-offline": "examples/embodiment/train_offline_rl.py",
    "train-realworld": "examples/embodiment/train_embodied_agent.py",
    "train-realworld-async": "examples/embodiment/train_async.py",
    "eval-realworld-teleop": "examples/reward/eval_realworld_teleop.py",
    "eval": "evaluations/eval_embodied_agent.py",
    "deploy": "evaluations/eval_embodied_agent.py",
    "eval-mani-ood": "evaluations/eval_embodied_agent.py",
    "eval-diffusion": "evaluations/eval_embodied_agent.py",
    "train-vla-sft": "examples/sft/train_vla_sft.py",
    "train-vlm-sft": "examples/sft/train_vlm_sft.py",
    "train-reward": "examples/reward/train_reward_model.py",
    "train-reasoning-math": "examples/reasoning/main_grpo.py",
    "train-reasoning-vqa": "examples/reasoning/main_grpo.py",
    "train-searchr1": "examples/agent/searchr1/train.py",
    "eval-searchr1": "examples/agent/searchr1/eval.py",
    "train-wideseek": "examples/agent/wideseek_r1/train.py",
    "eval-wideseek": "examples/agent/wideseek_r1/eval.py",
    "train-diffusion": "examples/embodiment/train_embodied_agent.py",
    "train-rstar2": "examples/agent/rstar2/main_rstar2.py",
    "train-coding": "examples/agent/coding_online_rl/main_coding_online_rl.py",
    "train-coding-judge": "examples/agent/coding_online_rl/main_coding_rl_llm_judge.py",
    "train-calc-x": "examples/agent/agentlightning/calc_x/main.py",
    "eval-calc-x": "examples/agent/agentlightning/calc_x/main.py",
    "train-cfg-rl": "examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py",
    "train-recap": "examples/offline_rl/advantage_labeling/recap/train_value.py",
    "train-steam": "examples/offline_rl/advantage_labeling/steam/train_steam.py",
}

DEFAULT_CONFIGS = {
    "train": "maniskill_ppo_openvlaoft",
    "train-async": "maniskill_sac_mlp_async",
    "train-offline": "d4rl_iql_mujoco",
    "train-realworld": "realworld_dummy_franka_sac_cnn",
    "train-realworld-async": "realworld_peginsertion_rlpd_cnn_async",
    "eval-realworld-teleop": "realworld_teleop",
    "eval": "libero_spatial_starvla_eval",
    "deploy": "realworld_eval",
    "eval-mani-ood": "maniskill_ood_openvlaoft_eval",
    "eval-diffusion": "wan22_ti2v_5b_nft_video_ocr",
    "train-vla-sft": "libero_sft_openpi",
    "train-vlm-sft": "qwen2_5_vl_sft_vlm",
    "train-reward": "reward_training",
    "train-reasoning-math": "qwen2.5-1.5b-grpo-megatron",
    "train-reasoning-vqa": "qwen2.5-vl-3b-grpo-fsdp",
    "train-searchr1": "train_qwen2.5",
    "eval-searchr1": "eval_qwen2.5",
    "train-wideseek": "train_qwen3_hybrid",
    "eval-wideseek": "eval_qwen3_widesearch",
    "train-diffusion": "sd3_nft_ocr",
    "train-rstar2": "rstar2-qwen2.5-1.5b-megatron",
    "train-coding": "qwen2.5-1.5b-ppo",
    "train-coding-judge": "qwen2.5-1.5b-grpo-llm_judge",
    "train-calc-x": "qwen2.5-1.5b-enginehttp-multiturn",
    "eval-calc-x": "qwen2.5-1.5b-enginehttp-multiturn_eval",
    "train-cfg-rl": "cfg_rl_openpi",
    "train-recap": "recap_value_model_sft",
    "train-steam": "steam_value_model_sft",
}

SCRIPT_ROUTES = {
    "examples/embodiment/run_embodiment.sh": "train",
    "examples/embodiment/run_async.sh": "train-async",
    "examples/embodiment/run_offline_rl.sh": "train-offline",
    "examples/embodiment/run_realworld.sh": "train-realworld",
    "examples/embodiment/run_realworld_async.sh": "train-realworld-async",
    "evaluations/run_eval.sh": "eval",
    "examples/reward/run_realworld_teleop.sh": "eval-realworld-teleop",
    "examples/sft/run_vla_sft.sh": "train-vla-sft",
    "examples/sft/run_vlm_sft.sh": "train-vlm-sft",
    "examples/reward/run_reward_training.sh": "train-reward",
    "examples/reasoning/run_main_grpo_math.sh": "train-reasoning-math",
    "examples/reasoning/run_main_grpo_vqa.sh": "train-reasoning-vqa",
    "examples/agent/searchr1/run_train.sh": "train-searchr1",
    "examples/agent/searchr1/run_eval.sh": "eval-searchr1",
    "examples/agent/wideseek_r1/run_train.sh": "train-wideseek",
    "examples/agent/wideseek_r1/run_eval.sh": "eval-wideseek",
    "examples/diffusion/run_diffusion.sh": "train-diffusion",
    "examples/diffusion/eval_diffusion.sh": "eval-diffusion",
    "examples/agent/rstar2/run_rstar2.sh": "train-rstar2",
    "examples/agent/coding_online_rl/run_main_coding_online_rl.sh": "train-coding",
    "examples/agent/coding_online_rl/run_main_coding_rl_llm_judge.sh": "train-coding-judge",
    "examples/agent/agentlightning/calc_x/run_calc_x.sh": "train-calc-x",
    "examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh": "train-cfg-rl",
    "examples/offline_rl/advantage_labeling/recap/run_value_sft.sh": "train-recap",
    "examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh": "train-steam",
}

MANI_OOD_TEST_ENVS = (
    "VisionImage",
    "VisionTexture03",
    "VisionTexture05",
    "VisionWhole03",
    "VisionWhole05",
    "Carrot",
    "Plate",
    "Instruct",
    "MultiCarrot",
    "MultiPlate",
    "Position",
    "EEPose",
    "PositionChangeTo",
)
MANI_OOD_TRAIN_ENVS = ("Carrot", "MultiCarrot", "MultiPlate")


def public_scripts() -> list[str]:
    """List the upstream repository-relative shell selectors."""
    return list(SCRIPT_ROUTES)


def short_command(script: Path) -> str:
    """Give each upstream route a short, unambiguous command."""
    command = script.stem.removeprefix("run_")
    if command in {"train", "eval"} and script.parent != Path("evaluations"):
        return f"{script.parent.name}/{command}"
    return command


def public_commands() -> list[str]:
    """Show each short command alongside its upstream script in the ELF help."""
    lines = [f"  {short_command(Path(script)):<26} {script}" for script in SCRIPT_ROUTES]
    lines.append("  deploy                     evaluations/run_eval.sh (real-world policy deployment)")
    return lines


def resolve_script(name: str) -> str:
    """Accept public short commands, upstream paths and legacy delivery modes."""
    selector = name.removeprefix("./").removesuffix(".sh")
    if selector in ENTRIES:
        return selector
    directory, _, basename = selector.rpartition("/")
    command = basename.removeprefix("run_")
    normalized = f"{directory}/{command}" if directory else command
    matches = []
    for route, mode in SCRIPT_ROUTES.items():
        script = Path(route)
        stem = script.stem.removeprefix("run_")
        if normalized in (short_command(script), (script.parent / stem).as_posix()):
            return mode
        if not directory and command == stem:
            matches.append(route)
    if len(matches) == 1:
        return SCRIPT_ROUTES[matches[0]]
    if matches:
        raise ValueError(f"Ambiguous script {name!r}. Use one of: {', '.join(matches)}")
    raise ValueError(f"Unknown script {name!r}. Use --help to list available commands.")


def config_roots(app: Path, mode: str) -> list[Path]:
    """Return the bundled Hydra directories for the mode, default first."""
    embodied_config = app / "examples" / "embodiment" / "config"
    if mode == "deploy":
        return [app / "evaluations" / "realworld", embodied_config]
    if mode == "eval":
        evaluations = app / "evaluations"
        default = evaluations / "libero"
        others = sorted(
            directory
            for directory in evaluations.glob("*")
            if directory.is_dir() and directory != default
        )
        return [default, *others, embodied_config]
    if mode == "eval-mani-ood":
        return [app / "evaluations" / "maniskill", embodied_config]
    if mode.startswith("train-reasoning-"):
        return [app / "examples" / "reasoning" / "config" / mode.rsplit("-", 1)[-1]]
    if mode in {"train-diffusion", "eval-diffusion"}:
        return [app / "examples" / "diffusion" / "config"]
    if mode in {"train-cfg-rl", "train-recap", "train-steam"}:
        return [app / "examples" / "offline_rl" / "config"]
    return [app / Path(ENTRIES[mode]).parent / "config"]


def run_local_evaluation() -> None:
    """Dispatch the composed job only if its topology fits the local Ray runtime."""
    import hydra

    @hydra.main(version_base="1.1", config_path=None)
    def run(config):
        if config.cluster.get("num_nodes", 1) > 1:
            raise SystemExit("This delivery runs evaluation and deployment on a single Ray node. "
                             "The selected configuration requires multiple nodes.")
        from evaluations.eval_embodied_agent import main as evaluate
        evaluate(config)

    run()


def entry_command(root: Path, mode: str) -> list[str]:
    """Start an upstream script or import its native extension by package name."""
    relative_entry = Path(ENTRIES[mode])
    path = root / relative_entry
    if path.is_file():
        return [sys.executable, str(path)]
    if relative_entry.as_posix() == "evaluations/eval_embodied_agent.py":
        return [sys.executable, "-c",
                "from _rlinf_launcher import run_local_evaluation; run_local_evaluation()"]


    # Hydra uses the real module name in native deliveries. The absolute config
    # path supplied by main() keeps lookup independent of the loader's format.
    module_name = ".".join(relative_entry.with_suffix("").parts)
    return [
        sys.executable,
        "-c",
        "import importlib, pathlib, sys; "
        "module_name = sys.argv.pop(1); "
        "sys.argv[0] = sys.argv.pop(1); "
        "sys.path = [str(pathlib.Path(sys.argv[0]).parent), *filter(None, sys.path)]; "
        "importlib.import_module(module_name).main()",
        module_name,
        str(path),
    ]


def set_environment(root: Path, mode: str, benchmark: str) -> None:
    """Set bundle paths and the upstream simulation defaults for the driver."""
    os.environ["REPO_PATH"] = str(root)
    # Some upstream families use EMBODIED_PATH for their own directory.
    family = {
        "train-vla-sft": "sft",
        "train-reward": "reward",
    }.get(mode, "embodiment")
    os.environ["EMBODIED_PATH"] = str(root / "examples" / family)
    os.environ["EVALUATIONS_PATH"] = str(root / "evaluations")
    os.environ["DIFFUSION_PATH"] = str(root / "examples" / "diffusion")
    os.environ["VLM_PATH"] = str(root / "examples" / "sft")
    os.environ["REWARD_PATH"] = str(root / "examples" / "reward")
    paths = [str(root), str(root / "examples")]
    for key in (
        "ROBOTWIN_PATH",
        "DREAMZERO_PATH",
        "MEGATRON_PATH",
        "LIBERO_REPO_PATH",
        "PYTHONPATH",
    ):
        if value := os.environ.get(key):
            paths.append(value)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if mode in {"eval", "deploy", "eval-diffusion", "eval-mani-ood", "train-realworld-async"}:
        os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    if mode.startswith("train-coding"):
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    if mode in {"train-diffusion", "eval-diffusion", "train-steam"}:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if mode.endswith("wideseek"):
        os.environ.setdefault("RAY_DEBUG", "1")
    if mode.endswith("calc-x") and (conda_prefix := os.environ.get("CONDA_PREFIX")):
        lib_paths = [str(Path(conda_prefix) / "lib")]
        if current_paths := os.environ.get("LD_LIBRARY_PATH"):
            lib_paths.append(current_paths)
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_paths)
    if any(
        task in mode
        for task in ("reasoning", "searchr1", "wideseek", "vlm-sft", "rstar2", "coding", "calc-x")
    ):
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        os.environ.setdefault("RAY_DEDUP_LOGS", "0")
        return
    if benchmark == "realworld" or mode in {
        "deploy",
        "train-realworld",
        "train-realworld-async",
        "eval-realworld-teleop",
    }:
        return
    renderer = "osmesa" if mode in {"eval", "eval-mani-ood"} else "egl"
    os.environ.setdefault("MUJOCO_GL", renderer)
    os.environ.setdefault("PYOPENGL_PLATFORM", renderer)
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    if mode == "train":
        os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
        os.environ.setdefault("POLARIS_DATA_PATH", "/path/to/dataset/PolaRiS-Hub")
        os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "1")
        os.environ.setdefault("OMNIGIBSON_DEBUG", "0")
    if data := os.environ.get("OMNIGIBSON_DATA_PATH"):
        data_path = Path(data)
        os.environ.setdefault(
            "OMNIGIBSON_DATASET_PATH", str(data_path / "behavior-1k-assets")
        )
        os.environ.setdefault("OMNIGIBSON_KEY_PATH", str(data_path / "omnigibson.key"))
        os.environ.setdefault(
            "OMNIGIBSON_ASSET_PATH", str(data_path / "omnigibson-robot-assets")
        )
    if isaac := os.environ.get("ISAAC_PATH"):
        os.environ.setdefault("EXP_PATH", str(Path(isaac) / "apps"))
        os.environ.setdefault("CARB_APP_PATH", str(Path(isaac) / "kit"))
    if benchmark == "robocasa365" and mode == "eval":
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    if benchmark == "libero":
        os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
        os.environ.setdefault("LIBERO_TYPE", "standard")
        if os.environ["LIBERO_TYPE"] == "pro":
            os.environ.setdefault("LIBERO_PERTURBATION", "all")
        elif os.environ["LIBERO_TYPE"] == "plus":
            os.environ.setdefault("LIBERO_SUFFIX", "all")


def run_mani_ood(root: Path, command: list[str], hydra_args: list[str]) -> None:
    """Run the same set of ManiSkill OOD cases as the upstream batch launcher."""
    environment_keys = {"EVAL_NAME", "CKPT_PATH", "TOTAL_NUM_ENVS", "EVAL_ROLLOUT_EPOCH"}
    missing = sorted(key for key in environment_keys if not os.environ.get(key))
    if missing:
        raise SystemExit("ManiSkill OOD requires environment variables " + ", ".join(missing))
    logs = root / "logs" / "eval" / os.environ["EVAL_NAME"]
    common = [
        f'runner.ckpt_path={os.environ["CKPT_PATH"]}',
        f'env.eval.rollout_epoch={os.environ["EVAL_ROLLOUT_EPOCH"]}',
        f'env.eval.total_num_envs={os.environ["TOTAL_NUM_ENVS"]}',
    ]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for object_set, variants in (
        ("test", MANI_OOD_TEST_ENVS),
        ("train", MANI_OOD_TRAIN_ENVS),
    ):
        for variant in variants:
            env_id = f"PutOnPlateInScene25{variant}-v1"
            log_path = logs / f"{stamp}-{env_id}-{object_set}"
            case = [
                f"env.eval.init_params.id={env_id}",
                f"env.eval.init_params.obj_set={object_set}",
                f"runner.logger.log_path={log_path}",
            ]
            subprocess.run([*command, *common, *case, *hydra_args], check=True)


def main() -> None:
    """Start a bundled script in its normal Python launch context."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Run training, evaluation or real-world policy deployment.",
        usage="%(prog)s MODE [Hydra arguments]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("mode", metavar="MODE")
    parser.epilog = (
        "Modes\n"
        "  Embodied       train, train-async, train-offline, eval, deploy, eval-mani-ood\n"
        "                 train-realworld, train-realworld-async, eval-realworld-teleop\n"
        "  Supervised     train-vla-sft, train-vlm-sft, train-reward\n"
        "  Reasoning      train-reasoning-math, train-reasoning-vqa\n"
        "  Agents         train-searchr1, eval-searchr1, train-wideseek, eval-wideseek,\n"
        "                 train-rstar2, train-coding, train-coding-judge,\n"
        "                 train-calc-x, eval-calc-x\n"
        "  Specialized    train-diffusion, eval-diffusion, train-cfg-rl, train-recap, train-steam\n\n"
        "Use eval --benchmark NAME to select a bundled evaluation directory.\n"
        "The eval-mani-ood batch requires EVAL_NAME, CKPT_PATH, TOTAL_NUM_ENVS,\n"
        "and EVAL_ROLLOUT_EPOCH.\n"
        "Use --config-path DIR and --config-name NAME for external YAML.\n"
        "Pass --help after a mode to see that entry point's Hydra options."
    )
    mode_argument = parser.parse_args(sys.argv[1:2]).mode
    try:
        mode = resolve_script(mode_argument)
    except ValueError as error:
        parser.error(str(error))
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument("--benchmark")
    config_parser.add_argument("-cp", "--config-path")
    config_parser.add_argument("-cn", "--config-name")
    config_args, hydra_args = config_parser.parse_known_args(sys.argv[2:])
    benchmark = config_args.benchmark
    config_path = config_args.config_path
    if benchmark and mode != "eval":
        parser.error("--benchmark is available for eval")
    if benchmark == "mani-ood":
        mode = "eval-mani-ood"
        benchmark = None
    if benchmark and config_path:
        parser.error("choose --benchmark or --config-path")
    config_name = config_args.config_name or DEFAULT_CONFIGS[mode]
    config_key = Path(config_name).name.removesuffix(".yaml")
    if not benchmark:
        benchmark = config_key.split("_", 1)[0].split("-", 1)[0]

    if not config_path:
        bundled_roots = config_roots(root, mode)
        config_dir = bundled_roots[0]
        if mode == "eval-mani-ood" and not (config_dir / f"{config_key}.yaml").is_file():
            config_dir = bundled_roots[1]
        if mode == "eval":
            eval_root = root / "evaluations"
            if config_args.benchmark:
                config_dir = eval_root / benchmark
                if Path(benchmark).name != benchmark or benchmark.startswith("."):
                    parser.error(f"unknown evaluation benchmark: {benchmark}")
            else:
                matches = sorted(
                    directory
                    for directory in bundled_roots[:-1]
                    if (directory / f"{config_key}.yaml").is_file()
                )
                if len(matches) > 1:
                    parser.error(
                        f"pass --benchmark to choose among evaluation configs named {config_name}"
                    )
                if matches:
                    config_dir = matches[0]
                    benchmark = config_dir.name
            if not (config_dir / f"{config_key}.yaml").is_file():
                config_dir = bundled_roots[-1]
        config_path = str(config_dir)
    elif "://" not in config_path:
        config_path = str(Path(config_path).expanduser().resolve())

    set_environment(root, mode, benchmark)
    if ENTRIES[mode] == "evaluations/eval_embodied_agent.py" and "://" not in config_path:
        # Some evaluation primaries reference shared groups without declaring a search path.
        # Respect an explicit path both in the primary YAML and in the caller's overrides.
        from omegaconf import OmegaConf
        primary = Path(config_path) / (config_name + ".yaml")
        if primary.is_file() and not any(arg.lstrip("+~").split("=", 1)[0] == "hydra.searchpath"
                                         for arg in hydra_args):
            cfg = OmegaConf.load(primary)
            if OmegaConf.select(cfg, "hydra.searchpath") is None:
                common = root / "examples" / "embodiment" / "config"
                uri = json.dumps("file://" + str(common), ensure_ascii=False)
                hydra_args.append(f"hydra.searchpath=[{uri}]")
    command = [
        *entry_command(root, mode),
        "--config-path",
        config_path,
        "--config-name",
        config_name,
    ]
    help_requested = {"--help", "-h", "--hydra-help"}.intersection(hydra_args)
    if mode == "eval-mani-ood" and not help_requested:
        run_mani_ood(root, command, hydra_args)
        return
    os.execv(sys.executable, [*command, *hydra_args])


if __name__ == "__main__":
    main()
