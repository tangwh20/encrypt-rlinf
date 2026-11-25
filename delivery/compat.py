"""Apply source-inspection compatibility changes to the private build stage."""

import ast
from pathlib import Path


def prepare_compatibility(staged_source: Path) -> None:
    """Keep activation and worker imports callable after source protection."""
    path = staged_source / "rlinf" / "config.py"
    source = path.read_text(encoding="utf-8")
    expected_decorators = {
        "gelu_impl": "torch.jit.script",
        "erf_gelu": "torch.jit.script",
        "squared_relu": "jit_fuser",
    }
    removed_lines = set()
    for function in ast.parse(source).body:
        if not isinstance(function, ast.FunctionDef):
            continue
        expected = expected_decorators.get(function.name)
        if expected is None:
            continue
        decorators = [
            decorator
            for decorator in function.decorator_list
            if ast.unparse(decorator) == expected
        ]
        if len(decorators) != 1:
            raise ValueError(f"Review protection compatibility for {path}:{function.name}")
        decorator = decorators[0]
        removed_lines.update(range(decorator.lineno, decorator.end_lineno + 1))
        del expected_decorators[function.name]
    if expected_decorators:
        missing = ", ".join(expected_decorators)
        raise ValueError(
            f"Review protection compatibility for missing functions in {path}: {missing}"
        )

    # Preserve line numbers for other code in this module.
    path.write_text(
        "".join(
            "\n" if number in removed_lines else line
            for number, line in enumerate(source.splitlines(keepends=True), start=1)
        ),
        encoding="utf-8",
    )
    prepare_ray_compatibility(staged_source)
    prepare_scheduler_compatibility(staged_source)


def prepare_ray_compatibility(staged_source: Path) -> None:
    """Load importable actors by module on nodes with the bundle installed."""
    scheduler = staged_source / "rlinf" / "scheduler"
    cluster = scheduler / "cluster" / "cluster.py"
    if not cluster.is_file():
        return

    # Ray's default export disassembles class methods to identify collisions.
    # The job search path lets Ray import protected actors on each worker.
    direct_init = "                logging_level=Cluster.LOGGING_LEVEL,\n"
    direct_with_job = (
        direct_init + "                job_config=Cluster._protected_job_config(),\n"
    )
    kwargs_level = '                "logging_level": Cluster.LOGGING_LEVEL,\n'
    kwargs_with_job = (
        kwargs_level + '                "job_config": self._protected_job_config(),\n'
    )
    helper_anchor = "    def _start_local_ray(self, ray_init_kwargs: dict[str, Any]):\n"
    helper = (
        "    @staticmethod\n"
        "    def _protected_job_config() -> JobConfig:\n"
        "        bundle = str(Path(__file__).resolve().parents[3])\n"
        "        return JobConfig(code_search_path=[bundle])\n\n"
    )
    sync_guard_anchor = '        lowered = raw.lower()\n'
    sync_guard = (
        sync_guard_anchor
        + '        if raw and lowered not in {"0", "false", "no", "off"}:\n'
        '            raise ValueError(\n'
        '                f"Set {env_key}=0 and install the protected bundle at the same "\n'
        '                "path on every Ray node."\n'
        '            )\n'
    )
    source = cluster.read_text(encoding="utf-8")
    anchors = (
        (
            "from ray.actor import ActorHandle\n",
            "from ray.actor import ActorHandle\nfrom ray.job_config import JobConfig\n",
            1,
        ),
        (kwargs_level, kwargs_with_job, 2),
        (direct_init, direct_with_job, 1),
        (helper_anchor, helper + helper_anchor, 1),
        (sync_guard_anchor, sync_guard, 1),
    )
    for old, new, count in anchors:
        if source.count(old) != count:
            raise ValueError(f"Review Ray protection compatibility for {cluster}")
        source = source.replace(old, new)
    cluster.write_text(source, encoding="utf-8")

    for relative in ("manager/manager.py", "worker/worker.py"):
        path = scheduler / relative
        source = path.read_text(encoding="utf-8")
        if source.count(direct_init) != 1:
            raise ValueError(f"Review Ray protection compatibility for {path}")
        path.write_text(source.replace(direct_init, direct_with_job), encoding="utf-8")


def prepare_scheduler_compatibility(staged_source: Path) -> None:
    """Keep compiled LambdaLR closure state compatible with the source build."""
    fsdp = staged_source / "rlinf" / "hybrid_engines" / "fsdp"
    utils = fsdp / "utils.py"
    strategy = fsdp / "strategy" / "base.py"

    replacements = {
        utils: (
            (
                "from torch.optim import Optimizer\n",
                "from torch.optim import Optimizer\n"
                "from torch.optim.lr_scheduler import LambdaLR\n",
                1,
            ),
            (
                "def get_lr_scheduler(\n",
                """class _CompiledClosureLambdaLR(LambdaLR):
    def state_dict(self):
        state = super().state_dict()
        # Upstream's stateless closures are Python functions and save None for
        # each group. Cython functions otherwise save an empty attribute dict.
        state["lr_lambdas"] = [None] * len(self.lr_lambdas)
        return state


def get_lr_scheduler(
""",
                1,
            ),
            (
                "        from torch.optim.lr_scheduler import LambdaLR\n\n",
                "",
                3,
            ),
            (
                "return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)",
                "return _CompiledClosureLambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)",
                3,
            ),
        ),
        strategy: (
            (
                "from torch.distributed.checkpoint.state_dict import (\n",
                "from torch.distributed.checkpoint import DefaultLoadPlanner\n"
                "from torch.distributed.checkpoint.state_dict import (\n",
                1,
            ),
            (
                "from rlinf.hybrid_engines.fsdp.utils import (\n    FSDPVersion,\n",
                "from rlinf.hybrid_engines.fsdp.utils import (\n"
                "    FSDPVersion,\n"
                "    _CompiledClosureLambdaLR,\n",
                1,
            ),
            (
                "class FSDPStrategyBase(ABC):\n",
                """class _SchedulerLoadPlanner(DefaultLoadPlanner):
    def __init__(self, lr_schedulers):
        super().__init__()
        self.closure_indices = [
            index
            for index, scheduler in enumerate(lr_schedulers)
            if isinstance(scheduler, _CompiledClosureLambdaLR)
        ]

    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        if metadata is not None and self.closure_indices:
            checkpoint_keys = metadata.state_dict_metadata
            schedulers = state_dict["fsdp_checkpoint"]["lr_schedulers"]
            for index in self.closure_indices:
                prefix = f"fsdp_checkpoint.lr_schedulers.{index}."
                key = prefix + "lr_lambdas"
                if (
                    prefix + "base_lrs" in checkpoint_keys
                    and key not in checkpoint_keys
                    and not any(name.startswith(key + ".") for name in checkpoint_keys)
                ):
                    # Match the legacy shape whose empty dictionaries disappear
                    # during DCP flattening. Loading them restores no attributes.
                    state = schedulers[index]
                    state["lr_lambdas"] = [{} for _ in state["lr_lambdas"]]
        super().set_up_planner(state_dict, metadata, is_coordinator)


class FSDPStrategyBase(ABC):
""",
                1,
            ),
            (
                '                    {"fsdp_checkpoint": training_state},\n'
                "                    checkpoint_id=dcp_load_path,\n",
                '                    {"fsdp_checkpoint": training_state},\n'
                "                    checkpoint_id=dcp_load_path,\n"
                "                    planner=_SchedulerLoadPlanner(training_state.lr_schedulers),\n",
                1,
            ),
        ),
    }
    for path, changes in replacements.items():
        source = path.read_text(encoding="utf-8")
        for old, new, count in changes:
            if source.count(old) != count:
                raise ValueError(f"Review scheduler checkpoint compatibility for {path}")
            source = source.replace(old, new)
        path.write_text(source, encoding="utf-8")
