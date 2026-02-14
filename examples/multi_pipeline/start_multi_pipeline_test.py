"""
SchedRL multi-pipeline example (ENG-123).

This ports the fork reference configs (`pipeline1_sokoban_grpo.yaml`, `pipeline2_sokoban_grpo.yaml`) and provides a
driver that runs 1+ pipelines concurrently under the SchedRL control plane.

Usage (from repo root):
  python third_party/ROLL/examples/multi_pipeline/start_multi_pipeline_test.py --config_name pipeline1_sokoban_grpo
  python third_party/ROLL/examples/multi_pipeline/start_multi_pipeline_test.py --config_name pipeline1_sokoban_grpo,pipeline2_sokoban_grpo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import ray
from dacite import from_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def _repo_root() -> Path:
    # .../third_party/ROLL/examples/multi_pipeline/start_multi_pipeline_test.py -> repo root
    return Path(__file__).resolve().parents[4]


def _ensure_import_paths() -> Path:
    repo_root = _repo_root()
    roll_root = (repo_root / "third_party" / "ROLL").resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(roll_root))
    return repo_root


def _resolve_hydra_config_path(*, roll_root: Path, arg_config_path: str) -> tuple[str, Path]:
    script_dir = Path(__file__).resolve().parent
    examples_dir = (roll_root / "examples").resolve()
    config_path = Path(arg_config_path)

    if config_path.is_absolute():
        return str(config_path), config_path

    script_relative_dir = (script_dir / config_path).resolve()
    if script_relative_dir.is_dir():
        return str(config_path), script_relative_dir

    examples_relative_dir = (examples_dir / config_path).resolve()
    if examples_relative_dir.is_dir():
        hydra_config_path = os.path.relpath(examples_relative_dir, script_dir)
        return hydra_config_path, examples_relative_dir

    roll_relative_dir = (roll_root / config_path).resolve()
    if roll_relative_dir.is_dir():
        hydra_config_path = os.path.relpath(roll_relative_dir, script_dir)
        return hydra_config_path, roll_relative_dir

    raise FileNotFoundError(
        f"Config directory not found. Received --config_path={arg_config_path!r} "
        f"(tried {script_relative_dir}, {examples_relative_dir}, {roll_relative_dir})"
    )


def _inject_system_envs(*, pipeline_config: Any, envs: Dict[str, str]) -> None:
    def _update_system_envs(obj: Any) -> None:
        if obj is None:
            return
        system_envs = getattr(obj, "system_envs", None)
        if system_envs is None:
            setattr(obj, "system_envs", dict(envs))
            return
        if not isinstance(system_envs, dict):
            raise RuntimeError(f"Expected system_envs to be dict, got {type(system_envs).__name__}")
        system_envs.update(envs)

    _update_system_envs(getattr(pipeline_config, "actor_train", None))
    _update_system_envs(getattr(pipeline_config, "actor_infer", None))
    _update_system_envs(getattr(pipeline_config, "reference", None))
    _update_system_envs(getattr(pipeline_config, "critic", None))
    _update_system_envs(getattr(pipeline_config, "reward", None))
    _update_system_envs(getattr(pipeline_config, "train_env_manager", None))
    _update_system_envs(getattr(pipeline_config, "val_env_manager", None))


def _cluster_registry_inputs(*, pipeline_config: Any) -> tuple[Dict[str, int], Dict[str, List[int]]]:
    cluster_tp_configs: Dict[str, int] = {}
    cluster_device_mappings: Dict[str, List[int]] = {}

    for key in ("actor_train", "actor_infer", "reference", "critic", "reward"):
        cfg = getattr(pipeline_config, key, None)
        if cfg is None:
            continue
        mapping = getattr(cfg, "device_mapping", None)
        if mapping is None:
            continue
        cluster_device_mappings[key] = list(mapping)
        cluster_tp_configs[key] = int(getattr(cfg, "num_gpus_per_worker", 1))

    if "actor_infer" not in cluster_tp_configs:
        raise RuntimeError("pipeline_config must include actor_infer device_mapping for SchedRL mode")
    return cluster_tp_configs, cluster_device_mappings


def main() -> None:
    repo_root = _ensure_import_paths()
    roll_root = (repo_root / "third_party" / "ROLL").resolve()

    from roll.pipeline.agentic.agentic_config import AgenticConfig
    from roll.schedrl_adapter.adapter import SchedRLAdapter, _get_pipeline_namespace

    import schedrl

    parser = argparse.ArgumentParser(description="SchedRL multi-pipeline example")
    parser.add_argument(
        "--config_path",
        default="multi_pipeline",
        help="Path to config directory (relative to third_party/ROLL/examples/)",
    )
    parser.add_argument(
        "--config_name",
        default="pipeline1_sokoban_grpo",
        help="Comma-separated config file names (without .yaml)",
    )
    parser.add_argument(
        "--admit-delay-s",
        type=float,
        default=0.0,
        help="Seconds to sleep after admitting each pipeline (except the last).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        default=False,
        help="Print the fully resolved Hydra config to logs (can be very large).",
    )
    args = parser.parse_args()

    config_names = [name.strip() for name in args.config_name.split(",") if name.strip()]
    if not config_names:
        raise ValueError("--config_name must be non-empty")

    hydra_config_path, _ = _resolve_hydra_config_path(roll_root=roll_root, arg_config_path=args.config_path)
    GlobalHydra.instance().clear()
    initialize(config_path=hydra_config_path, job_name="schedrl_multi_pipeline", version_base=None)

    pipeline_configs: List[AgenticConfig] = []
    for idx, cn in enumerate(config_names, start=1):
        cfg = compose(config_name=cn)
        suffix = f"mp{idx}"
        if hasattr(cfg, "exp_name") and cfg.exp_name:
            cfg.exp_name = f"{cfg.exp_name}-{suffix}"
        else:
            cfg.exp_name = f"{cn}-{suffix}"

        for key in ("model_name", "base_dir", "log_dir", "profiler_output_dir"):
            if hasattr(cfg, key):
                value = getattr(cfg, key)
                if isinstance(value, str) and value:
                    setattr(cfg, key, f"{value}-{suffix}")

        if args.print_config or os.environ.get("ROLL_PRINT_CONFIG", "0") == "1":
            print(OmegaConf.to_yaml(cfg, resolve=True))

        pipeline_config = from_dict(
            data_class=AgenticConfig,
            data=OmegaConf.to_container(cfg, resolve=True),
        )
        pipeline_configs.append(pipeline_config)

    # Ensure SchedRL control plane is up (creates orchestrator + scheduler actors).
    orchestrator = schedrl.init(create_if_missing=True)
    if orchestrator is None:
        raise RuntimeError("schedrl.init returned None (expected orchestrator actor handle on rank 0)")

    AdapterActor = ray.remote(SchedRLAdapter)

    adapters = []
    coordinators = []
    run_refs = []

    admit_delay_s = float(args.admit_delay_s)

    pipeline_ids: List[str] = []
    for pipeline_config in pipeline_configs:
        pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote())
        pipeline_ids.append(str(pipeline_id))

    for i, (pipeline_id, pipeline_config) in enumerate(zip(pipeline_ids, pipeline_configs)):
        ray_namespace = _get_pipeline_namespace(str(pipeline_id))
        cluster_tp_configs, cluster_device_mappings = _cluster_registry_inputs(pipeline_config=pipeline_config)

        ray.get(
            orchestrator.register_pipeline.remote(
                pipeline_id=str(pipeline_id),
                ray_namespace=ray_namespace,
                cluster_tp_configs=cluster_tp_configs,
                cluster_device_mappings=cluster_device_mappings,
            )
        )
        ray.get(orchestrator.admit_pipeline.remote(pipeline_id=str(pipeline_id)))

        adapter = AdapterActor.options(
            name=f"schedrl:adapter:{pipeline_id}",
            namespace=ray_namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
        ).remote(
            pipeline_id=pipeline_id,
            pipeline_config=pipeline_config,
            cluster_tp_configs=cluster_tp_configs,
            cluster_device_mappings=cluster_device_mappings,
        )
        adapters.append(adapter)

        envs = ray.get(adapter.get_pipeline_env_vars.remote())
        _inject_system_envs(pipeline_config=pipeline_config, envs=envs)

        coordinator = ray.get(adapter.ensure_coordinator.remote())
        coordinators.append(coordinator)
        run_refs.append(coordinator.run.remote(pipeline_config=pipeline_config))

        if admit_delay_s > 0 and i < len(pipeline_ids) - 1:
            import time
            time.sleep(admit_delay_s)

    # Block until all pipelines complete (fail-fast if any crashes).
    ray.get(run_refs)


if __name__ == "__main__":
    main()
