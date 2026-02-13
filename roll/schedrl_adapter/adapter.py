from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _require_ray():
    try:
        import ray  # noqa: F401
    except Exception as e:
        raise RuntimeError("roll.schedrl_adapter requires ray") from e


def _get_pipeline_namespace(pipeline_id: str) -> str:
    return f"pipeline_{pipeline_id}_NS"


def _build_pipeline_env_vars(*, pipeline_id: str, ray_namespace: str) -> Dict[str, str]:
    _require_ray()
    import ray

    job_id = ray.get_runtime_context().get_job_id()
    scratch_root = f"/tmp/schedrl/{pipeline_id}/{job_id}"
    shared_root = "/tmp/schedrl/shared"

    env_vars = {
        "PIPELINE_ID": pipeline_id,
        "ROLL_RAY_NAMESPACE": ray_namespace,
        "SCHEDRL_CONTROL_PLANE": "schedrl",
        # Used by upstream ROLL shims to avoid taking down the job-global Ray cluster.
        "SCHEDRL_LIBRARY_MODE": "1",
        # Shared weights/cache (big, reusable).
        "HF_HOME": f"{shared_root}/hf",
        "HUGGINGFACE_HUB_CACHE": f"{shared_root}/hf/hub",
        "TRANSFORMERS_CACHE": f"{shared_root}/hf/transformers",
        "HF_DATASETS_CACHE": f"{shared_root}/hf/datasets",
        # Job/pipeline-scoped scratch (write-hot / collision-prone).
        "HUGGINGFACE_AUTOMAP_CACHE": f"{scratch_root}/hf/automap",
        "VLLM_CACHE_ROOT": f"{scratch_root}/vllm",
        "FLASHINFER_WORKSPACE_DIR": f"{scratch_root}/flashinfer",
    }
    return env_vars


def _validate_cpu_only_reward(*, pipeline_config: Any) -> None:
    reward_cfg = getattr(pipeline_config, "reward", None)
    if reward_cfg is None:
        return
    device_mapping = getattr(reward_cfg, "device_mapping", None)
    if device_mapping is None:
        return
    if isinstance(device_mapping, list) and len(device_mapping) == 0:
        return
    if isinstance(device_mapping, str) and device_mapping.strip() in {"", "[]"}:
        return
    # TODO(ENG-123): lift this restriction to support GPU reward clusters.
    raise RuntimeError("ENG-123 Phase 3 only supports CPU-only reward (reward.device_mapping must be empty/None).")


def _validate_vllm_sleep_level(*, pipeline_config: Any) -> None:
    actor_infer = getattr(pipeline_config, "actor_infer", None)
    if actor_infer is None:
        return
    strategy_args = getattr(actor_infer, "strategy_args", None)
    if strategy_args is None:
        return
    strategy_name = getattr(strategy_args, "strategy_name", None)
    if strategy_name != "vllm":
        return
    strategy_config = getattr(strategy_args, "strategy_config", None) or {}
    sleep_level = strategy_config.get("sleep_level", 1)
    if int(sleep_level) != 2:
        raise RuntimeError("ENG-123 Phase 3 requires actor_infer vLLM sleep_level=2 (drop model weights on offload).")


@dataclass(frozen=True, slots=True)
class PipelineRegistration:
    pipeline_id: str
    ray_namespace: str
    cluster_tp_configs: Dict[str, int]
    cluster_device_mappings: Dict[str, List[int]]


class SchedRLAdapter:
    """Per-pipeline adapter actor (ENG-123 Phase 3).

    Contract:
    - Does NOT forward progress reports (progress is emitted in ROLL GroupQueueManager.put()).
    - Exposes shrink/expand RPCs for the SchedRL scheduler (fail-fast).
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        pipeline_config: Any,
        cluster_tp_configs: Dict[str, int],
        cluster_device_mappings: Dict[str, List[int]],
    ):
        _require_ray()
        import ray

        from schedrl.protocol.request_id import validate_pipeline_id

        validate_pipeline_id(pipeline_id)
        self._pipeline_id = pipeline_id
        self._ray_namespace = _get_pipeline_namespace(pipeline_id)
        self._pipeline_env_vars = _build_pipeline_env_vars(pipeline_id=pipeline_id, ray_namespace=self._ray_namespace)

        _validate_cpu_only_reward(pipeline_config=pipeline_config)
        _validate_vllm_sleep_level(pipeline_config=pipeline_config)

        if not isinstance(cluster_tp_configs, dict) or not cluster_tp_configs:
            raise ValueError("cluster_tp_configs must be non-empty dict[str,int]")
        if not isinstance(cluster_device_mappings, dict) or not cluster_device_mappings:
            raise ValueError("cluster_device_mappings must be non-empty dict[str,list[int]]")
        if set(cluster_tp_configs.keys()) != set(cluster_device_mappings.keys()):
            raise ValueError("cluster_tp_configs and cluster_device_mappings must have identical keys")
        if "actor_infer" not in cluster_tp_configs:
            raise ValueError("cluster_tp_configs must include 'actor_infer'")

        self._registration = PipelineRegistration(
            pipeline_id=pipeline_id,
            ray_namespace=self._ray_namespace,
            cluster_tp_configs={k: int(v) for k, v in cluster_tp_configs.items()},
            cluster_device_mappings={k: list(v) for k, v in cluster_device_mappings.items()},
        )

        self._schedrl_orchestrator = ray.get_actor("schedrl:orchestrator", namespace="schedrl")
        self._schedrl_scheduler = ray.get_actor("schedrl:scheduler", namespace="schedrl")
        self._request_scheduler_cache: Dict[str, Any] = {}
        self._coordinator = None

        ray.get(
            self._schedrl_orchestrator.register_pipeline.remote(
                pipeline_id=self._registration.pipeline_id,
                ray_namespace=self._registration.ray_namespace,
                cluster_tp_configs=self._registration.cluster_tp_configs,
                cluster_device_mappings=self._registration.cluster_device_mappings,
            )
        )
        ray.get(self._schedrl_orchestrator.admit_pipeline.remote(pipeline_id=self._registration.pipeline_id))

    def get_registration(self) -> PipelineRegistration:
        return self._registration

    def get_pipeline_env_vars(self) -> Dict[str, str]:
        return dict(self._pipeline_env_vars)

    def ensure_coordinator(self) -> Any:
        _require_ray()
        import ray

        if self._coordinator is not None:
            return self._coordinator

        from roll.schedrl_adapter.concurrent_pipeline import SchedRLConcurrentPipeline

        Coordinator = ray.remote(SchedRLConcurrentPipeline)
        self._coordinator = Coordinator.options(
            name=f"schedrl:pipeline:{self._pipeline_id}",
            namespace=self._ray_namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
            runtime_env={"env_vars": dict(self._pipeline_env_vars)},
        ).remote(pipeline_id=self._pipeline_id)
        return self._coordinator

    def start_pipeline(self, *, pipeline_config: Any) -> None:
        self._inject_pipeline_env_vars(pipeline_config=pipeline_config)
        coordinator = self.ensure_coordinator()
        coordinator.run.remote(pipeline_config=pipeline_config)

    def _inject_pipeline_env_vars(self, *, pipeline_config: Any) -> None:
        envs = dict(self._pipeline_env_vars)

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

        # Worker clusters
        _update_system_envs(getattr(pipeline_config, "actor_train", None))
        _update_system_envs(getattr(pipeline_config, "actor_infer", None))
        _update_system_envs(getattr(pipeline_config, "reference", None))
        _update_system_envs(getattr(pipeline_config, "critic", None))
        _update_system_envs(getattr(pipeline_config, "reward", None))

        # Env managers (spawn env actors/workers)
        _update_system_envs(getattr(pipeline_config, "train_env_manager", None))
        _update_system_envs(getattr(pipeline_config, "val_env_manager", None))

    def _get_or_lookup_request_scheduler(self, *, mode: str) -> Any:
        _require_ray()
        import ray

        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be 'train'|'val', got {mode!r}")

        cached = self._request_scheduler_cache.get(mode)
        if cached is not None:
            return cached

        name = f"{self._pipeline_id}_request_scheduler_{mode}"
        try:
            handle = ray.get_actor(name, namespace=self._ray_namespace)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve RequestScheduler actor {name!r} in namespace {self._ray_namespace!r}"
            ) from e
        self._request_scheduler_cache[mode] = handle
        return handle

    def _try_get_request_scheduler(self, *, mode: str) -> Optional[Any]:
        """Best-effort actor lookup.

        Contract:
        - Returns None if the named actor doesn't exist yet.
        - Any other failure is treated as fatal (fail-fast).
        """
        _require_ray()
        import ray

        cached = self._request_scheduler_cache.get(mode)
        if cached is not None:
            return cached

        name = f"{self._pipeline_id}_request_scheduler_{mode}"
        try:
            handle = ray.get_actor(name, namespace=self._ray_namespace)
        except ValueError:
            return None
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve RequestScheduler actor {name!r} in namespace {self._ray_namespace!r}"
            ) from e

        self._request_scheduler_cache[mode] = handle
        return handle

    def _dp_ranks_to_gpu_ids(self, *, dp_ranks: List[int]) -> List[int]:
        cfg = self._registration
        tp_size = int(cfg.cluster_tp_configs["actor_infer"])
        device_mapping = list(cfg.cluster_device_mappings["actor_infer"])
        if tp_size <= 0:
            raise RuntimeError(f"Invalid actor_infer tp_size={tp_size}")
        if not device_mapping:
            raise RuntimeError("actor_infer device_mapping is empty")
        if len(device_mapping) % tp_size != 0:
            raise RuntimeError("actor_infer device_mapping length must be divisible by tp_size")

        max_dp = len(device_mapping) // tp_size
        gpu_ids: List[int] = []
        for dp_rank in dp_ranks:
            r = int(dp_rank)
            if not (0 <= r < max_dp):
                raise ValueError(f"dp_rank {r} out of range [0, {max_dp})")
            start = r * tp_size
            gpu_ids.extend(device_mapping[start : start + tp_size])
        return sorted(set(int(x) for x in gpu_ids))

    async def shrink_workers(self, dp_ranks_to_remove: List[int]) -> Dict[str, Any]:
        """SchedRL scheduler shrink hook: dp_ranks -> RequestScheduler.shrink_workers(target_gpus=...)."""
        _require_ray()

        if not isinstance(dp_ranks_to_remove, list) or not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be a non-empty list[int]")

        target_gpus = self._dp_ranks_to_gpu_ids(dp_ranks=dp_ranks_to_remove)
        train_scheduler = self._get_or_lookup_request_scheduler(mode="train")
        val_scheduler = self._try_get_request_scheduler(mode="val")

        train_ref = train_scheduler.shrink_workers.remote(target_gpus)
        refs = [train_ref]
        if val_scheduler is not None:
            refs.append(val_scheduler.shrink_workers.remote(target_gpus))

        results = await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in refs])
        train_result = results[0]
        if len(results) > 1:
            train_result = dict(train_result)
            train_result["val_result"] = results[1]
        return train_result

    async def expand_workers(self, dp_ranks_to_add: List[int]) -> Dict[str, Any]:
        _require_ray()

        if not isinstance(dp_ranks_to_add, list) or not dp_ranks_to_add:
            raise ValueError("dp_ranks_to_add must be a non-empty list[int]")
        target_gpus = self._dp_ranks_to_gpu_ids(dp_ranks=dp_ranks_to_add)
        train_scheduler = self._get_or_lookup_request_scheduler(mode="train")
        val_scheduler = self._try_get_request_scheduler(mode="val")

        train_ref = train_scheduler.expand_workers.remote(target_gpus)
        refs = [train_ref]
        if val_scheduler is not None:
            refs.append(val_scheduler.expand_workers.remote(target_gpus))

        results = await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in refs])
        train_result = results[0]
        if len(results) > 1:
            train_result = dict(train_result)
            train_result["val_result"] = results[1]
        return train_result
