from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import ray

from schedrl.protocol.request_id import validate_pipeline_id
from schedrl.protocol.types import ActionResponse


def _get_pipeline_namespace(pipeline_id: str) -> str:
    return f"pipeline_{pipeline_id}_NS"


def _build_pipeline_env_vars(*, pipeline_id: str, ray_namespace: str) -> Dict[str, str]:
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
    ):
        validate_pipeline_id(pipeline_id)
        self._pipeline_id = pipeline_id
        self._ray_namespace = _get_pipeline_namespace(pipeline_id)
        self._pipeline_env_vars = _build_pipeline_env_vars(pipeline_id=pipeline_id, ray_namespace=self._ray_namespace)

        _validate_cpu_only_reward(pipeline_config=pipeline_config)
        _validate_vllm_sleep_level(pipeline_config=pipeline_config)

        self._coordinator = None
        # NOTE: infer resize serialization is owned by the per-pipeline pipeline-side resize actor.

        # Driver is responsible for:
        # - orchestrator.allocate_pipeline_id()
        # - orchestrator.register_pipeline(...)
        # - orchestrator.admit_pipeline(...)
        # before creating this adapter actor.

    def create_coordinator(self, *, pipeline_config: Any) -> Any:
        if self._coordinator is not None:
            return self._coordinator

        from roll.schedrl_adapter.concurrent_pipeline import SchedRLConcurrentPipeline

        Coordinator = ray.remote(SchedRLConcurrentPipeline)
        # Safety: always inject env vars before constructing the coordinator, so callers can't
        # accidentally create a pipeline with missing system_envs.
        self._inject_pipeline_env_vars(pipeline_config=pipeline_config)
        self._coordinator = Coordinator.options(
            name=f"schedrl:pipeline:{self._pipeline_id}",
            namespace=self._ray_namespace,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
            # Critical: allow resize RPCs to run while `run()` is in-flight.
            max_concurrency=1000,
            runtime_env={"env_vars": dict(self._pipeline_env_vars)},
        ).remote(pipeline_id=self._pipeline_id, pipeline_config=pipeline_config)
        return self._coordinator

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

    async def resize_infer(self, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]):
        """Pipeline-scoped resize for actor_infer (ENG-123).

        Contract: exactly one of {dp_ranks_to_remove, dp_ranks_to_add} must be non-empty.
        Applies to both train+val RequestSchedulers (shared infer cluster):
        - Shrink: train offloads; val routing-only (skip_offload=True).
        - Expand: train loads + optional selective update; val routing-only (skip_load=True).

        NOTE: This intentionally does NOT call suspend()/resume() globally. Upstream RequestScheduler.shrink_workers()
        removes shrinking ranks from active_dp_ranks under routing_lock and aborts/drains only impacted ranks; new
        requests continue on remaining ranks. Shrink-to-zero and expand-from-zero are handled internally via
        need_suspend/resume().
        """
        if not isinstance(dp_ranks_to_remove, list):
            raise ValueError("dp_ranks_to_remove must be list[int]")
        if not isinstance(dp_ranks_to_add, list):
            raise ValueError("dp_ranks_to_add must be list[int]")
        if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add):
            raise ValueError("Exactly one of dp_ranks_to_remove or dp_ranks_to_add must be non-empty")

        # NOTE: adapter does not coordinate train/val request schedulers directly; it delegates to the
        # per-pipeline coordinator actor (single serialization boundary owned by pipeline runtime).
        resize_actor_name = f"schedrl:pipeline:{self._pipeline_id}"
        try:
            resize_actor = ray.get_actor(resize_actor_name, namespace=self._ray_namespace)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve pipeline coordinator actor {resize_actor_name!r} in namespace {self._ray_namespace!r} "
                f"for pipeline_id={self._pipeline_id!r}"
            ) from e

        ref = resize_actor.resize_infer.remote(
            dp_ranks_to_remove=list(dp_ranks_to_remove),
            dp_ranks_to_add=list(dp_ranks_to_add),
        )
        await asyncio.wrap_future(ref.future())
        return ActionResponse(success=True)
