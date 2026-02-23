from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.timer import _Timer

from schedrl.protocol.types import ActionResponse

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
from roll.pipeline.agentic.agentic_pipeline import compute_rollout_traj_metrics
import threading
from roll.pipeline.agentic.utils import (
    agentic_compute_advantage,
    compute_discounted_returns,
    compute_response_level_rewards,
    dump_rollout_trajectories,
    get_agentic_response_level_mask,
)
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import (
    agg_loss,
    batch_balance,
    compute_token_reward,
    masked_mean,
    reduce_metrics,
)
from roll.utils.logging import get_logger
from roll.utils.train_infer_corrections import apply_train_infer_correction_to_batch

logger = get_logger()


class SchedRLConcurrentPipeline(AgenticPipeline):
    """SchedRL-controlled variant of ROLL AgenticPipeline (ENG-123 Phase 3).

    Key differences from upstream AgenticPipeline.run():
    - Before each rollout, request generation GPUs from SchedRL (scheduler drives expand via adapter).
    - After each rollout, shrink actor_infer to zero and release allocation back to SchedRL.
    - Validation runs synchronously to avoid racing with shrink/release.
    """

    def __init__(self, *, pipeline_id: str, pipeline_config: Any):
        # In SchedRL mode we should follow the ConcurrentAgenticPipeline semantics:
        if not isinstance(pipeline_id, str) or pipeline_id == "":
            raise ValueError("pipeline_id must be non-empty str")
        self._pipeline_id = pipeline_id
        self._pipeline_config = pipeline_config
        self._initialized = False
        # Ray actor can run with max_concurrency>1; guard init so resize/run can't race it.
        self._init_lock = threading.Lock()
        try:
            self._schedrl_scheduler = ray.get_actor("schedrl:scheduler", namespace="schedrl")
        except Exception as e:
            # Expectation: the central schedrl scheduler actor ('schedrl:scheduler')
            # must already be created before the pipeline is instantiated.
            # Fail loudly with a clear message to aid debugging of startup ordering.
            raise RuntimeError(
                "Failed to resolve schedrl:scheduler in namespace 'schedrl'. "
                "The pipeline expects the central scheduler actor to be present before startup; "
                "ensure the orchestrator created it earlier or that startup ordering is correct."
            ) from e
        self._actor_infer_cluster_id = f"{self._pipeline_id}_actor_infer"
        self._actor_train_cluster_id = f"{self._pipeline_id}_actor_train"
        self._critic_cluster_id = f"{self._pipeline_id}_critic"
        self._reference_cluster_id = f"{self._pipeline_id}_reference"

    def initialize_pipeline(self) -> ActionResponse:
        # In SchedRL mode we should follow the ConcurrentAgenticPipeline semantics:
        """Initialize pipeline clusters/schedulers and prepare selective sync cache before first rollout."""
        with self._init_lock:
            if self._initialized:
                return ActionResponse(success=True)

            # Inline the heavy init logic (based on ConcurrentAgenticPipeline + AgenticPipeline init).
            # Do not call AgenticPipeline.__init__ here: we need explicit ordering + central scheduler interaction.
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            from roll.distributed.executor.cluster import Cluster
            from roll.distributed.scheduler.generate_scheduler import RequestScheduler
            from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
            from roll.models.model_providers import default_tokenizer_provider
            from roll.pipeline.base_pipeline import BasePipeline
            from roll.utils.functionals import RunningMoments
            from roll.utils.kl_controller import get_kl_controller
            from roll.utils.constants import RAY_NAMESPACE, schedrl_env_vars

            pipeline_config = self._pipeline_config
            BasePipeline.__init__(self, pipeline_config)
            self.pipeline_config = pipeline_config

            self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
            actor_lora_target = getattr(self.pipeline_config.actor_train.model_args, "lora_target", None)
            self.use_ref_model = bool(self.pipeline_config.enable_reference and (actor_lora_target is None))
            self.partial_gpu_mode = False

            self.kl_ctrl = get_kl_controller(
                init_kl_coef=self.pipeline_config.init_kl_coef,
                target_kl=self.pipeline_config.target_kl,
                kl_horizon=self.pipeline_config.kl_horizon,
            )

            # INIT PHASE: Create clusters (use pipeline_id prefix to keep names readable in logs).
            self.actor_train = Cluster(
                name=f"{self._pipeline_id}_{self.pipeline_config.actor_train.name}",
                worker_cls=self.pipeline_config.actor_train.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.actor_train,
            )
            self.actor_infer = Cluster(
                name=f"{self._pipeline_id}_{self.pipeline_config.actor_infer.name}",
                worker_cls=self.pipeline_config.actor_infer.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.actor_infer,
            )

            download_clusters = [self.actor_train, self.actor_infer]

            if self.use_ref_model:
                self.reference = Cluster(
                    name=f"{self._pipeline_id}_{self.pipeline_config.reference.name}",
                    worker_cls=self.pipeline_config.reference.worker_cls,
                    resource_manager=self.resource_manager,
                    worker_config=self.pipeline_config.reference,
                )
                download_clusters.append(self.reference)

            if self.pipeline_config.adv_estimator == "gae":
                self.critic = Cluster(
                    name=f"{self._pipeline_id}_{self.pipeline_config.critic.name}",
                    worker_cls=self.pipeline_config.critic.worker_cls,
                    resource_manager=self.resource_manager,
                    worker_config=self.pipeline_config.critic,
                )
                download_clusters.append(self.critic)

            # Reward cluster is optional; keep consistent with AgenticPipeline behavior.
            self.reward = None
            self.reward_scheduler = None
            if self.pipeline_config.reward is not None and len(self.pipeline_config.reward.device_mapping) > 0:
                self.reward = Cluster(
                    name=f"{self._pipeline_id}_{self.pipeline_config.reward.name}",
                    worker_cls=self.pipeline_config.reward.worker_cls,
                    resource_manager=self.resource_manager,
                    worker_config=self.pipeline_config.reward,
                )
                download_clusters.append(self.reward)

            # INIT PHASE: Download models once per node/PG before strategy initialization.
            self.download_models(*download_clusters)
            self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

            # Reward scheduler (named actor for env managers) if reward cluster exists.
            if self.reward:
                reward_name = f"RewardScheduler-{self._pipeline_id}"
                self.reward_scheduler = RequestScheduler.options(
                    name=reward_name,
                    get_if_exists=True,
                    namespace=RAY_NAMESPACE,
                    runtime_env={"env_vars": schedrl_env_vars()},
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False,
                    ),
                ).remote(
                    infer_cluster=self.reward,
                    pipeline_config=self.pipeline_config,
                    resource_manager=self.resource_manager,
                )

            # shared RequestScheduler (named actor).
            request_scheduler_name = f"RequestScheduler-{self._pipeline_id}"
            # Standard control-plane env vars for RequestScheduler (same as RolloutScheduler uses internally)
            control_env_vars = {
                "TORCH_COMPILE_DISABLE": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "RAY_num_server_call_thread": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
            control_env_vars.update(schedrl_env_vars())

            self.generate_scheduler = RequestScheduler.options(
                name=request_scheduler_name,
                namespace=RAY_NAMESPACE,
                get_if_exists=True,
                runtime_env={"env_vars": control_env_vars},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                max_concurrency=1024, # Large enough for shared use
            ).remote(
                infer_cluster=self.actor_infer,
                pipeline_config=self.pipeline_config,
                resource_manager=self.resource_manager,
            )

            # Rollout schedulers (named actors).
            self.train_rollout_scheduler = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-{self._pipeline_id}-train",
                namespace=RAY_NAMESPACE,
                runtime_env={"env_vars": schedrl_env_vars()},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                config=self.pipeline_config,
                env_manager_config=self.pipeline_config.train_env_manager,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="train",
                request_scheduler=self.generate_scheduler,
            )
            self.val_rollout_scheduler = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-{self._pipeline_id}-val",
                namespace=RAY_NAMESPACE,
                runtime_env={"env_vars": schedrl_env_vars()},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                config=self.pipeline_config,
                env_manager_config=self.pipeline_config.val_env_manager,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="val",
                request_scheduler=self.generate_scheduler,
            )

            # Create val dataset manager as in AgenticPipeline.
            from roll.datasets.global_dataset import GlobalDatasetManager

            self.val_dataset_manager = GlobalDatasetManager.options(
                name="val_dataset_manager",
                get_if_exists=True,
                namespace=RAY_NAMESPACE,
                runtime_env={"env_vars": schedrl_env_vars()},
            ).remote()

            # Infer resize serialization boundary (ENG-123).
            infer_strategy_config = self.actor_infer.worker_config.strategy_args.strategy_config
            tp_size = int(infer_strategy_config.get("tensor_parallel_size", 1))
            pp_size = int(infer_strategy_config.get("pipeline_parallel_size", 1))
            self._infer_gpus_per_dp_rank = tp_size * pp_size
            self._infer_device_mapping = list(getattr(self.pipeline_config.actor_infer, "device_mapping", None) or [])
            if not self._infer_device_mapping:
                raise RuntimeError("actor_infer.device_mapping must be set")
            self._infer_resize_lock = threading.Lock()

            # INIT PHASE: Initialize clusters with central scheduler coordination and strict offload ordering.
            from schedrl.protocol.types import Priority

            init_global_step = -1
            self._request_static_cluster(
                cluster_id=self._actor_train_cluster_id,
                priority=Priority.INITIALIZATION,
                global_step=init_global_step,
            )
            try:
                refs: List[ray.ObjectRef] = []
                refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
                ray.get(refs)

                # Build and promote the initial base-model cache (-1/-1) before offload.
                # Under sleep_level=2 this cache must stay active so expand can rehydrate infer workers.
                init_checkpoint_version = -1
                init_bucket_step = -1
                self.actor_train.load_states(blocking=True)
                ray.get(
                    [
                        w.build_latest_bucket_cache.remote(
                            checkpoint_version=int(init_checkpoint_version),
                            global_step=int(init_bucket_step),
                        )
                        for w in self.actor_train.workers
                    ]
                )
                ray.get(
                    [
                        w.promote_active_checkpoint.remote(
                            checkpoint_version=int(init_checkpoint_version),
                            global_step=int(init_bucket_step),
                        )
                        for w in self.actor_train.workers
                    ]
                )

                # Offload training-side clusters before initializing actor_infer (avoid transient OOM).
                logger.info("[init][%s] offloading actor_train before actor_infer init", self._pipeline_id)
                self.actor_train.offload_states(blocking=True)
                logger.info("[init][%s] actor_train offload done", self._pipeline_id)
            finally:
                self._release_static_cluster(cluster_id=self._actor_train_cluster_id, global_step=init_global_step)
                logger.info("[init][%s] released actor_train cluster", self._pipeline_id)

            logger.info("[init][%s] requesting actor_infer cluster (INITIALIZATION)", self._pipeline_id)
            self._request_static_cluster(
                cluster_id=self._actor_infer_cluster_id,
                priority=Priority.INITIALIZATION,
                global_step=init_global_step,
            )
            logger.info("[init][%s] actor_infer cluster granted — starting init", self._pipeline_id)
            try:
                refs = []
                if self.reward:
                    refs.extend(self.reward.initialize(pipeline_config=self.pipeline_config, blocking=False))
                refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
                ray.get(refs)
                logger.info("[init][%s] actor_infer initialized — offloading (sleep_level=2: destroy weights+KV)", self._pipeline_id)
                if self.reward:
                    self.reward.offload_states(blocking=True)
                self.actor_infer.offload_states(blocking=True)
                logger.info("[init][%s] actor_infer offload done — GPU memory freed", self._pipeline_id)
            finally:
                self._release_static_cluster(cluster_id=self._actor_infer_cluster_id, global_step=init_global_step)
                logger.info("[init][%s] released actor_infer cluster", self._pipeline_id)

            if self.pipeline_config.adv_estimator == "gae":
                self._request_static_cluster(
                    cluster_id=self._critic_cluster_id,
                    priority=Priority.INITIALIZATION,
                    global_step=init_global_step,
                )
                try:
                    self.critic.initialize(pipeline_config=self.pipeline_config, blocking=True)
                    self.critic.offload_states(blocking=True)
                finally:
                    self._release_static_cluster(cluster_id=self._critic_cluster_id, global_step=init_global_step)

            if self.use_ref_model:
                self._request_static_cluster(
                    cluster_id=self._reference_cluster_id,
                    priority=Priority.INITIALIZATION,
                    global_step=init_global_step,
                )
                try:
                    self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)
                    self.reference.offload_states(blocking=True)
                finally:
                    self._release_static_cluster(cluster_id=self._reference_cluster_id, global_step=init_global_step)

            # Setup model update pair and checkpoint clusters (required by BasePipeline.model_update/do_checkpoint).
            self.set_model_update_pair(
                src_cluster=self.actor_train,
                tgt_cluster=self.actor_infer,
                frequency=self.pipeline_config.actor_train.model_update_frequency,
            )
            if self.pipeline_config.adv_estimator == "gae":
                self.set_checkpoint_clusters(self.actor_train, self.critic)
            else:
                self.set_checkpoint_clusters(self.actor_train)

            self.running = RunningMoments()

            # Validate partial GPU mode configuration and set self.partial_gpu_mode
            if getattr(self.pipeline_config, "partial_gpu_mode", False):
                self.partial_gpu_mode = self._validate_partial_gpu_config()
            else:
                self.partial_gpu_mode = False

            # Namespace contract: in SchedRL mode, require explicit per-pipeline env vars (fail fast).
            ray_namespace = os.environ.get("ROLL_RAY_NAMESPACE", "roll")
            if os.environ.get("SCHEDRL_CONTROL_PLANE", "") == "schedrl":
                env_namespace = os.environ.get("ROLL_RAY_NAMESPACE")
                pipeline_id_env = os.environ.get("PIPELINE_ID")
                if not env_namespace:
                    raise RuntimeError("SCHEDRL_CONTROL_PLANE=schedrl requires ROLL_RAY_NAMESPACE to be set")
                if not pipeline_id_env:
                    raise RuntimeError("SCHEDRL_CONTROL_PLANE=schedrl requires PIPELINE_ID to be set")
                if pipeline_id_env != self._pipeline_id:
                    raise RuntimeError(
                        f"PIPELINE_ID mismatch for coordinator: env PIPELINE_ID={pipeline_id_env!r} "
                        f"!= coordinator pipeline_id={self._pipeline_id!r}"
                    )
                ray_namespace = env_namespace

            # Align with ConcurrentAgenticPipeline: interact with central scheduler during init.
            # The initial (-1) cache bucket is built during actor_train init above under INITIALIZATION allocation.

            # Create ModelUpdateService in the per-pipeline namespace. This is used by
            # RequestScheduler.expand_workers() in SchedRL mode to sync selected dp ranks after load.
            from roll.schedrl_adapter.model_update_service import ModelUpdateService

            runtime_env = {
                "env_vars": {
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "PIPELINE_ID": os.environ.get("PIPELINE_ID", self._pipeline_id),
                    "ROLL_RAY_NAMESPACE": ray_namespace,
                    "SCHEDRL_CONTROL_PLANE": os.environ.get("SCHEDRL_CONTROL_PLANE", "schedrl"),
                    "SCHEDRL_LIBRARY_MODE": os.environ.get("SCHEDRL_LIBRARY_MODE", "1"),
                }
            }
            svc = ModelUpdateService.options(
                name=f"{self._pipeline_id}_model_update_service",
                namespace=ray_namespace,
                get_if_exists=True,
                max_restarts=0,
                max_task_retries=0,
                runtime_env=runtime_env,
                lifetime="detached",
            ).remote(
                pipeline_id=self._pipeline_id,
                src_cluster=self.actor_train,
                tgt_cluster=self.actor_infer,
            )
            ray.get(svc.__ray_ready__.remote())

            # Start from a well-defined state (ENG-123):
            # - disable routing until we request GPUs from SchedRL.
            # NOTE: avoid local suspend()/resume() state transitions; shrink-to-zero is the single
            # source of truth for pausing generation traffic, and expand-from-zero resumes internally.
            dp_ranks = self._actor_infer_all_dp_ranks()
            ray.get(self.train_rollout_scheduler.shrink_sampler.remote(dp_ranks, skip_offload=True))
            ray.get(self.val_rollout_scheduler.shrink_sampler.remote(dp_ranks, skip_offload=True))

            # Verify state: both schedulers must have empty active_dp_ranks after init shrink.
            train_active = ray.get(self.train_rollout_scheduler.get_active_dp_ranks.remote())
            val_active = ray.get(self.val_rollout_scheduler.get_active_dp_ranks.remote())
            if train_active or val_active:
                raise RuntimeError(
                    f"Initialization failed: active_dp_ranks not empty after shrink. "
                    f"train_active={sorted(train_active)}, val_active={sorted(val_active)}. "
                    f"This indicates state desync between SchedRL and ROLL."
                )

            self._initialized = True
            return ActionResponse(success=True)

    def _shrink_workers(self, *, dp_ranks_to_remove: List[int]) -> Dict[str, Any]:
        """Pipeline-local shrink helper (ENG-123).

        In SchedRL mode with shared RequestScheduler, a single call performs:
        - routing-only shrink (updates shared active_dp_ranks)
        - physical offload (skip_offload=False)
        """
        if not isinstance(dp_ranks_to_remove, list) or not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be a non-empty list[int]")
        with self._infer_resize_lock:
            # Both train and val share self.generate_scheduler.
            # One call with skip_offload=False is sufficient.
            return ray.get(
                self.train_rollout_scheduler.shrink_sampler.remote(dp_ranks_to_remove, skip_offload=False)
            )

    def _expand_workers(self, *, dp_ranks_to_add: List[int], train_skip_load: bool) -> Dict[str, Any]:
        """Pipeline-local expand helper (ENG-123).

        In SchedRL mode with shared RequestScheduler, a single call performs:
        - weight load (skip_load=train_skip_load)
        - routing-only expand (updates shared active_dp_ranks)
        """
        if not isinstance(dp_ranks_to_add, list) or not dp_ranks_to_add:
            raise ValueError("dp_ranks_to_add must be a non-empty list[int]")
        with self._infer_resize_lock:
            # Both train and val share self.generate_scheduler.
            return ray.get(
                self.train_rollout_scheduler.expand_sampler.remote(
                    dp_ranks_to_add, skip_load=bool(train_skip_load)
                )
            )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            resp = self.initialize_pipeline()
            if not getattr(resp, "success", False):
                raise RuntimeError(f"initialize_pipeline failed: {resp}")

    def _actor_infer_device_mapping(self) -> List[int]:
        mapping = getattr(self.pipeline_config.actor_infer, "device_mapping", None)
        if mapping is None:
            raise RuntimeError("actor_infer.device_mapping must be set for SchedRL mode")
        if not isinstance(mapping, list):
            raise RuntimeError(f"actor_infer.device_mapping must be list[int], got {type(mapping).__name__}")
        if not mapping:
            raise RuntimeError("actor_infer.device_mapping must be non-empty for SchedRL mode")
        if not all(isinstance(x, int) and x >= 0 for x in mapping):
            raise RuntimeError("actor_infer.device_mapping must be list[int>=0]")
        return list(mapping)

    def _actor_infer_all_dp_ranks(self) -> List[int]:
        infer_strategy_config = self.actor_infer.worker_config.strategy_args.strategy_config
        tp_size = int(infer_strategy_config.get("tensor_parallel_size", 1))
        pp_size = int(infer_strategy_config.get("pipeline_parallel_size", 1))
        gpus_per_dp_rank = tp_size * pp_size
        device_mapping = self._actor_infer_device_mapping()
        if len(device_mapping) % int(gpus_per_dp_rank) != 0:
            raise RuntimeError("actor_infer.device_mapping length must be divisible by gpus_per_dp_rank")
        max_dp = len(device_mapping) // int(gpus_per_dp_rank)
        return list(range(int(max_dp)))

    def _request_actor_infer_gpus(self, *, global_step: int) -> List[int]:
        from schedrl.protocol.types import Priority

        allocated = ray.get(
            self._schedrl_scheduler.request_gpus.remote(
                cluster_id=self._actor_infer_cluster_id,
                priority=Priority.GENERATION,
                global_step=global_step,
            )
        )
        if not isinstance(allocated, list):
            raise RuntimeError(f"schedrl:scheduler.request_gpus returned non-list: {type(allocated).__name__}")
        allocated = [int(x) for x in allocated]
        if not allocated:
            raise RuntimeError(
                f"schedrl:scheduler allocated empty GPU list for cluster_id={self._actor_infer_cluster_id!r}"
            )
        return allocated

    def _request_static_cluster(self, *, cluster_id: str, priority: Any, global_step: int) -> List[int]:
        allocated = ray.get(
            self._schedrl_scheduler.request_gpus.remote(
                cluster_id=str(cluster_id),
                priority=priority,
                global_step=global_step,
            )
        )
        if not isinstance(allocated, list):
            raise RuntimeError(f"schedrl:scheduler.request_gpus returned non-list: {type(allocated).__name__}")
        allocated = [int(x) for x in allocated]
        if not allocated:
            raise RuntimeError(f"schedrl:scheduler allocated empty GPU list for cluster_id={cluster_id!r}")
        return allocated

    def _release_static_cluster(self, *, cluster_id: str, global_step: int) -> None:
        ray.get(self._schedrl_scheduler.release_gpus.remote(cluster_id=str(cluster_id), global_step=global_step))

    def _release_and_request_static_cluster(
        self,
        *,
        release_cluster_id: str,
        release_global_step: int,
        request_cluster_id: str,
        request_priority: Any,
        request_global_step: int,
    ) -> List[int]:
        allocated = ray.get(
            self._schedrl_scheduler.release_and_request_gpus.remote(
                release_cluster_id=str(release_cluster_id),
                release_global_step=int(release_global_step),
                request_cluster_id=str(request_cluster_id),
                request_priority=request_priority,
                request_global_step=int(request_global_step),
            )
        )
        if not isinstance(allocated, list):
            raise RuntimeError(f"schedrl:scheduler.release_and_request_gpus returned non-list: {type(allocated).__name__}")
        allocated = [int(x) for x in allocated]
        if not allocated:
            raise RuntimeError(f"schedrl:scheduler allocated empty GPU list for cluster_id={request_cluster_id!r}")
        return allocated

    def _notify_ready_to_release_actor_infer(self, *, global_step: int) -> List[int]:
        timeout_s_raw = os.environ.get("SCHEDRL_NOTIFY_READY_TIMEOUT_S", "300")
        try:
            timeout_s = float(timeout_s_raw)
        except ValueError as e:
            raise RuntimeError(f"Invalid SCHEDRL_NOTIFY_READY_TIMEOUT_S={timeout_s_raw!r}") from e
        if timeout_s <= 0:
            raise RuntimeError(f"SCHEDRL_NOTIFY_READY_TIMEOUT_S must be > 0, got {timeout_s!r}")

        released = ray.get(
            self._schedrl_scheduler.notify_ready_to_release.remote(
                cluster_id=self._actor_infer_cluster_id,
                global_step=global_step,
                timeout_s=timeout_s,
            )
        )
        if not isinstance(released, list):
            raise RuntimeError(f"notify_ready_to_release returned non-list: {type(released).__name__}")
        released = [int(x) for x in released]
        logger.info(
            f"[schedrl][{self._pipeline_id}] notify_ready_to_release done: step={global_step} released={sorted(released)}"
        )
        return released

    @torch.no_grad()
    def run(self):
        # In SchedRL mode we should follow the ConcurrentAgenticPipeline semantics:
        self._ensure_initialized()
        tps_timer = _Timer(window_size=5)
        last_notify_ready_step: int | None = None

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"[schedrl][{self._pipeline_id}] pipeline global_step={global_step} start")
            metrics: Dict[str, Any] = {}
            should_checkpoint = bool(
                global_step > 0
                and (
                    global_step % self.pipeline_config.save_steps == 0
                    or global_step == self.pipeline_config.max_steps - 1
                )
            )
            defer_actor_train_release_for_checkpoint = False

            with Timer(name="pipeline_step_total", logger=None) as step_timer:
                with tps_timer:
                    # Phase 0 (Multi-pipeline semantics): at step start, block until the previous step's rollout
                    # workers are stopped/offloaded by the central scheduler. This ensures model update happens
                    # with maximum free GPU memory and without concurrent rollout activity.
                    if global_step > 0 and last_notify_ready_step != global_step - 1:
                        self._notify_ready_to_release_actor_infer(global_step=global_step - 1)
                        last_notify_ready_step = global_step - 1

                    # PHASE 1: Offload States
                    if self.pipeline_config.adv_estimator == "gae":
                        self.critic.offload_states(blocking=True)
                    if self.pipeline_config.enable_reference and self.use_ref_model:
                        self.reference.offload_states(blocking=True)
                    self.actor_train.offload_states(blocking=True)

                    # PHASE 2: (SchedRL) no local suspend; scheduler-driven shrink/expand owns routing state.

                    # PHASE 3: Model Update
                    # In SchedRL mode we should follow the ConcurrentAgenticPipeline semantics:
                    # the pipeline must not run model_update() itself.
                    #
                    # Selective model update is triggered by the central scheduler when it grants the next
                    # generation allocation and calls resize_infer/expand.
                    # Selective model update is triggered by the central scheduler when it grants the next
                    # generation allocation and calls resize_infer/expand.
                    with Timer(name="model_update", logger=None) as model_update_timer:
                        pass
                    metrics["time/step_model_update"] = model_update_timer.last

                    # PHASE 4: Request actor_infer GPUs (central scheduler will call resize_infer).
                    # Multi-pipeline semantics: for step>0, atomically release last step's actor_train
                    # allocation before requesting actor_infer generation GPUs.
                    #
                    # Note: actor_train is intentionally kept allocated (but offloaded) at the end of the
                    # previous step when actor training runs, and is released here via release_and_request.
                    from schedrl.protocol.types import Priority

                    if global_step > 0 and self.pipeline_config.critic_warmup <= (global_step - 1):
                        self._release_and_request_static_cluster(
                            release_cluster_id=self._actor_train_cluster_id,
                            release_global_step=global_step - 1,
                            request_cluster_id=self._actor_infer_cluster_id,
                            request_priority=Priority.GENERATION,
                            request_global_step=global_step,
                        )
                    else:
                        self._request_actor_infer_gpus(global_step=global_step)

                    batch: DataProto = DataProto()
                    batch.meta_info = {"global_step": global_step}

                    # PHASE 5: Validation (synchronous in SchedRL mode)
                    val_metrics = {}
                    with Timer(name="val", logger=None) as val_timer:
                        if self.pipeline_config.eval_steps > 0 and global_step % self.pipeline_config.eval_steps == 0:
                            val_metrics = self.val(global_step)

                    # PHASE 6: Rollout Get Batch
                    with Timer(name="rollout", logger=None) as rollout_timer:
                        batch = ray.get(
                            self.train_rollout_scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size)
                        )
                        sample_uuids = [f"{traj_id}_{i}" for i, traj_id in enumerate(batch.non_tensor_batch["traj_id"])]
                        batch.non_tensor_batch["sample_uuid"] = np.array(sample_uuids, dtype=object)
                        if "get_batch_return_start_time" in batch.meta_info:
                            metrics["time/get_batch_cost_train"] = time.time() - batch.meta_info.pop(
                                "get_batch_return_start_time"
                            )
                        actor_infer_metrics = self.actor_infer.get_metrics()
                        metrics.update(reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {})))
                        metrics.update(compute_rollout_traj_metrics(batch))

                        dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

                    metrics["time/step_rollout"] = rollout_timer.last
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    batch.meta_info["global_step"] = global_step
                    batch.meta_info["_broadcast_non_tensor_batch"] = True
                    batch.meta_info["loss_mask_keys"] = ["response_mask"]

                    if len(val_metrics) > 0:
                        metrics.update(val_metrics)
                        metrics["time/step_val"] = val_timer.last

                    batch = compute_discounted_returns(
                        batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma
                    )

                    batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    # PHASE 11: Reference Log Probs
                    if self.pipeline_config.enable_reference:
                        from schedrl.protocol.types import Priority

                        if self.use_ref_model:
                            self._request_static_cluster(
                                cluster_id=self._reference_cluster_id,
                                priority=Priority.REF_LOG_PROBS,
                                global_step=global_step,
                            )
                        else:
                            self._request_static_cluster(
                                cluster_id=self._actor_train_cluster_id,
                                priority=Priority.REF_LOG_PROBS,
                                global_step=global_step,
                            )
                    with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
                        if self.pipeline_config.enable_reference:
                            worker_config = (
                                self.pipeline_config.reference if self.use_ref_model else self.pipeline_config.actor_train
                            )
                            worker = self.reference if self.use_ref_model else self.pipeline_config.actor_train
                            if worker_config.use_dynamic_batching_in_infer:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    worker.dp_size,
                                    worker_config.max_tokens_per_microbatch_in_infer,
                                    worker_config.sequence_length_round_in_infer,
                                    worker_config.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                                    worker_config.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                                    "reference/compute_log_probs",
                                )
                                metrics.update(dynamic_batching_metrics)
                            if not self.use_ref_model:
                                batch.meta_info["disable_adapter"] = True
                                batch.meta_info["is_offload_states"] = False
                                batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                                ref_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(
                                    batch, blocking=False
                                )
                            else:
                                batch_balance(batch, dp_size=self.reference.dp_size, minibatch_size=len(batch))
                                ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(
                                    batch, blocking=False
                                )

                            ref_log_probs = DataProto.materialize_concat(data_refs=ref_log_probs_refs)
                            ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                            batch = batch.union(ref_log_probs)
                            avg_ref_log_prob = masked_mean(
                                batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:]
                            )
                            metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                            metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
                    metrics["time/step_ref_log_probs_values_reward"] = cal_timer.last
                    if self.pipeline_config.enable_reference:
                        if self.use_ref_model:
                            self.reference.offload_states(blocking=True)
                            self._release_static_cluster(cluster_id=self._reference_cluster_id, global_step=global_step)
                        else:
                            self.actor_train.offload_states(blocking=True)
                            self._release_static_cluster(cluster_id=self._actor_train_cluster_id, global_step=global_step)

                    # PHASE 12: Old Log Probs & Values
                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        critic_requested = False
                        if self.pipeline_config.enable_reference and not self.use_ref_model:
                            batch.meta_info["disable_adapter"] = False
                        batch.meta_info["is_offload_states"] = False
                        if self.pipeline_config.enable_old_logprobs_recompute:
                            from schedrl.protocol.types import Priority

                            self._request_static_cluster(
                                cluster_id=self._actor_train_cluster_id,
                                priority=Priority.OLD_LOG_PROBS,
                                global_step=global_step,
                            )
                            batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                                    self.pipeline_config.actor_train.sequence_length_round_in_infer,
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                        "pipeline_model_parallel_size", 1
                                    ),
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                        "virtual_pipeline_model_parallel_size", None
                                    ),
                                    "actor_train/compute_log_probs",
                                )
                                metrics.update(dynamic_batching_metrics)
                            old_log_probs: DataProto = self.actor_train.compute_log_probs(batch, blocking=True)
                            batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                            avg_old_log_prob = masked_mean(
                                batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:]
                            )
                            metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})
                            metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                            agg_entropy = agg_loss(
                                loss_mat=old_log_probs.batch["entropy"],
                                loss_mask=batch.batch["response_mask"][:, 1:],
                                loss_agg_mode="token-mean",
                            )
                            metrics.update({"critic/entropy/mean": agg_entropy.item()})
                            self.actor_train.offload_states(blocking=True)
                            if self.pipeline_config.adv_estimator == "gae":
                                self._release_and_request_static_cluster(
                                    release_cluster_id=self._actor_train_cluster_id,
                                    release_global_step=global_step,
                                    request_cluster_id=self._critic_cluster_id,
                                    request_priority=Priority.VALUE_COMPUTE,
                                    request_global_step=global_step,
                                )
                                critic_requested = True
                            else:
                                self._release_static_cluster(cluster_id=self._actor_train_cluster_id, global_step=global_step)
                        else:
                            batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])

                        if self.pipeline_config.adv_estimator == "gae":
                            from schedrl.protocol.types import Priority

                            if not critic_requested:
                                self._request_static_cluster(
                                    cluster_id=self._critic_cluster_id,
                                    priority=Priority.VALUE_COMPUTE,
                                    global_step=global_step,
                                )
                            values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                        if self.pipeline_config.adv_estimator == "gae":
                            values = DataProto.materialize_concat(data_refs=values_refs)
                            batch = batch.union(values)
                            metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))
                            self.critic.offload_states(blocking=True)
                            self._release_static_cluster(cluster_id=self._critic_cluster_id, global_step=global_step)

                        if not self.pipeline_config.enable_reference:
                            batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                            avg_ref_log_prob = masked_mean(
                                batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:]
                            )
                            metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})

                    metrics["time/step_old_log_probs_values"] = cal_old_logpb_timer.last

                    with Timer(name="cal_response_level_mask", logger=None) as timer:
                        batch, mask_metrics = get_agentic_response_level_mask(batch, self.pipeline_config)
                        metrics.update(mask_metrics)
                    metrics["time/step_cal_response_level_mask"] = timer.last

                    # PHASE 13: Advantage Computation
                    with Timer(name="cal_response_norm_rewards", logger=None) as timer:
                        batch, reward_metrics = compute_response_level_rewards(batch=batch, pipeline_config=self.pipeline_config)
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                        metrics.update(reward_metrics)
                    metrics["time/step_cal_norm_rewards"] = timer.last

                    with Timer(name="cal_token_reward", logger=None) as timer:
                        batch, token_level_metrics = compute_token_reward(batch, self.pipeline_config, self.kl_ctrl)
                        metrics.update(token_level_metrics)
                    metrics["time/step_cal_token_reward"] = timer.last

                    with Timer(name="compute_advantage", logger=None) as timer:
                        batch = agentic_compute_advantage(
                            data=batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                        )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    metrics["time/step_adv"] = timer.last

                    if self.pipeline_config.enable_old_logprobs_recompute:
                        batch, corr_metrics = apply_train_infer_correction_to_batch(
                            self.pipeline_config, batch, update_mask_keys=batch.meta_info["loss_mask_keys"]
                        )
                        metrics.update(corr_metrics)

                    # PHASE 14: Training (critic + actor)
                    with Timer(name="train_timer", logger=None) as train_timer:
                        if self.pipeline_config.adv_estimator == "gae":
                            from schedrl.protocol.types import Priority

                            self._request_static_cluster(
                                cluster_id=self._critic_cluster_id,
                                priority=Priority.CRITIC_TRAINING,
                                global_step=global_step,
                            )
                            critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                        if self.pipeline_config.critic_warmup <= global_step:
                            from schedrl.protocol.types import Priority

                            self._request_static_cluster(
                                cluster_id=self._actor_train_cluster_id,
                                priority=Priority.ACTOR_TRAINING,
                                global_step=global_step,
                            )
                            batch_balance_metrics = batch_balance(
                                batch,
                                dp_size=self.actor_train.dp_size,
                                minibatch_size=self.actor_train.dp_size
                                * self.pipeline_config.actor_train.training_args.per_device_train_batch_size
                                * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps,
                                logging_prefix="global_seqlen/actor_train",
                            )
                            metrics.update(batch_balance_metrics)
                            if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                                batch, dynamic_batching_metrics = dynamic_batching_shard(
                                    batch,
                                    self.actor_train.dp_size,
                                    self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                    self.pipeline_config.actor_train.sequence_length_round_in_train,
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                        "pipeline_model_parallel_size", 1
                                    ),
                                    self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                        "virtual_pipeline_model_parallel_size", None
                                    ),
                                    "actor_train/train_step",
                                )
                                metrics.update(dynamic_batching_metrics)
                            actor_train_metrics_refs = self.actor_train.train_step(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(data_refs=actor_train_metrics_refs)
                            metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))
                            checkpoint_version = int(batch.meta_info.get("checkpoint_version", global_step))
                            ray.get(
                                [
                                    worker.promote_active_checkpoint.remote(checkpoint_version, int(global_step))
                                    for worker in self.actor_train.workers
                                ]
                            )
                            self.actor_train.offload_states(blocking=True)
                            if should_checkpoint:
                                # Always defer: save_checkpoint calls load_states(), so we must
                                # re-offload after the checkpoint before any GPU release or handoff.
                                defer_actor_train_release_for_checkpoint = True
                            else:
                                # Keep actor_train allocated (but offloaded) so next step can perform an
                                # atomic release_and_request during the train→infer transition.
                                if global_step == self.pipeline_config.max_steps - 1:
                                    self._release_static_cluster(
                                        cluster_id=self._actor_train_cluster_id,
                                        global_step=global_step,
                                    )

                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                            metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                            self.critic.offload_states(blocking=True)
                            self._release_static_cluster(cluster_id=self._critic_cluster_id, global_step=global_step)
                        tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                    metrics["time/step_train"] = train_timer.last

                from roll.pipeline.agentic.agentic_pipeline import compute_train_data_metrics

                with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                    data_metrics = compute_train_data_metrics(batch=batch)

                metrics["time/step_compute_data_metrics"] = data_metrics_timer.last
                metrics.update(data_metrics)
                metrics["system/tps"] = tps_timer.mean_throughput
                metrics["system/samples"] = (global_step + 1) * self.pipeline_config.rollout_batch_size

                self.state.step = global_step
                self.state.log_history.append(metrics)

                self.do_checkpoint(global_step=global_step)
                if defer_actor_train_release_for_checkpoint:
                    # save_checkpoint calls load_states() internally to read weights for saving.
                    # Re-offload so peer pipelines see clean GPU state before any release or
                    # next-step Phase 4 handoff.
                    self.actor_train.offload_states(blocking=True)
                    if global_step == self.pipeline_config.max_steps - 1:
                        # Last step: no next-step Phase 4 to release actor_train, so release here.
                        self._release_static_cluster(cluster_id=self._actor_train_cluster_id, global_step=global_step)

                with Timer(name="log", logger=None) as log_timer:
                    if self.pipeline_config.logging_steps > 0 and global_step % self.pipeline_config.logging_steps == 0:
                        if int(os.environ.get("RAY_PROFILING", "0")):
                            timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                            os.makedirs(timeline_dir, exist_ok=True)
                            ray.timeline(filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"))

                        log_res = []
                        batch_grouped = batch.group_by(keys="traj_id")
                        for _, group_batch in batch_grouped.items():
                            if "step" in group_batch.non_tensor_batch.keys():
                                indices = torch.argsort(
                                    torch.from_numpy(group_batch.non_tensor_batch["step"].astype(np.int64))
                                )
                                group_batch.reorder(indices)

                            prompt_mask = group_batch.batch["prompt_mask"]
                            non_prompt_mask = (
                                torch.logical_not(group_batch.batch["prompt_mask"]) * group_batch.batch["attention_mask"]
                            )
                            input_ids = group_batch.batch["input_ids"]
                            prompt_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(prompt_mask)]
                            response_ids_list = [input_ids[i][mask.bool()] for i, mask in enumerate(non_prompt_mask)]
                            prompts = self.tokenizer.batch_decode(prompt_ids_list, skip_special_tokens=False)
                            responses = self.tokenizer.batch_decode(response_ids_list, skip_special_tokens=False)
                            episode_scores = group_batch.non_tensor_batch["episode_scores"].tolist()
                            step_scores = group_batch.non_tensor_batch["step_scores"].tolist()
                            if isinstance(step_scores[0], np.ndarray):
                                step_scores = [t.tolist() for t in step_scores]

                            log_item = []
                            for prompt, response, episode_score, step_score in zip(
                                prompts, responses, episode_scores, step_scores
                            ):
                                log_item.append(
                                    {
                                        "prompt": prompt,
                                        "response": response,
                                        "episode_score": episode_score,
                                        "step_score": step_score,
                                    }
                                )
                            log_res.append(log_item)
                            if len(log_res) >= 10:
                                break
                        logger.info(json.dumps(log_res, ensure_ascii=False))
                        logger.info(json.dumps(metrics, ensure_ascii=False))

                metrics["time/step_log"] = log_timer.last

            metrics["time/step_total"] = step_timer.last
            self.tracker.log(values=metrics, step=global_step)

            logger.info(f"[schedrl][{self._pipeline_id}] pipeline step {global_step} finished")

        # Final cleanup: release the last step's actor_infer allocation.
        # This matches ROLL_multi_pipeline pattern where notify_ready_to_release is called after the loop.
        if last_notify_ready_step != self.pipeline_config.max_steps - 1:
            self._notify_ready_to_release_actor_infer(global_step=self.pipeline_config.max_steps - 1)
            logger.info(f"[schedrl][{self._pipeline_id}] final notify_ready_to_release for step {self.pipeline_config.max_steps - 1}")

        ray.get([self.train_rollout_scheduler.shutdown.remote(), self.val_rollout_scheduler.shutdown.remote()])
        logger.info(f"[schedrl][{self._pipeline_id}] pipeline complete!")

    def resize_infer(self, *, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]):
        self._ensure_initialized()
        if not isinstance(dp_ranks_to_remove, list):
            raise ValueError("dp_ranks_to_remove must be list[int]")
        if not isinstance(dp_ranks_to_add, list):
            raise ValueError("dp_ranks_to_add must be list[int]")
        if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add):
            raise ValueError("Exactly one of dp_ranks_to_remove or dp_ranks_to_add must be non-empty")

        # Snapshot pre-state for verification
        train_active_before = ray.get(self.train_rollout_scheduler.get_active_dp_ranks.remote())
        val_active_before = ray.get(self.val_rollout_scheduler.get_active_dp_ranks.remote())

        if dp_ranks_to_remove:
            self._shrink_workers(dp_ranks_to_remove=list(dp_ranks_to_remove))
            # Verify shrink: ranks should be removed from active_dp_ranks
            train_active_after = ray.get(self.train_rollout_scheduler.get_active_dp_ranks.remote())
            val_active_after = ray.get(self.val_rollout_scheduler.get_active_dp_ranks.remote())
            expected_removed = set(dp_ranks_to_remove)
            still_active_train = train_active_after & expected_removed
            still_active_val = val_active_after & expected_removed
            if still_active_train or still_active_val:
                raise RuntimeError(
                    f"Shrink verification failed: ranks {sorted(expected_removed)} should be inactive. "
                    f"train still active: {sorted(still_active_train)}, val still active: {sorted(still_active_val)}. "
                    f"Before: train={sorted(train_active_before)}, val={sorted(val_active_before)}. "
                    f"After: train={sorted(train_active_after)}, val={sorted(val_active_after)}."
                )
        else:
            # PRE-condition check for expand: ranks should NOT already be active
            expected_added = set(dp_ranks_to_add)
            already_active_train = train_active_before & expected_added
            already_active_val = val_active_before & expected_added
            if already_active_train or already_active_val:
                raise RuntimeError(
                    f"Expand PRE-condition failed: ranks {sorted(expected_added)} should NOT be active. "
                    f"train already active: {sorted(already_active_train)}, val already active: {sorted(already_active_val)}. "
                    f"Current state: train={sorted(train_active_before)}, val={sorted(val_active_before)}. "
                    f"This indicates state desync between SchedRL and ROLL."
                )
            self._expand_workers(dp_ranks_to_add=list(dp_ranks_to_add), train_skip_load=False)
            # Verify expand: ranks should be added to active_dp_ranks
            train_active_after = ray.get(self.train_rollout_scheduler.get_active_dp_ranks.remote())
            val_active_after = ray.get(self.val_rollout_scheduler.get_active_dp_ranks.remote())
            missing_train = expected_added - train_active_after
            missing_val = expected_added - val_active_after
            if missing_train or missing_val:
                raise RuntimeError(
                    f"Expand verification failed: ranks {sorted(expected_added)} should be active. "
                    f"train missing: {sorted(missing_train)}, val missing: {sorted(missing_val)}. "
                    f"Before: train={sorted(train_active_before)}, val={sorted(val_active_before)}. "
                    f"After: train={sorted(train_active_after)}, val={sorted(val_active_after)}."
                )

        return ActionResponse(success=True)
