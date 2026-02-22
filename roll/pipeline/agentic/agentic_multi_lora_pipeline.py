import os
import time
import uuid
from dataclasses import replace
from typing import Any

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.timer import _Timer

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import AgenticConfig, EnvManagerConfig
from roll.pipeline.agentic.agentic_pipeline import compute_rollout_traj_metrics, compute_train_data_metrics
from roll.pipeline.agentic.utils import (
    agentic_compute_advantage,
    compute_discounted_returns,
    compute_response_level_rewards,
    dump_rollout_trajectories,
    get_agentic_response_level_mask,
)
from roll.pipeline.base_pipeline import BasePipeline
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import (
    RunningMoments,
    agg_loss,
    compute_token_reward,
    masked_mean,
    reduce_metrics,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType
from roll.utils.lora_routing import normalize_domain
from roll.utils.train_infer_corrections import apply_train_infer_correction_to_batch


logger = get_logger()


def is_lora_training(pipeline_config: AgenticConfig) -> bool:
    return pipeline_config.actor_train.model_args.adapters is not None


class AgenticMultiLoraPipeline(BasePipeline):
    """
    Async multi-LoRA Agentic pipeline:
    - multiple env tags sampled concurrently
    - each batch routes via non_tensor_batch["lora_name"]
    - per-adapter optimizer stepping via actor_train.train_step_lora([...])
    """

    def __init__(self, pipeline_config: AgenticConfig):
        super().__init__(pipeline_config)
        self.pipeline_config: AgenticConfig

        self.pipeline_config.set_max_steps(max_steps=self.pipeline_config.max_steps)
        if not is_lora_training(self.pipeline_config):
            raise RuntimeError(
                "AgenticMultiLoraPipeline requires LoRA adapters (actor_train.model_args.adapters). "
                "For full fine-tune (FFT), use AgenticPipeline. "
                "FFT reference requires a separate frozen reference model/cluster (not disable_adapter)."
            )

        actor_infer_strategy = getattr(self.pipeline_config.actor_infer, "strategy_args", None)
        if actor_infer_strategy is not None and getattr(actor_infer_strategy, "strategy_name", None) == "vllm":
            strategy_config = actor_infer_strategy.strategy_config or {}
            sleep_level = int(strategy_config.get("sleep_level", 1))
            if sleep_level != 1:
                raise RuntimeError(
                    "AgenticMultiLoraPipeline requires vLLM sleep_level=1. "
                    "In vLLM 0.8.4, sleep_level=2 discards weights (no CPU backup), so offload→load can restore garbage."
                )

        # For multi-LoRA training, reference is the same backbone with LoRA disabled.
        # Use actor_train.disable_adapter() to compute ref_log_probs; do not create a separate reference cluster.
        self.use_ref_model = False

        self.partial_gpu_mode: bool = False

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )

        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        download_clusters = [self.actor_train, self.actor_infer]

        if self.use_ref_model:
            self.reference: Any = Cluster(
                name=self.pipeline_config.reference.name,
                worker_cls=self.pipeline_config.reference.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.reference,
            )
            download_clusters.append(self.reference)

        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )
            download_clusters.append(self.critic)

        # INIT PHASE: Download models and tokenizer
        self.download_models(*download_clusters)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        # INIT PHASE: Initialize clusters
        refs: list[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)
        if self.use_ref_model:
            self.reference.initialize(pipeline_config=self.pipeline_config, blocking=True)

        # INIT PHASE: Model update pairing (train -> infer)
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

        # Hardcoded constraint: partial_gpu_mode must remain true for this standalone multi-LoRA pipeline.
        if hasattr(self.pipeline_config, "partial_gpu_mode") and self.pipeline_config.partial_gpu_mode is False:
            raise RuntimeError(
                "AgenticMultiLoraPipeline: partial_gpu_mode must be true (hardcoded constraint)."
            )
        self.partial_gpu_mode = self._validate_partial_gpu_config()

        # Per-tag rollout schedulers (shared actor_infer).
        self.rollout_schedulers: dict[str, Any] = {}
        base_env: EnvManagerConfig = self.pipeline_config.train_env_manager
        for tag, n_group in zip(base_env.tags, base_env.num_groups_partition):
            env_cfg = replace(base_env)
            env_cfg.tags = [tag]
            env_cfg.num_groups_partition = [n_group]
            env_cfg.num_env_groups = n_group
            env_cfg.name = f"train_env_{tag}"
            # Recompute derived fields (world_size, env_configs placeholder, etc) after mutation.
            env_cfg.__post_init__()
            # NOTE: AgenticConfig computes train_env_manager.max_traj_per_env based on the *global* env count,
            # but in this multi-tag pipeline each tag gets its own RolloutScheduler with its own env subset.
            # Ensure each per-tag scheduler can actually produce `rollout_batch_size` trajectories per tick;
            # otherwise GroupQueueManager.get_batch() can block forever once it exhausts its per-step groups.
            train_env_num = env_cfg.num_env_groups * env_cfg.group_size
            traj_per_env = (self.pipeline_config.rollout_batch_size + train_env_num - 1) // train_env_num
            if env_cfg.max_traj_per_env < traj_per_env:
                logger.warning(
                    "Overriding per-tag max_traj_per_env to avoid get_batch deadlock: "
                    f"tag={tag!r} max_traj_per_env={env_cfg.max_traj_per_env} -> {traj_per_env} "
                    f"(rollout_batch_size={self.pipeline_config.rollout_batch_size} train_env_num={train_env_num})"
                )
                env_cfg.max_traj_per_env = traj_per_env
            # Recompute env_configs for this per-tag manager.
            self.pipeline_config.make_env_configs(env_cfg)
            self.rollout_schedulers[tag] = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-train-{tag}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                config=self.pipeline_config,
                env_manager_config=env_cfg,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="train",
            )

        # Initial model update to register/load adapters on inference before first rollout.
        self._initial_model_update()
        self._maybe_init_ml_tracker_runs()

    def _maybe_init_ml_tracker_runs(self) -> None:
        """
        Eagerly initialize ml_tracker runs at startup (instead of init-on-first-log).

        This makes ml_tracker failures fail-fast and ensures the "ml_tracker init with ..."
        line appears even if the job crashes before the first training tick.
        """
        if self.pipeline_config.track_with != "ml_tracker":
            return
        adapters = self.pipeline_config.actor_train.model_args.adapters or {}
        if not adapters:
            return
        adapter_names = sorted(adapters.keys())
        logger.info("Initializing ml_tracker runs for adapters: %s", adapter_names)
        for name in adapter_names:
            self.tracker.log(
                values={"system/init": 1, "system/lora_name": name},
                step=0,
                lora_name=name,
            )

    def _verify_lora_model_update(self, *, adapters: set[str] | None, where: str) -> None:
        """Fail-fast verification that infer workers can see updated LoRA adapters."""
        if not adapters:
            return
        if self.pipeline_config.actor_infer.model_args.adapters is None:
            raise RuntimeError(
                f"{where}: actor_infer.model_args.adapters is not configured; cannot verify LoRA model update."
            )

        timeout_s = float(os.environ.get("ROLL_VERIFY_LORA_TIMEOUT_S", "30"))
        adapter_names = sorted(adapters)

        ray.get(
            [
                w.wait_loras_ready.remote(adapter_names=adapter_names, timeout_s=timeout_s)
                for w in self.actor_infer.workers
            ]
        )
        for adapter_name in adapter_names:
            lora_ids = ray.get([w.get_lora_id.remote(adapter_name) for w in self.actor_infer.workers])
            if not lora_ids or lora_ids[0] is None:
                raise RuntimeError(f"{where}: infer workers missing adapter id: adapter={adapter_name!r} ids={lora_ids!r}")
            first = lora_ids[0]
            if any(lora_id != first for lora_id in lora_ids):
                raise RuntimeError(
                    f"{where}: inconsistent adapter id across infer workers: adapter={adapter_name!r} ids={lora_ids!r}"
                )

    def _initial_model_update(self) -> None:
        if self.pipeline_config.async_pipeline:
            self.actor_infer.offload_states(include=OffloadStateType.other_params)
        adapters = set(self.pipeline_config.actor_train.model_args.adapters.keys()) if self.pipeline_config.actor_train.model_args.adapters else None
        _ = self.model_update_lora_subset(global_step=0, adapters_to_update=adapters)
        self.actor_infer.load_states()
        self._verify_lora_model_update(adapters=adapters, where="initial_model_update")

    def adjust_batch(self, data: DataProto, mode: str = "copy") -> DataProto:
        # Reuse AgenticPipeline.adjust_batch to keep behavior identical.
        from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline

        return AgenticPipeline.adjust_batch(self, data=data, mode=mode)  # type: ignore[misc]

    def _validate_partial_gpu_config(self) -> bool:
        train_devices = set(self.actor_train.worker_config.device_mapping)
        infer_devices = set(self.actor_infer.worker_config.device_mapping)
        critic_devices = set(self.critic.worker_config.device_mapping) if hasattr(self, "critic") and self.critic else set()
        use_ref_model = bool(getattr(self, "use_ref_model", False))
        ref_devices = set(self.reference.worker_config.device_mapping) if use_ref_model else set()

        if not train_devices or not infer_devices:
            raise ValueError(
                "device_mapping cannot be empty: "
                f"train={list(train_devices)}, infer={list(infer_devices)}"
            )

        if use_ref_model:
            assert ref_devices == train_devices, (
                "Reference device_mapping must match actor_train exactly: "
                f"ref={list(ref_devices)}, train={list(train_devices)}"
            )

        if train_devices.isdisjoint(infer_devices):
            raise RuntimeError(
                "AgenticMultiLoraPipeline does not support disjoint actor_train/actor_infer device_mapping. "
                "Use partial overlap (actor_train ⊂ actor_infer) so inference can continue on remaining GPUs while "
                "training runs."
            )

        if train_devices.issubset(infer_devices) and len(train_devices) < len(infer_devices):
            logger.info("Detected Configuration Model B: Subset device_mapping, partial_gpu_mode=True")
            infer_dp_size = self.actor_infer.worker_config.world_size
            assert infer_dp_size >= 2, (
                f"partial_gpu_mode requires actor_infer.dp_size >= 2, got {infer_dp_size}"
            )
            async_ratio = self.pipeline_config.async_generation_ratio
            assert async_ratio >= 0, f"async_generation_ratio must be >= 0, got {async_ratio}"

            if hasattr(self, "critic") and self.critic is not None:
                assert critic_devices.issubset(infer_devices), (
                    "Critic device_mapping must be subset of actor_infer: "
                    f"critic={list(critic_devices)}, infer={list(infer_devices)}"
                )
                assert critic_devices.isdisjoint(train_devices), (
                    "Critic device_mapping must be disjoint from actor_train: "
                    f"critic={list(critic_devices)}, train={list(train_devices)}"
                )

            infer_strategy_config = self.actor_infer.worker_config.strategy_args.strategy_config
            tp_size = infer_strategy_config.get("tensor_parallel_size", 1)
            pp_size = infer_strategy_config.get("pipeline_parallel_size", 1)
            assert tp_size >= 1 and pp_size >= 1, f"tp_size and pp_size must be >= 1: tp={tp_size}, pp={pp_size}"

            expected_gpu_count = tp_size * pp_size * infer_dp_size
            actual_gpu_count = len(infer_devices)
            assert expected_gpu_count == actual_gpu_count, (
                "Parallelism configuration mismatch: "
                f"tp_size * pp_size * dp_size = {tp_size} * {pp_size} * {infer_dp_size} = {expected_gpu_count}, "
                f"but device_mapping has {actual_gpu_count} GPUs"
            )

            gpus_per_dp_rank = tp_size * pp_size
            freed_gpus = train_devices | critic_devices
            self._validate_minimum_active_ranks(infer_dp_size, infer_devices, list(freed_gpus), gpus_per_dp_rank)
            logger.info(f"Partial GPU mode validated: infer_dp_size={infer_dp_size}, freed_gpus={sorted(freed_gpus)}")
            return True

        if len(train_devices) == len(infer_devices):
            raise RuntimeError(
                "AgenticMultiLoraPipeline does not support actor_train/actor_infer colocating mode "
                "(train device_mapping == infer device_mapping). Use partial overlap (actor_train ⊂ actor_infer)."
            )

        raise RuntimeError(
            "Unsupported device_mapping relationship for AgenticMultiLoraPipeline. "
            f"train={sorted(train_devices)} infer={sorted(infer_devices)}"
        )

    def _validate_minimum_active_ranks(
        self,
        infer_dp_size: int,
        infer_devices: set,
        freed_gpu_list: list,
        gpus_per_dp_rank: int,
    ) -> None:
        freed_gpu_set = set(freed_gpu_list)
        if not freed_gpu_set.issubset(infer_devices):
            raise ValueError(
                "Freed GPUs (train + critic) must be subset of infer device_mapping: "
                f"freed={sorted(freed_gpu_list)}, infer={sorted(infer_devices)}"
            )

        infer_devices_list = sorted(list(infer_devices))
        at_least_one_active = False
        for dp_rank in range(infer_dp_size):
            start_idx = dp_rank * gpus_per_dp_rank
            end_idx = start_idx + gpus_per_dp_rank
            dp_rank_gpus = set(infer_devices_list[start_idx:end_idx])
            if dp_rank_gpus.isdisjoint(freed_gpu_set):
                at_least_one_active = True
                break

        if not at_least_one_active:
            raise ValueError(
                "At least 1 DP rank must remain active after shrink. "
                f"All {infer_dp_size} DP ranks have at least one GPU in freed set. "
                f"infer_devices={sorted(infer_devices_list)}, freed_gpus={sorted(freed_gpu_list)}, "
                f"gpus_per_rank={gpus_per_dp_rank}"
            )

    def _ensure_sample_uuid(self, batch: DataProto) -> None:
        if "sample_uuid" in batch.non_tensor_batch:
            sample_uuid = batch.non_tensor_batch["sample_uuid"]
            if not (isinstance(sample_uuid, np.ndarray) and sample_uuid.dtype == object):
                raise RuntimeError(
                    f"Invalid non_tensor_batch['sample_uuid'] type: {type(sample_uuid)} dtype={getattr(sample_uuid, 'dtype', None)}"
                )
            return

        if batch.batch is None:
            raise RuntimeError("Cannot derive sample_uuid: batch.batch is None.")
        batch_size = int(batch.batch.batch_size[0])

        if "traj_id" in batch.non_tensor_batch:
            traj_id = batch.non_tensor_batch["traj_id"]
            if not (isinstance(traj_id, np.ndarray) and traj_id.dtype == object and len(traj_id) == batch_size):
                raise RuntimeError(
                    "Invalid non_tensor_batch['traj_id'] for sample_uuid derivation: "
                    f"type={type(traj_id)} dtype={getattr(traj_id, 'dtype', None)} len={len(traj_id) if hasattr(traj_id, '__len__') else None} "
                    f"expected_len={batch_size}"
                )
            sample_uuids = [f"{tid}_{i}" for i, tid in enumerate(traj_id.tolist())]
        else:
            sample_uuids = [str(uuid.uuid4()) for _ in range(batch_size)]

        batch.non_tensor_batch["sample_uuid"] = np.asarray(sample_uuids, dtype=object)

    def _prepare_batch(self, batch: DataProto, metrics: dict) -> DataProto:
        batch = compute_discounted_returns(batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma)

        batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
        self._ensure_sample_uuid(batch)

        # Reference log probs (per adapter)
        with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
            if self.pipeline_config.enable_reference:
                batch.meta_info["is_offload_states"] = False
                if self.use_ref_model:
                    ref_log_probs: DataProto = self.reference.compute_log_probs(batch, blocking=True)
                else:
                    batch.meta_info["disable_adapter"] = True
                    ref_log_probs = self.actor_train.compute_log_probs(batch, blocking=True)
                    batch.meta_info.pop("disable_adapter", None)
                batch.batch["ref_log_probs"] = ref_log_probs.batch["log_probs"]
                avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
        metrics["time/step_ref_log_probs_values_reward"] = cal_timer.last

        # Old logprobs (for PPO ratio)
        with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
            batch.meta_info["is_offload_states"] = False
            if self.pipeline_config.enable_old_logprobs_recompute:
                if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                    batch, dynamic_batching_metrics = dynamic_batching_shard(
                        batch,
                        self.actor_train.dp_size,
                        self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                        self.pipeline_config.actor_train.sequence_length_round_in_infer,
                        self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                        self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                        "actor_train/compute_log_probs",
                    )
                    metrics.update(dynamic_batching_metrics)
                old_log_probs: DataProto = self.actor_train.compute_log_probs(batch, blocking=True)
                batch.batch["old_log_probs"] = old_log_probs.batch["log_probs"]
                avg_old_log_prob = masked_mean(batch.batch["old_log_probs"], batch.batch["response_mask"][:, 1:])
                metrics.update({"critic/old_log_prob/mean": avg_old_log_prob.item()})
                metrics.update(reduce_metrics(old_log_probs.meta_info.pop("metrics", {})))
                agg_entropy = agg_loss(
                    loss_mat=old_log_probs.batch["entropy"],
                    loss_mask=batch.batch["response_mask"][:, 1:],
                    loss_agg_mode="token-mean",
                )
                metrics.update({"critic/entropy/mean": agg_entropy.item()})
            else:
                batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])

            if self.pipeline_config.adv_estimator == "gae":
                values_refs: list[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                values = DataProto.materialize_concat(data_refs=values_refs)
                batch = batch.union(values)
                metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))

            # Reference logprobs (if reference disabled, mock with old_log_probs)
            if not self.pipeline_config.enable_reference:
                batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
        metrics["time/step_old_log_probs_values"] = cal_old_logpb_timer.last

        # Token/segment response-level mask (filters)
        with Timer(name="cal_response_level_mask", logger=None) as timer:
            batch, mask_metrics = get_agentic_response_level_mask(batch, self.pipeline_config)
            metrics.update(mask_metrics)
        metrics["time/step_cal_response_level_mask"] = timer.last

        # Rewards
        with Timer(name="cal_response_norm_rewards", logger=None) as timer:
            batch, reward_metrics = compute_response_level_rewards(batch=batch, pipeline_config=self.pipeline_config)
            metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
            metrics.update(reward_metrics)
        metrics["time/step_cal_norm_rewards"] = timer.last

        # Token-level rewards (KL controller etc)
        with Timer(name="cal_token_reward", logger=None) as timer:
            batch, token_level_metrics = compute_token_reward(batch, self.pipeline_config, self.kl_ctrl)
            metrics.update(token_level_metrics)
        metrics["time/step_cal_token_reward"] = timer.last

        # Advantages
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
            # Generate train_infer_is_weight and apply optional correction filters before actor training.
            batch, corr_metrics = apply_train_infer_correction_to_batch(
                self.pipeline_config,
                batch,
                update_mask_keys=batch.meta_info["loss_mask_keys"],
            )
            metrics.update(corr_metrics)
        return batch

    @torch.no_grad()
    def run(self):
        if not is_lora_training(self.pipeline_config):
            raise RuntimeError("AgenticMultiLoraPipeline requires actor_train.model_args.adapters to be configured.")

        success = False
        try:
            max_steps_per_adapter = int(self.pipeline_config.max_steps)
            adapters = list(self.pipeline_config.actor_train.model_args.adapters.keys())
            lora_step: dict[str, int] = {name: 0 for name in adapters}
            global_tick = 0
            # Adapter keys in model_args.adapters are canonical lowercase (normalized in __post_init__).
            tag_to_adapter = {tag: normalize_domain(tag) for tag in self.rollout_schedulers.keys()}
            unknown = sorted({a for a in tag_to_adapter.values() if a not in lora_step})
            if unknown:
                raise RuntimeError(
                    f"Train env tags must map to configured LoRA adapters. Unknown adapters from tags: {unknown}. "
                    f"adapters={sorted(lora_step.keys())} tag_to_adapter={tag_to_adapter}"
                )

            # Calculate tokens-per-second system throughput
            tps_timer = _Timer(window_size=5)
            pipeline_start_mono = time.monotonic()

            # Kick off one in-flight get_batch per tag.
            in_flight: dict[str, ray.ObjectRef] = {}
            pending_by_tag: dict[str, DataProto] = {}
            submitted_at_mono: dict[str, float] = {}
            tags = list(self.rollout_schedulers.keys())
            for tag in tags:
                adapter = tag_to_adapter[tag]
                if lora_step.get(adapter, 0) >= max_steps_per_adapter:
                    continue
                data = DataProto(meta_info={"global_step": global_tick})
                in_flight[tag] = self.rollout_schedulers[tag].get_batch.remote(
                    data, self.pipeline_config.rollout_batch_size
                )
                submitted_at_mono[tag] = time.monotonic()

            stall_timeout_s = float("inf")
            wait_poll_s = 30.0
            last_any_ready_mono = time.monotonic()
            wait_ready_since_mono: float | None = None
            barrier_mode = bool(getattr(self.pipeline_config, "multi_lora_barrier_mode", False))
            last_get_batch_done_ts_by_adapter: dict[str, float] = {}
            last_train_step_done_ts_by_adapter: dict[str, float] = {}
            last_train_step_done_ts_global: float | None = None

            while any(lora_step[name] < max_steps_per_adapter for name in adapters):
                active_tags = [tag for tag in tags if lora_step.get(tag_to_adapter[tag], 0) < max_steps_per_adapter]
                active_tags_in_flight = [tag for tag in active_tags if tag in in_flight]
                active_refs = [in_flight[tag] for tag in active_tags_in_flight]
                assert len(active_refs) > 0

                if wait_ready_since_mono is None:
                    wait_ready_since_mono = time.monotonic()
                required_ready = len(active_refs) if barrier_mode else 1
                ready, _ = ray.wait(active_refs, num_returns=required_ready, timeout=wait_poll_s)
                if len(ready) < required_ready:
                    now_mono = time.monotonic()
                    oldest_age_s = 0.0
                    ages = {}
                    for tag in active_tags_in_flight:
                        submitted_mono = submitted_at_mono.get(tag)
                        if submitted_mono is None:
                            raise RuntimeError(f"Missing submitted_at timestamp for in_flight tag={tag!r}")
                        age = now_mono - submitted_mono
                        ages[tag] = round(age, 3)
                        oldest_age_s = max(oldest_age_s, age)
                    if barrier_mode:
                        logger.info(
                            "Waiting for get_batch (barrier)... "
                            f"global_tick={global_tick} lora_step={lora_step} "
                            f"in_flight={sorted(in_flight.keys())} pending={sorted(pending_by_tag.keys())} "
                            f"ready={len(ready)}/{len(active_refs)} ages_s={ages}"
                        )
                    else:
                        logger.info(
                            "Waiting for get_batch... "
                            f"global_tick={global_tick} lora_step={lora_step} "
                            f"in_flight={sorted(in_flight.keys())} pending={sorted(pending_by_tag.keys())} "
                            f"ages_s={ages}"
                        )
                    if ready:
                        last_any_ready_mono = now_mono
                    if now_mono - last_any_ready_mono >= stall_timeout_s or oldest_age_s >= stall_timeout_s:
                        raise RuntimeError(
                            f"Timeout waiting for get_batch (stall >= {stall_timeout_s:.0f}s). "
                            f"global_tick={global_tick} lora_step={lora_step} "
                            f"in_flight={sorted(in_flight.keys())} pending={sorted(pending_by_tag.keys())} ages_s={ages}"
                        )
                    continue

                ready_now_mono = time.monotonic()
                if wait_ready_since_mono is None:
                    raise RuntimeError("wait_ready_since_mono is None when ready refs are returned")
                tick_wait_ready_batch_s = ready_now_mono - wait_ready_since_mono
                wait_ready_since_mono = None
                last_any_ready_mono = ready_now_mono

                if barrier_mode:
                    for tag in active_tags_in_flight:
                        ref = in_flight[tag]
                        batch = ray.get(ref)
                        if batch is None:
                            raise RuntimeError(f"get_batch returned None for tag={tag!r}")
                        batch.meta_info.setdefault("metrics", {})
                        batch.meta_info["metrics"]["time/ray_wait_ready_batch_s"] = tick_wait_ready_batch_s
                        adapter_name = tag_to_adapter.get(tag, tag)
                        get_batch_done_ts = time.monotonic() - pipeline_start_mono
                        batch.meta_info["metrics"]["time/get_batch_done_ts"] = get_batch_done_ts
                        issue_mono = submitted_at_mono.get(tag)
                        if issue_mono is None:
                            raise RuntimeError(f"Missing submitted_at timestamp for ready tag={tag!r}")
                        issue_ts = issue_mono - pipeline_start_mono
                        batch.meta_info["metrics"]["time/get_batch_issue_ts"] = issue_ts
                        batch.meta_info["metrics"]["time/get_batch_latency_s"] = get_batch_done_ts - issue_ts
                        prev_done_ts = last_get_batch_done_ts_by_adapter.get(adapter_name)
                        batch.meta_info["metrics"]["time/get_batch_done_interval_s"] = (
                            0.0 if prev_done_ts is None else get_batch_done_ts - prev_done_ts
                        )
                        last_get_batch_done_ts_by_adapter[adapter_name] = get_batch_done_ts
                        if "get_batch_return_start_time" in batch.meta_info:
                            batch.meta_info["metrics"]["time/get_batch_cost_train"] = (
                                time.time() - batch.meta_info.pop("get_batch_return_start_time")
                            )
                        pending_by_tag[tag] = batch
                        in_flight.pop(tag, None)
                        start_mono = submitted_at_mono.pop(tag, None)
                        if start_mono is None:
                            raise RuntimeError(f"Missing submitted_at timestamp for popped tag={tag!r}")
                        wait_s = time.monotonic() - start_mono
                        batch.meta_info["metrics"]["time/get_batch_wait_s"] = wait_s
                        logger.info(f"get_batch done tag={tag!r} global_tick={global_tick} elapsed_s={wait_s:.3f}")
                else:
                    # Single-adapter tick: consume exactly one ready batch per train_step_lora call.
                    ready_ref = ready[0]
                    ready_tag = next((t for t, r in in_flight.items() if r == ready_ref), None)
                    if ready_tag is None:
                        raise RuntimeError("ray.wait returned a ref that is not tracked in in_flight")

                    batch = ray.get(ready_ref)
                    if batch is None:
                        raise RuntimeError(f"get_batch returned None for tag={ready_tag!r}")
                    # Align with AgenticPipeline timing metrics:
                    # - time/get_batch_cost_train from rollout scheduler's internal marker (if present)
                    # - time/step_rollout approximated later as (wait + preprocess) per adapter
                    batch.meta_info.setdefault("metrics", {})
                    batch.meta_info["metrics"]["time/ray_wait_ready_batch_s"] = tick_wait_ready_batch_s
                    adapter_name = tag_to_adapter.get(ready_tag, ready_tag)
                    get_batch_done_ts = time.monotonic() - pipeline_start_mono
                    batch.meta_info["metrics"]["time/get_batch_done_ts"] = get_batch_done_ts
                    issue_mono = submitted_at_mono.get(ready_tag)
                    if issue_mono is None:
                        raise RuntimeError(f"Missing submitted_at timestamp for ready tag={ready_tag!r}")
                    issue_ts = issue_mono - pipeline_start_mono
                    batch.meta_info["metrics"]["time/get_batch_issue_ts"] = issue_ts
                    batch.meta_info["metrics"]["time/get_batch_latency_s"] = get_batch_done_ts - issue_ts
                    prev_done_ts = last_get_batch_done_ts_by_adapter.get(adapter_name)
                    batch.meta_info["metrics"]["time/get_batch_done_interval_s"] = (
                        0.0 if prev_done_ts is None else get_batch_done_ts - prev_done_ts
                    )
                    last_get_batch_done_ts_by_adapter[adapter_name] = get_batch_done_ts
                    if "get_batch_return_start_time" in batch.meta_info:
                        batch.meta_info["metrics"]["time/get_batch_cost_train"] = (
                            time.time() - batch.meta_info.pop("get_batch_return_start_time")
                        )
                    pending_by_tag[ready_tag] = batch
                    in_flight.pop(ready_tag, None)
                    start_mono = submitted_at_mono.pop(ready_tag, None)
                    if start_mono is None:
                        raise RuntimeError(f"Missing submitted_at timestamp for popped tag={ready_tag!r}")
                    wait_s = time.monotonic() - start_mono
                    batch.meta_info["metrics"]["time/get_batch_wait_s"] = wait_s
                    logger.info(f"get_batch done tag={ready_tag!r} global_tick={global_tick} elapsed_s={wait_s:.3f}")

                if not pending_by_tag:
                    continue

                # Greedy tick: once any tag has a ready batch, proceed to train. In partial-GPU mode, `shrink_sampler`
                # relies on RequestScheduler to abort/remap + update routing safely for any in-flight requests.

                tick_metrics: dict = {}
                per_adapter_metrics: dict[str, dict] = {}
                shrink_duration_s: Optional[float] = None
                with Timer(name="pipeline_tick_total", logger=None) as tick_timer:
                    with tps_timer:
                        # Partial GPU: shrink inference off training GPUs before training.
                        if self.partial_gpu_mode:
                            with Timer(name="cal_ref_log_probs", logger=None) as shrink_timer:
                                target_gpus: list[int] = []
                                if hasattr(self.actor_train.worker_config, "device_mapping") and self.actor_train.worker_config.device_mapping:
                                    target_gpus.extend(self.actor_train.worker_config.device_mapping)
                                if hasattr(self, "critic") and self.critic is not None:
                                    if hasattr(self.critic.worker_config, "device_mapping") and self.critic.worker_config.device_mapping:
                                        target_gpus.extend(self.critic.worker_config.device_mapping)
                                if target_gpus:
                                    # We rely on RequestScheduler.shrink_workers() (under each RolloutScheduler) to
                                    # abort/remap in-flight requests and update routing atomically. Rollouts may
                                    # continue on the remaining (non-overlap) inference workers while training runs.
                                    if os.environ.get("ROLL_LOG_PARTIAL_GPU_OPS", "0") == "1":
                                        logger.info(
                                            "PartialGPU tick=%s shrink start: target_gpus=%s active_tags=%d pending_tags=%d",
                                            global_tick,
                                            target_gpus,
                                            len(active_tags),
                                            len(pending_by_tag),
                                        )
                                    # Multi-scheduler safety: shrink (routing update + abort/drain) must be applied to
                                    # every RequestScheduler that can dispatch to the soon-to-be-offloaded ranks.
                                    #
                                    # Barrier is applied to the target dp_ranks only:
                                    # 1) shrink ALL schedulers with skip_offload=True so none can route to offload ranks
                                    # 2) wait until ALL schedulers report zero in-flight on those ranks
                                    # 3) offload ONCE (scheduler[0]) for those ranks
                                    schedulers = list(self.rollout_schedulers.values())
                                    offload_ranks = ray.get(schedulers[0].get_offload_ranks_for_target_gpus.remote(target_gpus))
                                    shrink_metrics_list = ray.get(
                                        [sched.shrink_sampler.remote(target_gpus, skip_offload=True) for sched in schedulers]
                                    )

                                    drain_timeout_s = float(os.environ.get("ROLL_VLLM_DRAIN_TIMEOUT_S", "30"))
                                    deadline = time.monotonic() + max(1.0, drain_timeout_s)
                                    while True:
                                        inflight_list = ray.get(
                                            [sched.get_inflight_counts.remote(offload_ranks) for sched in schedulers]
                                        )
                                        if all(all(v == 0 for v in inflight.values()) for inflight in inflight_list):
                                            break
                                        if time.monotonic() >= deadline:
                                            raise RuntimeError(
                                                "PartialGPU shrink timed out waiting for in-flight drain on offload ranks: "
                                                f"offload_ranks={offload_ranks} inflight={inflight_list}"
                                            )
                                        time.sleep(0.2)

                                    offload_metrics = ray.get(schedulers[0].offload_dp_ranks.remote(offload_ranks))

                                    for idx, shrink_metrics in enumerate(shrink_metrics_list):
                                        tick_metrics.update({f"shrink/{idx}/{k}": v for k, v in shrink_metrics.items()})
                                    tick_metrics.update({f"shrink/offload/{k}": v for k, v in offload_metrics.items()})
                                    if os.environ.get("ROLL_LOG_PARTIAL_GPU_OPS", "0") == "1":
                                        logger.info(
                                            "PartialGPU tick=%s shrink done: metrics=%s",
                                            global_tick,
                                            [
                                                {
                                                    "idx": idx,
                                                    "aborted": m.get("aborted"),
                                                    "remapped": m.get("remapped"),
                                                    "offload_ranks": m.get("offload_ranks"),
                                                }
                                                for idx, m in enumerate(shrink_metrics_list)
                                            ],
                                        )
                            shrink_duration_s = float(shrink_timer.last)

                        # Collect actor inference metrics once per tick
                        actor_infer_metrics = self.actor_infer.get_metrics()
                        actor_infer_reduced = {}
                        if "metrics" in actor_infer_metrics.meta_info:
                            actor_infer_reduced = reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {}))

                        # Prepare each tag-batch independently, then train in one batched call.
                        prepared: list[DataProto] = []
                        prepared_by_adapter: dict[str, list[DataProto]] = {}
                        dirty_adapters: set[str] = set()
                        for tag, batch in pending_by_tag.items():
                            adapter_for_tag = tag_to_adapter[tag]
                            adapter_metrics = per_adapter_metrics.setdefault(adapter_for_tag, {})
                            if actor_infer_reduced:
                                adapter_metrics.update(actor_infer_reduced)
                            tick_wait_ready_batch_s = float(
                                batch.meta_info.get("metrics", {}).get("time/ray_wait_ready_batch_s", 0.0) or 0.0
                            )
                            tick_metrics["time/ray_wait_ready_batch_s"] = tick_wait_ready_batch_s
                            adapter_metrics["time/ray_wait_ready_batch_s"] = tick_wait_ready_batch_s

                            wait_s = float(batch.meta_info.get("metrics", {}).get("time/get_batch_wait_s", 0.0) or 0.0)
                            batch.meta_info.setdefault("global_step", global_tick)
                            batch.meta_info["_broadcast_non_tensor_batch"] = True
                            # Keep strategy token-count accounting contract identical to agentic_pipeline.
                            batch.meta_info["loss_mask_keys"] = ["response_mask"]
                            with Timer(name="rollout", logger=None) as rollout_timer:
                                adapter_metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                                adapter_metrics.update(compute_rollout_traj_metrics(batch))
                                dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_tick, batch)
                            adapter_metrics["time/step_rollout"] = rollout_timer.last + wait_s

                            prepared_batch = self._prepare_batch(batch, adapter_metrics)
                            prepared.append(prepared_batch)

                            # Track which adapter(s) stepped this tick.
                            lora_names = prepared_batch.non_tensor_batch["lora_name"]
                            unique = list(dict.fromkeys(lora_names.tolist()))
                            if len(unique) != 1:
                                raise RuntimeError(f"Expected homogeneous lora_name per prepared batch, got {unique}")
                            adapter_name = str(unique[0])
                            if adapter_name != adapter_for_tag:
                                merged = per_adapter_metrics.setdefault(adapter_name, {})
                                merged.update(adapter_metrics)
                                adapter_metrics = merged
                            dirty_adapters.add(adapter_name)
                            prepared_by_adapter.setdefault(adapter_name, []).append(prepared_batch)

                        # Train (per-adapter optimizer mode). In barrier mode this concatenates all tags' batches.
                        with Timer(name="train_timer", logger=None) as train_timer:
                            train_input = prepared[0] if len(prepared) == 1 else DataProto.concat(prepared)
                            if os.environ.get("ROLL_DEBUG_TRAIN_STEP_INPUTS", "0") == "1":
                                lora_arr = train_input.non_tensor_batch.get("lora_name", None)
                                if lora_arr is None:
                                    raise RuntimeError("ROLL_DEBUG_TRAIN_STEP_INPUTS requires non_tensor_batch['lora_name'] to exist.")
                                lora_list = [str(x) for x in lora_arr.tolist()]
                                lora_counts: dict[str, int] = {}
                                for name in lora_list:
                                    lora_counts[name] = lora_counts.get(name, 0) + 1

                                response_mask_sum = float(train_input.batch["response_mask"][:, 1:].sum().detach().item())
                                advantages_abs_sum = float(train_input.batch["advantages"].abs().sum().detach().item())
                                raw_advantages_abs_sum = float(
                                    train_input.batch.get("raw_advantages", train_input.batch["advantages"]).abs().sum().detach().item()
                                )
                                token_rewards_abs_sum = float(
                                    train_input.batch.get("token_level_rewards", torch.zeros_like(train_input.batch["advantages"]))
                                    .abs()
                                    .sum()
                                    .detach()
                                    .item()
                                )
                                seq_scores = train_input.batch["scores"].sum(dim=-1).detach()
                                seq_score_min = float(seq_scores.min().item())
                                seq_score_max = float(seq_scores.max().item())
                                logger.info(
                                    "train_step_lora inputs: global_tick=%s lora_counts=%s response_mask_sum=%s "
                                    "advantages_abs_sum=%s raw_advantages_abs_sum=%s token_rewards_abs_sum=%s seq_score_min=%s seq_score_max=%s",
                                    global_tick,
                                    lora_counts,
                                    response_mask_sum,
                                    advantages_abs_sum,
                                    raw_advantages_abs_sum,
                                    token_rewards_abs_sum,
                                    seq_score_min,
                                    seq_score_max,
                                )
                            if self.pipeline_config.adv_estimator == "gae":
                                critic_train_refs: list[ray.ObjectRef] = self.critic.train_step(train_input, blocking=False)
                            train_refs: list[ray.ObjectRef] = self.actor_train.train_step_lora(train_input, blocking=False)
                            train_metrics = DataProto.materialize_concat(data_refs=train_refs)
                            reduced_train_metrics = reduce_metrics(train_metrics.meta_info.pop("metrics", {}))
                            tick_metrics.update(reduced_train_metrics)
                            if self.pipeline_config.adv_estimator == "gae":
                                critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_refs)
                                tick_metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                            tps_timer.push_units_processed(n=torch.sum(train_input.batch["attention_mask"]).detach().item())
                        train_step_s = float(train_timer.last)
                        train_step_done_ts = time.monotonic() - pipeline_start_mono
                        tick_metrics["time/train_step_done_ts"] = train_step_done_ts
                        tick_metrics["time/train_step_done_interval_s"] = (
                            0.0
                            if last_train_step_done_ts_global is None
                            else train_step_done_ts - last_train_step_done_ts_global
                        )
                        last_train_step_done_ts_global = train_step_done_ts
                        tick_metrics["system/tps"] = tps_timer.mean_throughput
                        for name in dirty_adapters:
                            adapter_metrics = per_adapter_metrics.setdefault(name, {})
                            adapter_metrics["time/step_train"] = train_step_s
                            adapter_metrics["time/step_train_step_lora"] = train_step_s
                            adapter_metrics["time/train_step_done_ts"] = train_step_done_ts
                            prev_train_done_ts = last_train_step_done_ts_by_adapter.get(name)
                            adapter_step_interval_s = (
                                0.0 if prev_train_done_ts is None else train_step_done_ts - prev_train_done_ts
                            )
                            adapter_metrics["time/train_step_done_interval_s"] = adapter_step_interval_s
                            last_train_step_done_ts_by_adapter[name] = train_step_done_ts
                            for k, v in reduced_train_metrics.items():
                                if f"/{name}/" in k:
                                    adapter_metrics[k] = v
                                else:
                                    adapter_metrics.setdefault(k, v)

                        # Update step counters.
                        for name in dirty_adapters:
                            if name in lora_step:
                                lora_step[name] += 1
                        global_tick += 1

                        tick_metrics["system/global_tick"] = global_tick
                        for name, step in lora_step.items():
                            tick_metrics[f"system/lora_step/{name}"] = step
                        for name in dirty_adapters:
                            adapter_metrics = per_adapter_metrics.setdefault(name, {})
                            adapter_metrics["system/global_tick"] = global_tick
                            adapter_metrics["system/lora_step"] = lora_step.get(name, global_tick)

                        # Model update boundary: suspend rollouts only for model_update.
                        with Timer(name="model_update", logger=None) as model_update_timer:
                            if os.environ.get("ROLL_LOG_PARTIAL_GPU_OPS", "0") == "1":
                                logger.info(
                                    "PartialGPU tick=%s model_update: suspend all schedulers (dirty_adapters=%s)",
                                    global_tick,
                                    sorted(dirty_adapters),
                                )
                            ray.get([sched.suspend.remote() for sched in self.rollout_schedulers.values()])
                            if self.pipeline_config.async_pipeline:
                                self.actor_infer.offload_states(include=OffloadStateType.other_params)
                            model_update_metrics = self.model_update_lora_subset(global_tick, adapters_to_update=dirty_adapters)
                            tick_metrics.update(model_update_metrics)
                            for name in dirty_adapters:
                                per_adapter_metrics.setdefault(name, {}).update(model_update_metrics)

                            # Partial GPU: expand routing state after model_update reloads to all GPUs.
                            if self.partial_gpu_mode and global_tick > 0:
                                target_gpus = []
                                if hasattr(self.actor_train.worker_config, "device_mapping") and self.actor_train.worker_config.device_mapping:
                                    target_gpus.extend(self.actor_train.worker_config.device_mapping)
                                if hasattr(self, "critic") and self.critic is not None:
                                    if hasattr(self.critic.worker_config, "device_mapping") and self.critic.worker_config.device_mapping:
                                        target_gpus.extend(self.critic.worker_config.device_mapping)
                                if target_gpus:
                                    if os.environ.get("ROLL_LOG_PARTIAL_GPU_OPS", "0") == "1":
                                        logger.info(
                                            "PartialGPU tick=%s expand start: target_gpus=%s",
                                            global_tick,
                                            target_gpus,
                                        )
                                    # Expand should (1) reload offloaded inference workers and (2) restore routing state.
                                    # Only the first scheduler performs the actual load; others only update routing.
                                    expand_metrics_list = ray.get(
                                        [
                                            sched.expand_sampler.remote(target_gpus, skip_load=(idx != 0))
                                            for idx, sched in enumerate(self.rollout_schedulers.values())
                                        ]
                                    )
                                    for idx, expand_metrics in enumerate(expand_metrics_list):
                                        tick_metrics.update({f"expand/{idx}/{k}": v for k, v in expand_metrics.items()})
                                        for name in dirty_adapters:
                                            per_adapter_metrics.setdefault(name, {}).update(
                                                {f"expand/{idx}/{k}": v for k, v in expand_metrics.items()}
                                            )
                                    if os.environ.get("ROLL_LOG_PARTIAL_GPU_OPS", "0") == "1":
                                        logger.info(
                                            "PartialGPU tick=%s expand done: metrics=%s",
                                            global_tick,
                                            [
                                                {
                                                    "idx": idx,
                                                    "aborted": m.get("aborted"),
                                                    "remapped": m.get("remapped"),
                                                    "load_ranks": m.get("load_ranks"),
                                                }
                                                for idx, m in enumerate(expand_metrics_list)
                                            ],
                                        )
                            else:
                                # Non-partial-GPU path: ensure inference weights are loaded before resuming rollouts.
                                self.actor_infer.load_states()
                            self._verify_lora_model_update(adapters=dirty_adapters, where=f"tick={global_tick}:model_update")
                            if os.environ.get("ROLL_LOG_PARTIAL_GPU_OPS", "0") == "1":
                                logger.info("PartialGPU tick=%s model_update: resume all schedulers", global_tick)
                            # We explicitly resume schedulers after model_update as a safety/unblock point.
                            #
                            # Note: `RolloutScheduler.get_batch()` always calls `generate_scheduler.resume()` before
                            # waiting for env outputs, so in the single-pipeline flow this resume is not strictly
                            # required. In multi-LoRA, env rollout loops keep running in the background and can hit
                            # `RequestScheduler.generate_one_request()` while `need_suspend=True` (they block on
                            # `_check_suspend()`). If the next `get_batch()` is delayed/skipped (e.g., extra work
                            # like expand/rebalance/logging or an early-return path), leaving schedulers suspended
                            # would stall rollout. This ensures we always unblock request dispatch immediately.
                            ray.get([sched.resume.remote() for sched in self.rollout_schedulers.values()])
                        model_update_s = float(model_update_timer.last)
                        tick_metrics["time/step_model_update"] = model_update_s
                        for name in dirty_adapters:
                            per_adapter_metrics.setdefault(name, {})["time/step_model_update"] = model_update_s

                        # Basic data metrics
                        for name, batches in prepared_by_adapter.items():
                            if not batches:
                                continue
                            with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                                per_adapter_metrics.setdefault(name, {}).update(
                                    compute_train_data_metrics(batch=DataProto.concat(batches))
                                )
                            per_adapter_metrics.setdefault(name, {})["time/step_compute_data_metrics"] = data_metrics_timer.last

                tick_total_s = float(tick_timer.last)
                for name in dirty_adapters:
                    per_adapter_metrics.setdefault(name, {})["time/tick_total"] = tick_total_s
                    per_adapter_metrics.setdefault(name, {})["time/step_log"] = 0.0
                    if shrink_duration_s is not None:
                        per_adapter_metrics.setdefault(name, {})["time/step_shrink"] = shrink_duration_s

                if self.pipeline_config.logging_steps > 0 and global_tick % self.pipeline_config.logging_steps == 0:
                    logger.info(f"tick={global_tick} lora_step={lora_step}")
                    logger.info(tick_metrics)

                if self.pipeline_config.track_with == "ml_tracker":
                    # Log to one ml_tracker run per LoRA adapter (via Ray actor).
                    for name in sorted(dirty_adapters):
                        per_lora_metrics = dict(per_adapter_metrics.get(name, {}))
                        per_lora_metrics["system/lora_name"] = name
                        self.tracker.log(values=per_lora_metrics, step=lora_step.get(name, global_tick), lora_name=name)
                else:
                    self.tracker.log(values=tick_metrics, step=global_tick)

                pending_by_tag.clear()
                for tag in tags:
                    adapter = tag_to_adapter[tag]
                    if lora_step.get(adapter, 0) >= max_steps_per_adapter:
                        in_flight.pop(tag, None)
                        continue
                    if tag in in_flight:
                        # Keep the existing in-flight request; do not clobber it.
                        continue
                    data = DataProto(meta_info={"global_step": global_tick})
                    in_flight[tag] = self.rollout_schedulers[tag].get_batch.remote(
                        data, self.pipeline_config.rollout_batch_size
                    )
                    submitted_at_mono[tag] = time.monotonic()

            success = True
        finally:
            try:
                ray.get([sched.shutdown.remote() for sched in self.rollout_schedulers.values()])
            except Exception:
                logger.exception("Failed to shutdown rollout schedulers")
            try:
                self.tracker.finish()
            except Exception:
                logger.exception("tracker.finish failed")
            if success:
                logger.info("pipeline complete!")
