import threading
import time

from dataclasses import replace
from typing import Any, Dict, List, Optional

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
from roll.datasets.global_dataset import GlobalDatasetManager
from roll.pipeline.agentic.agentic_pipeline import (
    compute_rollout_traj_metrics,
    compute_train_data_metrics,
    get_episode_scores,
)
from roll.utils.constants import RAY_NAMESPACE
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
    batch_balance,
    compute_token_reward,
    masked_mean,
    reduce_metrics,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

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
                    "Level 1 offloads weights to CPU (restorable); Level 2 discards weights entirely. "
                    "Multi-LoRA needs restorable offload for train/infer weight sync cycles."
                )

        # For multi-LoRA training, reference is the same backbone with LoRA disabled.
        # Use actor_train.disable_adapter() to compute ref_log_probs; do not create a separate reference cluster.
        self.use_ref_model = False

        # TODO: support GAE with per-LoRA critics: frozen backbone + per-LoRA adapters + per-LoRA value heads.
        # Critic setup per LoRA task:
        #   - Value head: fully tuned linear layer (hidden_state → scalar value)
        #   - Backbone: frozen weights + LoRA adapters (only adapters updated to save memory)
        if self.pipeline_config.adv_estimator == "gae":
            raise NotImplementedError(
                "AgenticMultiLoraPipeline does not support adv_estimator='gae'. "
                "A single shared critic cannot produce accurate advantages across different LoRA tasks. "
                "Requires per-LoRA critic adapters and per-LoRA value heads on a shared backbone "
                "(not yet implemented). Use 'grpo' or 'reinforce_plus_plus' instead."
            )

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

        # INIT PHASE: Download models and tokenizer
        self.download_models(*download_clusters)
        self.tokenizer = default_tokenizer_provider(model_args=self.pipeline_config.actor_train.model_args)

        # INIT PHASE: Initialize clusters
        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)

        self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=True)

        # INIT PHASE: Model update pairing (train -> infer)
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        self.set_checkpoint_clusters(self.actor_train)

        self.running = RunningMoments()

        self.partial_gpu_mode: bool = False
        if hasattr(self.pipeline_config, "partial_gpu_mode") and self.pipeline_config.partial_gpu_mode:
            self._validate_partial_gpu_config()
            self.partial_gpu_mode = True

        # Per-tag rollout schedulers (shared actor_infer).
        self.rollout_schedulers: dict[str, Any] = {}
        base_env: EnvManagerConfig = self.pipeline_config.train_env_manager
        for tag, n_group in zip(base_env.tags, base_env.num_groups_partition):
            # Shallow-copy the base config so per-tag mutations don't affect other tags.
            env_cfg = replace(base_env)
            # Narrow the config to this single tag's env subset (one tag, one partition).
            env_cfg.tags = [tag]
            env_cfg.num_groups_partition = [n_group]
            env_cfg.num_env_groups = n_group
            env_cfg.name = f"train_env_{tag}"
            # Recompute derived fields (world_size, max_env_num_per_worker, etc.) for the reduced env count.
            env_cfg.__post_init__()
            # Ensure per-tag max_traj_per_env is sufficient after narrowing to this tag's env subset.
            self.pipeline_config.ensure_min_traj_per_env(env_cfg, self.pipeline_config.rollout_batch_size)
            # Rebuild env_configs so worker_rank → env_id mapping reflects only this tag's envs.
            self.pipeline_config.make_env_configs(env_cfg)
            self.rollout_schedulers[tag] = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-train-{tag}",
                namespace=RAY_NAMESPACE,
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

        # Per-tag val rollout schedulers (mirrors train schedulers for per-adapter eval).
        val_env: EnvManagerConfig = self.pipeline_config.val_env_manager
        val_tags = list(val_env.tags) if getattr(val_env, "tags", None) else []
        # Val tags must match train tags exactly for correct per-adapter eval.
        assert val_tags == list(base_env.tags), (
            f"val_env_manager.tags must match train_env_manager.tags: "
            f"val={val_tags} train={list(base_env.tags)}"
        )
        num_tags = len(val_tags)

        # Validate val partition: no fallback, require explicit valid config.
        val_num_groups_partition = list(getattr(val_env, "num_groups_partition", []) or [])
        assert len(val_num_groups_partition) == num_tags, (
            f"val_env_manager.num_groups_partition length ({len(val_num_groups_partition)}) "
            f"must match num_tags ({num_tags})"
        )
        assert all(n_group > 0 for n_group in val_num_groups_partition), (
            f"val_env_manager.num_groups_partition entries must all be > 0: {val_num_groups_partition}"
        )
        assert sum(val_num_groups_partition) == val_env.num_env_groups, (
            f"sum(val_env_manager.num_groups_partition) = {sum(val_num_groups_partition)} "
            f"must equal val_env_manager.num_env_groups = {val_env.num_env_groups}"
        )

        # Per-tag val_batch_size: equal split, validated per-tag.
        assert self.pipeline_config.val_batch_size % num_tags == 0, (
            f"val_batch_size ({self.pipeline_config.val_batch_size}) must be divisible by "
            f"num_tags ({num_tags})"
        )
        val_batch_size_per_tag = self.pipeline_config.val_batch_size // num_tags
        self._val_batch_size_per_tag: dict[str, int] = {}
        for tag, val_n_group in zip(val_tags, val_num_groups_partition):
            tag_val_env_num = val_n_group * val_env.group_size
            assert val_batch_size_per_tag % tag_val_env_num == 0, (
                f"per-tag val_batch_size ({val_batch_size_per_tag}) must be divisible by "
                f"tag {tag!r} val_env_num ({tag_val_env_num} = {val_n_group} * {val_env.group_size})"
            )
            self._val_batch_size_per_tag[tag] = val_batch_size_per_tag

        self.val_rollout_schedulers: dict[str, Any] = {}
        for tag, val_n_group in zip(val_tags, val_num_groups_partition):
            val_env_cfg = replace(val_env)
            val_env_cfg.tags = [tag]
            val_env_cfg.num_groups_partition = [val_n_group]
            val_env_cfg.num_env_groups = val_n_group
            val_env_cfg.name = f"val_env_{tag}"
            val_env_cfg.__post_init__()
            # Ensure per-tag max_traj_per_env is sufficient for the proportional val batch.
            self.pipeline_config.ensure_min_traj_per_env(val_env_cfg, self._val_batch_size_per_tag[tag])
            self.pipeline_config.make_env_configs(val_env_cfg)
            self.val_rollout_schedulers[tag] = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-val-{tag}",
                namespace=RAY_NAMESPACE,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                config=self.pipeline_config,
                env_manager_config=val_env_cfg,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="val",
            )

        self.val_dataset_manager: Any = GlobalDatasetManager.options(
            name="val_dataset_manager",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

        # Serialize concurrent shrink/expand calls from partial-GPU mode.
        self._infer_resize_lock = threading.Lock()

        # Initial model update to register/load adapters on inference before first rollout.
        self._initial_model_update()
        self._create_lora_trackers()

    def _create_lora_trackers(self) -> None:
        """Create one metrics tracker per LoRA adapter for independent per-adapter tracking."""
        from roll.utils.tracking import create_lora_tracker

        adapters = self.pipeline_config.actor_train.model_args.adapters or {}
        if not adapters:
            return
        adapter_names = sorted(adapters.keys())
        tracker_name = self.pipeline_config.track_with

        self.lora_trackers: dict[str, Any] = {}
        for name in adapter_names:
            self.lora_trackers[name] = create_lora_tracker(
                tracker_name=tracker_name,
                lora_name=name,
                config=self.pipeline_config.to_dict(),
                **self.pipeline_config.tracker_kwargs,
            )
        logger.info("Created per-LoRA trackers for adapters: %s", adapter_names)

    def _initial_model_update(self) -> None:
        # Full offload: discard model weights, KV cache, and all LoRA tensors before initial sync.
        self.actor_infer.offload_states()
        adapters = set(self.pipeline_config.actor_train.model_args.adapters.keys()) if self.pipeline_config.actor_train.model_args.adapters else None
        _ = self.model_update_lora_subset(global_step=0, adapters_to_update=adapters)
        self.actor_infer.load_states()

    def adjust_batch(self, data: DataProto, mode: str = "copy") -> DataProto:
        # TODO: extract adjust_batch into a standalone utility function instead of
        # calling an unbound method from a sibling class (fragile, bypasses inheritance).
        from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline

        return AgenticPipeline.adjust_batch(self, data=data, mode=mode)  # type: ignore[misc]



    def val(self, lora_name: str, global_step: int) -> dict:
        """Validate a single adapter by running only its matching tag's val scheduler."""
        metrics: dict = {}
        ray.get(self.val_dataset_manager.reset.remote())

        for tag, val_scheduler in self.val_rollout_schedulers.items():
            # Only validate the tag that maps to the given adapter.
            if normalize_domain(tag) != lora_name:
                continue
            metrics.update(self._val_tag(tag, val_scheduler, global_step))

        logger.info(f"val lora={lora_name} metrics: {metrics}")
        return metrics

    def _val_tag(self, tag: str, val_scheduler: Any, global_step: int) -> dict:
        """Run validation for a single tag and return prefixed metrics."""
        metrics: dict = {}
        batch = DataProto(meta_info={"is_offload_states": False, "global_step": global_step})
        eval_batch = ray.get(val_scheduler.get_batch.remote(batch, self._val_batch_size_per_tag[tag]))

        if "get_batch_return_start_time" in eval_batch.meta_info:
            metrics[f"time/get_batch_cost_val/{tag}"] = (
                time.time() - eval_batch.meta_info.pop("get_batch_return_start_time")
            )

        dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, eval_batch)
        eval_metrics = reduce_metrics(eval_batch.meta_info.get("metrics", {}))
        eval_score = get_episode_scores(eval_batch)
        eval_metrics["score/mean"] = torch.mean(eval_score).detach().item()
        eval_metrics["score/max"] = torch.max(eval_score).detach().item()
        eval_metrics["score/min"] = torch.min(eval_score).detach().item()

        metrics.update({f"val/{tag}/{k}": v for k, v in eval_metrics.items()})
        return metrics

    def _validate_partial_gpu_config(self) -> bool:
        train_devices = set(self.actor_train.worker_config.device_mapping)
        infer_devices = set(self.actor_infer.worker_config.device_mapping)

        if not train_devices or not infer_devices:
            raise ValueError(
                "device_mapping cannot be empty: "
                f"train={list(train_devices)}, infer={list(infer_devices)}"
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
            freed_gpus = train_devices
            self._validate_minimum_active_ranks(infer_dp_size, infer_devices, list(freed_gpus), gpus_per_dp_rank)
            # Store TP/PP-aware attributes for GPU→dp_rank translation in shrink/expand.
            self._infer_gpus_per_dp_rank = gpus_per_dp_rank
            self._infer_device_mapping = list(self.actor_infer.worker_config.device_mapping)
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
        # TODO: extract _validate_minimum_active_ranks into a shared utility instead of
        # calling an unbound method from a sibling class (fragile, bypasses inheritance).
        from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline

        AgenticPipeline._validate_minimum_active_ranks(
            self, infer_dp_size, infer_devices, freed_gpu_list, gpus_per_dp_rank
        )

    def _prepare_batch(self, batch: DataProto, metrics: dict) -> DataProto:
        """Transform raw rollout data into a training-ready batch for the actor update.

        Multi-LoRA pipelines do NOT use a critic (GAE is explicitly unsupported because a
        single shared critic cannot produce accurate values across different LoRA tasks).
        Instead, critic-free estimators like GRPO, Reinforce++, or GIGPO are used.

        Processing pipeline (in order):
        1. Discounted returns — collapse multi-step rewards into per-token returns.
        2. Batch adjustment — filter/transform samples (e.g. drop low-quality trajectories).
        3. Reference log-probs — dynamic_batching_shard (if enabled), disable LoRA adapter,
           batch_balance, then compute log-probs under the frozen base model for KL penalty.
           Matches agentic_pipeline.py:404-436.
        4. Old log-probs — compute log-probs under the *current* policy (LoRA enabled) to form
           the importance-sampling ratio (π_new / π_old) used by the clipped surrogate objective.
           If old-logprob recompute is disabled, zeros are used (ratio = 1, i.e. on-policy).
        5. Response-level mask — build segment/token masks that select which parts of the
           response are included in the loss.
        6. Response-level rewards — normalize and reshape rewards per response segment.
        7. Token-level rewards — apply KL penalty and per-token reward shaping.
        8. Advantage estimation — critic-free estimator (GRPO / Reinforce++ / GIGPO)
            over the shaped rewards.
        9. Train-infer correction — optionally down-weight stale samples whose old log-probs
            diverge too far from the current policy (importance-weight clipping).

        Args:
            batch: Raw rollout output containing token ids, attention masks, rewards,
                and meta_info produced by the environment / rollout workers.
            metrics: Mutable dict; timing and scalar metrics are added in-place.

        Returns:
            The same ``batch`` object, enriched with fields required by the actor
            training step: ``old_log_probs``, ``ref_log_probs``, ``advantages``,
            ``returns``, token-level rewards, and response-level masks.
        """
        # Step 1: collapse multi-step rewards into discounted returns.
        batch = compute_discounted_returns(batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma)

        # Step 2: filter/transform samples (e.g. drop low-quality trajectories).
        batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

        # Step 3: reference log-probs — run the base model (LoRA disabled) to get the
        # KL-divergence anchor used in the training loss (matching agentic_pipeline.py:404-436).
        with Timer(name="cal_ref_log_probs", logger=None) as cal_timer:
            if self.pipeline_config.enable_reference:
                # Dynamic batching for ref path (same guard as old log-prob path below).
                if self.pipeline_config.actor_train.use_dynamic_batching_in_infer:
                    batch, dynamic_batching_metrics = dynamic_batching_shard(
                        batch,
                        self.actor_train.dp_size,
                        self.pipeline_config.actor_train.max_tokens_per_microbatch_in_infer,
                        self.pipeline_config.actor_train.sequence_length_round_in_infer,
                        self.pipeline_config.actor_train.strategy_args.strategy_config.get("pipeline_model_parallel_size", 1),
                        self.pipeline_config.actor_train.strategy_args.strategy_config.get("virtual_pipeline_model_parallel_size", None),
                        "reference/compute_log_probs",
                    )
                    metrics.update(dynamic_batching_metrics)
                # For multi-LoRA, reference logprobs are computed by disabling the LoRA adapter on the actor.
                batch.meta_info["disable_adapter"] = True
                batch.meta_info["is_offload_states"] = False
                batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))
                ref_log_probs: DataProto = self.actor_train.compute_log_probs(batch, blocking=True)
                batch.meta_info.pop("disable_adapter", None)
                # Use rename + union to preserve all fields from the ref DataProto (matching agentic_pipeline.py:431-432).
                ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                batch = batch.union(ref_log_probs)
                avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                metrics.update(reduce_metrics(ref_log_probs.meta_info.pop("metrics", {})))
                metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
        metrics["time/step_ref_log_probs_values_reward"] = cal_timer.last

        # Re-balance after ref log-prob compute may have changed padding.
        batch_balance(batch, dp_size=self.actor_train.dp_size, minibatch_size=len(batch))

        # Step 4: old log-probs — compute π_old(a|s) under the current LoRA-enabled policy.
        # These form the denominator of the importance-sampling ratio π_new / π_old.
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

            # Reference logprobs (if reference disabled, mock with old_log_probs)
            if not self.pipeline_config.enable_reference:
                batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                avg_ref_log_prob = masked_mean(batch.batch["ref_log_probs"], batch.batch["response_mask"][:, 1:])
                metrics.update({"critic/ref_log_prob/mean": avg_ref_log_prob.item()})
        metrics["time/step_old_log_probs_values"] = cal_old_logpb_timer.last

        # Step 5: build response-level masks that select which tokens/segments contribute to the loss.
        with Timer(name="cal_response_level_mask", logger=None) as timer:
            batch, mask_metrics = get_agentic_response_level_mask(batch, self.pipeline_config)
            metrics.update(mask_metrics)
        metrics["time/step_cal_response_level_mask"] = timer.last

        # Step 6: normalize and reshape rewards per response segment.
        with Timer(name="cal_response_norm_rewards", logger=None) as timer:
            batch, reward_metrics = compute_response_level_rewards(batch=batch, pipeline_config=self.pipeline_config)
            metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
            metrics.update(reward_metrics)
        metrics["time/step_cal_norm_rewards"] = timer.last

        # Step 7: apply KL penalty and per-token reward shaping via the adaptive KL controller.
        with Timer(name="cal_token_reward", logger=None) as timer:
            batch, token_level_metrics = compute_token_reward(batch, self.pipeline_config, self.kl_ctrl)
            metrics.update(token_level_metrics)
        metrics["time/step_cal_token_reward"] = timer.last

        # Step 8: compute advantages using critic-free estimator (GRPO / Reinforce++ / GIGPO).
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
        # Step 9: importance-weight correction — down-weight stale samples whose old
        # log-probs diverge too far from the current policy (only when old logprobs are recomputed).
        if self.pipeline_config.enable_old_logprobs_recompute:
            batch, corr_metrics = apply_train_infer_correction_to_batch(
                self.pipeline_config,
                batch,
                update_mask_keys=batch.meta_info["loss_mask_keys"],
            )
            metrics.update(corr_metrics)
        return batch

    def _shrink_workers(self, *, dp_ranks_to_remove: List[int]) -> Dict[str, Any]:
        """Shrink inference off training GPUs across all per-tag schedulers.

        2-phase pattern (mirrors agentic_pipeline._shrink_workers):
        - Phase 1: all schedulers except last do routing-only shrink (skip_offload=True).
        - Phase 2: last scheduler does routing + physical offload (skip_offload=False).
        """
        if not isinstance(dp_ranks_to_remove, list) or not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be a non-empty list[int]")
        with self._infer_resize_lock:
            all_schedulers = list(self.rollout_schedulers.values()) + list(self.val_rollout_schedulers.values())
            # Phase 1: routing-only shrink on all except last.
            if len(all_schedulers) > 1:
                phase1_metrics = ray.get(
                    [sched.shrink_sampler.remote(dp_ranks_to_remove, skip_offload=True) for sched in all_schedulers[:-1]]
                )
            else:
                phase1_metrics = []
            # Phase 2: last scheduler does routing + physical offload.
            phase2_metrics = ray.get(all_schedulers[-1].shrink_sampler.remote(dp_ranks_to_remove, skip_offload=False))
            shrink_metrics_list = phase1_metrics + [phase2_metrics]
            result: Dict[str, Any] = {}
            for idx, shrink_metrics in enumerate(shrink_metrics_list):
                result.update({f"shrink/{idx}/{k}": v for k, v in shrink_metrics.items()})
            return result

    def _expand_workers(self, *, dp_ranks_to_add: List[int], train_skip_load: bool) -> Dict[str, Any]:
        """Expand inference back to training GPUs across all per-tag schedulers.

        Sequential pattern (mirrors agentic_pipeline._expand_workers):
        - First scheduler does physical load (skip_load determined by train_skip_load).
        - Rest do routing-only expand (skip_load=True).
        """
        if not isinstance(dp_ranks_to_add, list) or not dp_ranks_to_add:
            raise ValueError("dp_ranks_to_add must be a non-empty list[int]")
        with self._infer_resize_lock:
            all_schedulers = list(self.rollout_schedulers.values()) + list(self.val_rollout_schedulers.values())
            # First scheduler loads model states (skip if model_update already loaded them).
            first_metrics = ray.get(all_schedulers[0].expand_sampler.remote(dp_ranks_to_add, skip_load=bool(train_skip_load)))
            # Rest do routing-only expand.
            rest_metrics = ray.get(
                [sched.expand_sampler.remote(dp_ranks_to_add, skip_load=True) for sched in all_schedulers[1:]]
            )
            expand_metrics_list = [first_metrics] + rest_metrics
            result: Dict[str, Any] = {}
            for idx, expand_metrics in enumerate(expand_metrics_list):
                result.update({f"expand/{idx}/{k}": v for k, v in expand_metrics.items()})
            return result

    @torch.no_grad()
    def run(self):
        if not is_lora_training(self.pipeline_config):
            raise RuntimeError("AgenticMultiLoraPipeline requires actor_train.model_args.adapters to be configured.")

        success = False
        try:
            max_steps_per_lora = int(self.pipeline_config.max_steps)
            adapters = list(self.pipeline_config.actor_train.model_args.adapters.keys())
            lora_step: dict[str, int] = {name: 0 for name in adapters}
            global_tick = 0
            # Adapter keys in model_args.adapters are canonical lowercase (normalized in __post_init__).
            tag_to_adapter = {tag: normalize_domain(tag) for tag in self.rollout_schedulers.keys()}

            # Resume per-lora state from checkpoint if available.
            if "lora_step_by_adapter" in self.state.kv:
                saved_mapping = self.state.kv["tag_to_adapter"]
                if saved_mapping != tag_to_adapter:
                    raise RuntimeError(
                        f"Checkpoint tag_to_adapter mismatch: saved={saved_mapping} current={tag_to_adapter}"
                    )
                lora_step = dict(self.state.kv["lora_step_by_adapter"])
                global_tick = int(self.state.kv["global_tick"])
                logger.info(f"Resumed from checkpoint: global_tick={global_tick} lora_step={lora_step}")

            unknown = sorted({a for a in tag_to_adapter.values() if a not in lora_step})
            if unknown:
                raise RuntimeError(
                    f"Train env tags must map to configured LoRA adapters. Unknown adapters from tags: {unknown}. "
                    f"adapters={sorted(lora_step.keys())} tag_to_adapter={tag_to_adapter}"
                )

            # Calculate tokens-per-second system throughput
            tps_timer = _Timer(window_size=5)
            # Monotonic clock origin for all relative timestamps in this pipeline run.
            pipeline_start_mono = time.monotonic()

            # Kick off one in-flight get_batch per tag.
            in_flight: dict[str, ray.ObjectRef] = {}
            pending_by_tag: dict[str, DataProto] = {}
            # Per-tag monotonic timestamp of when get_batch.remote() was issued,
            # used to measure rollout latency (submission → ray.wait ready).
            submitted_at_mono: dict[str, float] = {}
            tags = list(self.rollout_schedulers.keys())
            for tag in tags:
                adapter = tag_to_adapter[tag]
                if lora_step.get(adapter, 0) >= max_steps_per_lora:
                    continue
                # Use per-adapter step for rollout-facing operations (not global_tick).
                data = DataProto(meta_info={"global_step": lora_step.get(adapter, 0)})
                in_flight[tag] = self.rollout_schedulers[tag].get_batch.remote(
                    data, self.pipeline_config.rollout_batch_size
                )
                submitted_at_mono[tag] = time.monotonic()

            # Monotonic timestamp when the current ray.wait polling started (None when not waiting).
            # Used to measure wall-clock time spent blocked in ray.wait per tick.
            wait_ready_since_mono: float | None = None
            # Single-adapter first-ready tick: each tick processes one ready tag batch.
            last_get_batch_done_ts_by_adapter: dict[str, float] = {}
            last_train_step_done_ts_by_adapter: dict[str, float] = {}
            last_train_step_done_ts_global: float | None = None

            while any(lora_step[name] < max_steps_per_lora for name in adapters):
                active_tags = [tag for tag in tags if lora_step.get(tag_to_adapter[tag], 0) < max_steps_per_lora]
                active_tags_in_flight = [tag for tag in active_tags if tag in in_flight]
                active_refs = [in_flight[tag] for tag in active_tags_in_flight]
                assert len(active_refs) > 0

                if wait_ready_since_mono is None:
                    wait_ready_since_mono = time.monotonic()

                # ray.wait with no timeout blocks until num_returns refs are ready.
                ready, _ = ray.wait(active_refs, num_returns=1)

                ready_now_mono = time.monotonic()

                tick_wait_ready_batch_s = ready_now_mono - wait_ready_since_mono
                wait_ready_since_mono = None

                # Single-adapter tick: consume exactly one ready batch per train_step_lora call.
                ready_ref = ready[0]
                ready_tag = next((t for t, r in in_flight.items() if r == ready_ref), None)
                if ready_tag is None:
                    raise RuntimeError("ray.wait returned a ref that is not tracked in in_flight")

                batch = ray.get(ready_ref)
                if batch is None:
                    raise RuntimeError(f"get_batch returned None for tag={ready_tag!r}")
                # Derive sample UUIDs from traj_id (same as agentic_pipeline.py:338-339).
                sample_uuids = [f"{traj_id}_{idx}" for idx, traj_id in enumerate(batch.non_tensor_batch['traj_id'])]
                batch.non_tensor_batch['sample_uuid'] = np.array(sample_uuids, dtype=object)
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

                # Greedy tick: once any tag has a ready batch, proceed to train. In partial-GPU mode, `shrink_sampler`
                # relies on RequestScheduler to abort/remap + update routing safely for any in-flight requests.

                tick_metrics: dict = {}
                shrink_duration_s: Optional[float] = None
                with Timer(name="pipeline_tick_total", logger=None) as tick_timer:
                    with tps_timer:
                        # Partial GPU: shrink inference off training GPUs before training.
                        if self.partial_gpu_mode:
                            with Timer(name="exec_shrink", logger=None) as shrink_timer:
                                target_gpus: list[int] = []
                                if hasattr(self.actor_train.worker_config, "device_mapping") and self.actor_train.worker_config.device_mapping:
                                    target_gpus.extend(self.actor_train.worker_config.device_mapping)
                                if target_gpus:
                                    dp_ranks = self._target_gpus_to_dp_ranks_to_remove(
                                        target_gpus=target_gpus,
                                    )
                                    tick_metrics.update(self._shrink_workers(dp_ranks_to_remove=dp_ranks))
                            shrink_duration_s = float(shrink_timer.last)

                        # Collect actor inference metrics once per tick
                        actor_infer_metrics = self.actor_infer.get_metrics()
                        actor_infer_reduced = {}
                        if "metrics" in actor_infer_metrics.meta_info:
                            actor_infer_reduced = reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {}))

                        # Exactly one batch is ready per tick (ray.wait returns 1,
                        # cleared before next iteration).
                        if len(pending_by_tag) != 1:
                            raise RuntimeError(
                                f"Expected exactly 1 pending batch per tick, got {len(pending_by_tag)}: "
                                f"{sorted(pending_by_tag.keys())}"
                            )
                        (ready_tag_for_tick, ready_batch_for_tick), = pending_by_tag.items()

                        dirty_adapters: set[str] = set()
                        lora_metrics: dict[str, dict] = {}

                        adapter_for_tag = tag_to_adapter[ready_tag_for_tick]
                        adapter_metrics = lora_metrics.setdefault(adapter_for_tag, {})
                        if actor_infer_reduced:
                            adapter_metrics.update(actor_infer_reduced)
                        tick_wait_ready_batch_s = float(
                            ready_batch_for_tick.meta_info.get("metrics", {}).get("time/ray_wait_ready_batch_s", 0.0) or 0.0
                        )
                        tick_metrics["time/ray_wait_ready_batch_s"] = tick_wait_ready_batch_s
                        adapter_metrics["time/ray_wait_ready_batch_s"] = tick_wait_ready_batch_s

                        wait_s = float(ready_batch_for_tick.meta_info.get("metrics", {}).get("time/get_batch_wait_s", 0.0) or 0.0)
                        # Use per-adapter step for rollout-facing metadata (not global_tick).
                        ready_batch_for_tick.meta_info["global_step"] = lora_step[adapter_for_tag]
                        ready_batch_for_tick.meta_info["_broadcast_non_tensor_batch"] = True
                        # Keep strategy token-count accounting contract identical to agentic_pipeline.
                        ready_batch_for_tick.meta_info["loss_mask_keys"] = ["response_mask"]
                        with Timer(name="rollout", logger=None) as rollout_timer:
                            adapter_metrics.update(reduce_metrics(ready_batch_for_tick.meta_info.pop("metrics", {})))
                            adapter_metrics.update(compute_rollout_traj_metrics(ready_batch_for_tick))
                            dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, lora_step[adapter_for_tag], ready_batch_for_tick)
                        adapter_metrics["time/step_rollout"] = rollout_timer.last + wait_s

                        prepared_batch = self._prepare_batch(ready_batch_for_tick, adapter_metrics)

                        # Extract the single adapter name from the prepared batch.
                        lora_names = prepared_batch.non_tensor_batch["lora_name"]
                        unique = list(dict.fromkeys(lora_names.tolist()))
                        if len(unique) != 1:
                            raise RuntimeError(f"Expected homogeneous lora_name per prepared batch, got {unique}")
                        adapter_name = str(unique[0])
                        # Fail fast on adapter mismatch: adapter_for_tag is the canonical
                        # step source for rollout dump, model_update, and checkpoint.
                        if adapter_name != adapter_for_tag:
                            raise RuntimeError(
                                f"Adapter mismatch: tag={ready_tag_for_tick!r} expected adapter={adapter_for_tag!r} "
                                f"but prepared batch contains adapter={adapter_name!r}"
                            )
                        dirty_adapters.add(adapter_name)

                        # Per-adapter data metrics inline (single batch, no deferred concat needed).
                        with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                            adapter_metrics.update(compute_train_data_metrics(batch=prepared_batch))
                        adapter_metrics["time/step_compute_data_metrics"] = data_metrics_timer.last

                        # Balance batch for training (production pattern: agentic_pipeline.py:534-537).
                        batch_balance_metrics = batch_balance(
                            batch=prepared_batch,
                            dp_size=self.actor_train.dp_size,
                            minibatch_size=self.actor_train.dp_size
                            * self.pipeline_config.actor_train.training_args.per_device_train_batch_size
                            * self.pipeline_config.actor_train.training_args.gradient_accumulation_steps,
                            logging_prefix="global_seqlen/actor_train",
                        )
                        tick_metrics.update(batch_balance_metrics)
                        adapter_metrics.update(batch_balance_metrics)

                        # Dynamic batching: shard prepared_batch before train_step_lora
                        # (same pattern as agentic_pipeline.py train_step path).
                        if self.pipeline_config.actor_train.use_dynamic_batching_in_train:
                            prepared_batch, dynamic_batching_metrics = dynamic_batching_shard(
                                prepared_batch,
                                self.actor_train.dp_size,
                                self.pipeline_config.actor_train.max_tokens_per_microbatch_in_train,
                                self.pipeline_config.actor_train.sequence_length_round_in_train,
                                self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                    "pipeline_model_parallel_size", 1
                                ),
                                self.pipeline_config.actor_train.strategy_args.strategy_config.get(
                                    "virtual_pipeline_model_parallel_size", None
                                ),
                                "actor_train/train_step_lora",
                            )
                            adapter_metrics.update(dynamic_batching_metrics)

                        # Train single adapter.
                        with Timer(name="train_timer", logger=None) as train_timer:
                            train_refs: list[ray.ObjectRef] = self.actor_train.train_step_lora(prepared_batch, blocking=False)
                            train_metrics = DataProto.materialize_concat(data_refs=train_refs)
                            reduced_train_metrics = reduce_metrics(train_metrics.meta_info.pop("metrics", {}))
                            tick_metrics.update(reduced_train_metrics)
                            tps_timer.push_units_processed(n=torch.sum(prepared_batch.batch["attention_mask"]).detach().item())
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
                            adapter_metrics = lora_metrics.setdefault(name, {})
                            adapter_metrics["time/step_train"] = train_step_s
                            adapter_metrics["time/step_train_step_lora"] = train_step_s
                            adapter_metrics["time/train_step_done_ts"] = train_step_done_ts
                            prev_train_done_ts = last_train_step_done_ts_by_adapter.get(name)
                            lora_train_step_interval_s = (
                                0.0 if prev_train_done_ts is None else train_step_done_ts - prev_train_done_ts
                            )
                            adapter_metrics["time/train_step_done_interval_s"] = lora_train_step_interval_s
                            last_train_step_done_ts_by_adapter[name] = train_step_done_ts
                            for k, v in reduced_train_metrics.items():
                                if f"/{name}/" in k:
                                    adapter_metrics[k] = v
                                else:
                                    adapter_metrics.setdefault(k, v)

                        # Update step counters.
                        lora_step[adapter_for_tag] += 1
                        global_tick += 1

                        tick_metrics["system/global_tick"] = global_tick
                        for name, step in lora_step.items():
                            tick_metrics[f"system/lora_step/{name}"] = step
                        # Cumulative sample count (pattern from agentic_pipeline.py:569).
                        tick_metrics["system/samples"] = global_tick * self.pipeline_config.rollout_batch_size
                        for name in dirty_adapters:
                            adapter_metrics = lora_metrics.setdefault(name, {})
                            adapter_metrics["system/global_tick"] = global_tick
                            adapter_metrics["system/lora_step"] = lora_step[adapter_for_tag]

                        # Model update boundary: suspend rollouts only for model_update.
                        # TODO: fine-granular rollout interruption — currently we abort ALL loras' rollouts
                        # and force-sync ALL adapters. Better approach: only abort/interrupt requests for the
                        # just-trained adapter (dirty_adapters), leave other loras' in-flight rollouts running,
                        # and sync only the updated adapter weights instead of all_adapters.
                        with Timer(name="model_update", logger=None) as model_update_timer:
                            ray.get([sched.suspend.remote() for sched in self.rollout_schedulers.values()])

                            if self.pipeline_config.async_pipeline:
                                # Full offload: stop generation server, discard KV cache + all LoRA tensors.
                                self.actor_infer.offload_states()
                            # Full offload destroys all LoRA tensors on infer side — must re-sync every adapter.
                            # Train-side weights are preserved in pinned CPU memory across offload cycles.
                            all_adapters = set(self.pipeline_config.actor_train.model_args.adapters.keys()) if self.pipeline_config.actor_train.model_args.adapters else None
                            model_update_metrics = self.model_update_lora_subset(global_tick, adapters_to_update=all_adapters)
                            tick_metrics.update(model_update_metrics)
                            for name in dirty_adapters:
                                lora_metrics.setdefault(name, {}).update(model_update_metrics)
                            self.actor_infer.load_states()
                            # Partial GPU: expand routing state after model_update reloads to all GPUs.
                            if self.partial_gpu_mode and global_tick > 0:
                                target_gpus = []

                                target_gpus.extend(self.actor_train.worker_config.device_mapping)


                                # but the lost rank is silent — only the alignment warning in the callee signals it.
                                dp_ranks_to_add = self._target_gpus_to_dp_ranks_to_add(target_gpus=target_gpus)
                                expand_result = self._expand_workers(dp_ranks_to_add=dp_ranks_to_add,
                                                                      train_skip_load=True)


                                tick_metrics.update(expand_result)
                                for name in dirty_adapters:
                                    lora_metrics.setdefault(name, {}).update(expand_result)

                        model_update_s = float(model_update_timer.last)
                        tick_metrics["time/step_model_update"] = model_update_s
                        for name in dirty_adapters:
                            lora_metrics.setdefault(name, {})["time/step_model_update"] = model_update_s

                        # Per-adapter validation: run after model_update + expand so inference
                        # weights are current and schedulers are resumed.
                        if self.pipeline_config.eval_steps > 0:
                            for name in dirty_adapters:
                                if lora_step[name] % self.pipeline_config.eval_steps == 0:
                                    with Timer(name="val", logger=None) as val_timer:
                                        val_metrics = self.val(lora_name=name, global_step=lora_step[name])
                                    val_metrics["time/step_val"] = val_timer.last
                                    lora_metrics.setdefault(name, {}).update(val_metrics)

                tick_total_s = float(tick_timer.last)
                for name in dirty_adapters:
                    lora_metrics.setdefault(name, {})["time/step_total"] = tick_total_s
                    if shrink_duration_s is not None:
                        lora_metrics.setdefault(name, {})["time/step_shrink"] = shrink_duration_s

                if self.pipeline_config.logging_steps > 0 and global_tick % self.pipeline_config.logging_steps == 0:
                    logger.info(f"tick={global_tick} lora_step={lora_step}")
                    logger.info(tick_metrics)

                # Per-LoRA metrics to per-LoRA trackers (independent step counters).
                if hasattr(self, "lora_trackers"):
                    for name in sorted(dirty_adapters):
                        per_lora_metrics = dict(lora_metrics.get(name, {}))
                        per_lora_metrics["system/lora_name"] = name
                        self.lora_trackers[name].log(values=per_lora_metrics, step=lora_step[name])
                # Global tick metrics to pipeline-level tracker (shared step counter).
                self.tracker.log(values=tick_metrics, step=global_tick)

                # Persist per-lora state for checkpoint resume.
                all_done = all(lora_step[name] >= max_steps_per_lora for name in adapters)
                self.state.kv["lora_step_by_adapter"] = dict(lora_step)
                self.state.kv["global_tick"] = global_tick
                self.state.kv["tag_to_adapter"] = dict(tag_to_adapter)
                self.state.step = global_tick
                # Minimal log_history entry for do_checkpoint (reads log_history[-1] for system/step).
                # Do not persist full tick_metrics: base resume replay lacks lora_name context.
                self.state.log_history.append({"system/step": global_tick})
                self.do_checkpoint(global_step=global_tick, is_last_step=all_done)

                pending_by_tag.clear()
                for tag in tags:
                    adapter = tag_to_adapter[tag]
                    if lora_step.get(adapter, 0) >= max_steps_per_lora:
                        in_flight.pop(tag, None)
                        continue
                    if tag in in_flight:
                        # Keep the existing in-flight request; do not clobber it.
                        continue
                    # Use post-increment lora_step for next tick's rollout.
                    data = DataProto(meta_info={"global_step": lora_step[adapter]})
                    in_flight[tag] = self.rollout_schedulers[tag].get_batch.remote(
                        data, self.pipeline_config.rollout_batch_size
                    )
                    submitted_at_mono[tag] = time.monotonic()

            success = True
        finally:
            try:
                ray.get(
                    [sched.shutdown.remote() for sched in self.rollout_schedulers.values()]
                    + [sched.shutdown.remote() for sched in self.val_rollout_schedulers.values()]
                )
            except Exception:
                logger.exception("Failed to shutdown rollout schedulers")
            try:
                if hasattr(self, "lora_trackers"):
                    for lora_tracker in self.lora_trackers.values():
                        lora_tracker.finish()
                self.tracker.finish()
            except Exception:
                logger.exception("tracker.finish failed")
            if success:
                logger.info("pipeline complete!")
