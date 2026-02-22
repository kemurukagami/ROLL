"""SchedRL Multi-LoRA Pipeline.

Sequential cycle for adapter-aware agentic training under SchedRL's sleep_level=2:
  Expand -> Rollout (all tags) -> Shrink -> Train (dirty adapters) -> Repeat

Key constraints vs AgenticMultiLoraPipeline:
  - sleep_level=2 (GPU weights released; actors stay alive in CPU RAM)
  - No partial_gpu_mode (sequential, not overlapping)
  - megatron_train strategy required
  - lora_optimizer_mode='per_adapter' required
  - Per-tag RolloutSchedulers (one per env tag / adapter)
"""
from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.timer import _Timer

from schedrl.protocol.types import ActionResponse

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_pipeline import compute_rollout_traj_metrics, compute_train_data_metrics
from roll.pipeline.agentic.utils import (
    agentic_compute_advantage,
    compute_discounted_returns,
    compute_response_level_rewards,
    dump_rollout_trajectories,
    get_agentic_response_level_mask,
)
from roll.schedrl_adapter.concurrent_pipeline import SchedRLConcurrentPipeline
from roll.utils.dynamic_batching import dynamic_batching_shard
from roll.utils.functionals import (
    agg_loss,
    batch_balance,
    compute_token_reward,
    masked_mean,
    reduce_metrics,
)
from roll.utils.logging import get_logger
from roll.utils.lora_routing import normalize_domain
from roll.utils.train_infer_corrections import apply_train_infer_correction_to_batch

logger = get_logger()


class SchedRLMultiLoraPipeline(SchedRLConcurrentPipeline):
    """SchedRL-controlled multi-LoRA agentic pipeline.

    Cycle: Expand → Rollout (all tags) → Shrink → Train (dirty adapters) → Repeat.

    Constraints:
    - actor_infer.strategy_args.strategy_config.sleep_level == 2
    - actor_train.strategy_args.strategy_name == 'megatron_train'
    - actor_train.strategy_args.strategy_config.lora_optimizer_mode == 'per_adapter'
    - actor_train.model_args.adapters is not None
    """

    def initialize_pipeline(self) -> ActionResponse:
        """Initialize pipeline with per-tag rollout schedulers and multi-LoRA validation."""
        # super() owns _init_lock + _initialized guard; do not re-acquire here (not reentrant).
        result = super().initialize_pipeline()
        if not getattr(result, "success", False):
            return result

        # Guard child-specific init (idempotent: Ray may call twice if actor restarts are enabled).
        if getattr(self, "_rollout_schedulers_initialized", False):
            return ActionResponse(success=True)

        pipeline_config = self._pipeline_config

        # --- Multi-LoRA validation ---
        train_strategy_name = (
            getattr(getattr(pipeline_config.actor_train, "strategy_args", None), "strategy_name", None)
        )
        if train_strategy_name != "megatron_train":
            raise RuntimeError(
                f"SchedRLMultiLoraPipeline requires actor_train strategy_name='megatron_train', "
                f"got {train_strategy_name!r}"
            )
        train_strategy_config = (
            getattr(getattr(pipeline_config.actor_train, "strategy_args", None), "strategy_config", None) or {}
        )
        lora_optimizer_mode = train_strategy_config.get("lora_optimizer_mode", "shared")
        if lora_optimizer_mode != "per_adapter":
            raise RuntimeError(
                "SchedRLMultiLoraPipeline requires actor_train strategy_config.lora_optimizer_mode='per_adapter', "
                f"got {lora_optimizer_mode!r}"
            )
        adapters = getattr(pipeline_config.actor_train.model_args, "adapters", None) or {}
        if not adapters:
            raise RuntimeError(
                "SchedRLMultiLoraPipeline requires actor_train.model_args.adapters to be non-empty"
            )

        # --- Static VRAM cap (Phase 2) ---
        max_resident = getattr(pipeline_config, "max_resident_adapters", None)
        if max_resident is not None and len(adapters) > int(max_resident):
            raise RuntimeError(
                f"SchedRLMultiLoraPipeline: number of adapters ({len(adapters)}) exceeds "
                f"max_resident_adapters ({max_resident}). Reduce the adapter count or raise the cap."
            )

        # --- Build tag → adapter mapping ---
        base_env = pipeline_config.train_env_manager
        tags = list(base_env.tags) if getattr(base_env, "tags", None) else []
        if not tags:
            raise RuntimeError("train_env_manager.tags must be non-empty for SchedRLMultiLoraPipeline")
        self._tag_to_adapter: Dict[str, str] = {tag: normalize_domain(tag) for tag in tags}
        unknown = sorted({a for a in self._tag_to_adapter.values() if a not in adapters})
        if unknown:
            raise RuntimeError(
                f"SchedRLMultiLoraPipeline: env tags map to unknown adapters: {unknown}. "
                f"Configured adapters: {sorted(adapters.keys())}"
            )

        # --- Per-tag rollout schedulers ---
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler
        from roll.utils.constants import schedrl_env_vars

        ray_namespace = os.environ.get("ROLL_RAY_NAMESPACE", "roll")
        num_groups_partition = list(getattr(base_env, "num_groups_partition", []) or [])
        if len(num_groups_partition) != len(tags):
            # Fall back: equal partition
            num_groups_partition = [getattr(base_env, "num_env_groups", 1)] * len(tags)

        self.rollout_schedulers: Dict[str, Any] = {}
        for tag, n_group in zip(tags, num_groups_partition):
            env_cfg = replace(base_env)
            env_cfg.tags = [tag]
            env_cfg.num_groups_partition = [n_group]
            env_cfg.num_env_groups = n_group
            env_cfg.name = f"train_env_{tag}"
            env_cfg.__post_init__()
            # Ensure each per-tag scheduler can produce rollout_batch_size trajectories per step.
            train_env_num = env_cfg.num_env_groups * env_cfg.group_size
            traj_per_env = (pipeline_config.rollout_batch_size + train_env_num - 1) // train_env_num
            if env_cfg.max_traj_per_env < traj_per_env:
                env_cfg.max_traj_per_env = traj_per_env
            pipeline_config.make_env_configs(env_cfg)

            self.rollout_schedulers[tag] = ray.remote(RolloutScheduler).options(
                name=f"RolloutScheduler-{self._pipeline_id}-{tag}",
                namespace=ray_namespace,
                runtime_env={"env_vars": schedrl_env_vars()},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(
                config=pipeline_config,
                env_manager_config=env_cfg,
                resource_manager=self.resource_manager,
                infer_cluster=self.actor_infer,
                mode="train",
                request_scheduler=self.generate_scheduler,
            )

        # Build and promote initial per-adapter caches so first expand can sync all adapters.
        all_adapters = list(dict.fromkeys(self._tag_to_adapter.values()))
        for adapter_name in all_adapters:
            ray.get([
                worker.build_latest_bucket_cache.remote(0, 0, adapter_name)
                for worker in self.actor_train.workers
            ])
            ray.get([
                worker.promote_active_adapter_checkpoint.remote(adapter_name, 0, 0)
                for worker in self.actor_train.workers
            ])

        # Shrink all per-tag schedulers to zero (initial state, before first expand).
        dp_ranks = self._actor_infer_all_dp_ranks()
        for scheduler in self.rollout_schedulers.values():
            ray.get(scheduler.shrink_sampler.remote(dp_ranks, skip_offload=True))

        self._rollout_schedulers_initialized = True
        logger.info(
            f"[init][{self._pipeline_id}] SchedRLMultiLoraPipeline ready: "
            f"adapters={sorted(adapters.keys())} tags={tags}"
        )
        return ActionResponse(success=True)

    @torch.no_grad()
    def run(self):
        """Multi-LoRA SchedRL training loop.

        Adapted from SchedRLConcurrentPipeline.run() with these changes:
        - PHASE 6: collect batches from ALL per-tag schedulers (not a single one)
        - PHASE 14: use actor_train.train_step_lora() instead of train_step()
        """
        self._ensure_initialized()
        tps_timer = _Timer(window_size=5)
        last_notify_ready_step: Optional[int] = None

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"[schedrl][{self._pipeline_id}] multi-lora step={global_step} start")
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
                    # Phase 0: ensure previous step's notify_ready_to_release was called.
                    if global_step > 0 and last_notify_ready_step != global_step - 1:
                        self._notify_ready_to_release_actor_infer(global_step=global_step - 1)
                        last_notify_ready_step = global_step - 1

                    # PHASE 1: Offload States
                    if self.pipeline_config.adv_estimator == "gae":
                        self.critic.offload_states(blocking=True)
                    if self.pipeline_config.enable_reference and self.use_ref_model:
                        self.reference.offload_states(blocking=True)
                    self.actor_train.offload_states(blocking=True)

                    # PHASE 3: Model update (no-op: done via expand_sampler on next expand)
                    with Timer(name="model_update", logger=None) as model_update_timer:
                        pass
                    metrics["time/step_model_update"] = model_update_timer.last

                    # PHASE 4: Request actor_infer GPUs from SchedRL.
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

                    # PHASE 5: Validation (synchronous)
                    val_metrics = {}
                    with Timer(name="val", logger=None) as val_timer:
                        if self.pipeline_config.eval_steps > 0 and global_step % self.pipeline_config.eval_steps == 0:
                            val_metrics = self.val(global_step)

                    # PHASE 6: Rollout - collect from ALL per-tag schedulers and concatenate.
                    with Timer(name="rollout", logger=None) as rollout_timer:
                        tag_batches: List[DataProto] = []
                        for tag, scheduler in self.rollout_schedulers.items():
                            tag_batch = ray.get(
                                scheduler.get_batch.remote(batch, self.pipeline_config.rollout_batch_size)
                            )
                            if "get_batch_return_start_time" in tag_batch.meta_info:
                                metrics[f"time/get_batch_cost_{tag}"] = time.time() - tag_batch.meta_info.pop(
                                    "get_batch_return_start_time"
                                )
                            tag_batches.append(tag_batch)

                        batch = DataProto.concat(tag_batches)
                        sample_uuids = [f"{traj_id}_{i}" for i, traj_id in enumerate(batch.non_tensor_batch["traj_id"])]
                        batch.non_tensor_batch["sample_uuid"] = np.array(sample_uuids, dtype=object)
                        actor_infer_metrics = self.actor_infer.get_metrics()
                        metrics.update(reduce_metrics(actor_infer_metrics.meta_info.pop("metrics", {})))
                        metrics.update(compute_rollout_traj_metrics(batch))
                        dump_rollout_trajectories(self.pipeline_config.rollout_dump_dir, global_step, batch)

                    metrics["time/step_rollout"] = rollout_timer.last
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))
                    batch.meta_info["global_step"] = global_step
                    batch.meta_info["_broadcast_non_tensor_batch"] = True
                    batch.meta_info["loss_mask_keys"] = ["response_mask"]

                    if val_metrics:
                        metrics.update(val_metrics)
                        metrics["time/step_val"] = val_timer.last

                    batch = compute_discounted_returns(
                        batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma
                    )
                    batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    # PHASE 11: Reference Log Probs
                    if self.pipeline_config.enable_reference:
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
                            worker = self.reference if self.use_ref_model else self.actor_train
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

                    # PHASE 14: Training (multi-LoRA: use train_step_lora)
                    with Timer(name="train_timer", logger=None) as train_timer:
                        if self.pipeline_config.adv_estimator == "gae":
                            self._request_static_cluster(
                                cluster_id=self._critic_cluster_id,
                                priority=Priority.CRITIC_TRAINING,
                                global_step=global_step,
                            )
                            critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                        if self.pipeline_config.critic_warmup <= global_step:
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

                            # Multi-LoRA: use train_step_lora instead of train_step.
                            actor_train_metrics_refs = self.actor_train.train_step_lora(batch, blocking=False)
                            actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=actor_train_metrics_refs
                            )
                            metrics.update(reduce_metrics(actor_train_metrics.meta_info.pop("metrics", {})))
                            checkpoint_version = int(batch.meta_info.get("checkpoint_version", global_step))

                            # Determine trained adapters from canonical lora_name and fail fast on missing/unknown values.
                            if "lora_name" not in batch.non_tensor_batch:
                                raise RuntimeError(
                                    "multi_lora_pipeline.run(): missing non_tensor_batch['lora_name']. "
                                    "Env managers must inject lora_name before the training step."
                                )
                            lora_name_arr = batch.non_tensor_batch["lora_name"]
                            valid_adapter_names = set(self._tag_to_adapter.values())
                            trained_adapters = list(dict.fromkeys(
                                str(name) for name in lora_name_arr.tolist() if str(name) in valid_adapter_names
                            ))
                            if not trained_adapters:
                                raise RuntimeError(
                                    "multi_lora_pipeline.run(): no recognized adapters in lora_name. "
                                    f"lora_name values={lora_name_arr.tolist()!r} "
                                    f"valid_adapters={sorted(valid_adapter_names)!r}"
                                )

                            # Build per-adapter CPU bucket caches (BEFORE offload_states — needs GPU).
                            for adapter_name in trained_adapters:
                                ray.get([
                                    worker.build_latest_bucket_cache.remote(
                                        checkpoint_version, int(global_step), adapter_name
                                    )
                                    for worker in self.actor_train.workers
                                ])

                            # Promote active adapter versions.
                            for adapter_name in trained_adapters:
                                ray.get([
                                    worker.promote_active_adapter_checkpoint.remote(
                                        adapter_name, checkpoint_version, int(global_step)
                                    )
                                    for worker in self.actor_train.workers
                                ])

                            # Notify scheduler to sync updated adapters to all currently active rollout workers.
                            # All per-tag schedulers share the same underlying RequestScheduler.
                            first_scheduler = next(iter(self.rollout_schedulers.values()))
                            ray.get(first_scheduler.notify_adapter_updated.remote(trained_adapters))

                            # Offload train states (AFTER cache build; cache is CPU-resident).
                            self.actor_train.offload_states(blocking=True)
                            if should_checkpoint:
                                defer_actor_train_release_for_checkpoint = True
                            else:
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
                    self.actor_train.offload_states(blocking=True)
                    if global_step == self.pipeline_config.max_steps - 1:
                        self._release_static_cluster(cluster_id=self._actor_train_cluster_id, global_step=global_step)

                with Timer(name="log", logger=None) as log_timer:
                    if self.pipeline_config.logging_steps > 0 and global_step % self.pipeline_config.logging_steps == 0:
                        logger.info(json.dumps(metrics, ensure_ascii=False))
                metrics["time/step_log"] = log_timer.last

            metrics["time/step_total"] = step_timer.last
            self.tracker.log(values=metrics, step=global_step)
            logger.info(f"[schedrl][{self._pipeline_id}] multi-lora step={global_step} done")

        # Final cleanup.
        if last_notify_ready_step != self.pipeline_config.max_steps - 1:
            self._notify_ready_to_release_actor_infer(global_step=self.pipeline_config.max_steps - 1)

        ray.get([scheduler.shutdown.remote() for scheduler in self.rollout_schedulers.values()])
        ray.get(self.val_rollout_scheduler.shutdown.remote())
        logger.info(f"[schedrl][{self._pipeline_id}] multi-lora pipeline complete!")

    def resize_infer(self, *, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]):
        """SchedRL hook for per-tag scheduler shrink/expand."""
        self._ensure_initialized()
        if not isinstance(dp_ranks_to_remove, list):
            raise ValueError("dp_ranks_to_remove must be list[int]")
        if not isinstance(dp_ranks_to_add, list):
            raise ValueError("dp_ranks_to_add must be list[int]")
        if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add):
            raise ValueError("Exactly one of dp_ranks_to_remove or dp_ranks_to_add must be non-empty")

        if dp_ranks_to_remove:
            self._shrink_all_schedulers(dp_ranks_to_remove=list(dp_ranks_to_remove))
        else:
            try:
                self._expand_all_schedulers(dp_ranks_to_add=list(dp_ranks_to_add))
            except Exception as e:
                error_msg = str(e)
                logger.fatal(
                    f"[schedrl][{self._pipeline_id}] expand failed (possible partial TP group failure): {error_msg}"
                )
                raise RuntimeError(f"PARTIAL_TP_GROUP_FAILURE: {error_msg}") from e

        return ActionResponse(success=True)

    def _shrink_all_schedulers(self, *, dp_ranks_to_remove: List[int]) -> None:
        """Shrink all per-tag rollout schedulers (atomically via shared RequestScheduler)."""
        if not dp_ranks_to_remove:
            raise ValueError("dp_ranks_to_remove must be non-empty")
        with self._infer_resize_lock:
            # All per-tag schedulers and val_rollout_scheduler share the same RequestScheduler actor.
            # A single call with skip_offload=False updates routing state and performs physical offload.
            # We use val_rollout_scheduler as the handle, but any would work.
            ray.get(self.val_rollout_scheduler.shrink_sampler.remote(dp_ranks_to_remove, skip_offload=False))

    def _expand_all_schedulers(self, *, dp_ranks_to_add: List[int]) -> None:
        """Expand all per-tag rollout schedulers (atomically via shared RequestScheduler)."""
        if not dp_ranks_to_add:
            raise ValueError("dp_ranks_to_add must be non-empty")
        with self._infer_resize_lock:
            # All per-tag schedulers and val_rollout_scheduler share the same RequestScheduler actor.
            # A single call with skip_load=False performs weight load/selection sync and updates routing.
            ray.get(self.val_rollout_scheduler.expand_sampler.remote(dp_ranks_to_add, skip_load=False))
            # Fail fast on adapter ID skew after expand/load, before workers serve requests.
            adapters = set(self._tag_to_adapter.values())
            self._verify_lora_model_update(adapters=adapters, where="multi_lora_pipeline._expand_all_schedulers")
            # TODO(item-6): Run a dummy forward pass (batch_size=1) on newly expanded workers to
            # initialize CUDA kernels before exposing them to the scheduler (prevents first-request
            # timeout). Not implemented yet — monitor expand latency before adding.

    def _verify_lora_model_update(self, *, adapters: Optional[set], where: str) -> None:
        """Fail-fast: verify all infer workers agree on adapter_name → lora_int_id mapping."""
        if not adapters:
            return
        if getattr(self.pipeline_config.actor_infer.model_args, "adapters", None) is None:
            raise RuntimeError(
                f"{where}: actor_infer.model_args.adapters not configured; cannot verify LoRA model update."
            )
        timeout_s = float(os.environ.get("ROLL_VERIFY_LORA_TIMEOUT_S", "30"))
        adapter_names = sorted(adapters)
        ray.get(
            [w.wait_loras_ready.remote(adapter_names=adapter_names, timeout_s=timeout_s)
             for w in self.actor_infer.workers]
        )
        for adapter_name in adapter_names:
            lora_ids = ray.get([w.get_lora_id.remote(adapter_name) for w in self.actor_infer.workers])
            if not lora_ids or lora_ids[0] is None:
                raise RuntimeError(
                    f"{where}: infer workers missing adapter id: adapter={adapter_name!r} ids={lora_ids!r}"
                )
            first = lora_ids[0]
            if any(lid != first for lid in lora_ids):
                raise RuntimeError(
                    f"{where}: inconsistent adapter id across infer workers: "
                    f"adapter={adapter_name!r} ids={lora_ids!r}"
                )
