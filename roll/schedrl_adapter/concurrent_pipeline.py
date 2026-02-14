from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.util.timer import _Timer

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
from roll.pipeline.agentic.utils import (
    agentic_compute_advantage,
    compute_discounted_returns,
    compute_response_level_rewards,
    compute_rollout_traj_metrics,
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


class _SchedRLAgenticPipeline(AgenticPipeline):
    """SchedRL-controlled variant of ROLL AgenticPipeline (ENG-123 Phase 3).

    Key differences from upstream AgenticPipeline.run():
    - Before each rollout, request generation GPUs from SchedRL (scheduler drives expand via adapter).
    - After each rollout, shrink actor_infer to zero and release allocation back to SchedRL.
    - Validation runs synchronously to avoid racing with shrink/release.
    """

    def __init__(self, *, pipeline_id: str, pipeline_config: Any):
        if not isinstance(pipeline_id, str) or pipeline_id == "":
            raise ValueError("pipeline_id must be non-empty str")
        self._pipeline_id = pipeline_id
        super().__init__(pipeline_config=pipeline_config)
        try:
            self._schedrl_scheduler = ray.get_actor("schedrl:scheduler", namespace="schedrl")
        except Exception as e:
            raise RuntimeError("Failed to resolve schedrl:scheduler in namespace 'schedrl'") from e
        self._actor_infer_cluster_id = f"{self._pipeline_id}_actor_infer"
        self._ensure_model_update_service()

    def _ensure_model_update_service(self) -> None:
        from roll.schedrl_adapter.model_update_service import ModelUpdateService
        from roll.utils.constants import RAY_NAMESPACE

        ModelUpdateSvc = ModelUpdateService.options(
            name=f"{self._pipeline_id}_model_update_service",
            namespace=RAY_NAMESPACE,
            get_if_exists=True,
            max_restarts=0,
            max_task_retries=0,
        )
        ModelUpdateSvc.remote(
            pipeline_id=self._pipeline_id,
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
        )

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

    def _request_and_expand_actor_infer(self, *, global_step: int) -> List[int]:
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

    def _notify_ready_to_release_actor_infer(self, *, global_step: int, planned_release_gpu_ids: List[int]) -> List[int]:
        timeout_s_raw = os.environ.get("SCHEDRL_NOTIFY_READY_TIMEOUT_S", "300")
        try:
            timeout_s = float(timeout_s_raw)
        except ValueError as e:
            raise RuntimeError(f"Invalid SCHEDRL_NOTIFY_READY_TIMEOUT_S={timeout_s_raw!r}") from e
        if timeout_s <= 0:
            raise RuntimeError(f"SCHEDRL_NOTIFY_READY_TIMEOUT_S must be > 0, got {timeout_s!r}")

        ray.get(self.train_rollout_scheduler.suspend.remote())
        ray.get(self.val_rollout_scheduler.suspend.remote())

        released = ray.get(
            self._schedrl_scheduler.notify_ready_to_release.remote(
                cluster_id=self._actor_infer_cluster_id,
                global_step=global_step,
                timeout_s=timeout_s,
                planned_release_gpu_ids=list(planned_release_gpu_ids),
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
        tps_timer = _Timer(window_size=5)

        # Start from a well-defined state: actor_infer offloaded + routing disabled until we request GPUs.
        ray.get(self.train_rollout_scheduler.suspend.remote())
        try:
            dp_ranks = self._actor_infer_all_dp_ranks()
            ray.get(self.train_rollout_scheduler.shrink_sampler.remote(dp_ranks))
            ray.get(self.val_rollout_scheduler.suspend.remote())
            ray.get(self.val_rollout_scheduler.shrink_sampler.remote(dp_ranks))
        except Exception:
            # Fail-fast semantics: if this doesn't work, the pipeline can't be safely controlled by SchedRL.
            raise

        for global_step in range(self.pipeline_config.max_steps):
            if global_step <= self.state.step:
                global_step += 1
                continue
            logger.info(f"[schedrl][{self._pipeline_id}] pipeline global_step={global_step} start")
            metrics: Dict[str, Any] = {}

            with Timer(name="pipeline_step_total", logger=None) as step_timer:
                with tps_timer:
                    # PHASE 1: Offload States
                    if self.pipeline_config.adv_estimator == "gae":
                        self.critic.offload_states(blocking=True)
                    self.actor_train.offload_states(blocking=True)

                    # PHASE 2: Suspend rollout scheduler to pause request processing
                    ray.get(self.train_rollout_scheduler.suspend.remote())

                    # PHASE 3: Model Update
                    with Timer(name="model_update", logger=None) as model_update_timer:
                        model_update_metrics: Dict = self.model_update(global_step)
                    metrics["time/step_model_update"] = model_update_timer.last
                    metrics.update(model_update_metrics)

                    # PHASE 4: Request + expand actor_infer to SchedRL allocation
                    allocated_gpus = self._request_and_expand_actor_infer(global_step=global_step)

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

                    # Release generation GPUs during training phase (scheduler-driven shrink).
                    self._notify_ready_to_release_actor_infer(
                        global_step=global_step,
                        planned_release_gpu_ids=allocated_gpus,
                    )

                    batch = compute_discounted_returns(
                        batch, self.pipeline_config.adv_estimator, self.pipeline_config.step_reward_gamma
                    )

                    batch = self.adjust_batch(batch, mode=self.pipeline_config.batch_adjust_mode)
                    metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    # PHASE 11: Reference Log Probs
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

                    # PHASE 12: Old Log Probs & Values
                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        if self.pipeline_config.enable_reference and not self.use_ref_model:
                            batch.meta_info["disable_adapter"] = False
                        batch.meta_info["is_offload_states"] = False
                        if self.pipeline_config.enable_old_logprobs_recompute:
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
                        else:
                            batch.batch["old_log_probs"] = torch.zeros_like(batch.batch["attention_mask"][:, 1:])

                        if self.pipeline_config.adv_estimator == "gae":
                            values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)

                        if self.pipeline_config.adv_estimator == "gae":
                            values = DataProto.materialize_concat(data_refs=values_refs)
                            batch = batch.union(values)
                            metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))

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
                            critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                        if self.pipeline_config.critic_warmup <= global_step:
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

                        if self.pipeline_config.adv_estimator == "gae":
                            critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                            metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))
                        tps_timer.push_units_processed(n=torch.sum(batch.batch["attention_mask"]).detach().item())
                    metrics["time/step_train"] = train_timer.last

                from roll.pipeline.agentic.utils import compute_train_data_metrics

                with Timer(name="compute_data_metrics", logger=None) as data_metrics_timer:
                    data_metrics = compute_train_data_metrics(batch=batch)

                metrics["time/step_compute_data_metrics"] = data_metrics_timer.last
                metrics.update(data_metrics)
                metrics["system/tps"] = tps_timer.mean_throughput
                metrics["system/samples"] = (global_step + 1) * self.pipeline_config.rollout_batch_size

                self.state.step = global_step
                self.state.log_history.append(metrics)

                self.do_checkpoint(global_step=global_step)

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

        ray.get([self.train_rollout_scheduler.shutdown.remote(), self.val_rollout_scheduler.shutdown.remote()])
        logger.info(f"[schedrl][{self._pipeline_id}] pipeline complete!")


class SchedRLConcurrentPipeline:
    def __init__(self, *, pipeline_id: str):
        if not isinstance(pipeline_id, str) or pipeline_id == "":
            raise ValueError("pipeline_id must be non-empty str")
        self._pipeline_id = pipeline_id
        self._pipeline: Optional[_SchedRLAgenticPipeline] = None

    def resize_infer(self, *, dp_ranks_to_remove: List[int], dp_ranks_to_add: List[int]) -> Dict[str, Any]:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized; call run() first")
        if not isinstance(dp_ranks_to_remove, list):
            raise ValueError("dp_ranks_to_remove must be list[int]")
        if not isinstance(dp_ranks_to_add, list):
            raise ValueError("dp_ranks_to_add must be list[int]")
        if bool(dp_ranks_to_remove) == bool(dp_ranks_to_add):
            raise ValueError("Exactly one of dp_ranks_to_remove or dp_ranks_to_add must be non-empty")
        if dp_ranks_to_remove:
            return self._pipeline._shrink_workers(dp_ranks_to_remove=list(dp_ranks_to_remove))
        return self._pipeline._expand_workers(dp_ranks_to_add=list(dp_ranks_to_add), train_skip_load=False)

    def run(self, *, pipeline_config: Any) -> None:
        self._pipeline = _SchedRLAgenticPipeline(pipeline_id=self._pipeline_id, pipeline_config=pipeline_config)
        self._pipeline.run()
