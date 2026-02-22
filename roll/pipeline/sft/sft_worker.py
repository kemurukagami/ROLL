import os
from typing import Dict, Union, Optional

import torch
from codetiming import Timer

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import register, Dispatch
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.utils.functionals import reduce_metrics
from roll.utils.lora_routing import ensure_lora_name_in_batch
from roll.models.model_providers import default_actor_model_provider
from roll.platforms import current_platform


class SFTWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.tokenizer = None
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config)
        self.strategy = create_strategy(worker=self)
        self.strategy.initialize(model_provider=default_actor_model_provider)
        self.logger.info(f"{self.worker_name} initialized")

    @register(Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def train_step(self, data: DataProto):
        if data.meta_info is None:
            data.meta_info = {}
        data.meta_info.setdefault("_broadcast_non_tensor_batch", True)
        data = self.strategy.get_data_input(data)
        data = data.to(current_platform.device_type)

        metrics = self.strategy.train_step(batch=data, loss_func=self.loss_func)

        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        return output

    @register(Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def train_step_lora(self, data: DataProto):
        """Multi-LoRA training step.

        Routes to ``MegatronTrainStrategy.train_step_lora`` which dispatches
        per-adapter optimizer.step() when ``lora_optimizer_mode='per_adapter'``.

        The microbatch must carry ``non_tensor_batch["lora_name"]`` to
        identify which adapter owns the batch.
        """
        if data.meta_info is None:
            data.meta_info = {}
        # Broadcast non_tensor_batch (including lora_name) to all TP/PP ranks first.
        # ensure_lora_name_in_batch runs after so every rank has the full non_tensor_batch.
        data.meta_info.setdefault("_broadcast_non_tensor_batch", True)
        data = self.strategy.get_data_input(data)
        # Validate/fill lora_name after broadcast — all ranks now have non_tensor_batch.
        _bs = data.batch.batch_size[0] if data.batch is not None else None
        ensure_lora_name_in_batch(
            data.non_tensor_batch,
            adapters=self.worker_config.model_args.adapters,
            batch_size=_bs,
        )
        data = data.to(current_platform.device_type)
        metrics = self.strategy.train_step_lora(data, loss_func=self.loss_func)
        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        return output

    @register(Dispatch.DP_MP_DISPATCH_FIRST, clear_cache=False)
    def val_step(self, data: DataProto):
        data.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size
        data = self.strategy.get_data_input(data)
        data = data.to(current_platform.device_type)
        metrics = self.strategy.forward_step(batch=data, forward_func=self.loss_func)
        if metrics is None:
            metrics = {}
        metrics = reduce_metrics(metrics)
        output = DataProto(meta_info={"metrics": metrics}).to("cpu")
        return output

    @register(Dispatch.ONE_TO_ALL)
    def do_checkpoint(self, global_step, is_last_step=False):
        with Timer("do_checkpoint") as total_timer:
            ckpt_id = f"checkpoint-{global_step}"
            save_dir = os.path.join(self.pipeline_config.output_dir, self.worker_name, ckpt_id, self.cluster_name)
            self.logger.info(f"save checkpoint-{global_step} to {save_dir}")
            exec_metrics: Dict = self.strategy.save_checkpoint(save_dir, global_step, ckpt_id, is_last_step=is_last_step)

        metrics = {
            f"time/{self.cluster_name}/do_checkpoint/total": total_timer.last,
        }
        metric_prefix = f"time/{self.cluster_name}/do_checkpoint"
        metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        output = DataProto(meta_info={"metrics": metrics})
        return output

    # ------------------------------------------------------------------
    # Per-adapter LoRA weight management (Phase-1 multi-LoRA port)
    # ------------------------------------------------------------------

    @register(Dispatch.ONE_TO_ALL)
    def get_lora_tensors(self, adapter_name: str) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of all LoRA parameter tensors for *adapter_name*.

        Called on all workers; caller typically uses ``result[0]`` (rank-0)
        since all DP/TP ranks hold the same LoRA weights.
        """
        return self.strategy.get_lora_tensors(adapter_name)

    @register(Dispatch.ONE_TO_ALL)
    def set_lora_tensors(self, adapter_name: str, tensors: Dict[str, torch.Tensor]) -> int:
        """Overwrite LoRA parameters for *adapter_name* in-place on all workers."""
        return self.strategy.set_lora_tensors(adapter_name=adapter_name, tensors=tensors)

    @register(Dispatch.ONE_TO_ALL)
    def copy_lora_params(self, src_adapter: str, dst_adapter: str) -> int:
        """Copy LoRA parameters from *src_adapter* to *dst_adapter* on all workers."""
        return self.strategy.copy_lora_params(src_adapter=src_adapter, dst_adapter=dst_adapter)

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        labels = data.batch["labels"]
        batch_num_tokens = data.meta_info['batch_num_tokens']['labels']
        loss, metrics = self.strategy.op_compute_language_loss(output_tensor, labels, batch_num_tokens)
        return loss, metrics
