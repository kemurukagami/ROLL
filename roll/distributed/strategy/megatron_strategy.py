import math
import os
import random
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import ray
import ray.actor
import torch
import torch.distributed as dist
from codetiming import Timer
from megatron.core import DistributedDataParallel, dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.optimizer import MegatronOptimizer, OptimizerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
)
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.packed_seq_params import PackedSeqParams

from mcore_adapter import TrainingArguments
from mcore_adapter.checkpointing import get_checkpoint_dir, load_state_dict_from_checkpoint
from mcore_adapter.parallel_functions import context_parallel_gather, vocab_parallel_logprobs
from mcore_adapter.patcher import patch_torch_find_nd_overlapping_shards, patch_torch_validate_global_plan
from mcore_adapter.trainer.utils import get_megatron_lr_scheduler
from roll.datasets.collator import collate_fn_to_dict_list
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_processor_provider, default_tokenizer_provider
from roll.platforms import current_platform
from roll.third_party.megatron.model_update import (
    MegatronWeightUpdater,
    gather_all_hf_weights,
    gather_weights_meta_cross_pp,
)
from roll.third_party.megatron.offload_states_patch import (
    MegatronOffloadStateType,
    bind_megatron_offload_states_func,
    offload_megatron_no_grad_module,
    reload_megatron_no_grad_module,
)
from roll.third_party.megatron.optimizer import get_megatron_optimizer
from roll.third_party.megatron.tensor_parallel import vocab_parallel_entropy
from roll.utils.constants import (
    DIST_OPTIMIZER_DIR,
    IGNORE_INDEX,
    OPTIMIZER_NAME,
    RNG_STATE_DIR,
    SCHEDULER_NAME,
)
from roll.utils.context_managers import disable_gradients
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.dynamic_batching import make_micro_batch_iter_for_dynamic_batching
from roll.utils.functionals import append_to_dict, reduce_metrics, adjust_sequence_length
from roll.utils.collective import collective
from roll.utils.logging import get_logger
from roll.utils.lora_routing import resolve_microbatch_lora_name
from roll.utils.network_utils import collect_free_port, get_node_ip
from roll.utils.offload_states import OffloadStateType
from roll.utils.send_recv_utils import _bucket_named_tensors, named_tensors_from_bucket
from roll.utils.sequence_packing import make_micro_batch_iter_for_sequence_packing, restore_results_order


if TYPE_CHECKING:
    from mcore_adapter.models.model_factory import VirtualModels

logger = get_logger()


def _safe_dist_barrier(group=None):
    if not dist.is_available() or not dist.is_initialized():
        return
    kwargs = {}
    if dist.get_backend() == "nccl" and current_platform.is_available():
        kwargs["device_ids"] = [current_platform.current_device()]
    if group is None:
        dist.barrier(**kwargs)
    else:
        dist.barrier(group=group, **kwargs)


class MegatronInferStrategy(InferenceStrategy):
    strategy_name = "megatron_infer"

    def __init__(self, worker: Worker):
        #TODO remove the patches when the latest pytorch version > v2.9.1
        patch_torch_find_nd_overlapping_shards()
        patch_torch_validate_global_plan()
        super().__init__(worker)
        config_dict = self.worker_config.training_args.to_dict()
        config_dict.update(self.worker_config.strategy_args.strategy_config)
        # maybe put max_grad_norm into training_args as transformers do, rather
        # than in pipeline_config (PPOConfig)
        config_dict.update({"max_grad_norm": self.worker.pipeline_config.max_grad_norm})
        supported_keys = set(TrainingArguments.__dataclass_fields__.keys())
        dropped_keys = [k for k in config_dict if k not in supported_keys]
        if dropped_keys:
            logger.warn(f"Ignore non-TrainingArguments keys: {dropped_keys}")
            config_dict = {k: v for k, v in config_dict.items() if k in supported_keys}
        logger.info(f"training_args: {config_dict}")
        self.megatron_train_args = TrainingArguments(**config_dict)
        self.model = None
        self.forward_backward_func = None
        self.seq_length = None
        self.use_sequence_packing = self.worker_config.use_sequence_packing
        # hard to impl with offload states
        assert not self.megatron_train_args.overlap_param_gather, "overlap_param_gather is not supported"

    def initialize(self, model_provider):
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.model: "VirtualModels" = model_provider(
            tokenizer=self.tokenizer,
            model_args=self.worker_config.model_args,
            training_args=self.megatron_train_args,
            is_trainable=False,
        )
        self.model.config.finalize_model_grads_func = finalize_model_grads

        self.models_unwrapped = self.model.get_models()
        self.forward_backward_func = get_forward_backward_func()

        self.seq_length = self.worker.pipeline_config.sequence_length
        self.is_lora = self.worker_config.model_args.adapters is not None

        self.worker.rank_info.dp_rank = mpu.get_data_parallel_rank(with_context_parallel=False)
        self.worker.rank_info.dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
        self.worker.rank_info.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.worker.rank_info.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.worker.rank_info.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.worker.rank_info.pp_size = mpu.get_pipeline_model_parallel_world_size()
        self.worker.rank_info.cp_size = mpu.get_context_parallel_world_size()
        self.worker.rank_info.cp_rank = mpu.get_context_parallel_rank()

        if (self.worker_config.use_dynamic_batching_in_infer or self.worker_config.use_sequence_packing) and self.worker.rank_info.pp_size > 1:
            self.model.config.variable_seq_lengths = True
            logger.info("Set variable_seq_lengths to True when use dynamic batching and pipeline parallel.")

        logger.info(f"{self.model.get_models()}")
        _safe_dist_barrier()

    def get_data_input(self, batch: DataProto):
        def broadcast_obj(obj, group):
            obj_list = [obj if dist.get_rank(group) == 0 else None]
            src_rank = dist.get_process_group_ranks(group)[0]
            dist.broadcast_object_list(obj_list, src=src_rank, group=group)
            return obj_list[0]

        # to avoid making side-effect on LLM, if want to broadcast non_tensor_batch,
        # set _broadcast_non_tensor_batch into meta_info
        broadcast_non_tensor_batch = batch.meta_info.get("_broadcast_non_tensor_batch", False)

        if mpu.get_pipeline_model_parallel_rank() == 0 and mpu.get_tensor_and_context_parallel_world_size() > 1:
            if broadcast_non_tensor_batch:
                tmp_batch = broadcast_obj(batch, mpu.get_tensor_and_context_parallel_group())
                batch.batch = tmp_batch.batch
                batch.non_tensor_batch = tmp_batch.non_tensor_batch
            else:
                batch.batch = broadcast_obj(batch.batch, mpu.get_tensor_and_context_parallel_group())

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            if broadcast_non_tensor_batch:
                tmp_batch = broadcast_obj(batch, mpu.get_pipeline_model_parallel_group())
                batch.batch = tmp_batch.batch
                batch.non_tensor_batch = tmp_batch.non_tensor_batch
            else:
                batch.batch = broadcast_obj(batch.batch, mpu.get_pipeline_model_parallel_group())

        return batch

    def forward_step(
        self,
        batch: DataProto,
        forward_func: Callable[[DataProto, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        batch.meta_info['batch_num_tokens'] = self._get_batch_num_tokens(batch, dp_group=mpu.get_data_parallel_group())
        batch.meta_info['global_valid_samples'] = self._get_global_valid_samples(batch, dp_group=mpu.get_data_parallel_group())

        output_on_all_tp_cp_ranks = batch.meta_info.get("output_on_all_tp_cp_ranks", False)
        if self.worker_config.use_dynamic_batching_in_infer:
            micro_batches_list = list(make_micro_batch_iter_for_dynamic_batching(batch))
            num_microbatches = batch.meta_info["num_micro_batchs"]
            micro_batch_size = 1
        elif self.use_sequence_packing:
            vp_size = self.worker_config.strategy_args.strategy_config['virtual_pipeline_model_parallel_size'] \
                if 'virtual_pipeline_model_parallel_size' in self.worker_config.strategy_args.strategy_config else 1
            micro_batches_list = list(
                make_micro_batch_iter_for_sequence_packing(batch, tp_size=self.worker.rank_info.tp_size,
                                                           cp_size=self.worker.rank_info.cp_size,
                                                           vp_size=vp_size, is_train=False,
                                                           dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
                                                           micro_batch_size=batch.meta_info["micro_batch_size"],
                                                           config=self.worker_config.sequence_packing_args))
            num_microbatches = micro_batches_list[0].meta_info["num_micro_batchs"]
            micro_batch_size = 1
        else:
            batch_size = batch.batch.batch_size[0]
            micro_batch_size = batch.meta_info["micro_batch_size"]
            num_microbatches = max(batch_size // micro_batch_size, 1)
            micro_batches_list = batch.chunk(chunks=num_microbatches)

        disable_adapter = batch.meta_info.get("disable_adapter", False)
        adapter_context = self.models_unwrapped[0].disable_adapter() if disable_adapter else nullcontext()

        for micro_batch in micro_batches_list:
            micro_batch.meta_info['loss_scale'] = num_microbatches * mpu.get_data_parallel_world_size()
            micro_batch.meta_info['micro_batch_size'] = micro_batch.batch.batch_size[0]

        data_iterator = [iter(micro_batches_list) for _ in range(len(self.model))]
        with disable_gradients(models=self.model.get_models()), adapter_context:
            # List 是每个 micro-batch 构成的
            losses_reduced: List[Dict[str, torch.Tensor]] = self.forward_backward_func(
                forward_step_func=partial(self.inner_forward_step, forward_func),
                data_iterator=data_iterator,
                model=self.model.get_models(),
                num_microbatches=num_microbatches,
                seq_length=self.seq_length,
                micro_batch_size=micro_batch_size,
                forward_only=True,
            )
        if self.worker_config.use_dynamic_batching_in_infer:
            for data in losses_reduced:
                for k, v in data.items():
                    data[k] = torch.nn.functional.pad(v, (0, self.seq_length - data[k].size(-1) - 1), "constant", 0)
        results = collate_fn_to_dict_list(losses_reduced)

        if self.use_sequence_packing:
            results = restore_results_order(results, micro_batches_list[0].meta_info['partition_indices_list'],
                                  self.worker_config.sequence_packing_args)


        if not (
                ((self.worker.rank_info.tp_rank == 0
                and self.worker.rank_info.cp_rank == 0) or output_on_all_tp_cp_ranks)
                and self.worker.rank_info.is_pipeline_last_stage
        ):
            return None
        return results

    def _get_feature_on_this_cp_rank(self, feature: torch.Tensor, feature_name: str = "input_ids") -> torch.Tensor:
        # Debugging aid: detect unexpected device transition during CP slicing.
        out = self.models_unwrapped[0].get_batch_on_this_cp_rank({feature_name: feature}, dim3_keys=[])[feature_name]
        if (
            feature is not None
            and out is not None
            and isinstance(feature, torch.Tensor)
            and isinstance(out, torch.Tensor)
            and feature.device != out.device
        ):
            logger.info(
                "[device_trace][cp_rank_slice] rank=%s feature=%s in_device=%s out_device=%s",
                self.worker.rank_info.rank,
                feature_name,
                feature.device,
                out.device,
            )
        return out

    def _get_unpad_seqlen(self, attention_mask: torch.Tensor, pad_to_multiple_of: int = 256) -> int:
        max_seqlen = attention_mask.sum(dim=1).max().item()

        cp_size = mpu.get_context_parallel_world_size()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        pad_factor = 2 * cp_size * tp_size if cp_size > 1 else tp_size
        pad_factor = math.lcm(pad_factor, pad_to_multiple_of)

        padded_max_seqlen = (max_seqlen + pad_factor - 1) // pad_factor * pad_factor

        return padded_max_seqlen

    def _get_pad_factor(self):
        # caculate pad_factor in sequence packing
        cp_size = mpu.get_context_parallel_world_size()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        pad_factor = cp_size * 2 * tp_size if cp_size > 1 else tp_size
        pad_factor = math.lcm(16, pad_factor)
        return pad_factor

    def _pack_sequences(self, input_tensor, attention_mask, pad_packed_seq_to=None, pad_val=0):
        """
        Pack multiple sequences into a single continuous sequence by removing padding.

        Implements sequence packing for efficient batch processing with variable-length sequences.
        Removes per-sample padding and concatenates sequences while maintaining cumulative length info.

        Args:
            input_tensor (torch.Tensor): Shape [batch_size, seq_len, ...], padded sequences.
            attention_mask (torch.Tensor): Shape [batch_size, seq_len], 1=valid, 0=padding.
            pad_packed_seq_to (int, optional): Target length for packed sequence. Defaults to None.
            pad_val (int): Padding value. Defaults to 0.

        Returns:
            tuple: (packed_input_tensor, packed_seq_params, cu_seqlens, cu_seqlens_padded)
                - packed_input_tensor: Shape [1, total_packed_length, ...], ready for current CP rank
                - packed_seq_params: PackedSeqParams with cumulative lengths and max_seqlen
                - cu_seqlens: Shape [batch_size + 1], cumulative lengths of original sequences
                - cu_seqlens_padded: Shape [batch_size + 1], cumulative lengths after alignment

        Note:
            - Sequences padded to alignment boundaries if pad_factor > 1 or pad_packed_seq_to is set
            - For CP training, sequences distributed across CP ranks
            - attention_mask not needed after packing
        """

        batch_size = input_tensor.shape[0]
        seq_lens = attention_mask.sum(dim=-1)
        pad_factor = self._get_pad_factor()

        # Remove padding from each sequence
        # Note: attention_mask is not needed in sequence packing mode
        input_tensor_unpadded = [input_tensor[b][:seq_lens[b]] for b in range(batch_size)]

        # Build cumulative sequence lengths
        cu_seqlens = [0]
        cu_seqlens_padded = ([0] if pad_factor > 1 or pad_packed_seq_to is not None
                             else None
                             )

        # Calculate cumulative lengths for both original and padded sequences
        for b in range(batch_size):
            seq_len = seq_lens[b].item() if torch.is_tensor(seq_lens[b]) else seq_lens[b]
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
            if pad_factor > 1 or pad_packed_seq_to is not None:
                # Pad sequence length to multiple of pad_factor
                padded_seq_len = ((seq_len + pad_factor - 1) // pad_factor) * pad_factor
                cu_seqlens_padded.append(cu_seqlens_padded[-1] + padded_seq_len)

        # Convert to tensors
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=current_platform.device_type)
        if pad_factor > 1 or pad_packed_seq_to is not None:
            cu_seqlens_padded = torch.tensor(cu_seqlens_padded, dtype=torch.int32, device=current_platform.device_type)
            if pad_packed_seq_to is not None:
                cu_seqlens_padded[-1] = pad_packed_seq_to

        # Calculate maximum sequence length
        if pad_factor > 1 or pad_packed_seq_to is not None:
            seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
            max_seqlen = seq_lens_padded.max().item()
        else:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = seq_lens.max().item()

        cp_size = mpu.get_context_parallel_world_size()

        # Track running sequence length for padding
        running_seq_len = 0
        all_input_tensor_padded = []
        padded_tokens = []
        for b in range(batch_size):
            seq_len = seq_lens[b].item() if torch.is_tensor(seq_lens[b]) else seq_lens[b]
            if b == batch_size - 1 and pad_packed_seq_to is not None:
                # Different from original implementation: calculate remaining length
                padded_seq_len = pad_packed_seq_to - running_seq_len
            else:
                # Align to pad_factor boundary
                padded_seq_len = ((seq_len + pad_factor - 1) // pad_factor) * pad_factor

            running_seq_len += padded_seq_len

            seq_tokens = input_tensor_unpadded[b]

            # Pad sequence if needed
            if padded_seq_len > seq_len:
                seq_tokens = torch.nn.functional.pad(
                    seq_tokens, (0, padded_seq_len - seq_len), value=pad_val
                )
            all_input_tensor_padded.append(seq_tokens)

            if cp_size > 1:
                # Handle Context Parallel distribution
                # Add batch dimension for processing
                seq_tokens_with_batch = seq_tokens.unsqueeze(0)  # [1, seq_len]
                seq_tokens_with_batch = self._get_feature_on_this_cp_rank(
                    seq_tokens_with_batch, "seq_tokens"
                )
                seq_tokens = seq_tokens_with_batch.squeeze(0)  # Remove batch dimension

            padded_tokens.append(seq_tokens)

        # Concatenate all sequences
        packed_input_tensor = torch.cat(padded_tokens, dim=0).unsqueeze(0)
        all_input_tensor_padded = torch.cat(all_input_tensor_padded, dim=0).unsqueeze(0)

        if cu_seqlens_padded is None:
            cu_seqlens_padded = cu_seqlens.clone()

        # Create packed sequence parameters for attention computation
        # Only use padded cumulative sequence lengths
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            # Individual sequence length
            max_seqlen_q=int(max_seqlen),
            max_seqlen_kv=int(max_seqlen),
            qkv_format="thd",
        )

        return (
            # Packed input tensor for current rank (especially CP rank) computation
            # Contains all tokens from the batch with individual sample padding/alignment preserved
            packed_input_tensor.contiguous(),

            # Parameters required for sequence packing
            packed_seq_params,

            # Cumulative sequence lengths of original unpadded data
            cu_seqlens,

            # Cumulative sequence lengths after padding/alignment
            cu_seqlens_padded,
        )

    def _unpack_sequences(self, output_tensor, cu_seqlens_padded):
        """
        Unpack concatenated sequences into individual padded sequences.
        """
        cp_size = mpu.get_context_parallel_world_size()
        seq_starts = cu_seqlens_padded[:-1] // cp_size
        seq_ends = cu_seqlens_padded[1:] // cp_size

        for seq_idx, (seq_start, seq_end) in enumerate(zip(seq_starts, seq_ends)):
            local_chunk = output_tensor[:, seq_start:seq_end]
            yield local_chunk

    def inner_forward_step(self, loss_func, data_iterator: Iterator[DataProto], model):
        data = next(data_iterator)
        logger.info(f"inner_forward_step enter rank={self.worker.rank_info.rank}")
        if self.is_lora:
            routing = resolve_microbatch_lora_name(data.non_tensor_batch)
            for m in self.models_unwrapped:
                m.set_adapter(routing.lora_name)
        is_pp_first = mpu.is_pipeline_first_stage()
        is_pp_last = mpu.is_pipeline_last_stage()

        input_ids = data.batch["input_ids"] if is_pp_first else None
        attention_mask = data.batch["attention_mask"] if is_pp_first else None
        labels = data.batch["labels"] if (is_pp_last and "labels" in data.batch) else None  # labels is only used for sft
        packed_seq_params = None
        # Root-cause tracing: per-call logs for LoRA train forwards. One-time logs are insufficient because
        # earlier compute_log_probs forwards can consume the once-only guard before train_step_lora executes.
        is_lora_train_forward = bool(data.meta_info and ("grad_accumulation_loss_scale" in data.meta_info))
        # Root-cause tracing: log once per strategy instance before CP split/transforms.
        if is_pp_first and input_ids is not None and not getattr(self, "_logged_lora_inner_pre_cp_once", False):
            logger.info(
                "[device_trace][inner_forward_step/pre_cp] rank=%s input_ids=%s attention_mask=%s labels=%s",
                self.worker.rank_info.rank,
                input_ids.device,
                attention_mask.device if attention_mask is not None else None,
                labels.device if labels is not None else None,
            )
            self._logged_lora_inner_pre_cp_once = True
        if is_pp_first and input_ids is not None and is_lora_train_forward:
            logger.info(
                "[device_trace][inner_forward_step/pre_cp_lora_train] rank=%s input_ids=%s attention_mask=%s labels=%s",
                self.worker.rank_info.rank,
                input_ids.device,
                attention_mask.device if attention_mask is not None else None,
                labels.device if labels is not None else None,
            )

        if self.use_sequence_packing and is_pp_first:
            input_ids, packed_seq_params, cu_seqlens, cu_seqlens_padded = self._pack_sequences(
                input_ids, attention_mask,
            )
            if labels is not None:
                labels, _, _, _ = self._pack_sequences(labels, attention_mask, pad_val=IGNORE_INDEX)
            attention_mask = None
        elif is_pp_first:
            input_ids = self._get_feature_on_this_cp_rank(input_ids, "input_ids")
            attention_mask = self._get_feature_on_this_cp_rank(attention_mask, "attention_mask")
            if labels is not None:
                labels = self._get_feature_on_this_cp_rank(labels, "labels")
            # Root-cause tracing: log once per strategy instance after CP split/transforms.
            if not getattr(self, "_logged_lora_inner_post_cp_once", False):
                logger.info(
                    "[device_trace][inner_forward_step/post_cp] rank=%s input_ids=%s attention_mask=%s labels=%s",
                    self.worker.rank_info.rank,
                    input_ids.device if input_ids is not None else None,
                    attention_mask.device if attention_mask is not None else None,
                    labels.device if labels is not None else None,
                )
                self._logged_lora_inner_post_cp_once = True
            if is_lora_train_forward:
                logger.info(
                    "[device_trace][inner_forward_step/post_cp_lora_train] rank=%s input_ids=%s attention_mask=%s labels=%s",
                    self.worker.rank_info.rank,
                    input_ids.device if input_ids is not None else None,
                    attention_mask.device if attention_mask is not None else None,
                    labels.device if labels is not None else None,
                )
        if attention_mask is not None and attention_mask.dtype != torch.bool and not torch.is_floating_point(attention_mask):
            attention_mask = attention_mask.bool()
        position_ids = None
        # attention_mask: SelfAttention defalt to te DotProductAttention with
        # AttnMaskType.causal in which attention_mask would not be used, pass
        # it mainly for moe aux loss without pad token and it is 2D
        # position_ids: not used in LLM
        # While MCA Qwen2VlModel requires 4D attention_mask, and
        # attention_mask and position_ids would be chunked for cp with dim 2 as
        # seq dim in it if they are provided
        forward_args = data.meta_info.get("forward_args", {})
        if is_pp_first and "position_ids" in data.batch.keys() and data.batch["position_ids"].dim() == 3:  # qwen2vl mrope
            # not support MoE VLM, not used temperarily
            attention_mask = None
            position_ids = data.batch["position_ids"]
            if position_ids.size(1) == 4:
                position_ids = position_ids[:, 1:, :].contiguous()  # (bsz, 4, seqlen) -> (bsz, 3, seqlen)
            position_ids = position_ids.transpose(0, 1)  # (bsz, C, seqlen) -> (C, bsz, seqlen)
        if "multi_modal_inputs" in data.non_tensor_batch:
            multi_modal_inputs = data.non_tensor_batch["multi_modal_inputs"]
            multi_modal_data = defaultdict(list)
            # mm inputs of some samples would be empty to allow text and mm
            # mixed data
            for sample_mm_inputs in multi_modal_inputs:
                for key in sample_mm_inputs.keys():
                    multi_modal_data[key].append(sample_mm_inputs[key])
            for key in multi_modal_data.keys():
                assert key not in forward_args
                # DataProto.to('cuda') in upper frame not work for non_tensor_batch
                target_device = input_ids.device if input_ids is not None else labels.device
                forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(target_device)
            forward_args.update({"force_vit_image": True})

        # megatron_llama_core need loss_mask to compute aux loss
        if "loss_mask" not in forward_args:
            if labels is not None:
                forward_args["loss_mask"] = (labels != IGNORE_INDEX).float()
            elif input_ids is not None:
                forward_args["loss_mask"] = torch.ones_like(input_ids)
            else:
                forward_args["loss_mask"] = None

        # Debugging aid: log exact devices at model-call boundary for LoRA train forwards.
        if is_lora_train_forward and is_pp_first:
            loss_mask = forward_args.get("loss_mask", None)
            loss_mask_device = loss_mask.device if isinstance(loss_mask, torch.Tensor) else None
            # Try best-effort lookup for embedding weight device to compare against input_ids.
            embedding_weight_device = None
            try:
                for n, p in self.models_unwrapped[0].named_parameters():
                    if "word_embeddings.weight" in n:
                        embedding_weight_device = p.device
                        break
            except Exception:
                embedding_weight_device = None
            logger.info(
                "[device_trace][inner_forward_step/model_call_lora_train] rank=%s input_ids=%s attention_mask=%s position_ids=%s labels=%s loss_mask=%s emb_weight=%s",
                self.worker.rank_info.rank,
                input_ids.device if input_ids is not None else None,
                attention_mask.device if attention_mask is not None else None,
                position_ids.device if isinstance(position_ids, torch.Tensor) else None,
                labels.device if labels is not None else None,
                loss_mask_device,
                embedding_weight_device,
            )

        output_tensor = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels,
            packed_seq_params=packed_seq_params, **forward_args
        )
        logger.info(f"inner_forward_step model_done rank={self.worker.rank_info.rank}")

        if self.use_sequence_packing:
            cp_size = mpu.get_context_parallel_world_size()
            def loss_wrapper(output_tensor):
                unpacked_output_iter = self._unpack_sequences(
                    output_tensor,
                    cu_seqlens_padded,
                )
                loss_result = torch.tensor(0.0, device=output_tensor.device)
                metrics_result_list = []
                num_samples = len(data)
                for i in range(num_samples):
                    single_output_tensor = next(unpacked_output_iter)
                    full_seq_len = single_output_tensor.size(1) * cp_size
                    if full_seq_len == 0:
                    # Create a mock output tensor when the sample is empty to ensure the subsequent pipeline works correctly.
                        full_seq_len = self._get_pad_factor()
                        local_seq_len = max(1, full_seq_len // cp_size)
                        new_shape = list(single_output_tensor.shape)
                        new_shape[1] = local_seq_len
                        single_output_tensor = torch.zeros(new_shape, dtype=single_output_tensor.dtype,
                                                           device=single_output_tensor.device)
                    single_data = data[i:i+1]
                    for key, val in single_data.batch.items():
                        single_data.batch[key] = adjust_sequence_length(val, full_seq_len, self.seq_length, pad_value=IGNORE_INDEX
                                                                  if key in {'labels', 'labels_for_loss'} else 0)
                    loss, metrics = loss_func(single_data, single_output_tensor)
                    loss_result += loss
                    for key, val in metrics.items():
                        if isinstance(val, torch.Tensor):
                            metrics[key] = adjust_sequence_length(val, self.seq_length, full_seq_len, pad_value=0)
                    metrics_result_list.append(metrics)
                    del single_output_tensor
                metrics_result_dict = collate_fn_to_dict_list(metrics_result_list)
                if self.worker_config.apply_loss_scale:
                    loss_result *= data.meta_info['loss_scale']
                return loss_result, reduce_metrics(metrics_result_dict)

            return output_tensor, loss_wrapper
        else:
            def loss_wrapper(output_tensor):
                loss, metrics = loss_func(data, output_tensor)
                if self.worker_config.apply_loss_scale:
                    loss *= data.meta_info['loss_scale']
                return loss, metrics
            return output_tensor, loss_wrapper

    def broadcast_parameter(self, *args, **kwargs):
        pass

    def load_states(self, include=None, non_blocking=False):
        reload_megatron_no_grad_module(model_chunks=self.model.get_models())

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            offload_megatron_no_grad_module(model_chunks=self.model.get_models())
        RotaryEmbedding.forward.cache_clear()
        current_platform.empty_cache()

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids [[p, p, r, r, r, 0, 0]] p: prompt, r: response, 0: pad
        response_mask [[0, 0, 1, 1, 1, 0, 0]]
        """
        ori_seq_length = attention_mask.size(1)
        cp_size = mpu.get_context_parallel_world_size()
        seq_len = ori_seq_length

        labels: torch.Tensor = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:seq_len] == 0] = 0  # avoid invalid token id
        # TODO: don't pad here but process this shift after generation
        labels = torch.cat([labels, torch.zeros_like(labels[:, :1])], dim=1)
        labels = self._get_feature_on_this_cp_rank(labels, "labels")
        # compute logprobs in remove padding token
        log_probs = vocab_parallel_logprobs(logits, labels)
        if mpu.get_context_parallel_world_size() > 1:
            log_probs = context_parallel_gather(log_probs, parallel_dim=1)
        log_probs = log_probs[:, :-1] * attention_mask[:, 1:]
        return log_probs

    def op_compute_entropy(self, logits: torch.Tensor, attention_mask: torch.Tensor):
        if self.worker_config.logits_in_fp32:
            logits = logits.float()
        entropy = vocab_parallel_entropy(logits)
        if mpu.get_context_parallel_world_size() > 1:
            entropy = context_parallel_gather(entropy, parallel_dim=1)
        entropy = entropy[:, :-1] * attention_mask[:, 1:]
        return entropy

    def op_compute_language_loss_from_logits(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            reduction: str = "mean"
    ):
        """
        Compute cross-entropy language modeling loss with TP and CP support.

        Handles causal next-token prediction with proper sequence boundary alignment
        in distributed training scenarios.

        Args:
            logits (torch.Tensor): Shape [batch_size, local_seq_len, vocab_size/tp_size].
                                  TP-sharded (vocab) and CP-sharded (sequence).
            targets (torch.Tensor): Shape [batch_size, global_seq_len].
                                   Global vocab IDs, padding marked with IGNORE_INDEX.
            reduction (str): "mean" or "sum". Default: "mean".

        Returns:
            tuple: (loss, token_count)
                - loss: Scalar tensor based on reduction method
                - token_count: int64 tensor, number of valid tokens

        Sequence Alignment:
            - No CP: Simple shift, logits[:, :-1] predicts targets[:, 1:]
            - With CP (2 chunks/rank): Handle chunk boundaries carefully
                * Chunk 0: logits[:, :chunk_size-1] → targets[:, 1:chunk_size]
                * Chunk 1: logits[:, chunk_size:-1] → targets[:, chunk_size+1:]

        Note:
            - vocab_parallel_cross_entropy handles TP all-reduce internally
            - CP all-reduce performed explicitly for loss_sum and token_count
            - Assumes 2 chunks per rank in CP mode for load balancing
        """
        cp_size = mpu.get_context_parallel_world_size()

        # Slice targets to current CP rank's sequence portion
        targets = self._get_feature_on_this_cp_rank(targets, "targets")

        if cp_size == 1:
            # Simple causal shift: logits[t] predicts targets[t+1]
            logits = logits[:, :-1, :].contiguous()
            targets = targets[:, 1:].contiguous()
        else:
            # CP mode: Handle chunk boundaries with load balancing
            local_seq_len = logits.size(1)
            chunk_size = local_seq_len // 2  # 2 chunks per rank

            # Chunk 0: Remove last position (its target is in Chunk 1)
            chunk_0_logits = logits[:, :chunk_size - 1, :]
            chunk_0_targets = targets[:, 1:chunk_size]

            # Chunk 1: Remove last position and skip first target (belongs to Chunk 0)
            chunk_1_logits = logits[:, chunk_size:-1, :]
            chunk_1_targets = targets[:, chunk_size + 1:]

            # Merge chunks
            logits = torch.cat([chunk_0_logits, chunk_1_logits], dim=1)
            targets = torch.cat([chunk_0_targets, chunk_1_targets], dim=1)

        # Transpose to sequence-first layout for Megatron CE
        logits_tp = logits.transpose(0, 1).contiguous()
        labels_tp = targets.transpose(0, 1).contiguous()

        # Compute per-token CE loss (handles TP all-reduce)
        loss_per_token = vocab_parallel_cross_entropy(
            logits_tp, labels_tp, label_smoothing=0.0
        )

        # Apply ignore_index mask
        mask = (labels_tp != IGNORE_INDEX)
        loss_sum_local = (loss_per_token * mask).sum()
        token_count_local = mask.sum()

        # All-reduce across CP ranks
        if cp_size > 1:
            cp_group = mpu.get_context_parallel_group()
            stats_tensor = torch.stack([
                loss_sum_local.float(),
                token_count_local.float()
            ], dim=0)
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM, group=cp_group)
            loss_sum, token_count = stats_tensor[0], stats_tensor[1]
        else:
            loss_sum = loss_sum_local.float()
            token_count = token_count_local.float()

        # Apply reduction
        if reduction == "sum":
            loss = loss_sum
        elif reduction == "mean":
            loss = loss_sum / torch.clamp(token_count, min=1.0)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}. Use 'mean' or 'sum'.")

        return loss, token_count.to(torch.int64)

    def op_compute_topk_logits(
            self,
            logits: torch.Tensor,
            topk: int = 0
    ):
        """
        Compute top-k logits with memory-efficient two-stage approach for TP and CP training.

        Strategy:
            - topk=0: Gather full vocab across TP ranks
            - topk>0: Two-stage TopK (local → gather K values → global TopK → CP gather)

        Args:
            logits (torch.Tensor): Shape [batch_size, local_seq_len, local_vocab_size].
                                  TP-sharded along vocabulary.
            topk (int): 0=full vocab, >0=top-k mode.

        Returns:
            tuple: (values, indices)
                - topk=0: (logits [B, S, V], None)
                - topk>0: (values [B, S, K], indices [B, S, K] in global vocab space)

        Note:
            - Indices adjusted to global vocabulary space
            - Intermediate tensors deleted early
            - CP gathering after TP operations
        """

        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()

        # ========== TopK Mode: Two-Stage Memory Optimization ==========
        if topk > 0:
            # Stage 1: Local TopK on each TP rank's vocabulary shard
            # Memory reduction: [B, local_seq, local_vocab] -> [B, local_seq, K]
            local_topk_values, local_topk_indices = torch.topk(
                logits, k=topk, dim=-1, sorted=False
            )

            # Adjust indices to global vocabulary space
            # Each TP rank owns a contiguous vocabulary range [vocab_start, vocab_end)
            vocab_start_index = mpu.get_tensor_model_parallel_rank() * logits.shape[-1]
            local_topk_indices = local_topk_indices + vocab_start_index

            # Release original logits immediately to save memory
            del logits

            # Stage 2: Gather local TopK results across TP ranks
            # Memory: [B, local_seq, K] -> [B, local_seq, K * tp_world_size]
            # Only gather K values per rank instead of full vocabulary
            gathered_values = local_topk_values
            gathered_indices = local_topk_indices
            if tp_size > 1:
                gathered_values = gather_from_tensor_model_parallel_region(local_topk_values)
                gathered_indices = gather_from_tensor_model_parallel_region(local_topk_indices)
            del local_topk_values, local_topk_indices

            # Stage 3: Global TopK on gathered candidates
            # Select final top-k from K * tp_size candidates
            # Memory: [B, local_seq, K * tp_world_size] -> [B, local_seq, K]
            final_topk_values, topk_positions = torch.topk(
                gathered_values, k=topk, dim=-1, sorted=True
            )
            # Use topk_positions to gather corresponding global indices
            final_topk_indices = torch.gather(
                gathered_indices, dim=-1, index=topk_positions
            )
            del gathered_values, gathered_indices, topk_positions

            # Stage 4: CP gather for sequence parallel training
            if cp_size > 1:
                final_topk_values = context_parallel_gather(final_topk_values, parallel_dim=1)
                final_topk_indices = context_parallel_gather(final_topk_indices, parallel_dim=1)

            return final_topk_values, final_topk_indices

        # ========== Full Vocabulary Mode: Traditional Gather Path ==========
        result = logits
        # Gather full vocabulary across TP ranks
        if tp_size > 1:
            result = gather_from_tensor_model_parallel_region(result)

        # Gather across CP ranks for sequence parallelism
        if cp_size > 1:
            result = context_parallel_gather(result, parallel_dim=1)

        # Return full vocabulary logits
        if topk == 0:
            return result, None

        # Fallback: TopK mode without TP optimization (when TP is not used)
        topk_values, topk_indices = torch.topk(result, k=topk, dim=-1)
        del result

        return topk_values, topk_indices

    def op_compute_gather_by_teacher_indices(
            self,
            student_logits: torch.Tensor,
            teacher_indices: torch.Tensor
    ):
        """
        Gather student logits at teacher indices with TP support via sparse gather.

        Strategy:
            - No TP: Direct torch.gather
            - TP mode: Sparse gather + all-reduce
                1. Mask indices belonging to local vocab shard
                2. Gather local values, zero out non-local
                3. All-reduce sum across TP ranks

        Args:
            student_logits (torch.Tensor): Shape [batch_size, seq_len, local_vocab_size].
                                           TP-sharded along vocabulary.
            teacher_indices (torch.Tensor): Shape [batch_size, seq_len, k] or [batch_size, seq_len].
                                           Global vocabulary indices (not sharded).

        Returns:
            torch.Tensor: Gathered logits matching teacher_indices shape.

        Note:
            - Returns original logits if teacher_indices is None
            - Handles 2D/3D indices, restores original shape
            - Vocab range per rank: [tp_rank * local_vocab_size, (tp_rank+1) * local_vocab_size)
        """

        # Early return if no teacher indices provided
        if teacher_indices is None:
            return student_logits

        # Ensure indices are long type for indexing
        if teacher_indices.dtype != torch.long:
            teacher_indices = teacher_indices.long()

        # Handle 2D input by adding dimension (will be removed before return)
        squeeze_output = False
        if teacher_indices.dim() == 2:
            teacher_indices = teacher_indices.unsqueeze(-1)
            squeeze_output = True

        tp_world_size = mpu.get_tensor_model_parallel_world_size()

        # Non-TP mode: Direct gather operation
        if tp_world_size == 1:
            gathered = torch.gather(student_logits, dim=-1, index=teacher_indices)
            return gathered.squeeze(-1) if squeeze_output else gathered

        # ========== TP-Sharded Sparse Gather ==========
        tp_rank = mpu.get_tensor_model_parallel_rank()
        local_vocab_size = student_logits.shape[-1]

        # Calculate vocabulary range owned by current TP rank
        vocab_start = tp_rank * local_vocab_size
        vocab_end = vocab_start + local_vocab_size

        # Create mask for indices that belong to local vocabulary shard
        local_mask = (teacher_indices >= vocab_start) & (teacher_indices < vocab_end)

        # Convert global indices to local vocabulary space
        # Clamp to valid range to avoid index errors (non-local indices will be masked out)
        local_indices = teacher_indices - vocab_start
        local_indices = torch.clamp(local_indices, 0, local_vocab_size - 1)

        # Gather values from local vocabulary shard
        local_gathered = torch.gather(student_logits, dim=-1, index=local_indices)

        # Mask out values that don't belong to local vocabulary
        # Non-local positions are set to zero (will not contribute to final sum)
        local_gathered = torch.where(local_mask, local_gathered, torch.zeros_like(local_gathered))

        # All-reduce sum across TP ranks (fully differentiable)
        # Forward: Sum contributions from all ranks (only one rank contributes non-zero per index)
        # Backward: Each rank receives full gradient, but only masked portion affects local parameters
        gathered = reduce_from_tensor_model_parallel_region(local_gathered)

        # Restore original shape if input was 2D
        return gathered.squeeze(-1) if squeeze_output else gathered

    def op_compute_various_divergence(
            self,
            loss_callable, logits, teacher_topk_probs, teacher_topk_log_probs, teacher_topk_indices,
            teacher_topk_inf_mask, labels, attention_mask=None, reduction="mean"
    ):
        """
        Compute divergence losses (KL, JSD, RKL, etc.) with TP and CP support.

        Strategy:
            1. Slice teacher outputs to current CP rank's sequence
            2. Gather student logits at teacher's top-k indices (TP-aware)
            3. Compute per-token divergence loss
            4. Gather loss across CP ranks
            5. Apply padding mask and reduction

        Args:
            loss_callable (callable): Divergence function (KL/JSD/RKL).
                                     Takes: logits, teacher_probs, teacher_log_probs, teacher_inf_mask.
            logits (torch.Tensor): Shape [batch_size, local_seq_len, local_vocab_size].
                                  TP and CP sharded.
            teacher_topk_probs (torch.Tensor): Shape [batch_size, global_seq_len, topk].
                                              Full tensor (not sharded).
            teacher_topk_log_probs (torch.Tensor): Shape [batch_size, global_seq_len, topk].
            teacher_topk_indices (torch.Tensor): Shape [batch_size, global_seq_len, topk].
                                                Global vocabulary indices.
            teacher_topk_inf_mask (torch.Tensor): Shape [batch_size, global_seq_len, topk].
            labels (torch.Tensor): Shape [batch_size, global_seq_len].
                                  Padding marked with IGNORE_INDEX.
            attention_mask (torch.Tensor, optional): Shape [batch_size, global_seq_len].
                                                    0=padding. Used if labels is None.
            reduction (str): "mean", "sum", or "none".

        Returns:
            tuple: (loss, token_count)
                - loss: Scalar (mean/sum) or tensor [B, S] (none)
                - token_count: Scalar, number of valid tokens

        Note:
            - Teacher outputs sliced to CP rank's sequence
            - Student logits TP-sharded, handled by sparse gather
            - Token count from full sequence for correct normalization
        """

        # Preserve full tensors for final mask computation
        labels_full = labels
        attention_mask_full = attention_mask

        # (1) Slice teacher outputs to current CP rank's sequence portion
        # Each CP rank processes a contiguous chunk of the sequence
        if teacher_topk_probs is not None:
            teacher_topk_probs = self._get_feature_on_this_cp_rank(teacher_topk_probs, "teacher_topk_probs")
        if teacher_topk_indices is not None:
            teacher_topk_indices = self._get_feature_on_this_cp_rank(teacher_topk_indices, "teacher_topk_indices")
        if teacher_topk_log_probs is not None:
            teacher_topk_log_probs = self._get_feature_on_this_cp_rank(teacher_topk_log_probs,"teacher_topk_log_probs")
        if teacher_topk_inf_mask is not None:
            teacher_topk_inf_mask = self._get_feature_on_this_cp_rank(teacher_topk_inf_mask, "teacher_topk_inf_mask")

        # (2) Gather student logits at teacher's top-k indices
        # Handles TP-sharded logits with sparse gather operation
        # Input: [batch_size, local_seq_len, local_vocab_size] (TP-sharded)
        # Output: [batch_size, local_seq_len, topk] (aligned with teacher indices)
        full_logits = self.op_compute_gather_by_teacher_indices(logits, teacher_topk_indices)

        # (3) Compute per-token divergence loss
        # loss_callable computes divergence (e.g., KL, JSD) between student and teacher distributions
        # Returns: [batch_size, local_seq_len] per-token loss
        kld_per_token = loss_callable(
            logits=full_logits,
            teacher_probs=teacher_topk_probs,
            teacher_log_probs=teacher_topk_log_probs,
            teacher_inf_mask=teacher_topk_inf_mask,
        )

        # (4) Gather per-token loss across CP ranks to restore full sequence
        # Input: [batch_size, local_seq_len] (CP-sharded sequence)
        # Output: [batch_size, global_seq_len] (full sequence)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size > 1:
            kld_per_token = context_parallel_gather(kld_per_token, parallel_dim=1)

        # (5) Compute total number of valid (non-padded) tokens
        # Uses full labels/attention_mask to count across entire batch
        if labels_full is not None:
            # Padding positions marked with IGNORE_INDEX in labels
            pad_mask = labels_full.eq(IGNORE_INDEX)
        else:
            # Alternatively use attention_mask where 0 indicates padding
            pad_mask = attention_mask_full.eq(0)
        token_count = (~pad_mask).sum().float()

        # (6) Early return for 'none' reduction (per-token loss)
        if reduction == 'none':
            return kld_per_token, token_count

        # (7) Apply padding mask and compute aggregated loss
        # Mask out padding positions by setting their loss to 0
        kld_masked = kld_per_token.masked_fill_(pad_mask, 0.0)
        loss_sum = kld_masked.sum()

        # (8) Return loss based on reduction method
        if reduction == "sum":
            # Return sum of loss over all valid tokens
            return loss_sum, token_count
        elif reduction == "mean":
            # Return average loss per valid token
            # Clamp token_count to avoid division by zero
            return loss_sum / token_count.clamp(min=1.0), token_count
        else:
            raise ValueError(f"Unsupported reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

    def op_compute_language_loss(self, losses: torch.Tensor, labels: torch.Tensor, batch_num_tokens: int):
        labels = self._get_feature_on_this_cp_rank(labels, "labels")

        loss_mask = (labels != IGNORE_INDEX).float()
        loss_mask = loss_mask.view(-1).float()
        losses = torch.sum(losses.view(-1) * loss_mask)

        if mpu.get_context_parallel_world_size() > 1:
            loss_info = torch.cat([losses.view(1)])
            torch.distributed.all_reduce(
                loss_info, op=torch.distributed.ReduceOp.SUM, group=mpu.get_context_parallel_group()
            )
            losses = loss_info[0]

        loss = losses.clone() / batch_num_tokens# clone to make sure loss is not a view

        metrics = {f"{self.worker_config.name}/loss@sum": loss.clone().detach().item()}

        return loss, metrics

class MegatronTrainStrategy(MegatronInferStrategy, TrainStrategy):
    strategy_name = "megatron_train"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.models_wrapped = None
        self.models_unwrapped = None
        self.processor = None
        self._validate_access_integrity = True

        # ENG-123 Phase 4: sender-side cached buckets + promotion + selective sync.
        self._cache_lock = threading.Lock()
        self._cache_map: Dict[Tuple[int, int], List[Any]] = {}
        self._latest_cached: Optional[Tuple[int, int]] = None
        self._active_cached: Optional[Tuple[int, int]] = None
        self._selective_update_weights_meta = None
        self._selective_sync_cpu_group = None
        self._selective_sync_cpu_group_size: Optional[int] = None

        # Per-adapter versioned cache (multi-LoRA selective sync)
        self._adapter_cache_map: Dict[str, Dict[Tuple[int, int], List[Any]]] = {}
        self._latest_adapter_cached: Dict[str, Optional[Tuple[int, int]]] = {}
        self._active_adapter_cached: Dict[str, Optional[Tuple[int, int]]] = {}

    def initialize(self, model_provider):
        self.seq_length = self.worker.pipeline_config.sequence_length
        self.weight_updaters: dict[str, MegatronWeightUpdater] = {}

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.processor = default_processor_provider(model_args=self.worker_config.model_args)
        # model provider will initialize megatron distributed groups
        self.model: "VirtualModels" = model_provider(
            tokenizer=self.tokenizer,
            model_args=self.worker_config.model_args,
            training_args=self.megatron_train_args,
            is_trainable=True,
        )
        self.forward_backward_func = get_forward_backward_func()
        self.model.config.finalize_model_grads_func = finalize_model_grads
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=self.megatron_train_args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=self.megatron_train_args.overlap_grad_reduce,
            use_distributed_optimizer=self.megatron_train_args.use_distributed_optimizer,
            check_for_nan_in_grad=self.megatron_train_args.check_for_nan_in_loss_and_grad,
            bucket_size=self.megatron_train_args.ddp_bucket_size,
        )
        self.models_wrapped = [
            DistributedDataParallel(
                config=m.config,
                ddp_config=ddp_config,
                module=m,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_index > 0),
            )
            for model_index, m in enumerate(self.model.get_models())
        ]
        self.models_unwrapped = self.model.get_models()
        self.model.models = self.models_wrapped
        self.is_lora = (self.worker_config.model_args.adapters is not None) or \
                       (getattr(self.worker_config.model_args, "lora_target", None) is not None)

        params_dtype = (
            torch.float16
            if self.megatron_train_args.fp16
            else torch.bfloat16 if self.megatron_train_args.bf16 else torch.float32
        )

        # ---- lora_optimizer_mode: 'shared' (default) or 'per_adapter' ----
        self.lora_optimizer_mode: str = (
            self.worker_config.strategy_args.strategy_config.get("lora_optimizer_mode", "shared")
            if self.worker_config.strategy_args and self.worker_config.strategy_args.strategy_config
            else "shared"
        )
        if self.lora_optimizer_mode not in ("shared", "per_adapter"):
            raise ValueError(
                f"Unknown lora_optimizer_mode={self.lora_optimizer_mode!r} "
                "(expected 'shared' | 'per_adapter')"
            )

        optimizer_config = OptimizerConfig(
            optimizer=self.megatron_train_args.optimizer,
            lr=self.megatron_train_args.learning_rate,
            min_lr=self.megatron_train_args.lr_scheduler_kwargs.get("min_lr", 0.0),
            weight_decay=self.megatron_train_args.weight_decay,
            adam_beta1=self.megatron_train_args.adam_beta1,
            adam_beta2=self.megatron_train_args.adam_beta2,
            adam_eps=self.megatron_train_args.adam_epsilon,
            fp16=self.megatron_train_args.fp16,
            bf16=self.megatron_train_args.bf16,
            params_dtype=params_dtype,
            # per_adapter prototype requires non-distributed optimizer.
            use_distributed_optimizer=(
                False
                if self.lora_optimizer_mode == "per_adapter"
                else self.megatron_train_args.use_distributed_optimizer
            ),
            clip_grad=self.megatron_train_args.max_grad_norm,
        )

        self.adapter_optimizers: Dict[str, MegatronOptimizer] | None = None
        self.adapter_schedulers: Dict[str, Any] | None = None

        if self.lora_optimizer_mode == "shared":
            self.optimizer: MegatronOptimizer = get_megatron_optimizer(optimizer_config, self.models_wrapped)
            logger.info(f"megatron optimizer: {self.optimizer}")
            bind_megatron_offload_states_func(optimizer=self.optimizer)
        else:
            # ---- per_adapter mode: one optimizer + scheduler per adapter ----
            if self.megatron_train_args.use_distributed_optimizer:
                raise ValueError(
                    "lora_optimizer_mode='per_adapter' requires use_distributed_optimizer=False"
                )
            if self.megatron_train_args.overlap_grad_reduce:
                raise ValueError(
                    "lora_optimizer_mode='per_adapter' requires overlap_grad_reduce=False. "
                    "With overlap_grad_reduce=True, idle adapters' DDP backward hooks never fire "
                    "during another adapter's sequential pass, causing a hang in finish_grad_sync()."
                )
            if not self.is_lora:
                raise ValueError(
                    "lora_optimizer_mode='per_adapter' requires LoRA adapters to be configured"
                )
            if getattr(self.worker_config.model_args, "model_type", None) == "trl":
                raise ValueError(
                    "lora_optimizer_mode='per_adapter' does not support TRL value-head models "
                    "(model_type='trl'). Disable value head or use lora_optimizer_mode='shared'."
                )

            adapter_names = list(self.worker_config.model_args.adapters.keys())
            if not adapter_names:
                raise ValueError(
                    "lora_optimizer_mode='per_adapter' requires at least one adapter"
                )

            # PEFT activates trainability only for the currently active adapter.
            # For per-adapter optimizer construction we need a stable snapshot where
            # *all* adapters' LoRA params are considered trainable.
            for model in self.models_unwrapped:
                base_model = getattr(model, "base_model", None)
                if base_model is not None and hasattr(base_model, "set_adapter"):
                    base_model.set_adapter(adapter_names)

            # Verify all trainable params are adapter-scoped (no shared trainables like a value head).
            name_to_param: Dict[str, torch.nn.Parameter] = dict(
                self.models_unwrapped[0].named_parameters()
            )
            original_requires_grad: Dict[str, bool] = {
                n: bool(p.requires_grad) for n, p in name_to_param.items()
            }
            markers = {a: f".{a}." for a in adapter_names}

            shared_trainables: List[str] = []
            for name, param in name_to_param.items():
                if not original_requires_grad[name]:
                    continue
                if not any(marker in name for marker in markers.values()):
                    shared_trainables.append(name)
            if shared_trainables:
                preview = ", ".join(repr(n) for n in shared_trainables[:10])
                likely_value_head = any(
                    ("v_head" in n or "value_head" in n) for n in shared_trainables
                )
                hint = (
                    " This looks like a value head / TRL wrapper. Set model_type: ~ to disable."
                    if likely_value_head
                    else ""
                )
                raise ValueError(
                    "lora_optimizer_mode='per_adapter' requires all trainable parameters to be "
                    f"adapter-scoped (name must include one of: {sorted(markers.values())}). "
                    f"Found shared trainables (first 10): {preview}. "
                    "Either freeze these parameters or use lora_optimizer_mode='shared'."
                    + hint
                )

            # Check that BN/LN running-stats buffers are adapter-scoped (plan item 16).
            # These buffers have requires_grad=False so they are NOT caught by the param check above.
            _NORM_BUFFER_TAGS = ("running_mean", "running_var", "num_batches_tracked")
            shared_norm_buffers: List[str] = [
                name
                for name, _ in self.models_unwrapped[0].named_buffers()
                if any(tag in name for tag in _NORM_BUFFER_TAGS)
                and not any(marker in name for marker in markers.values())
            ]
            if shared_norm_buffers:
                preview = ", ".join(repr(n) for n in shared_norm_buffers[:10])
                raise ValueError(
                    "lora_optimizer_mode='per_adapter' requires BN/LN running-stats buffers to be "
                    f"adapter-scoped (name must include one of: {sorted(markers.values())}). "
                    f"Found shared norm buffers (first 10): {preview}. "
                    "Wrap BN/LN layers in nn.ModuleDict keyed by adapter name."
                )

            def _apply_trainability_mask_for_adapter(active_adapter: str) -> None:
                marker = markers[active_adapter]
                for n, p in name_to_param.items():
                    p.requires_grad_(bool(original_requires_grad[n] and (marker in n)))

            self.adapter_optimizers = {}
            self.adapter_schedulers = {}
            param_id_to_name = {id(p): n for n, p in name_to_param.items()}
            seen_param_ids: Set[int] = set()
            for adapter_name in adapter_names:
                self.models_unwrapped[0].set_adapter(adapter_name)
                _apply_trainability_mask_for_adapter(adapter_name)
                adapter_opt = get_megatron_optimizer(optimizer_config, self.models_wrapped)
                bind_megatron_offload_states_func(optimizer=adapter_opt)

                # Assert optimizer param ownership is isolated to this adapter.
                marker = markers[adapter_name]
                for group in getattr(adapter_opt, "param_groups", []):
                    for param in group.get("params", []):
                        pid = id(param)
                        pname = param_id_to_name.get(pid)
                        if pname is None:
                            # Megatron optimizers may create FP32 "main params" (new Parameter
                            # objects) for FP16/BF16 model params. Those parameters are not
                            # present in model.named_parameters(), so we cannot verify their
                            # adapter ownership here.
                            continue
                        if marker not in pname:
                            raise RuntimeError(
                                f"Per-adapter optimizer for {adapter_name!r} captured "
                                f"non-scoped param {pname!r}"
                            )
                        if pid in seen_param_ids:
                            raise RuntimeError(
                                f"Parameter {pname!r} appears in multiple per-adapter optimizers; "
                                "expected disjoint param sets"
                            )
                        seen_param_ids.add(pid)

                self.adapter_optimizers[adapter_name] = adapter_opt
                self.adapter_schedulers[adapter_name] = get_megatron_lr_scheduler(
                    self.megatron_train_args,
                    self.megatron_train_args.max_steps,
                    optimizer=adapter_opt,
                )

            # Restore original trainability.
            for n, p in name_to_param.items():
                p.requires_grad_(original_requires_grad[n])

            # Chained optimizer for generic offload/load hooks.
            from megatron.core.optimizer import ChainedOptimizer
            self.optimizer = ChainedOptimizer(list(self.adapter_optimizers.values()))
            bind_megatron_offload_states_func(optimizer=self.optimizer)

            # Initialize per-adapter RNG states for sequential training (plan item 15).
            # Each adapter starts from the current global RNG state; they diverge as training progresses.
            self.adapter_rng_states: Dict[str, Dict[str, Any]] = {
                name: {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                }
                for name in adapter_names
            }

        self.worker.rank_info.dp_rank = mpu.get_data_parallel_rank()
        self.worker.rank_info.dp_size = mpu.get_data_parallel_world_size()
        self.worker.rank_info.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.worker.rank_info.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.worker.rank_info.pp_rank = mpu.get_pipeline_model_parallel_rank()
        self.worker.rank_info.pp_size = mpu.get_pipeline_model_parallel_world_size()
        self.worker.rank_info.cp_size = mpu.get_context_parallel_world_size()
        self.worker.rank_info.cp_rank = mpu.get_context_parallel_rank()

        logger.info(f"max steps pipeline {self.worker_config.training_args.max_steps}")
        self.worker_config.training_args.max_steps = (
            self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
        )
        self.megatron_train_args.max_steps = self.worker_config.training_args.max_steps
        logger.info(f"max steps worker train {self.worker_config.training_args.max_steps}")

        # Per-adapter schedulers must use DP-adjusted max_steps. They were initially
        # created before dp_size was known, so rebuild here with the final step budget.
        if self.lora_optimizer_mode == "per_adapter" and self.adapter_optimizers:
            self.adapter_schedulers = {
                adapter_name: get_megatron_lr_scheduler(
                    self.megatron_train_args,
                    self.megatron_train_args.max_steps,
                    optimizer=adapter_opt,
                )
                for adapter_name, adapter_opt in self.adapter_optimizers.items()
            }

        self.scheduler = get_megatron_lr_scheduler(
            self.megatron_train_args, self.megatron_train_args.max_steps, optimizer=self.optimizer
        )

        if self.megatron_train_args.use_distributed_optimizer:
            self.save_strategy = FullyParallelSaveStrategyWrapper(
                dist_checkpointing.serialization.get_default_save_sharded_strategy(),
                mpu.get_data_parallel_group(with_context_parallel=True),
                do_cache_distribution=True,
            )

        if self.megatron_train_args.overlap_grad_reduce:
            model_config = self.model.config
            assert model_config.no_sync_func is None, (
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
            model_config.no_sync_func = [model_wrapped.no_sync for model_wrapped in self.models_wrapped]
            if len(self.models_wrapped) == 1:
                model_config.no_sync_func = model_config.no_sync_func[0]
            if self.megatron_train_args.delay_grad_reduce:
                model_config.grad_sync_func = [model_wrapped.start_grad_sync for model_wrapped in self.models_wrapped]
                if len(self.models_wrapped) == 1:
                    model_config.grad_sync_func = model_config.grad_sync_func[0]

        if (self.worker_config.use_dynamic_batching_in_train or self.worker_config.use_sequence_packing or
            self.worker_config.use_sequence_packing) and self.worker.rank_info.pp_size > 1:
            self.model.config.variable_seq_lengths = True
            logger.info("Set variable_seq_lengths to True when use dynamic batching and pipeline parallel.")

        logger.info(f"{self.model.get_models()}")
        _safe_dist_barrier()

    def train_step(self, batch: DataProto, loss_func: Callable):
        self.model.train()
        logger.info(f"train_step start rank={self.worker.rank_info.rank} pp={self.worker.rank_info.pp_size}")

        global_step = batch.meta_info.get("global_step", 0)
        is_offload_optimizer_states_in_train_step = batch.meta_info.get("is_offload_optimizer_states_in_train_step", True)
        batch.meta_info['batch_num_tokens'] = self._get_batch_num_tokens(batch, dp_group=mpu.get_data_parallel_group())
        batch.meta_info['global_valid_samples'] = self._get_global_valid_samples(batch, dp_group=mpu.get_data_parallel_group())

        if self.worker_config.use_dynamic_batching_in_train:
            micro_batches_list = list(make_micro_batch_iter_for_dynamic_batching(batch))
            num_microbatches = batch.meta_info["num_micro_batchs"]
            mini_batch_size = 1
        elif self.use_sequence_packing:
            vp_size = self.worker_config.strategy_args.strategy_config['virtual_pipeline_model_parallel_size']\
                if 'virtual_pipeline_model_parallel_size' in self.worker_config.strategy_args.strategy_config else 1
            micro_batches_list = list(make_micro_batch_iter_for_sequence_packing(batch, tp_size=self.worker.rank_info.tp_size,
                                                                cp_size=self.worker.rank_info.cp_size,
                                                                vp_size=vp_size, is_train=True,
                                                                dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
                                                                micro_batch_size=self.worker_config.training_args.per_device_train_batch_size,
                                                                                 config=self.worker_config.sequence_packing_args))
            num_microbatches = micro_batches_list[0].meta_info["num_micro_batchs"]
            mini_batch_size = 1
        else:
            mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
            num_microbatches = batch.batch.batch_size[0] // self.worker_config.training_args.per_device_train_batch_size
            assert (
                num_microbatches == self.megatron_train_args.gradient_accumulation_steps
            ), f"num_microbatches={num_microbatches} gradient_accumulation_steps={self.megatron_train_args.gradient_accumulation_steps}"
            micro_batches_list = batch.chunk(chunks=num_microbatches)

        for micro_batch in micro_batches_list:
            micro_batch.meta_info['loss_scale'] = num_microbatches * mpu.get_data_parallel_world_size()
            micro_batch.meta_info['micro_batch_size'] = micro_batch.batch.batch_size[0]
        logger.info(
            f"train_step before fwd_bwd rank={self.worker.rank_info.rank} num_microbatches={num_microbatches}"
        )

        data_iterator = [iter(micro_batches_list) for _ in range(len(self.model))]

        metrics_tensors: List[Dict[str, "torch.Tensor"]] = self.forward_backward_func(
            forward_step_func=partial(self.inner_forward_step, loss_func),
            data_iterator=data_iterator,
            model=self.model.get_models(),
            num_microbatches=num_microbatches,
            seq_length=self.seq_length,
            micro_batch_size=mini_batch_size,
            forward_only=False,
        )
        logger.info(f"train_step after fwd_bwd rank={self.worker.rank_info.rank}")

        # 只有step的时候需要load optimizer states
        self.load_states(include=[OffloadStateType.optimizer_states])

        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        if is_offload_optimizer_states_in_train_step:
            self.offload_states(include=[OffloadStateType.optimizer_states], non_blocking=True)

        if update_successful:
            self.scheduler.step()
        else:
            raise NotImplementedError("megatron optimizer step failed!")

        for model in self.model:
            model.zero_grad_buffer()
            # Offload/reload does not update cached_param_buffer_shard_list/cached_grad_buffer_shard_list,
            # resulting using old params in `start_param_sync`, which leads to wrong results. So we clear the cache.
            for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
                if hasattr(bucket_group, "cached_param_buffer_shard_list"):
                    bucket_group.cached_param_buffer_shard_list = [None] * len(bucket_group.buckets)
                if hasattr(bucket_group, "cached_grad_buffer_shard_list"):
                    bucket_group.cached_grad_buffer_shard_list = [None] * len(bucket_group.buckets)
        self.optimizer.zero_grad()

        metrics = {}
        for mini_metrics in metrics_tensors:
            append_to_dict(metrics, mini_metrics)

        metrics.update({self.worker_config.name + "/" + "grad_norm": grad_norm})

        if self.model.config.num_moe_experts is not None and self.model.config.num_moe_experts > 1:
            reduce_aux_losses_tracker_across_ranks()
            tracker = get_moe_layer_wise_logging_tracker()
            loss_scale = 1 / self.megatron_train_args.gradient_accumulation_steps
            moe_losses = {
                self.worker_config.name + "/" + k: (v["values"].float() * loss_scale).mean().item()
                for k, v in tracker.items()
            }
            clear_aux_losses_tracker()
            metrics.update(moe_losses)

        if self.model.config.mtp_num_layers is not None and self.model.config.mtp_num_layers > 0:
            mtp_total_loss_dict = {}
            MTPLossLoggingHelper.reduce_loss_in_tracker()
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                loss_scale = 1 / self.megatron_train_args.gradient_accumulation_steps
                mtp_losses = tracker["values"] * loss_scale
                mtp_num_layers = mtp_losses.shape[0]
                for i in range(mtp_num_layers):
                    name = self.worker_config.name + "/" + f"mtp_{i+1} loss"
                    mtp_total_loss_dict[name] = mtp_losses[i].item()
                MTPLossLoggingHelper.clean_loss_in_tracker()
                metrics.update(mtp_total_loss_dict)

        if os.environ.get("SCHEDRL_CONTROL_PLANE", "") == "schedrl":
            checkpoint_version = int(batch.meta_info.get("checkpoint_version", global_step))
            self._build_latest_bucket_cache(checkpoint_version=checkpoint_version, global_step=int(global_step))
            # fixme(tao) it need an if test, default to false, and only promt after cache explicitly  
            # Ensure selective sync has a valid promoted cache for the next expand/broadcast.
            self.promote_active_checkpoint(checkpoint_version=checkpoint_version, global_step=int(global_step))
        return metrics

    def model_update(self, model_update_name: str, adapters_to_update: list[str] | None = None):
        # Forward optional adapter subset to weight updater for multi-LoRA selective sync.
        return self.weight_updaters[model_update_name].model_update(adapters_to_update=adapters_to_update)

    # ------------------------------------------------------------------
    # Per-adapter multi-LoRA helpers (Phase 1 port)
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        """Zero Megatron DDP grad buffers and optimizer grad state."""
        for model in self.model:
            model.zero_grad_buffer()
        self.optimizer.zero_grad()

    def forward_backward_only(self, batch: DataProto, loss_func: Callable) -> dict:
        """
        Run forward/backward to accumulate gradients but do NOT optimizer.step().

        Supports ``batch.meta_info["num_microbatches_override"]`` to bypass the
        default ``gradient_accumulation_steps`` check (needed for per-adapter
        one-microbatch-at-a-time accumulation).

        ``batch.meta_info["grad_accumulation_loss_scale"]`` (optional float) is
        applied as a pre-multiplier on the loss before backward so that several
        forward_backward_only calls can be composed into a single effective step.
        """
        self.model.train()

        if self.worker_config.use_dynamic_batching_in_train:
            raise RuntimeError("forward_backward_only does not support dynamic batching in train.")
        if batch.meta_info is None:
            batch.meta_info = {}
        batch.meta_info.setdefault(
            "batch_num_tokens", self._get_batch_num_tokens(batch, dp_group=mpu.get_data_parallel_group())
        )
        batch.meta_info.setdefault(
            "global_valid_samples", self._get_global_valid_samples(batch, dp_group=mpu.get_data_parallel_group())
        )

        mini_batch_size = self.worker_config.training_args.per_device_train_batch_size
        override = batch.meta_info.get("num_microbatches_override", None) if batch.meta_info else None
        if override is None:
            num_microbatches = batch.batch.batch_size[0] // mini_batch_size
            assert (
                num_microbatches == self.megatron_train_args.gradient_accumulation_steps
            ), (
                f"num_microbatches={num_microbatches} gradient_accumulation_steps="
                f"{self.megatron_train_args.gradient_accumulation_steps}"
            )
            micro_batches_list = batch.chunk(chunks=num_microbatches)
        else:
            num_microbatches = int(override)
            if num_microbatches <= 0:
                raise ValueError(f"num_microbatches_override must be > 0, got {override!r}")
            if num_microbatches == 1:
                micro_batches_list = [batch]
            else:
                micro_batches_list = batch.chunk(chunks=num_microbatches)

        if self.use_sequence_packing:
            mini_batch_size = 1
            self.max_packed_len = self._get_max_packed_len(micro_batches_list)

        # Optionally populate batch_num_tokens so loss_func can use it.
        for mb in micro_batches_list:
            if mb.meta_info is None:
                mb.meta_info = {}
            mb.meta_info.setdefault(
                "loss_scale", num_microbatches * mpu.get_data_parallel_world_size()
            )
            mb.meta_info.setdefault("micro_batch_size", mb.batch.batch_size[0])
            mb.meta_info.setdefault("batch_num_tokens", batch.meta_info["batch_num_tokens"])
            mb.meta_info.setdefault("global_valid_samples", batch.meta_info["global_valid_samples"])

        loss_scale = (
            batch.meta_info.get("grad_accumulation_loss_scale", None)
            if batch.meta_info
            else None
        )
        if loss_scale is not None:
            loss_scale = float(loss_scale)
            if loss_scale <= 0:
                raise ValueError(f"grad_accumulation_loss_scale must be > 0, got {loss_scale}")

            def scaled_loss_func(data: DataProto, output_tensor: torch.Tensor):
                out = loss_func(data, output_tensor)
                if not isinstance(out, tuple):
                    raise TypeError(f"loss_func must return a tuple, got {type(out)}")
                if len(out) == 2:
                    raw_loss, metrics = out
                    return raw_loss * loss_scale, metrics
                if len(out) == 3:
                    raw_loss, num_tokens, metrics = out
                    return raw_loss * loss_scale, num_tokens, metrics
                raise TypeError(
                    f"loss_func returned a {len(out)}-tuple; expected 2 or 3 elements"
                )

            effective_loss_func = scaled_loss_func
        else:
            effective_loss_func = loss_func

        data_iterator = [iter(micro_batches_list) for _ in range(len(self.model))]
        metrics_tensors: List[Dict[str, "torch.Tensor"]] = self.forward_backward_func(
            forward_step_func=partial(self.inner_forward_step, effective_loss_func),
            data_iterator=data_iterator,
            model=self.model.get_models(),
            num_microbatches=num_microbatches,
            seq_length=self.seq_length if not self.use_sequence_packing else self.max_packed_len,
            micro_batch_size=mini_batch_size,
            forward_only=False,
        )

        metrics: dict = {}
        for mini_metrics in metrics_tensors:
            append_to_dict(metrics, mini_metrics)
        return metrics

    def optimizer_step_only(
        self, *, adapter_name: str | None = None, batch_meta: dict | None = None
    ) -> dict:
        """
        Perform optimizer.step() + scheduler.step() + zero_grad assuming gradients are already
        accumulated via forward_backward_only().

        When ``adapter_name`` is provided (per_adapter mode), only that adapter's
        optimizer is stepped. Otherwise the shared optimizer is used.
        """
        if self.lora_optimizer_mode == "per_adapter" and adapter_name is None:
            raise RuntimeError(
                "optimizer_step_only requires adapter_name when lora_optimizer_mode='per_adapter'"
            )
        if self.lora_optimizer_mode == "shared" and adapter_name is not None:
            raise RuntimeError(
                "optimizer_step_only: adapter_name must be None for lora_optimizer_mode='shared'"
            )

        is_offload = True
        if batch_meta is not None:
            is_offload = bool(batch_meta.get("is_offload_optimizer_states_in_train_step", True))

        if adapter_name is not None:
            opt = self.adapter_optimizers[adapter_name]
            sch = self.adapter_schedulers[adapter_name]
        else:
            opt = self.optimizer
            sch = self.scheduler

        self.load_states(include=[OffloadStateType.optimizer_states])
        grad_norm_unclip = opt.get_grad_norm()
        update_successful, grad_norm, _num_zeros_in_grad = opt.step()
        if is_offload:
            self.offload_states(include=[OffloadStateType.optimizer_states], non_blocking=True)

        if update_successful:
            sch.step()
        else:
            raise NotImplementedError("megatron optimizer step failed!")

        for model in self.model:
            model.zero_grad_buffer()
        self.optimizer.zero_grad()

        prefix = self.worker_config.name
        name_prefix = f"{prefix}/{adapter_name}" if adapter_name else prefix
        return {
            f"{name_prefix}/grad_norm": grad_norm,
            f"{name_prefix}/grad_norm_unclip": grad_norm_unclip,
        }

    def train_step_lora(self, batch_or_microbatches: Any, loss_func: Callable) -> dict:
        """
        LoRA training step with two possible modes.

        - ``lora_optimizer_mode='shared'``: accumulate gradients across all
          microbatches then do one optimizer step (existing shared semantics).
        - ``lora_optimizer_mode='per_adapter'``: per-adapter optimizer + scheduler
          state; one optimizer step per adapter that appears in this call.
          A single call with N adapters is equivalent to N separate single-adapter
          calls — the key correctness claim of adapter isolation.

        Adapter routing requires ``non_tensor_batch["lora_name"]`` as the
        canonical key; the legacy ``domain`` fallback is removed.
        """
        if not self.is_lora:
            raise RuntimeError(
                "train_step_lora called but LoRA is not enabled for this strategy."
            )

        def _merge_metrics(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            # Keep train_step_lora metric shapes consistent with train_step: values are flat lists.
            for key, val in src.items():
                if key not in dst:
                    dst[key] = []
                if isinstance(val, list):
                    dst[key].extend(val)
                else:
                    dst[key].append(val)

        # ----------------------------------------------------------------
        # Shared mode: forward existing train_step logic via forward/backward
        # ----------------------------------------------------------------
        if self.lora_optimizer_mode == "shared":
            if isinstance(batch_or_microbatches, list):
                if len(batch_or_microbatches) == 0:
                    raise ValueError("train_step_lora(shared) received empty microbatch list")
                self.zero_grad()
                loss_scale = 1.0 / len(batch_or_microbatches)
                metrics: Dict[str, Any] = {}
                for mb in batch_or_microbatches:
                    if mb.meta_info is None:
                        mb.meta_info = {}
                    mb.meta_info.setdefault("num_microbatches_override", 1)
                    mb.meta_info.setdefault("grad_accumulation_loss_scale", loss_scale)
                    _merge_metrics(metrics, self.forward_backward_only(mb, loss_func))
                _merge_metrics(
                    metrics, self.optimizer_step_only(batch_meta=batch_or_microbatches[0].meta_info)
                )
                return metrics
            self.zero_grad()
            metrics = self.forward_backward_only(batch_or_microbatches, loss_func)
            _merge_metrics(metrics, self.optimizer_step_only(batch_meta=batch_or_microbatches.meta_info))
            return metrics

        # ----------------------------------------------------------------
        # Per-adapter mode
        # ----------------------------------------------------------------
        if self.adapter_optimizers is None or self.adapter_schedulers is None:
            raise RuntimeError(
                "train_step_lora(per_adapter) requires adapter_optimizers/adapter_schedulers "
                "to be initialized"
            )

        if isinstance(batch_or_microbatches, list):
            microbatches = batch_or_microbatches
        else:
            if self.worker_config.use_dynamic_batching_in_train:
                raise RuntimeError(
                    "train_step_lora(per_adapter) does not support dynamic batching in train."
                )
            micro_batch_size = self.worker_config.training_args.per_device_train_batch_size
            if batch_or_microbatches.batch.batch_size[0] % micro_batch_size != 0:
                raise RuntimeError(
                    f"batch_size {batch_or_microbatches.batch.batch_size[0]} must be divisible "
                    f"by micro_batch_size {micro_batch_size}"
                )
            num_microbatches = batch_or_microbatches.batch.batch_size[0] // micro_batch_size
            microbatches = batch_or_microbatches.chunk(chunks=num_microbatches)
        # Root-cause tracing: log once before per-adapter grouping/chunking.
        if not getattr(self, "_logged_lora_train_step_once", False):
            if not microbatches:
                logger.info("[device_trace][strategy/train_step_lora] microbatches=0")
            else:
                first_mb = microbatches[0]
                if first_mb.batch is not None and "input_ids" in first_mb.batch:
                    logger.info(
                        "[device_trace][strategy/train_step_lora] mb_count=%s first_input_ids_device=%s",
                        len(microbatches),
                        first_mb.batch["input_ids"].device,
                    )
            self._logged_lora_train_step_once = True

        first_meta = (
            microbatches[0].meta_info if microbatches and microbatches[0].meta_info else {}
        )
        is_offload_optimizer_states_in_train_step = bool(
            first_meta.get("is_offload_optimizer_states_in_train_step", True)
        )

        # Group microbatches by adapter (preserve encounter order for adapter ordering).
        adapters_in_order: List[str] = []
        adapter_to_mbs: Dict[str, List] = {}
        for mb in microbatches:
            if mb.non_tensor_batch:
                routing = resolve_microbatch_lora_name(mb.non_tensor_batch)
                adapter_name = routing.lora_name
            else:
                adapter_name = mb.meta_info.get("lora_name") if mb.meta_info is not None else None
                if not isinstance(adapter_name, str) or not adapter_name:
                    raise RuntimeError(
                        "Missing LoRA routing key for microbatch. "
                        "Expected non_tensor_batch['lora_name'] or meta_info['lora_name']."
                    )
            if adapter_name not in adapter_to_mbs:
                adapters_in_order.append(adapter_name)
                adapter_to_mbs[adapter_name] = []
            adapter_to_mbs[adapter_name].append(mb)

        metrics: Dict[str, Any] = {}

        # Sequential per-adapter loop (plan item 15): for each adapter, restore its RNG state,
        # run forward/backward for its microbatches, save its RNG state, then step its optimizer.
        # This guarantees RNG isolation between adapters (dropout masks are deterministic per-adapter).
        # Requires overlap_grad_reduce=False (checked at init): finalize_model_grads() does a
        # synchronous all-reduce that safely handles zero grads for idle adapters — no DDP hang.
        self.load_states(include=[OffloadStateType.optimizer_states])
        for adapter_name in adapters_in_order:
            opt = self.adapter_optimizers.get(adapter_name)
            sch = self.adapter_schedulers.get(adapter_name)
            if opt is None or sch is None:
                raise RuntimeError(f"Missing optimizer/scheduler for adapter {adapter_name!r}")

            # Restore this adapter's RNG state before forward passes.
            rng = self.adapter_rng_states[adapter_name]
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state(rng["cuda"])
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])

            # Forward/backward for this adapter's microbatches only.
            self.zero_grad()
            adapter_mbs = adapter_to_mbs[adapter_name]
            count = len(adapter_mbs)
            # Debugging aid: verify per-adapter microbatch tensor devices before forward/backward.
            if count > 0 and adapter_mbs[0].batch is not None:
                first_mb = adapter_mbs[0]
                pos_ids = first_mb.batch.get("position_ids", None)
                logger.info(
                    "[device_trace][train_step_lora/per_adapter_first_mb] rank=%s adapter=%s count=%s input_ids=%s attention_mask=%s position_ids=%s",
                    self.worker.rank_info.rank,
                    adapter_name,
                    count,
                    first_mb.batch["input_ids"].device if "input_ids" in first_mb.batch else None,
                    first_mb.batch["attention_mask"].device if "attention_mask" in first_mb.batch else None,
                    pos_ids.device if isinstance(pos_ids, torch.Tensor) else None,
                )
            logger.info(
                f"train_step_lora(per_adapter) adapter={adapter_name} microbatches={count} "
                f"pp={self.worker.rank_info.pp_size} rank={self.worker.rank_info.rank}"
            )
            if self.worker.rank_info.pp_size > 1 and count > 1:
                merged = DataProto.concat(adapter_mbs)
                if merged.meta_info is None:
                    merged.meta_info = {}
                merged.meta_info["num_microbatches_override"] = count
                merged.meta_info["grad_accumulation_loss_scale"] = 1.0 / float(count)
                _merge_metrics(metrics, self.forward_backward_only(merged, loss_func))
            else:
                for mb in adapter_mbs:
                    if mb.meta_info is None:
                        mb.meta_info = {}
                    mb.meta_info["num_microbatches_override"] = 1
                    mb.meta_info["grad_accumulation_loss_scale"] = 1.0 / float(count)
                    _merge_metrics(metrics, self.forward_backward_only(mb, loss_func))
            logger.info(
                f"train_step_lora(per_adapter) adapter={adapter_name} forward_backward_done "
                f"rank={self.worker.rank_info.rank}"
            )

            # Save this adapter's RNG state after its forward passes.
            self.adapter_rng_states[adapter_name] = {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            }

            grad_norm_unclip = opt.get_grad_norm()
            update_successful, grad_norm, _ = opt.step()
            if update_successful:
                sch.step()
            else:
                raise NotImplementedError("megatron optimizer step failed!")
            logger.info(
                f"train_step_lora(per_adapter) adapter={adapter_name} optimizer_step_done "
                f"rank={self.worker.rank_info.rank}"
            )

            # Mirror train_step (lines 1337-1341): clear bucket caches after each adapter step.
            # Offload/reload does not update cached_param_buffer_shard_list/cached_grad_buffer_shard_list;
            # stale caches cause wrong params in start_param_sync (relevant when use_distributed_optimizer=True).
            for m in self.model:
                for bucket_group in m.bucket_groups + m.expert_parallel_bucket_groups:
                    if hasattr(bucket_group, "cached_param_buffer_shard_list"):
                        bucket_group.cached_param_buffer_shard_list = [None] * len(bucket_group.buckets)
                    if hasattr(bucket_group, "cached_grad_buffer_shard_list"):
                        bucket_group.cached_grad_buffer_shard_list = [None] * len(bucket_group.buckets)

            _merge_metrics(
                metrics,
                {
                    f"{self.worker_config.name}/{adapter_name}/grad_norm": grad_norm,
                    f"{self.worker_config.name}/{adapter_name}/grad_norm_unclip": grad_norm_unclip,
                },
            )

        if is_offload_optimizer_states_in_train_step:
            self.offload_states(include=[OffloadStateType.optimizer_states], non_blocking=True)

        # Restore all adapters active (PEFT sometimes expects list of active adapters).
        active_adapters = list(self.worker_config.model_args.adapters.keys())
        for model in self.models_unwrapped:
            model.base_model.set_adapter(active_adapters)

        return metrics

    def get_lora_tensors(self, adapter_name: str) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of all LoRA parameter tensors for *adapter_name*."""
        if not self.is_lora:
            raise RuntimeError(
                "get_lora_tensors called but LoRA is not enabled for this strategy."
            )
        marker = f".{adapter_name}."
        tensors: Dict[str, torch.Tensor] = {}
        for name, param in self.models_unwrapped[0].named_parameters():
            if "lora_" not in name:
                continue
            if marker not in name:
                continue
            tensors[name] = param.detach().cpu().clone()
        if not tensors:
            raise RuntimeError(
                f"No LoRA tensors found for adapter {adapter_name!r}; check adapter naming."
            )
        return tensors

    def set_lora_tensors(
        self, *, adapter_name: str, tensors: Dict[str, torch.Tensor]
    ) -> int:
        """Overwrite the LoRA parameters for *adapter_name* with *tensors* (in-place)."""
        if not self.is_lora:
            raise RuntimeError(
                "set_lora_tensors called but LoRA is not enabled for this strategy."
            )
        marker = f".{adapter_name}."
        name_to_param = dict(self.models_unwrapped[0].named_parameters())
        copied = 0
        for name, value in tensors.items():
            if "lora_" not in name:
                continue
            if marker not in name:
                continue
            if name not in name_to_param:
                raise KeyError(
                    f"Unknown LoRA param name {name!r} when setting adapter {adapter_name!r}"
                )
            param = name_to_param[name]
            src = value.detach()
            if src.device != param.device or src.dtype != param.dtype:
                src = src.to(device=param.device, dtype=param.dtype)
            param.data.copy_(src)
            copied += 1
        copied_total = copied
        if dist.is_initialized():
            copied_total_tensor = torch.tensor([copied], dtype=torch.int64, device=current_platform.current_device())
            dist.all_reduce(copied_total_tensor, op=dist.ReduceOp.SUM)
            copied_total = int(copied_total_tensor.item())
        if copied_total == 0:
            raise RuntimeError(
                f"No LoRA tensors applied for adapter {adapter_name!r}; "
                "check naming and tensor keys."
            )

        # Megatron mixed-precision optimizers keep FP32 "main params" copies of BF16/FP16
        # model weights. Since we just mutated model params in-place, refresh the main params
        # so the next optimizer.step() starts from the updated weights.
        self.optimizer.reload_model_params()
        return copied

    def copy_lora_params(self, *, src_adapter: str, dst_adapter: str) -> int:
        """Copy LoRA parameters in-place from *src_adapter* to *dst_adapter*."""
        if not self.is_lora:
            raise RuntimeError(
                "copy_lora_params called but LoRA is not enabled for this strategy."
            )
        src_marker = f".{src_adapter}."
        dst_marker = f".{dst_adapter}."
        name_to_param = dict(self.models_unwrapped[0].named_parameters())
        copied = 0
        for name, param in name_to_param.items():
            if "lora_" not in name:
                continue
            if src_marker not in name:
                continue
            dst_name = name.replace(src_marker, dst_marker)
            if dst_name not in name_to_param:
                raise KeyError(
                    f"Expected destination param {dst_name!r} for source {name!r}"
                )
            name_to_param[dst_name].data.copy_(param.data)
            copied += 1
        if copied == 0:
            raise RuntimeError(
                "No LoRA parameters copied; check adapter naming and parameter patterns."
            )

        # Keep optimizer FP32 main params in sync with the mutated model params.
        self.optimizer.reload_model_params()
        return copied

    def _ensure_selective_sync_cpu_group(self, *, infer_tp_size: int) -> None:
        if self._selective_sync_cpu_group is not None and self._selective_sync_cpu_group_size == int(infer_tp_size):
            return

        infer_tp_size = int(infer_tp_size)
        if infer_tp_size <= 0:
            raise ValueError(f"infer_tp_size must be positive int, got {infer_tp_size}")

        world_size = dist.get_world_size()
        if world_size % infer_tp_size != 0:
            raise RuntimeError(f"train world_size={world_size} must be divisible by infer_tp_size={infer_tp_size}")

        self._selective_sync_cpu_group = None
        for start_rank in range(0, world_size, infer_tp_size):
            end_rank = start_rank + infer_tp_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._selective_sync_cpu_group = new_group

        if self._selective_sync_cpu_group is None:
            raise RuntimeError("Failed to resolve selective_sync cpu group for this rank")
        self._selective_sync_cpu_group_size = infer_tp_size

    def _build_latest_bucket_cache(
        self, *, checkpoint_version: int, global_step: int, adapter_name: Optional[str] = None
    ) -> None:
        buffer_size = int(self.worker.pipeline_config.model_update_buffer_size_mb) * 1024 * 1024
        cache_key = (int(checkpoint_version), int(global_step))

        with self._cache_lock:
            if self._selective_update_weights_meta is None:
                self._selective_update_weights_meta = gather_weights_meta_cross_pp(self.models_unwrapped)

            cached_buckets: List[Any] = []
            for hf_named_weights in gather_all_hf_weights(
                self.models_unwrapped,
                buffer_size=buffer_size,
                weights_meta=self._selective_update_weights_meta,
                adapter_name=adapter_name,
            ):
                # Important: cache must be CPU-resident and must not pickle torch Tensors.
                #
                # If we pickle torch Tensors (even CPU tensors), torch's multiprocessing reductions can create
                # resource-sharer connections with authkeys that are not consistent with vLLM v1 engine worker
                # processes, resulting in "digest sent was rejected" when applying IPC updates.
                #
                # So we serialize the flattened bucket as raw bytes + metadata only.
                cpu_named_weights = [(str(name), weight.detach().to("cpu").contiguous()) for name, weight in hf_named_weights]
                bucket, tensors_meta = _bucket_named_tensors(cpu_named_weights)  # CPU int8
                cached_buckets.append(
                    MultiprocessingSerializer.serialize(
                        {
                            "bucket_bytes": memoryview(bucket.numpy()).tobytes(),
                            "tensors_meta": tensors_meta,
                        }
                    )
                )

            if adapter_name is not None:
                self._adapter_cache_map.setdefault(adapter_name, {})[cache_key] = cached_buckets
                self._latest_adapter_cached[adapter_name] = cache_key
            else:
                self._cache_map[cache_key] = cached_buckets
                self._latest_cached = cache_key

    def promote_active_checkpoint(self, checkpoint_version: int, global_step: int) -> None:
        if os.environ.get("SCHEDRL_CONTROL_PLANE", "") != "schedrl":
            raise RuntimeError("promote_active_checkpoint is only supported under SchedRL control plane")

        cache_key = (int(checkpoint_version), int(global_step))
        with self._cache_lock:
            if cache_key not in self._cache_map:
                raise RuntimeError(f"promote_active_checkpoint missing cache_key={cache_key}")
            self._active_cached = cache_key

            keep: Set[Tuple[int, int]] = set()
            if self._latest_cached is not None:
                keep.add(self._latest_cached)
            keep.add(self._active_cached)

            for key in list(self._cache_map.keys()):
                if key not in keep:
                    del self._cache_map[key]

    def promote_active_adapter_checkpoint(
        self, adapter_name: str, checkpoint_version: int, global_step: int
    ) -> None:
        cache_key = (int(checkpoint_version), int(global_step))
        with self._cache_lock:
            if cache_key not in self._adapter_cache_map.get(adapter_name, {}):
                raise RuntimeError(
                    f"promote_active_adapter_checkpoint missing cache for adapter={adapter_name!r} key={cache_key}"
                )
            self._active_adapter_cached[adapter_name] = cache_key
            keep: Set[Tuple[int, int]] = set()
            if self._latest_adapter_cached.get(adapter_name) is not None:
                keep.add(self._latest_adapter_cached[adapter_name])
            keep.add(self._active_adapter_cached[adapter_name])
            for key in list(self._adapter_cache_map[adapter_name].keys()):
                if key not in keep:
                    del self._adapter_cache_map[adapter_name][key]

    def selective_sync_active_cache(
        self,
        *,
        sync_id: str,
        tgt_dp_ranks: List[int],
        tgt_workers,
        tgt_device_mapping: List[int],
        tgt_num_gpus_per_worker: int,
        model_update_name: Optional[str] = None,
        comm_plan: Optional[dict] = None,
        is_leader: bool = False,
        adapters_to_sync: Optional[List[str]] = None,
    ) -> None:
        if os.environ.get("SCHEDRL_CONTROL_PLANE", "") != "schedrl":
            raise RuntimeError("selective_sync_active_cache is only supported under SchedRL control plane")

        tgt_dp_ranks = sorted(set(int(r) for r in tgt_dp_ranks))
        if not tgt_dp_ranks:
            raise ValueError("tgt_dp_ranks must be non-empty")
        if not tgt_device_mapping:
            raise ValueError("tgt_device_mapping must be non-empty")
        if not isinstance(tgt_num_gpus_per_worker, int) or int(tgt_num_gpus_per_worker) <= 0:
            raise ValueError("tgt_num_gpus_per_worker must be positive int")
        if len(tgt_device_mapping) % int(tgt_num_gpus_per_worker) != 0:
            raise RuntimeError("tgt_device_mapping length must be divisible by tgt_num_gpus_per_worker")

        sync_t0 = time.perf_counter()
        logger.info(
            "[schedrl][selective_sync] enter "
            f"sync_id={sync_id} world_rank={dist.get_rank()} "
            f"tgt_dp_ranks={tgt_dp_ranks} tgt_num_gpus_per_worker={tgt_num_gpus_per_worker} "
            f"tgt_device_mapping={list(tgt_device_mapping)} "
            f"train_device_mapping={list(self.worker_config.device_mapping or [])}"
        )

        def _dp_rank_gpus(dp_rank: int) -> List[int]:
            start = int(dp_rank) * int(tgt_num_gpus_per_worker)
            end = start + int(tgt_num_gpus_per_worker)
            return [int(x) for x in tgt_device_mapping[start:end]]

        world_rank = dist.get_rank()
        adapter_names_to_register: List[str] = []
        base_cached_buckets: List[Any] = []
        adapter_cached_buckets: Dict[str, List[Any]] = {}

        with self._cache_lock:
            # Multi-LoRA under sleep_level=2 requires replaying base + adapter weights to infer workers.
            # Base model is pinned at an active cache version (typically init checkpoint -1/-1).
            # Keep base and adapter bucket streams separate so infer replay can run in phases:
            # base weights first, then per-adapter stage+register.
            if adapters_to_sync is not None:
                # Sync specified adapters using their active versions
                missing = [a for a in adapters_to_sync if self._active_adapter_cached.get(a) is None]
                if missing:
                    raise RuntimeError(f"selective_sync_active_cache: no active version for adapters {missing}")
                adapter_names_to_register = list(dict.fromkeys(str(a) for a in adapters_to_sync))
                if self._active_cached is None:
                    raise RuntimeError(
                        "selective_sync_active_cache(is_lora): active base cache is unset; "
                        "call promote_active_checkpoint first"
                    )
                if self._active_cached not in self._cache_map:
                    raise RuntimeError(f"selective_sync_active_cache: base active cache missing key={self._active_cached}")
                base_cached_buckets = list(self._cache_map[self._active_cached])
                for a in adapters_to_sync:
                    key = self._active_adapter_cached[a]
                    adapter_cached_buckets[a] = list(self._adapter_cache_map[a][key])
            elif self.is_lora:
                # adapters_to_sync=None + LoRA mode: sync ALL active adapters (expand path)
                active_entries = {a: k for a, k in self._active_adapter_cached.items() if k is not None}
                if not active_entries:
                    raise RuntimeError(
                        "selective_sync_active_cache(is_lora, adapters_to_sync=None): no active adapter caches promoted yet"
                    )
                adapter_names_to_register = list(sorted(active_entries.keys()))
                if self._active_cached is None:
                    raise RuntimeError(
                        "selective_sync_active_cache(is_lora): active base cache is unset; "
                        "call promote_active_checkpoint first"
                    )
                if self._active_cached not in self._cache_map:
                    raise RuntimeError(f"selective_sync_active_cache: base active cache missing key={self._active_cached}")
                base_cached_buckets = list(self._cache_map[self._active_cached])
                for a, key in active_entries.items():
                    adapter_cached_buckets[a] = list(self._adapter_cache_map[a][key])
            else:
                # Full fine-tune path (unchanged)
                if self._active_cached is None:
                    raise RuntimeError(
                        "selective_sync_active_cache requires an active promoted cache (active_cached is unset)"
                    )
                if self._active_cached not in self._cache_map:
                    raise RuntimeError(f"active_cached={self._active_cached} missing from cache_map")
                base_cached_buckets = list(self._cache_map[self._active_cached])
            logger.info(
                "[schedrl][selective_sync] cache "
                f"sync_id={sync_id} world_rank={world_rank} active_cached={self._active_cached} "
                f"adapters_to_sync={adapters_to_sync} base_num_buckets={len(base_cached_buckets)} "
                f"adapter_num_buckets={sum(len(v) for v in adapter_cached_buckets.values())}"
            )

            train_devices = set(int(x) for x in (self.worker_config.device_mapping or []))
            infer_devices = set(int(x) for x in tgt_device_mapping)
            is_colocated = bool(train_devices.intersection(infer_devices))

            ipc_target_dp_ranks: Set[int] = set()
            broadcast_target_dp_ranks: Set[int] = set()
            for dp_rank in tgt_dp_ranks:
                gpus = _dp_rank_gpus(dp_rank)
                if any(g in train_devices for g in gpus) and is_colocated:
                    ipc_target_dp_ranks.add(int(dp_rank))
                else:
                    broadcast_target_dp_ranks.add(int(dp_rank))

            logger.info(
                "[schedrl][selective_sync] targets "
                f"sync_id={sync_id} world_rank={world_rank} is_colocated={int(is_colocated)} "
                f"ipc_target_dp_ranks={sorted(ipc_target_dp_ranks)} "
                f"broadcast_target_dp_ranks={sorted(broadcast_target_dp_ranks)}"
            )

            # IPC path (colocated overlapped workers): reuse upstream Megatron mapping/group behavior.
            if ipc_target_dp_ranks:
                train_mapping = [int(x) for x in (self.worker_config.device_mapping or [])]
                if not train_mapping:
                    raise RuntimeError("train device_mapping is empty; cannot perform IPC selective sync")

                device_start_diff = min(train_mapping) - min(int(x) for x in tgt_device_mapping)
                device_end_diff = max(train_mapping) - max(int(x) for x in tgt_device_mapping)
                if device_start_diff % int(tgt_num_gpus_per_worker) != 0 or device_end_diff % int(tgt_num_gpus_per_worker) != 0:
                    raise RuntimeError(
                        "device_mapping diff must be divisible by tgt_num_gpus_per_worker "
                        f"({device_start_diff=}, {device_end_diff=}, {tgt_num_gpus_per_worker=})"
                    )

                self._ensure_selective_sync_cpu_group(infer_tp_size=int(tgt_num_gpus_per_worker))
                co_infer_rank = dist.get_rank(self._selective_sync_cpu_group)
                infer_parallel_size = dist.get_world_size(self._selective_sync_cpu_group)
                infer_worker_idx = (int(world_rank) + int(device_start_diff)) // int(tgt_num_gpus_per_worker)
                logger.info(
                    "[schedrl][selective_sync] ipc "
                    f"sync_id={sync_id} world_rank={world_rank} co_infer_rank={co_infer_rank} "
                    f"infer_parallel_size={infer_parallel_size} infer_worker_idx={infer_worker_idx} "
                    f"device_start_diff={device_start_diff} device_end_diff={device_end_diff}"
                )

                if 0 <= infer_worker_idx < len(tgt_workers) and infer_worker_idx in ipc_target_dp_ranks:
                    co_infer_worker = tgt_workers[infer_worker_idx]
                    # Keep gather_object calls rank-consistent by applying the same phase/bucket sequence on all ranks.
                    def _ipc_apply_bucket_sequence(
                        bucket_sequence: List[Any], *, is_lora_stage: bool, phase_tag: str, adapter_name: Optional[str] = None
                    ) -> None:
                        for bucket_idx, serialized_tensors in enumerate(bucket_sequence):
                            infer_parallel_tensors = [None] * infer_parallel_size if co_infer_rank == 0 else None
                            logger.info(
                                "[schedrl][selective_sync] ipc_gather_enter "
                                f"sync_id={sync_id} world_rank={world_rank} phase={phase_tag} "
                                f"adapter={adapter_name} bucket_idx={bucket_idx} "
                                f"serialized_len={len(serialized_tensors) if serialized_tensors is not None else 'None'}"
                            )
                            dist.gather_object(
                                serialized_tensors,
                                infer_parallel_tensors,
                                group_dst=0,
                                group=self._selective_sync_cpu_group,
                            )
                            if co_infer_rank == 0:
                                logger.info(
                                    "[schedrl][selective_sync] ipc_apply_enter "
                                    f"sync_id={sync_id} world_rank={world_rank} phase={phase_tag} "
                                    f"adapter={adapter_name} bucket_idx={bucket_idx}"
                                )
                                ray.get(
                                    co_infer_worker.update_parameter_in_bucket.remote(
                                        infer_parallel_tensors,
                                        is_lora=is_lora_stage,
                                    )
                                )
                                logger.info(
                                    "[schedrl][selective_sync] ipc_apply_exit "
                                    f"sync_id={sync_id} world_rank={world_rank} phase={phase_tag} "
                                    f"adapter={adapter_name} bucket_idx={bucket_idx}"
                                )

                    # Apply base tensors first so load_weights restores model state before adapter staging.
                    _ipc_apply_bucket_sequence(base_cached_buckets, is_lora_stage=False, phase_tag="base")
                    if self.is_lora and adapter_names_to_register:
                        peft_configs = getattr(self.models_unwrapped[0], "peft_config", None) or {}
                        missing_cfg = [a for a in adapter_names_to_register if a not in peft_configs]
                        if missing_cfg:
                            raise RuntimeError(
                                f"selective_sync_active_cache: missing peft_config for adapters {missing_cfg}"
                            )
                        # Stage one adapter at a time, then register so custom_add_lora consumes the correct tensors.
                        for adapter_name in adapter_names_to_register:
                            buckets = adapter_cached_buckets.get(adapter_name, [])
                            if not buckets:
                                raise RuntimeError(
                                    f"selective_sync_active_cache: no cached buckets for adapter={adapter_name!r}; "
                                    "promote_active_adapter_checkpoint must be called before sync"
                                )
                            _ipc_apply_bucket_sequence(
                                buckets,
                                is_lora_stage=True,
                                phase_tag="adapter",
                                adapter_name=adapter_name,
                            )
                            if co_infer_rank == 0:
                                ray.get(
                                    co_infer_worker.add_lora.remote(
                                        adapter_name=adapter_name, peft_config=asdict(peft_configs[adapter_name])
                                    )
                                )

            # Broadcast path (separated workers): ephemeral collective group managed by ModelUpdateService.
            # comm_plan=None is valid for leaders when all targets are colocated (IPC-only path):
            # ModelUpdateService intentionally passes None in that case (no NCCL group needed).
            assert comm_plan is not None or not is_leader or not broadcast_target_dp_ranks, (
                "selective_sync_active_cache: comm_plan must be provided for leader ranks that have "
                "broadcast targets. Self-setup (comm_plan is None) is no longer supported; use ModelUpdateService."
            )
            group_name = None
            broadcast_workers = None
            if broadcast_target_dp_ranks and comm_plan is not None and bool(is_leader):
                # ModelUpdateService set up the group ahead of time; retrieve group_name and receivers.
                model_update_name = str(model_update_name) if model_update_name is not None else str(sync_id)
                if int(self.worker.rank) not in comm_plan:
                    raise RuntimeError(
                        "selective_sync_active_cache comm_plan missing sender rank. "
                        f"sender_rank={int(self.worker.rank)} keys={sorted(int(k) for k in comm_plan.keys())}"
                    )
                comm_plan_args = comm_plan[int(self.worker.rank)]
                group_name = str(comm_plan_args["group_name"])
                planned_ranks = sorted({int(td["rank"]) for td in comm_plan_args.get("tgt_devices", [])})
                broadcast_workers = [tgt_workers[r] for r in planned_ranks]
                logger.info(
                    "[schedrl][selective_sync] broadcast_setup_from_comm_plan "
                    f"sync_id={sync_id} model_update_name={model_update_name} group_name={group_name} "
                    f"broadcast_dp_ranks={planned_ranks}"
                )
                # Reuse one broadcast helper for base and adapter phases to avoid diverging send/apply behavior.
                def _broadcast_apply_bucket_sequence(
                    bucket_sequence: List[Any], *, is_lora_stage: bool, phase_tag: str, adapter_name: Optional[str] = None
                ) -> None:
                    for bucket_idx, serialized_tensors in enumerate(bucket_sequence):
                        bucket_with_meta = MultiprocessingSerializer.deserialize(serialized_tensors)
                        # Cache stores bucket as raw bytes; reconstruct to sender GPU for NCCL broadcast.
                        bucket_bytes = bucket_with_meta.get("bucket_bytes")
                        tensors_meta = bucket_with_meta.get("tensors_meta")
                        if bucket_bytes is None or tensors_meta is None:
                            raise RuntimeError("selective_sync_active_cache cache missing bucket_bytes/tensors_meta")
                        bucket_cpu = torch.frombuffer(memoryview(bucket_bytes), dtype=torch.int8)
                        bucket = bucket_cpu.to(current_platform.device_type).contiguous()
                        named_params = named_tensors_from_bucket(bucket=bucket, tensors_meta=tensors_meta)

                        names = [n for n, _ in named_params]
                        dtypes = [t.dtype for _, t in named_params]
                        shapes = [t.shape for _, t in named_params]

                        logger.info(
                            "[schedrl][selective_sync] broadcast_bucket_enter "
                            f"sync_id={sync_id} group_name={group_name} phase={phase_tag} "
                            f"adapter={adapter_name} bucket_idx={bucket_idx} num_tensors={len(names)}"
                        )
                        recv_refs = [
                            worker.broadcast_parameter.remote(
                                group_name=group_name,
                                names=names,
                                dtypes=dtypes,
                                shapes=shapes,
                                is_lora=is_lora_stage,
                            )
                            for worker in broadcast_workers
                        ]

                        handles = []
                        for _, weight in named_params:
                            handles.append(
                                collective.broadcast(
                                    tensor=weight,
                                    src_rank=0,
                                    group_name=group_name,
                                    async_op=True,
                                )
                            )
                        logger.info(
                            "[schedrl][selective_sync] broadcast_wait_enter "
                            f"sync_id={sync_id} group_name={group_name} phase={phase_tag} "
                            f"adapter={adapter_name} bucket_idx={bucket_idx} num_handles={len(handles)}"
                        )
                        for handle in handles:
                            handle.wait()
                        logger.info(
                            "[schedrl][selective_sync] broadcast_wait_exit "
                            f"sync_id={sync_id} group_name={group_name} phase={phase_tag} "
                            f"adapter={adapter_name} bucket_idx={bucket_idx}"
                        )
                        logger.info(
                            "[schedrl][selective_sync] broadcast_apply_enter "
                            f"sync_id={sync_id} group_name={group_name} phase={phase_tag} "
                            f"adapter={adapter_name} bucket_idx={bucket_idx} num_workers={len(broadcast_workers)}"
                        )
                        ray.get(recv_refs)
                        logger.info(
                            "[schedrl][selective_sync] broadcast_apply_exit "
                            f"sync_id={sync_id} group_name={group_name} phase={phase_tag} "
                            f"adapter={adapter_name} bucket_idx={bucket_idx}"
                        )

                # Apply base tensors first so vLLM model weights are restored before adapter registration.
                _broadcast_apply_bucket_sequence(base_cached_buckets, is_lora_stage=False, phase_tag="base")
                if self.is_lora and adapter_names_to_register and broadcast_workers:
                    peft_configs = getattr(self.models_unwrapped[0], "peft_config", None) or {}
                    missing_cfg = [a for a in adapter_names_to_register if a not in peft_configs]
                    if missing_cfg:
                        raise RuntimeError(
                            f"selective_sync_active_cache: missing peft_config for adapters {missing_cfg}"
                        )
                    # Stage one adapter at a time, then register it so staged tensors are consumed immediately.
                    for adapter_name in adapter_names_to_register:
                        buckets = adapter_cached_buckets.get(adapter_name, [])
                        if not buckets:
                            raise RuntimeError(
                                f"selective_sync_active_cache: no cached buckets for adapter={adapter_name!r}; "
                                "promote_active_adapter_checkpoint must be called before sync"
                            )
                        _broadcast_apply_bucket_sequence(
                            buckets,
                            is_lora_stage=True,
                            phase_tag="adapter",
                            adapter_name=adapter_name,
                        )
                        ray.get(
                            [
                                worker.add_lora.remote(
                                    adapter_name=adapter_name, peft_config=asdict(peft_configs[adapter_name])
                                )
                                for worker in broadcast_workers
                            ]
                        )
                # Destroy groups before dist.barrier(): ncclCommDestroy blocks if called after barrier.
                logger.info(
                    "[schedrl][selective_sync] broadcast_teardown_enter "
                    f"sync_id={sync_id} group_name={group_name}"
                )
                collective.destroy_collective_group(group_name)
                ray.get([w.destroy_collective_group.remote(group_name) for w in broadcast_workers])
                logger.info(
                    "[schedrl][selective_sync] broadcast_teardown_exit "
                    f"sync_id={sync_id} group_name={group_name}"
                )

            # Critical: ensure all sender ranks complete this sync before allowing another to start.
            logger.info("[schedrl][selective_sync] barrier_enter " f"sync_id={sync_id} world_rank={world_rank}")
            _safe_dist_barrier()
            logger.info(
                "[schedrl][selective_sync] barrier_exit "
                f"sync_id={sync_id} world_rank={world_rank} elapsed_s={time.perf_counter() - sync_t0:.3f}"
            )

    def load_states(self, include=None, non_blocking=False):
        # Per-adapter mode must honor include semantics so SchedRL can fully release GPU memory
        # during train->infer handoff (model + optimizer states), then restore on demand.
        if getattr(self, "lora_optimizer_mode", "shared") == "per_adapter":
            include_states = []
            if include is None or OffloadStateType.model_params in include:
                # Include optimizer-managed trainable model params (e.g., active LoRA weights) in per-adapter mode.
                reload_megatron_no_grad_module(model_chunks=self.model.get_models())
                include_states.append(MegatronOffloadStateType.model_params)
            if include is None or OffloadStateType.other_params in include:
                include_states.append(MegatronOffloadStateType.other_params)
            if include is None or OffloadStateType.optimizer_states in include:
                include_states.append(MegatronOffloadStateType.optimizer_states)
            if include_states:
                self.optimizer.reload_states(include=include_states, non_blocking=non_blocking)
            return

        if include is not None:
            include_states = []
            if OffloadStateType.model_params in include:
                reload_megatron_no_grad_module(model_chunks=self.model.get_models())
                include_states.append(MegatronOffloadStateType.model_params)
            if OffloadStateType.other_params in include:
                include_states.append(MegatronOffloadStateType.other_params)
            if OffloadStateType.optimizer_states in include:
                include_states.append(MegatronOffloadStateType.optimizer_states)
            include = include_states
        self.optimizer.reload_states(include=include, non_blocking=non_blocking)

    def offload_states(self, include=None, non_blocking=False, pin_memory=True):
        # Per-adapter mode must honor include semantics so SchedRL can fully release GPU memory
        # during train->infer handoff (model + optimizer states), then restore on demand.
        if getattr(self, "lora_optimizer_mode", "shared") == "per_adapter":
            include_states = []
            if include is None or OffloadStateType.model_params in include:
                # Include optimizer-managed trainable model params (e.g., active LoRA weights) in per-adapter mode.
                offload_megatron_no_grad_module(
                    model_chunks=self.model.get_models(), pin_memory=pin_memory
                )
                include_states.append(MegatronOffloadStateType.model_params)
            if include is None or OffloadStateType.other_params in include:
                include_states.append(MegatronOffloadStateType.other_params)
            if include is None or OffloadStateType.optimizer_states in include:
                include_states.append(MegatronOffloadStateType.optimizer_states)
            if include_states:
                self.optimizer.offload_states(
                    include=include_states,
                    non_blocking=non_blocking,
                    pin_memory=pin_memory,
                )
            RotaryEmbedding.forward.cache_clear()
            current_platform.empty_cache()
            return

        if include is not None:
            include_states = []
            if OffloadStateType.model_params in include:
                offload_megatron_no_grad_module(
                    model_chunks=self.model.get_models(), pin_memory=pin_memory
                )
                include_states.append(MegatronOffloadStateType.model_params)
            if OffloadStateType.other_params in include:
                include_states.append(MegatronOffloadStateType.other_params)
            if OffloadStateType.optimizer_states in include:
                include_states.append(MegatronOffloadStateType.optimizer_states)
            include = include_states
        self.optimizer.offload_states(
            include=include, non_blocking=non_blocking, pin_memory=pin_memory
        )
        RotaryEmbedding.forward.cache_clear()
        current_platform.empty_cache()

    def setup_model_update(self, infer_cluster, model_update_name: str):
        assert model_update_name not in self.weight_updaters
        self.weight_updaters[model_update_name] = MegatronWeightUpdater(
            pipeline_config=self.worker.pipeline_config,
            worker_config=self.worker_config,
            model_update_name=model_update_name,
            models_unwrapped=self.models_unwrapped,
            infer_cluster=infer_cluster,
        )

    def save_checkpoint(self, save_dir, global_step, ckpt_id, tag="checkpoint", local_state_path=None, **kwargs):
        logger.info(f"save_dir: {save_dir}")
        if local_state_path is None:
            local_state_path = save_dir
        with Timer("load") as load_timer:
            self.load_states()

        is_last_step = kwargs.get("is_last_step", False)

        if self.megatron_train_args.save_hf_model:
            self.model.save_pretrained_as_hf(save_dir)

        # save model and tokenizer
        if len(self.models_unwrapped) == 1:
            self.models_unwrapped[0].save_pretrained(save_dir)
        else:
            state_dict = {f"model{i}": model.state_dict_for_save_checkpoint() for i, model in
                          enumerate(self.models_unwrapped)}
            self.models_unwrapped[0].save_pretrained(save_dir, state_dict=state_dict)
        if dist.get_rank() == 0:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_dir)
            if self.processor is not None:
                self.processor.save_pretrained(save_dir)

        # save optimizer
        checkpoint_dir = get_checkpoint_dir(save_dir,
                                            return_base_dir=self.megatron_train_args.use_distributed_optimizer)
        if self.megatron_train_args.use_distributed_optimizer:
            checkpoint_dir = os.path.join(checkpoint_dir, DIST_OPTIMIZER_DIR)
        os.makedirs(checkpoint_dir, exist_ok=True)
        if self.megatron_train_args.use_distributed_optimizer:
            model_shared_state_dict = self.model.sharded_state_dict()
            optimizer_state_dict = self.optimizer.sharded_state_dict(model_shared_state_dict,
                                                                     sharding_type="fully_sharded_model_space")
            dist_checkpointing.save(
                optimizer_state_dict,
                checkpoint_dir=checkpoint_dir,
                sharded_strategy=self.save_strategy,
                async_sharded_save=False,
                validate_access_integrity=self._validate_access_integrity,
            )
            self._validate_access_integrity = False
        # Compatibility: older Megatron builds do not expose get_data_modulo_expert_parallel_rank().
        elif not dist.is_initialized() or (
            mpu.get_data_modulo_expert_parallel_rank()
            if hasattr(mpu, "get_data_modulo_expert_parallel_rank")
            else mpu.get_data_parallel_rank(with_context_parallel=False)
        ) == 0:
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, OPTIMIZER_NAME))
            logger.info(f"Saving optimizer state to {os.path.join(checkpoint_dir, OPTIMIZER_NAME)}")

        if dist.is_initialized():
            _safe_dist_barrier()

        # save lr_scheduler
        if dist.get_rank() == 0:
            if self.adapter_schedulers is not None:
                scheduler_state = {
                    "mode": "per_adapter",
                    "schedulers": {k: v.state_dict() for k, v in self.adapter_schedulers.items()},
                }
            else:
                scheduler_state = self.scheduler.state_dict()
            torch.save(scheduler_state, os.path.join(save_dir, SCHEDULER_NAME))

        # save rng state
        rng_states = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": current_platform.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }
        if getattr(self, "adapter_rng_states", None) is not None:
            rng_states["adapter_rng_states"] = self.adapter_rng_states
        rgn_path = os.path.join(save_dir, RNG_STATE_DIR, f"rng_state_{dist.get_rank()}.pth")
        os.makedirs(os.path.dirname(rgn_path), exist_ok=True)
        torch.save(rng_states, rgn_path)

        if self.worker_config.checkpoint_config.get("async_upload", True) and not is_last_step:
            self.thread_executor.submit(self.checkpoint_manager.upload, ckpt_id=ckpt_id, local_state_path=local_state_path)
        else:
            self.checkpoint_manager.upload(ckpt_id=ckpt_id, local_state_path=local_state_path)

        metrics = {
            "load": load_timer.last,
        }
        return metrics

    def load_checkpoint(self, load_dir, tag="checkpoint", **kwargs):
        logger.info(f"load checkpoint from {load_dir}")

        # load optimizer
        optimizer_checkpoint = get_checkpoint_dir(
            load_dir, iteration=1, return_base_dir=self.megatron_train_args.use_distributed_optimizer
        )
        if self.megatron_train_args.use_distributed_optimizer:
            optimizer_checkpoint = os.path.join(optimizer_checkpoint, DIST_OPTIMIZER_DIR)
        logger.info(
            f"Loading optimizer from {optimizer_checkpoint}, process_index: {self.megatron_train_args.process_index}"
        )

        self.offload_states()

        if self.megatron_train_args.use_distributed_optimizer:
            model_shared_state_dict = self.model.sharded_state_dict()
            sharded_state_dict = self.optimizer.sharded_state_dict(
                model_shared_state_dict, is_loading=True, sharding_type="fully_sharded_model_space"
            )
            load_strategy = dist_checkpointing.serialization.get_default_load_sharded_strategy(optimizer_checkpoint)
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
            )
            state_dict = dist_checkpointing.load(sharded_state_dict, optimizer_checkpoint, load_strategy)
        else:
            state_dict = torch.load(
                os.path.join(optimizer_checkpoint, OPTIMIZER_NAME), map_location=self.megatron_train_args.device,
                weights_only=False
            )
        self.optimizer.load_state_dict(state_dict)

        # load lr_scheduler
        scheduler_state = torch.load(os.path.join(load_dir, SCHEDULER_NAME), weights_only=False)
        if isinstance(scheduler_state, dict) and scheduler_state.get("mode") == "per_adapter":
            if self.adapter_schedulers is None:
                raise RuntimeError(
                    "Checkpoint was saved in per_adapter scheduler mode but current strategy "
                    "has no adapter_schedulers (lora_optimizer_mode mismatch)."
                )
            for adapter_name, state in scheduler_state["schedulers"].items():
                if adapter_name not in self.adapter_schedulers:
                    raise RuntimeError(
                        f"Checkpoint contains scheduler state for adapter {adapter_name!r} "
                        "but this adapter is not registered in the current strategy."
                    )
                self.adapter_schedulers[adapter_name].load_state_dict(state)
            logger.info(f"Loaded per-adapter scheduler states for: {sorted(scheduler_state['schedulers'].keys())}")
        else:
            self.scheduler.load_state_dict(scheduler_state)

        # load model state dict
        state_dict = load_state_dict_from_checkpoint(load_dir)
        assert state_dict is not None, "No model state_dict found in checkpoint."
        self.model.models = self.models_unwrapped
        self.model.load_state_dict(state_dict)
        self.model.models = self.models_wrapped

        # load rng state
        rng_file = os.path.join(load_dir, RNG_STATE_DIR, f"rng_state_{dist.get_rank()}.pth")
        if os.path.exists(rng_file):
            logger.info(f"Loading rng states from {rng_file}")
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
            random.setstate(checkpoint_rng_state["random_rng_state"])
            np.random.set_state(checkpoint_rng_state["np_rng_state"])
            torch.set_rng_state(checkpoint_rng_state["torch_rng_state"])
            current_platform.set_rng_state(checkpoint_rng_state["cuda_rng_state"])
            # Check for empty states array
            if not checkpoint_rng_state["rng_tracker_states"]:
                raise KeyError
            tensor_parallel.get_cuda_rng_tracker().set_states(checkpoint_rng_state["rng_tracker_states"])
            if "adapter_rng_states" in checkpoint_rng_state and getattr(self, "adapter_rng_states", None) is not None:
                self.adapter_rng_states.update(checkpoint_rng_state["adapter_rng_states"])
                logger.info(f"Loaded adapter RNG states for: {sorted(checkpoint_rng_state['adapter_rng_states'].keys())}")
        else:
            logger.info(f"not load rng state, not found file: {rng_file}")

        self.load_states()
