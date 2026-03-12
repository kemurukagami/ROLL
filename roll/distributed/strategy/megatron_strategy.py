import math
import os
import pickle
import random
import threading
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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
    DO_TIME_SHARING,
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
from roll.utils.send_recv_utils import (
    _bucket_named_tensors,
    compute_weight_stats,
    monkey_patch_torch_reductions,
    named_tensors_from_bucket,
)
from roll.utils.sequence_packing import make_micro_batch_iter_for_sequence_packing, restore_results_order


if TYPE_CHECKING:
    from mcore_adapter.models.model_factory import VirtualModels

logger = get_logger()


def _safe_dist_barrier(subgroup=None):
    """Synchronize ranks at a barrier, handling two common failure modes.

    Safe to call even when ``dist`` is not initialized (single-process or
    workers that skipped dist init) — the barrier becomes a no-op in that case.

    For NCCL backend, passes ``device_ids`` explicitly to avoid a hang that
    occurs when no default CUDA device is set (NCCL requires an explicit device
    for the barrier collective; see PyTorch issue fixed after v2.9.1).

    Args:
        subgroup: Optional process-group subset (e.g. TP group, PP group).
            When None, synchronizes all ranks in the global default group.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    kwargs = {}
    if dist.get_backend() == "nccl" and current_platform.is_available():
        kwargs["device_ids"] = [current_platform.current_device()]
    if subgroup is None:
        dist.barrier(**kwargs)
    else:
        dist.barrier(group=subgroup, **kwargs)


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
        # Filter out strategy_config keys (e.g., is_lora_optimizer_isolated) that are not
        # valid TrainingArguments fields — otherwise TrainingArguments(**config_dict) raises TypeError.
        supported_keys = set(TrainingArguments.__dataclass_fields__.keys())
        dropped_keys = [k for k in config_dict if k not in supported_keys]
        if dropped_keys:
            logger.warning(f"Ignore non-TrainingArguments keys: {dropped_keys}")
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
        # True when PEFT LoRA adapters are configured; gates adapter-routing code paths.
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
        """Slice a feature tensor for this Context Parallel rank."""
        return self.models_unwrapped[0].get_batch_on_this_cp_rank({feature_name: feature}, dim3_keys=[])[feature_name]

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
        """Single micro-batch forward step called by Megatron's forward_backward_func.

        Multi-LoRA: ``set_adapter`` is called per microbatch because different
        microbatches may target different LoRA adapters.

        """
        data = next(data_iterator)
        # Multi-LoRA: activate the correct adapter for this microbatch before forward.
        if self.is_lora:
            routing = resolve_microbatch_lora_name(data.non_tensor_batch)
            for m in self.models_unwrapped:
                m.set_adapter(routing.lora_name)
        # get_data_input broadcasts batch.batch to all PP/TP/CP ranks, so tensors are always available.
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        labels = data.batch["labels"] if "labels" in data.batch else None  # labels is only used for sft
        packed_seq_params = None

        if self.use_sequence_packing:
            input_ids, packed_seq_params, cu_seqlens, cu_seqlens_padded = self._pack_sequences(
                input_ids, attention_mask,
            )
            if labels is not None:
                labels, _, _, _ = self._pack_sequences(labels, attention_mask, pad_val=IGNORE_INDEX)
            attention_mask = None
        else:
            input_ids = self._get_feature_on_this_cp_rank(input_ids, "input_ids")
            attention_mask = self._get_feature_on_this_cp_rank(attention_mask, "attention_mask")
            if labels is not None:
                labels = self._get_feature_on_this_cp_rank(labels, "labels")
        # Megatron TransformerEngine expects bool attention_mask; some pipelines produce int tensors.
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
        if "position_ids" in data.batch.keys() and data.batch["position_ids"].dim() == 3:  # qwen2vl mrope
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
                # DataProto.to('cuda') in upper frame not work for non_tensor_batch.
                forward_args[key] = torch.concat(multi_modal_data[key], dim=0).to(input_ids.device)
            forward_args.update({"force_vit_image": True})

        # megatron_llama_core need loss_mask to compute aux loss
        if "loss_mask" not in forward_args:
            if labels is not None:
                forward_args["loss_mask"] = (labels != IGNORE_INDEX).float()
            else:
                forward_args["loss_mask"] = torch.ones_like(input_ids)

        output_tensor = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels,
            packed_seq_params=packed_seq_params, **forward_args
        )

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


@dataclass
class SplitBatchResult:
    """Result of splitting a batch into microbatches for training."""

    microbatches: List[DataProto]
    num_microbatches: int
    # 1 for dynamic batching / sequence packing; per_device_train_batch_size otherwise.
    micro_batch_size: int


class MegatronTrainStrategy(MegatronInferStrategy, TrainStrategy):
    strategy_name = "megatron_train"

    def __init__(self, worker: Worker):
        super().__init__(worker)
        self.models_wrapped = None
        self.models_unwrapped = None
        self.processor = None
        self._validate_access_integrity = True

        # ---------- Versioned Bucket Cache for Selective Sync (Time-Sharing) ----------
        # Design: after each train_step, weights are gathered across PP ranks into CPU
        # buckets and stored in a versioned cache. Only one rank (pp0/dp0/tp0/cp0, the
        # "cache owner") stores the buckets; other ranks participate in the PP collective
        # but discard results. When selective_sync_active_cache is called, the cache
        # owner replays the "active" version's buckets to inference workers via CUDA IPC
        # (colocated) or NCCL broadcast (remote), avoiding a full model_update cycle.
        #
        # _latest_cached: version just built (may not yet be promoted)
        # _active_cached: version promoted for the next selective sync
        # GC policy: keep latest + active; evict everything else.
        self._cache_lock = threading.Lock()
        self._cache_map: Dict[int, List[Any]] = {}
        self._latest_cached: Optional[int] = None
        self._active_cached: Optional[int] = None
        # weights_meta is computed per-adapter inside _build_latest_bucket_cache()
        # so that metadata names match the adapter-specific state dict keys.
        # Single global cache owner: pp0/dp0/tp0/cp0 only; set during initialize().
        self._is_cache_owner: bool = False

        # Sender stats for post-sync verification, keyed by cache version.
        self._cache_stats: Dict[int, dict] = {}
        # Per-adapter versioned cache (multi-LoRA selective sync): same design as base
        # cache but keyed by adapter name, so each adapter's LoRA weights can be synced
        # independently at different versions.
        self._adapter_cache_map: Dict[str, Dict[int, List[Any]]] = {}
        self._latest_adapter_cached: Dict[str, Optional[int]] = {}
        self._active_adapter_cached: Dict[str, Optional[int]] = {}
        # Per-adapter sender stats keyed by (adapter_name, cache_key).
        self._adapter_cache_stats: Dict[tuple, dict] = {}

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

        # Capture unwrapped models before DDP replaces self.model.models.
        self.models_unwrapped = self.model.get_models()

        # LoRA detection: check both explicit adapter configs and the legacy lora_target field.
        self.is_lora = (self.worker_config.model_args.adapters is not None) or \
                       (getattr(self.worker_config.model_args, "lora_target", None) is not None)
        # Multi-adapter discriminator: True only for RLix multi-adapter LoRA configs.
        # Legacy single-LoRA (lora_target only, no adapters dict) uses train_step + shared optimizer.
        self.has_multi_adapter = self.worker_config.model_args.adapters is not None and len(self.worker_config.model_args.adapters) > 1

        # --- Config validation: reject incompatible configs before DDP wrapping ---

        # Read boolean flag; defaults to False when absent.
        self.is_lora_optimizer_isolated: bool = bool(
            self.worker_config.strategy_args.strategy_config.get("is_lora_optimizer_isolated", False)
            if self.worker_config.strategy_args and self.worker_config.strategy_args.strategy_config
            else False
        )
        # Multi-adapter requires isolated optimizers — one per adapter.
        if self.has_multi_adapter and not self.is_lora_optimizer_isolated:
            raise ValueError(
                "model_args.adapters is configured but is_lora_optimizer_isolated is not set. "
                "Set strategy_config.is_lora_optimizer_isolated=true."
            )

        if self.is_lora_optimizer_isolated:
            if self.megatron_train_args.use_distributed_optimizer:
                raise ValueError(
                    "Isolated multi-adapter LoRA requires use_distributed_optimizer=False. "
                    "Distributed optimizer shards state across ranks, which conflicts "
                    "with per-adapter optimizer isolation."
                )
            if self.megatron_train_args.overlap_grad_reduce:
                raise ValueError(
                    "Isolated multi-adapter LoRA requires overlap_grad_reduce=False. "
                    "With overlap_grad_reduce=True, idle adapters' DDP backward hooks "
                    "never fire during another adapter's sequential pass, causing a "
                    "hang in finish_grad_sync()."
                )
            if getattr(self.worker_config.model_args, "model_type", None) == "trl":
                raise ValueError(
                    "Isolated multi-adapter LoRA does not support TRL value-head models "
                    "(model_type='trl'). Disable value head."
                )

        # --- Model-structure validation: needs instantiated model, not DDP ---
        # When is_lora_optimizer_isolated=True, each adapter has its own optimizer.
        # This requires every trainable parameter to belong to exactly one adapter.
        # Shared trainable parameters (e.g., a value head not scoped to any adapter)
        # would receive gradient updates from multiple optimizers, corrupting state.
        #
        # Example of VALID param names (adapter-scoped):
        #   "layers.0.self_attn.q_proj.lora_A.adapter_A.weight"
        #   "layers.0.self_attn.q_proj.lora_B.adapter_B.weight"
        #
        # Example of INVALID shared trainable (would cause error):
        #   "v_head.weight"  # not scoped to any adapter → shared across optimizers
        if self.is_lora_optimizer_isolated:
            adapter_names = list(self.worker_config.model_args.adapters.keys())
            if not adapter_names:
                raise ValueError(
                    "Multi-adapter LoRA requires at least one adapter in model_args.adapters"
                )

            # Activate all adapters so their LoRA params are marked trainable for inspection.
            for model in self.models_unwrapped:
                base_model = getattr(model, "base_model", None)
                if base_model is not None and hasattr(base_model, "set_adapter"):
                    base_model.set_adapter(adapter_names)

            # Aggregate params from all chunks with a chunk-index prefix so names are unique.
            # Virtual-pipeline chunks each hold different layers; the same local name (e.g.
            # "layers.0.weight") can appear in multiple chunks, so the prefix is required.
            name_to_param: Dict[str, torch.nn.Parameter] = {}
            for chunk_idx, chunk_model in enumerate(self.models_unwrapped):
                for param_name, param in chunk_model.named_parameters():
                    name_to_param[f"chunk{chunk_idx}.{param_name}"] = param

            original_requires_grad: Dict[str, bool] = {
                n: bool(p.requires_grad) for n, p in name_to_param.items()
            }

            # Build adapter markers for name-matching. Example: {adapter_A: ".adapter_A.", ...}
            markers = {adapter_name: f".{adapter_name}." for adapter_name in adapter_names}

            # Find shared trainables: params that are trainable but not scoped to any adapter.
            # A param is adapter-scoped if its name contains one of the markers (e.g., ".adapter_A.")
            shared_trainables: List[str] = []
            for name, param in name_to_param.items():
                if not original_requires_grad[name]:
                    # Skip frozen params — they don't participate in optimizer updates.
                    continue
                if not any(marker in name for marker in markers.values()):
                    # Trainable but not adapter-scoped → shared across all adapters.
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
                    "Multi-adapter LoRA requires all trainable parameters to be "
                    f"adapter-scoped (name must include one of: {sorted(markers.values())}). "
                    f"Found shared trainables (first 10): {preview}. "
                    "Freeze these parameters to use per-adapter optimizer mode."
                    + hint
                )

        # --- DDP wrapping: all config and model-structure checks passed ---
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
            for model_index, m in enumerate(self.models_unwrapped)
        ]
        self.model.models = self.models_wrapped

        params_dtype = (
            torch.float16
            if self.megatron_train_args.fp16
            else torch.bfloat16 if self.megatron_train_args.bf16 else torch.float32
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
            use_distributed_optimizer=self.megatron_train_args.use_distributed_optimizer,
            clip_grad=self.megatron_train_args.max_grad_norm,
        )

        self.adapter_optimizers: Dict[str, MegatronOptimizer] | None = None
        self.adapter_schedulers: Dict[str, Any] | None = None

        if not self.has_multi_adapter:
            # Non-LoRA or legacy single-LoRA: single optimizer (upstream v0.2.0 path).
            self.optimizer: MegatronOptimizer = get_megatron_optimizer(optimizer_config, self.models_wrapped)
            logger.info(f"megatron optimizer: {self.optimizer}")
            bind_megatron_offload_states_func(optimizer=self.optimizer)
        else:
            # ---- Isolated mode: one optimizer + scheduler per adapter ----
            # adapter_names, name_to_param, original_requires_grad, markers already
            # computed during model-structure validation above.

            def _apply_trainability_mask_for_adapter(active_adapter: str) -> None:
                """Freeze all params except this adapter's LoRA weights.

                Used before ``get_megatron_optimizer`` so the optimizer only captures
                parameters that belong to ``active_adapter``. The trainability mask
                is restored after all per-adapter optimizers are constructed.
                """
                marker = markers[active_adapter]
                for n, p in name_to_param.items():
                    p.requires_grad_(bool(original_requires_grad[n] and (marker in n)))

            self.adapter_optimizers = {}
            self.adapter_schedulers = {}
            param_id_to_name = {id(p): n for n, p in name_to_param.items()}
            seen_param_ids: Set[int] = set()
            for adapter_name in adapter_names:
                # Activate the current adapter on every chunk so PEFT routes forward
                # correctly; chunk 0 alone is not sufficient for virtual-pipeline models.
                for chunk_model in self.models_unwrapped:
                    chunk_model.set_adapter(adapter_name)
                _apply_trainability_mask_for_adapter(adapter_name)
                adapter_opt = get_megatron_optimizer(optimizer_config, self.models_wrapped)
                # bind_megatron_offload_states_func is deferred to the ChainedOptimizer
                # call below (line ~1306), which recursively binds all sub-optimizers.

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

            # ChainedOptimizer wraps all per-adapter optimizers so that generic
            # offload/reload/state_dict calls (which expect a single self.optimizer)
            # fan out to every adapter optimizer transparently.
            # Tradeoff: all-or-nothing handling means all adapters are reloaded/offloaded together,
            # even when train_step_lora() only trains one adapter at a time.
            # fixme(tao) HACK can we do lora granular swap of optimizer?
            # Each sub-optimizer already has reload_states/offload_states bound by
            # bind_megatron_offload_states_func, so adapter_optimizers[adapter_name].reload_states()
            # would work mechanically — but train_step_lora still calls self.load_states()/
            # self.offload_states() which go through ChainedOptimizer and swap all adapters.
            from megatron.core.optimizer import ChainedOptimizer
            self.optimizer = ChainedOptimizer(list(self.adapter_optimizers.values()))
            bind_megatron_offload_states_func(optimizer=self.optimizer)

            # Initialize per-adapter RNG states for sequential training (plan item 15).
            # Each adapter starts from the current global RNG state; they diverge as training progresses.
            # Includes Megatron TP CUDA RNG tracker for deterministic TP-parallel dropout per adapter.
            self.adapter_rng_states: Dict[str, Dict[str, Any]] = {
                name: {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(),
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
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

        # Single global cache owner: the unique rank with all parallel dimensions at 0.
        self._is_cache_owner = (
            mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_data_parallel_rank() == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
        )

        logger.info(f"max steps pipeline {self.worker_config.training_args.max_steps}")
        self.worker_config.training_args.max_steps = (
            self.worker_config.training_args.max_steps // self.worker.rank_info.dp_size
        )
        self.megatron_train_args.max_steps = self.worker_config.training_args.max_steps
        logger.info(f"max steps worker train {self.worker_config.training_args.max_steps}")

        # Per-adapter schedulers must use DP-adjusted max_steps. They were initially
        # created before dp_size was known, so rebuild here with the final step budget.
        if self.has_multi_adapter and self.adapter_optimizers:
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

        global_step = batch.meta_info.get("global_step", 0)
        is_offload_optimizer_states_in_train_step = batch.meta_info.get(
            "is_offload_optimizer_states_in_train_step", True
        )

        # Shared: populate batch-level metadata.
        self._ensure_train_batch_meta(batch)

        # Shared: split batch into microbatches.
        split = self._split_batch_to_microbatches(batch)

        # Shared: stamp loss_scale, micro_batch_size, batch_num_tokens, global_valid_samples.
        self._annotate_microbatches_for_train(
            split.microbatches, split.num_microbatches, batch.meta_info
        )

        # Shared: forward/backward passes.
        # train_step always uses self.seq_length, even for sequence packing (current RLIX behavior).
        metrics = self._run_forward_backward(
            microbatches=split.microbatches,
            loss_func=loss_func,
            num_microbatches=split.num_microbatches,
            micro_batch_size=split.micro_batch_size,
            seq_length=self.seq_length,
        )

        # 只有step的时候需要load optimizer states
        self.load_states(include=[OffloadStateType.optimizer_states])

        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        if is_offload_optimizer_states_in_train_step:
            self.offload_states(include=[OffloadStateType.optimizer_states], non_blocking=True)

        if update_successful:
            self.scheduler.step()
        else:
            raise NotImplementedError("megatron optimizer step failed!")

        # Shared: zero grad buffers and optimizer state, then clear stale bucket caches.
        self._zero_grad()
        self._clear_bucket_caches()

        metrics.update({self.worker_config.name + "/" + "grad_norm": grad_norm})
        self._collect_auxiliary_loss_metrics(metrics)

        # Time-sharing: build a versioned bucket cache of the current weights.
        # Promotion is NOT done here — the RLix pipeline calls
        # promote_active_checkpoint explicitly after train_step to control which
        # version is broadcast via selective_sync_active_cache.
        if DO_TIME_SHARING:
            checkpoint_version = int(batch.meta_info["checkpoint_version"])
            self._build_latest_bucket_cache(checkpoint_version=checkpoint_version)
        return metrics


    # ------------------------------------------------------------------
    # Shared helpers extracted from train_step (Changes 2-6)
    # ------------------------------------------------------------------
    def _zero_grad(self) -> None:
        """Zero Megatron DDP grad buffers and optimizer grad state."""
        for model in self.model:
            model.zero_grad_buffer()
        self.optimizer.zero_grad()

    def _ensure_train_batch_meta(self, batch: DataProto) -> None:
        """Populate batch_num_tokens and global_valid_samples on batch.meta_info.

        Uses direct assignment matching train_step baseline.
        DataProto.chunk()/make_iterator() share the same meta_info dict reference
        across microbatches, so setdefault would preserve stale values from a
        previous mini-batch iteration.
        """
        if batch.meta_info is None:
            batch.meta_info = {}
        batch.meta_info['batch_num_tokens'] = self._get_batch_num_tokens(
            batch, dp_group=mpu.get_data_parallel_group()
        )
        batch.meta_info['global_valid_samples'] = self._get_global_valid_samples(
            batch, dp_group=mpu.get_data_parallel_group()
        )

    def _split_batch_to_microbatches(
        self,
        batch: DataProto,
    ) -> SplitBatchResult:
        """Split a DataProto batch into microbatches for training.

        Three splitting strategies, selected by worker config:
        - Dynamic batching: variable-length microbatches via make_micro_batch_iter_for_dynamic_batching.
        - Sequence packing: load-balanced packed partitions via make_micro_batch_iter_for_sequence_packing.
        - Standard: equal-size chunks by per_device_train_batch_size, with
          num_microbatches == gradient_accumulation_steps assertion.
        """
        if self.worker_config.use_dynamic_batching_in_train:
            # Fail fast if upstream caller did not run dynamic_batching_shard() to prepare
            # required batch metadata. See dynamic_batching.py:118.
            if not batch.meta_info or "micro_batch_indices" not in batch.meta_info:
                raise RuntimeError(
                    "use_dynamic_batching_in_train requires batch metadata from "
                    "dynamic_batching_shard(). Ensure the pipeline calls "
                    "dynamic_batching_shard() before train_step/train_step_lora."
                )
            microbatches = list(make_micro_batch_iter_for_dynamic_batching(batch))
            num_microbatches = batch.meta_info["num_micro_batchs"]
            return SplitBatchResult(
                microbatches=microbatches,
                num_microbatches=num_microbatches,
                micro_batch_size=1,
            )

        if self.use_sequence_packing:
            vp_size = self.worker_config.strategy_args.strategy_config.get(
                "virtual_pipeline_model_parallel_size", 1
            )
            microbatches = list(
                make_micro_batch_iter_for_sequence_packing(
                    batch,
                    tp_size=self.worker.rank_info.tp_size,
                    cp_size=self.worker.rank_info.cp_size,
                    vp_size=vp_size,
                    is_train=True,
                    dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
                    micro_batch_size=self.worker_config.training_args.per_device_train_batch_size,
                    config=self.worker_config.sequence_packing_args,
                )
            )
            num_microbatches = microbatches[0].meta_info["num_micro_batchs"]
            return SplitBatchResult(
                microbatches=microbatches,
                num_microbatches=num_microbatches,
                micro_batch_size=1,
            )

        # Standard path: equal chunks by per_device_train_batch_size.
        per_device_batch_size = self.worker_config.training_args.per_device_train_batch_size
        total_batch_size = batch.batch.batch_size[0]
        num_microbatches = total_batch_size // per_device_batch_size
        assert num_microbatches == self.megatron_train_args.gradient_accumulation_steps, (
            f"num_microbatches={num_microbatches} "
            f"gradient_accumulation_steps={self.megatron_train_args.gradient_accumulation_steps}"
        )
        microbatches = batch.chunk(chunks=num_microbatches)
        return SplitBatchResult(
            microbatches=microbatches,
            num_microbatches=num_microbatches,
            micro_batch_size=per_device_batch_size,
        )

    def _annotate_microbatches_for_train(
        self,
        microbatches: List[DataProto],
        num_microbatches: int,
        batch_meta: Dict[str, Any],
    ) -> None:
        """Stamp loss_scale, micro_batch_size, and batch-level metadata on each microbatch.

        loss_scale = num_microbatches * dp_world_size. This is the standard train_step
        convention — inner_forward_step multiplies loss by this value to normalize
        gradient accumulation across microbatches and data parallel ranks.
        """
        for micro_batch in microbatches:
            if micro_batch.meta_info is None:
                micro_batch.meta_info = {}
            # Direct assignment for loss_scale and micro_batch_size, matching train_step
            # baseline. These are always set fresh by the training step.
            micro_batch.meta_info['loss_scale'] = num_microbatches * mpu.get_data_parallel_world_size()
            micro_batch.meta_info['micro_batch_size'] = micro_batch.batch.batch_size[0]
            # setdefault for batch-level metadata that may already be populated.
            micro_batch.meta_info.setdefault("batch_num_tokens", batch_meta.get("batch_num_tokens"))
            micro_batch.meta_info.setdefault("global_valid_samples", batch_meta.get("global_valid_samples"))

    def _run_forward_backward(
        self,
        microbatches: List[DataProto],
        loss_func: Callable,
        num_microbatches: int,
        micro_batch_size: int,
        seq_length: int,
    ) -> Dict[str, Any]:
        """Run forward/backward passes on explicit microbatch list. Does NOT step optimizer.

        Builds the data_iterator from the provided microbatch list and calls
        forward_backward_func. Does not re-split — the microbatch list is used as-is,
        preserving packed partition boundaries for sequence packing.

        Loss scaling is handled by _annotate_microbatches_for_train which stamps
        loss_scale = num_microbatches * dp_world_size on each microbatch. The
        inner_forward_step loss_wrapper applies this scale.
        """
        data_iterator = [iter(microbatches) for _ in range(len(self.model))]
        metrics_tensors: List[Dict[str, "torch.Tensor"]] = self.forward_backward_func(
            forward_step_func=partial(self.inner_forward_step, loss_func),
            data_iterator=data_iterator,
            model=self.model.get_models(),
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )

        metrics: Dict[str, Any] = {}
        for mini_metrics in metrics_tensors:
            append_to_dict(metrics, mini_metrics)
        return metrics

    def _collect_auxiliary_loss_metrics(self, metrics: Dict[str, Any]) -> None:
        """Collect MoE and MTP auxiliary loss metrics after a training step.

        Called by both train_step and train_step_lora to ensure auxiliary losses
        are always reported regardless of training path.
        """
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
            mtp_total_loss_dict: Dict[str, Any] = {}
            MTPLossLoggingHelper.reduce_loss_in_tracker()
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                loss_scale = 1 / self.megatron_train_args.gradient_accumulation_steps
                mtp_losses = tracker["values"] * loss_scale
                mtp_num_layers = mtp_losses.shape[0]
                for layer_idx in range(mtp_num_layers):
                    name = self.worker_config.name + "/" + f"mtp_{layer_idx+1} loss"
                    mtp_total_loss_dict[name] = mtp_losses[layer_idx].item()
                MTPLossLoggingHelper.clean_loss_in_tracker()
                metrics.update(mtp_total_loss_dict)

    def _clear_bucket_caches(self) -> None:
        """Clear cached param/grad buffer shard lists after optimizer step.

        Offload/reload does not update these caches, so stale params in
        start_param_sync would lead to wrong results.
        """
        for model in self.model:
            for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
                if hasattr(bucket_group, "cached_param_buffer_shard_list"):
                    bucket_group.cached_param_buffer_shard_list = [None] * len(bucket_group.buckets)
                if hasattr(bucket_group, "cached_grad_buffer_shard_list"):
                    bucket_group.cached_grad_buffer_shard_list = [None] * len(bucket_group.buckets)

    def train_step_lora(self, batch: DataProto, loss_func: Callable) -> dict:
        """Single-adapter-per-call LoRA training step.

        Callers guarantee exactly one adapter per call. The adapter's per-adapter
        optimizer and scheduler are stepped independently.
        """
        self.model.train()

        if not self.is_lora_optimizer_isolated:
            raise RuntimeError(
                "train_step_lora requires model_args.adapters. "
                "Legacy (lora_target only) should use train_step."
            )

        if self.adapter_optimizers is None or self.adapter_schedulers is None:
            raise RuntimeError(
                "train_step_lora requires adapter_optimizers/adapter_schedulers "
                "to be initialized"
            )

        # Shared: populate batch-level metadata.
        self._ensure_train_batch_meta(batch)

        # Shared: split batch into microbatches (same contract as train_step).
        split = self._split_batch_to_microbatches(batch)
        microbatches = split.microbatches

        # Shared: stamp loss_scale, micro_batch_size, batch_num_tokens, global_valid_samples.
        # loss_scale = num_microbatches * dp_world_size, matching train_step semantics.
        self._annotate_microbatches_for_train(
            microbatches, split.num_microbatches, batch.meta_info
        )

        # LoRA-specific: resolve adapter name from non_tensor_batch.
        # resolve_microbatch_lora_name validates homogeneity within each microbatch.
        # All callers set lora_name via non_tensor_batch (pipeline routing).
        adapter_name = resolve_microbatch_lora_name(microbatches[0].non_tensor_batch).lora_name
        # Validate all microbatches target the same adapter (single-adapter-per-call contract).
        for mb_idx, mb in enumerate(microbatches[1:], start=1):
            mb_adapter = resolve_microbatch_lora_name(mb.non_tensor_batch).lora_name
            if mb_adapter != adapter_name:
                raise ValueError(
                    f"train_step_lora expects single adapter per call, but microbatch[{mb_idx}] "
                    f"has adapter={mb_adapter!r}, expected {adapter_name!r}"
                )

        is_offload_optimizer_states_in_train_step = bool(
            batch.meta_info.get("is_offload_optimizer_states_in_train_step", True)
        )

        opt = self.adapter_optimizers.get(adapter_name)
        sch = self.adapter_schedulers.get(adapter_name)
        if opt is None or sch is None:
            raise RuntimeError(f"Missing optimizer/scheduler for adapter {adapter_name!r}")

        # LoRA-specific: restore adapter RNG state (including TP CUDA RNG tracker for dropout).
        self.load_states(include=[OffloadStateType.optimizer_states])
        rng = self.adapter_rng_states[adapter_name]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        tensor_parallel.get_cuda_rng_tracker().set_states(rng["rng_tracker_states"])

        # Shared: forward/backward passes (same call signature as train_step).
        metrics = self._run_forward_backward(
            microbatches=microbatches,
            loss_func=loss_func,
            num_microbatches=split.num_microbatches,
            micro_batch_size=split.micro_batch_size,
            seq_length=self.seq_length,
        )

        # LoRA-specific: save adapter RNG state (including TP CUDA RNG tracker for dropout).
        self.adapter_rng_states[adapter_name] = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }

        # LoRA-specific: per-adapter optimizer step.
        update_successful, grad_norm, _ = opt.step()
        if update_successful:
            sch.step()
        else:
            raise NotImplementedError("megatron optimizer step failed!")

        # Shared: zero grad buffers and optimizer state, then clear stale bucket caches.
        self._zero_grad()
        self._clear_bucket_caches()

        # Time-sharing: build per-adapter bucket cache while GPU weights are still resident.
        # Must run before offload_states moves weights to CPU.
        # Promotion is NOT done here — the pipeline calls promote methods explicitly.
        if DO_TIME_SHARING:
            checkpoint_version = int(batch.meta_info["checkpoint_version"])
            self._build_latest_bucket_cache(
                checkpoint_version=checkpoint_version,
                adapter_name=adapter_name,
            )

        metrics.update({
            f"{self.worker_config.name}/{adapter_name}/grad_norm": grad_norm,
        })
        self._collect_auxiliary_loss_metrics(metrics)

        if is_offload_optimizer_states_in_train_step:
            self.offload_states(include=[OffloadStateType.optimizer_states], non_blocking=True)

        # Restore all adapters active (PEFT sometimes expects list of active adapters).
        active_adapters = list(self.worker_config.model_args.adapters.keys())
        for model in self.models_unwrapped:
            model.base_model.set_adapter(active_adapters)

        return metrics

    def model_update(self, model_update_name: str, adapters_to_update: list[str] | None = None):
        # Forward optional adapter subset to weight updater for multi-LoRA selective sync.
        return self.weight_updaters[model_update_name].model_update(adapters_to_update=adapters_to_update)


    def get_lora_tensors(self, adapter_name: str) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of all LoRA parameter tensors for *adapter_name*.

        Reads parameters from models_unwrapped[0] (TP/DP ranks share identical
        LoRA weights, so rank 0 is sufficient).

        Note: used only by integration tests for weight inspection and snapshot
        comparison. Not called in any production pipeline.
        """
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
        """Overwrite the LoRA parameters for *adapter_name* with *tensors* (in-place).

        Also refreshes the optimizer's FP32 main-param copies via
        ``optimizer.reload_model_params()`` so the next step starts from the
        updated weights, not stale copies.

        Note: used only by integration tests to reset adapter weights to a known
        state before a reference run. Not called in any production pipeline.
        """
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

        # Sync BF16 model params → FP32 main params.
        # Megatron's mixed-precision optimizers keep a separate FP32 "main params" copy of
        # BF16/FP16 model weights and use it as the authoritative source in optimizer.step().
        # We just mutated the BF16 side directly (bypassing the optimizer), so push those
        # changes into FP32 now — otherwise the next step() would overwrite our writes.
        self.optimizer.reload_model_params()
        return copied

    def copy_lora_params(self, *, src_adapter: str, dst_adapter: str) -> int:
        """Copy LoRA parameters in-place from *src_adapter* to *dst_adapter*.

        Matches source parameter names to destination names by substituting the
        adapter marker (``.<src_adapter>.`` → ``.<dst_adapter>.``) and raises
        ``KeyError`` if the expected destination parameter does not exist.

        Note: used only by integration tests to synchronize all adapters to the
        same initial weights. Not called in any production pipeline.
        """
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

        # Sync BF16 model params → FP32 main params (same reason as set_lora_tensors).
        self.optimizer.reload_model_params()
        return copied

    def _build_latest_bucket_cache(
        self, *, checkpoint_version: int, adapter_name: Optional[str] = None
    ) -> None:
        """Gather current model weights across PP ranks and store as CPU buckets.

        All PP ranks must participate in ``gather_all_hf_weights`` (it uses PP
        collectives internally). Only the cache owner (pp0/dp0/tp0/cp0) stores
        the resulting buckets; non-owners drain the generator to keep the
        collective moving but discard results.

        When ``adapter_name`` is given, only that adapter's LoRA weights are
        cached (stored in ``_adapter_cache_map``); otherwise base weights are
        cached in ``_cache_map``.
        """
        buffer_size = int(self.worker.pipeline_config.model_update_buffer_size_mb) * 1024 * 1024
        cache_key = int(checkpoint_version)

        with self._cache_lock:
            # Compute weights_meta with the actual adapter_name so metadata names match
            # the state dict keys used by gather_all_hf_weights (base vs LoRA names).
            weights_meta = gather_weights_meta_cross_pp(self.models_unwrapped, adapter_name=adapter_name)

            # All PP ranks must participate in gather_all_hf_weights (PP collective).
            # Only the cache owner stores results; non-owners drain and discard each batch.
            cached_buckets: List[Any] = []
            # Accumulate sender stats from globally-gathered weights for verification.
            # Gated by config flag to skip stats computation when verification is disabled.
            compute_stats = self.worker.pipeline_config.verify_model_after_sync
            running_sum = 0.0
            running_max = float("-inf")
            running_min = float("inf")
            batch_count = 0
            for hf_named_weights in gather_all_hf_weights(
                self.models_unwrapped,
                buffer_size=buffer_size,
                weights_meta=weights_meta,
                adapter_name=adapter_name,
            ):
                if not self._is_cache_owner:
                    # Non-owner must consume the generator element to keep the PP collective moving,
                    # but does not store anything.
                    continue
                # Compute sender stats on GPU tensors before CPU copy (GPU reductions are
                # ~20-40x faster than CPU for large models).
                if compute_stats:
                    batch_stats = compute_weight_stats(hf_named_weights)
                    if batch_stats:
                        running_sum += batch_stats["sum"]
                        running_max = max(running_max, batch_stats["max"])
                        running_min = min(running_min, batch_stats["min"])
                        batch_count += 1
                # Cache as raw CPU tensors. GPU staging + serialization happens at transport
                # time because IPC handles are ephemeral (tied to specific GPU allocations).
                cpu_named_weights = [
                    (str(name), weight.detach().to("cpu").contiguous())
                    for name, weight in hf_named_weights
                ]

                bucket, tensors_meta = _bucket_named_tensors(cpu_named_weights)  # CPU int8
                cached_buckets.append((tensors_meta, bucket))

            if not self._is_cache_owner:
                return

            # Store sender stats alongside cached buckets for later verification.
            sender_stats = {}
            if batch_count > 0:
                sender_stats = {"sum": running_sum, "max": running_max, "min": running_min}

            if adapter_name is not None:
                self._adapter_cache_map.setdefault(adapter_name, {})[cache_key] = cached_buckets
                self._latest_adapter_cached[adapter_name] = cache_key
                # Store per-adapter stats keyed by (adapter_name, cache_key).
                self._adapter_cache_stats[(adapter_name, cache_key)] = sender_stats
            else:
                self._cache_map[cache_key] = cached_buckets
                self._latest_cached = cache_key
                self._cache_stats[cache_key] = sender_stats

    def promote_active_checkpoint(self, checkpoint_version: int) -> None:
        """Mark a cached version as the "active" snapshot for selective sync.

        The distinction between "latest" and "active" allows a new cache to be
        built concurrently while selective_sync_active_cache reads the previous
        active version. After promotion, all versions except latest and active
        are garbage-collected.
        """
        if not DO_TIME_SHARING:
            raise RuntimeError("promote_active_checkpoint is only supported under RLix control plane")
        # Non-owners hold no cache, so there is nothing to promote.
        if not self._is_cache_owner:
            return

        cache_key = int(checkpoint_version)
        with self._cache_lock:
            if cache_key not in self._cache_map:
                raise RuntimeError(f"promote_active_checkpoint missing cache_key={cache_key}")
            self._active_cached = cache_key

            keep: Set[int] = set()
            if self._latest_cached is not None:
                keep.add(self._latest_cached)
            keep.add(self._active_cached)

            for key in list(self._cache_map.keys()):
                if key not in keep:
                    del self._cache_map[key]

    def promote_active_adapter_checkpoint(
        self, adapter_name: str, checkpoint_version: int
    ) -> None:
        """Same as ``promote_active_checkpoint`` but for a single adapter's LoRA cache."""
        if not DO_TIME_SHARING:
            raise RuntimeError("promote_active_adapter_checkpoint is only supported under RLix control plane")
        # Non-owners hold no cache, so there is nothing to promote.
        if not self._is_cache_owner:
            return
        cache_key = int(checkpoint_version)
        with self._cache_lock:
            if cache_key not in self._adapter_cache_map.get(adapter_name, {}):
                raise RuntimeError(
                    f"promote_active_adapter_checkpoint missing cache for adapter={adapter_name!r} key={cache_key}"
                )
            self._active_adapter_cached[adapter_name] = cache_key
            keep: Set[int] = set()
            if self._latest_adapter_cached.get(adapter_name) is not None:
                keep.add(self._latest_adapter_cached[adapter_name])
            keep.add(self._active_adapter_cached[adapter_name])
            for key in list(self._adapter_cache_map[adapter_name].keys()):
                if key not in keep:
                    del self._adapter_cache_map[adapter_name][key]

    def selective_sync_active_cache(
        self,
        *,
        tgt_dp_ranks: List[int],
        tgt_workers,
        tgt_device_mapping: List[int],
        tgt_num_gpus_per_worker: int,
        comm_plan: Optional[dict] = None,
        adapters_to_sync: Optional[List[str]] = None,
    ) -> None:
        """Replay the active bucket cache to inference workers (time-sharing).

        Transport flow (executed only by the single cache-owner rank):
        1. **Cache lookup**: read the promoted "active" version from ``_cache_map``
           (base weights) and optionally ``_adapter_cache_map`` (per-adapter LoRA).
        2. **Decode comm_plan**: the ``ModelUpdateService`` builds a per-rank plan
           specifying IPC targets (colocated workers) and NCCL broadcast targets.
        3. **Transport**: for each cached bucket, stage to GPU once, then:
           - IPC path: serialize the GPU tensor via CUDA IPC and push to colocated workers.
           - Broadcast path: NCCL broadcast to remote workers.
        4. **LoRA registration**: after adapter weights are transported, call
           ``add_lora`` on each target worker to register the adapter with its PEFT config.
        5. **Group teardown**: destroy the temporary NCCL broadcast group.

        Non-owner ranks return immediately; ``ray.get(sync_refs)`` in
        ``ModelUpdateService`` provides the cross-worker sync barrier.
        """
        if not DO_TIME_SHARING:
            raise RuntimeError("selective_sync_active_cache is only supported under RLix control plane")

        tgt_dp_ranks = sorted(set(int(r) for r in tgt_dp_ranks))
        if not tgt_dp_ranks:
            raise ValueError("tgt_dp_ranks must be non-empty")
        if not tgt_device_mapping:
            raise ValueError("tgt_device_mapping must be non-empty")
        if not isinstance(tgt_num_gpus_per_worker, int) or int(tgt_num_gpus_per_worker) <= 0:
            raise ValueError("tgt_num_gpus_per_worker must be positive int")
        if len(tgt_device_mapping) % int(tgt_num_gpus_per_worker) != 0:
            raise RuntimeError("tgt_device_mapping length must be divisible by tgt_num_gpus_per_worker")

        world_rank = int(self.worker.rank)
        logger.info(f"[rlix][selective_sync_active_cache] enter world_rank={world_rank} is_cache_owner={self._is_cache_owner}")

        # Non-owners have no cache and do no transport.
        # ray.get(sync_refs) in ModelUpdateService provides the sync barrier for all train workers.
        if not self._is_cache_owner:
            return None

        # Owner acquires lock for the entire replay (cache lookup + all transport + group teardown).
        # This prevents concurrent promote_active_checkpoint or _build_latest_bucket_cache from
        # racing with in-flight transport.
        logger.info("[rlix][selective_sync_active_cache] acquiring _cache_lock")
        with self._cache_lock:
            logger.info("[rlix][selective_sync_active_cache] _cache_lock acquired")
            # --- Cache lookup ---
            adapter_names_to_register: List[str] = []
            base_cached_buckets: List[Any] = []
            adapter_cached_buckets: Dict[str, List[Any]] = {}

            if adapters_to_sync is not None:
                # Sync specified adapters using their active versions.
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
                # adapters_to_sync=None + LoRA mode: sync ALL active adapters (expand path).
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
                # Full fine-tune path.
                if self._active_cached is None:
                    raise RuntimeError(
                        "selective_sync_active_cache requires an active promoted cache (active_cached is unset)"
                    )
                if self._active_cached not in self._cache_map:
                    raise RuntimeError(f"active_cached={self._active_cached} missing from cache_map")
                base_cached_buckets = list(self._cache_map[self._active_cached])

            # --- Decode comm_plan for the single owner ---
            # comm_plan is always non-None for the owner (ModelUpdateService guarantees this).
            if comm_plan is None:
                raise RuntimeError(
                    "selective_sync_active_cache: comm_plan must be non-None for the cache owner. "
                    "ModelUpdateService must always build a comm_plan keyed by the owner's src_rank."
                )
            if world_rank not in comm_plan:
                raise RuntimeError(
                    "selective_sync_active_cache comm_plan missing owner rank. "
                    f"owner_rank={world_rank} keys={sorted(int(k) for k in comm_plan.keys())}"
                )
            comm_plan_args = comm_plan[world_rank]
            group_name: Optional[str] = str(comm_plan_args["group_name"])
            ipc_targets: List[Dict[str, Any]] = comm_plan_args.get("ipc_targets", [])
            broadcast_local_ranks_by_dp_rank: Dict[int, List[int]] = comm_plan_args.get(
                "broadcast_local_ranks_by_dp_rank", {}
            )
            planned_broadcast_ranks = sorted({int(td["rank"]) for td in comm_plan_args.get("tgt_devices", [])})
            broadcast_workers = [tgt_workers[r] for r in planned_broadcast_ranks]
            logger.info(
                f"[rlix][selective_sync_active_cache] comm_plan decoded: "
                f"group_name={group_name} ipc_targets={len(ipc_targets)} "
                f"broadcast_ranks={planned_broadcast_ranks} "
                f"base_buckets={len(base_cached_buckets)} is_lora={self.is_lora}"
            )

            def _transport_bucket_sequence(
                bucket_sequence: List[Any],
                *,
                is_lora_stage: bool,
                phase_tag: str,
                adapter_label: Optional[str] = None,
            ) -> None:
                """Transport one bucket sequence (base or adapter) to all target workers.

                For each bucket: stage CPU->GPU once, then fan out via IPC to
                colocated workers and NCCL broadcast to remote workers. GPU staging
                buffer is freed after each bucket to limit peak VRAM.

                When model_update_transport="cpu_pickle", the IPC path serializes the
                CPU bucket directly with standard pickle (avoiding CUDA IPC). GPU
                staging is skipped when there are no broadcast workers.
                """
                transport = self.worker.pipeline_config.model_update_transport
                for bucket_idx, (tensors_meta, cpu_bucket) in enumerate(bucket_sequence):
                    logger.info(f"[rlix][transport] bucket={bucket_idx}/{len(bucket_sequence)} phase={phase_tag} transport={transport}")

                    # GPU staging is needed for NCCL broadcast or CUDA IPC serialization.
                    # With cpu_pickle and no broadcast workers, skip GPU staging entirely.
                    need_gpu_staging = bool(broadcast_workers) or transport == "cuda_ipc"
                    gpu_bucket = None
                    if need_gpu_staging:
                        gpu_bucket = cpu_bucket.to(current_platform.device_type).contiguous()
                        logger.info(f"[rlix][transport] bucket={bucket_idx} staged_to_gpu")

                    # Transport workflow (IPC + NCCL overlap):
                    # 1. Fire async: IPC sends to colocated workers (same node)
                    # 2. Fire async: NCCL broadcasts to remote workers (cross-node, GPU-to-GPU)
                    # 3. Barrier: wait on all IPC + NCCL to finish
                    # 4. Free gpu_bucket — safe because all consumers have copied the data
                    # IPC and NCCL run concurrently to hide transfer latency.

                    # Step 1: IPC path — serialize bucket once, then fan out to all colocated workers.
                    # Payload is identical for every IPC target, so serialize before the loop.
                    ipc_payload: Optional[bytes] = None
                    if ipc_targets:
                        if transport == "cpu_pickle":
                            # CPU byte serialization: serialize CPU bucket directly with
                            # standard pickle. Avoids CUDA IPC in restricted containers.
                            ipc_payload = pickle.dumps(
                                {"bucket": cpu_bucket.contiguous(), "tensors_meta": tensors_meta}
                            )
                        elif transport == "cuda_ipc":
                            # CUDA IPC: serialize GPU tensor via ForkingPickler.
                            # Ensure pickle uses GPU UUIDs instead of raw device indices,
                            # so the receiver resolves the correct local device even when
                            # CUDA_VISIBLE_DEVICES orderings differ between processes.
                            monkey_patch_torch_reductions()
                            ipc_payload = MultiprocessingSerializer.serialize(
                                {"bucket": gpu_bucket, "tensors_meta": tensors_meta}
                            )
                        else:
                            raise ValueError(
                                f"Unsupported model_update_transport: {transport!r}. "
                                f"Expected 'cuda_ipc' or 'cpu_pickle'."
                            )

                    ipc_refs: List[ray.ObjectRef] = []
                    for ipc_entry in ipc_targets:
                        tgt_dp_rank = int(ipc_entry["dp_rank"])
                        ipc_local_ranks: List[int] = [int(r) for r in ipc_entry["local_ranks"]]
                        # Build a list long enough to cover all TP ranks (worker indexes by self.rank).
                        payload_list = [ipc_payload] * tgt_num_gpus_per_worker
                        ipc_refs.append(
                            tgt_workers[tgt_dp_rank].update_parameter_in_bucket.remote(
                                payload_list,
                                is_lora=is_lora_stage,
                                ipc_local_ranks=ipc_local_ranks,
                            )
                        )

                    # Step 2: NCCL path — broadcast to remote (non-colocated) workers.
                    # Requires gpu_bucket; only entered when broadcast_workers is non-empty
                    # (which guarantees gpu_bucket was staged above).
                    nccl_handles: List[Any] = []
                    recv_refs: List[ray.ObjectRef] = []
                    named_params: List[Any] = []
                    if broadcast_workers and gpu_bucket is not None:
                        named_params = list(named_tensors_from_bucket(bucket=gpu_bucket, tensors_meta=tensors_meta))
                        names = [n for n, _ in named_params]
                        dtypes = [t.dtype for _, t in named_params]
                        shapes = [t.shape for _, t in named_params]

                        recv_refs = [
                            worker.broadcast_parameter.remote(
                                group_name=group_name,
                                names=names,
                                dtypes=dtypes,
                                shapes=shapes,
                                is_lora=is_lora_stage,
                                broadcast_local_ranks=broadcast_local_ranks_by_dp_rank.get(
                                    int(planned_broadcast_ranks[worker_idx])
                                ),
                            )
                            for worker_idx, worker in enumerate(broadcast_workers)
                        ]

                        for _, weight in named_params:
                            nccl_handles.append(
                                collective.broadcast(
                                    tensor=weight,
                                    src_rank=0,
                                    group_name=group_name,
                                    async_op=True,
                                )
                            )

                    # Step 3+4: barrier — wait for all transfers, then free GPU memory.
                    logger.info(f"[rlix][transport] bucket={bucket_idx} waiting nccl_handles={len(nccl_handles)} ipc_refs={len(ipc_refs)} recv_refs={len(recv_refs)}")
                    for nccl_handle in nccl_handles:
                        nccl_handle.wait()
                    logger.info(f"[rlix][transport] bucket={bucket_idx} nccl_done, waiting ray.get")
                    ray.get(ipc_refs + recv_refs)
                    logger.info(f"[rlix][transport] bucket={bucket_idx} all_done")
                    del nccl_handles, named_params
                    if gpu_bucket is not None:
                        del gpu_bucket
                        current_platform.empty_cache()

            # --- Transport: base buckets first, then per-adapter buckets ---
            _transport_bucket_sequence(base_cached_buckets, is_lora_stage=False, phase_tag="base")

            if self.is_lora and adapter_names_to_register:
                peft_configs = getattr(self.models_unwrapped[0], "peft_config", None) or {}
                missing_cfg = [a for a in adapter_names_to_register if a not in peft_configs]
                if missing_cfg:
                    raise RuntimeError(
                        f"selective_sync_active_cache: missing peft_config for adapters {missing_cfg}"
                    )
                for adapter_label in adapter_names_to_register:
                    buckets = adapter_cached_buckets.get(adapter_label, [])
                    if not buckets:
                        raise RuntimeError(
                            f"selective_sync_active_cache: no cached buckets for adapter={adapter_label!r}; "
                            "promote_active_adapter_checkpoint must be called before sync"
                        )
                    _transport_bucket_sequence(
                        buckets,
                        is_lora_stage=True,
                        phase_tag="adapter",
                        adapter_label=adapter_label,
                    )
                    # Compute the union of IPC and broadcast local ranks for this adapter's add_lora call.
                    # Collect all unique target actors across both paths.
                    adapter_target_actor_dp_ranks: Set[int] = set()
                    ipc_local_ranks_by_dp: Dict[int, List[int]] = {
                        int(entry["dp_rank"]): [int(r) for r in entry["local_ranks"]]
                        for entry in ipc_targets
                    }
                    for entry in ipc_targets:
                        adapter_target_actor_dp_ranks.add(int(entry["dp_rank"]))
                    for dp_rank in planned_broadcast_ranks:
                        adapter_target_actor_dp_ranks.add(int(dp_rank))

                    for dp_rank in sorted(adapter_target_actor_dp_ranks):
                        ipc_lr = ipc_local_ranks_by_dp.get(dp_rank, [])
                        broadcast_lr = broadcast_local_ranks_by_dp_rank.get(dp_rank, [])
                        lora_local_ranks = sorted(set(ipc_lr) | set(broadcast_lr)) or None
                        ray.get(
                            tgt_workers[dp_rank].add_lora.remote(
                                adapter_name=adapter_label,
                                peft_config=asdict(peft_configs[adapter_label]),
                                lora_local_ranks=lora_local_ranks,
                            )
                        )

            # --- Teardown broadcast group once after all replay completes ---
            if broadcast_workers:
                logger.info(f"[rlix][selective_sync_active_cache] teardown: destroying sender group {group_name}")
                collective.destroy_collective_group(group_name)
                logger.info(f"[rlix][selective_sync_active_cache] teardown: sender destroyed, destroying receiver groups")
                ray.get([w.destroy_collective_group.remote(group_name) for w in broadcast_workers])
                logger.info(f"[rlix][selective_sync_active_cache] teardown: all groups destroyed")

        # Collect sender stats from cached versions for post-sync verification.
        weight_stats: dict = {}
        if base_cached_buckets:
            base_key = self._active_cached
            base_stats = self._cache_stats.get(base_key, {})
            if base_stats:
                weight_stats["base"] = base_stats
        if adapter_cached_buckets:
            lora_stats: dict = {}
            for adapter_label in adapter_names_to_register:
                adapter_key = self._active_adapter_cached.get(adapter_label)
                adapter_stats = self._adapter_cache_stats.get((adapter_label, adapter_key), {})
                if adapter_stats:
                    lora_stats[adapter_label] = adapter_stats
            if lora_stats:
                weight_stats["lora"] = lora_stats

        # Lock released. No dist.barrier() here: ray.get(sync_refs) in ModelUpdateService
        # waits for all train workers to complete before the next sync is allowed.
        return {"weight_stats": weight_stats} if weight_stats else None

    def _translate_offload_include(
        self, include: Optional[List[OffloadStateType]]
    ) -> Tuple[bool, List[MegatronOffloadStateType]]:
        """Derive request intent from caller's include arg.

        Returns:
            wants_model_params: whether model_params reload/offload is requested.
            translated: Megatron-internal state types corresponding to requested states.
                When include is None (all states), returns all three types explicitly.
        """
        if include is None:
            return True, [
                MegatronOffloadStateType.model_params,
                MegatronOffloadStateType.other_params,
                MegatronOffloadStateType.optimizer_states,
            ]
        translated: List[MegatronOffloadStateType] = []
        if OffloadStateType.model_params in include:
            translated.append(MegatronOffloadStateType.model_params)
        if OffloadStateType.other_params in include:
            translated.append(MegatronOffloadStateType.other_params)
        if OffloadStateType.optimizer_states in include:
            translated.append(MegatronOffloadStateType.optimizer_states)
        return OffloadStateType.model_params in include, translated

    def load_states(self, include=None, non_blocking=False):
        """Reload optimizer and model states back to GPU.

        Behavior by caller context:
        - isolated + include=None: no-grad swap runs, optimizer gets explicit full list
        - non-isolated + include=None: no no-grad swap, optimizer gets raw None
        - either + explicit include with model_params: no-grad swap runs, optimizer gets translated list
        - either + explicit include without model_params: no no-grad swap
        - isolated + include=[]: no no-grad swap, optimizer call skipped
        - non-isolated + include=[]: no no-grad swap, optimizer called with []
        """
        wants_model_params, translated_include = self._translate_offload_include(include)

        # Manual no-grad reload needed when:
        # - Isolated optimizer: always (optimizer doesn't manage frozen base params)
        # - Explicit include with model_params: optimizer gets a filtered list,
        #   so it won't reload no-grad params on its own
        # Skipped only for non-isolated + include=None where the optimizer handles all.
        if wants_model_params and (self.is_lora_optimizer_isolated or include is not None):
            reload_megatron_no_grad_module(model_chunks=self.model.get_models())

        # Isolated path: always pass explicit translated list (never raw None).
        # Non-isolated path: preserve raw None so optimizer uses its default "all" handling.
        if self.is_lora_optimizer_isolated:
            if translated_include:
                self.optimizer.reload_states(include=translated_include, non_blocking=non_blocking)
        else:
            optimizer_include = None if include is None else translated_include
            self.optimizer.reload_states(include=optimizer_include, non_blocking=non_blocking)

    def offload_states(self, include=None, non_blocking=False, pin_memory=True):
        """Offload optimizer and model states from GPU.

        Behavior by caller context:
        - isolated + include=None: no-grad swap runs, optimizer gets explicit full list
        - non-isolated + include=None: no no-grad swap, optimizer gets raw None
        - either + explicit include with model_params: no-grad swap runs, optimizer gets translated list
        - either + explicit include without model_params: no no-grad swap
        - isolated + include=[]: no no-grad swap, optimizer call skipped
        - non-isolated + include=[]: no no-grad swap, optimizer called with []
        - rotary cache clear + CUDA cache clear always runs
        """
        wants_model_params, translated_include = self._translate_offload_include(include)

        # Same manual no-grad condition as load_states.
        if wants_model_params and (self.is_lora_optimizer_isolated or include is not None):
            offload_megatron_no_grad_module(
                model_chunks=self.model.get_models(), pin_memory=pin_memory,
            )

        if self.is_lora_optimizer_isolated:
            if translated_include:
                self.optimizer.offload_states(
                    include=translated_include, non_blocking=non_blocking, pin_memory=pin_memory,
                )
        else:
            optimizer_include = None if include is None else translated_include
            self.optimizer.offload_states(
                include=optimizer_include, non_blocking=non_blocking, pin_memory=pin_memory,
            )

        # Unconditional cleanup after offload (both paths, matches current behavior).
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
        # Save optimizer when single-process (no dist) OR when data-parallel rank is 0.
        elif (not dist.is_initialized()) or (
            (
                mpu.get_data_modulo_expert_parallel_rank()
                if hasattr(mpu, "get_data_modulo_expert_parallel_rank")
                else mpu.get_data_parallel_rank(with_context_parallel=False)
            )
            == 0
        ):
            torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, OPTIMIZER_NAME))
            logger.info(f"Saving optimizer state to {os.path.join(checkpoint_dir, OPTIMIZER_NAME)}")

        if dist.is_initialized():
            _safe_dist_barrier()

        # save lr_scheduler — isolated mode saves a dict with {"mode": "isolated",
        # "schedulers": {adapter_name: state_dict, ...}} so load_checkpoint can restore each
        # adapter's LR schedule independently.
        if dist.get_rank() == 0:
            if self.adapter_schedulers is not None:
                scheduler_state = {
                    "mode": "isolated",
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
        # Per-adapter RNG states enable deterministic per-adapter dropout across checkpoint restarts.
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
        if isinstance(scheduler_state, dict) and scheduler_state.get("mode") == "isolated":
            if self.adapter_schedulers is None:
                raise RuntimeError(
                    "Checkpoint contains shared-mode LoRA scheduler state which is no longer supported. "
                    "Only per-adapter LoRA checkpoints can be resumed."
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
