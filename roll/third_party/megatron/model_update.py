import time
from dataclasses import asdict
from typing import Generator, Optional

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from transformers.utils import is_peft_available

from mcore_adapter.models.converter.model_converter import ModelConverter
from mcore_adapter.models.model_factory import McaGPTModel
from roll.configs.base_config import PPOConfig
from roll.configs.worker_config import WorkerConfig, is_actor_infer_overlapping_with_any_cluster
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.driver_utils import Locker
from roll.platforms import current_platform
from roll.utils.collective import collective
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.utils.network_utils import collect_free_port, get_node_ip
from roll.utils.send_recv_utils import serialize_named_weights


if is_peft_available():
    from peft import PeftModel, get_peft_model_state_dict

logger = get_logger()


def gather_and_convert_weights(
    weights_info: list[tuple[str, torch.Tensor]],
    model_converter: ModelConverter,
    tp_group: Optional[dist.ProcessGroup] = None,
    ep_group: Optional[dist.ProcessGroup] = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    weights_info: list of tuples, each tuple is (mcore_name, weight)
    """
    if model_converter.mca_config.hf_model_type == "qwen3_vl_moe" and ep_group is not None:
        # qwen3_vl_moe has fused moe weights, so we need to gather weights in ep_group before convert
        handles, gathered_named_weights = [], []
        group_size = dist.get_world_size(ep_group)
        for mcore_name, weight in weights_info:
            if group_size == 1:
                gathered_named_weights.append((mcore_name, [weight]))
                handles.append(None)
                continue
            gathered_weights = [torch.empty_like(weight) for _ in range(group_size)]
            gathered_named_weights.append((mcore_name, gathered_weights))
            handles.append(dist.all_gather(gathered_weights, weight, group=ep_group, async_op=True))

        def extract_suffix_number(s):
            import re

            match = re.search(r"\d+$", s)
            return match.group() if match else None

        hf_named_weights = []
        for handle, (mcore_name, weights) in zip(handles, gathered_named_weights):
            if handle is not None:
                handle.wait()
            local_moe_index = extract_suffix_number(mcore_name)
            for ep_rank, weight in enumerate(weights):
                global_moe_index = model_converter.dist_converter.num_layers_for_expert * ep_rank + int(
                    local_moe_index
                )
                name = mcore_name[: -len(local_moe_index)] + str(global_moe_index)
                converted_weights = (
                    model_converter.convert_to_hf(
                        {name: [weight]}, layer_index_preprocessed=True, moe_index_preprocessed=True, **kwargs
                    )
                    or {}
                )
                hf_named_weights.extend([(name, weight) for name, weight in converted_weights.items()])

        return hf_named_weights

    handles, gathered_named_weights = [], []
    group_size = 1 if tp_group is None else dist.get_world_size(tp_group)
    for mcore_name, weight in weights_info:
        if group_size == 1:
            gathered_named_weights.append((mcore_name, [weight]))
            handles.append(None)
            continue
        gathered_weights = [torch.empty_like(weight) for _ in range(group_size)]
        gathered_named_weights.append((mcore_name, gathered_weights))
        handles.append(dist.all_gather(gathered_weights, weight, group=tp_group, async_op=True))

    hf_named_weights = []
    for handle, (mcore_name, weights) in zip(handles, gathered_named_weights):
        if handle is not None:
            handle.wait()
        converted_weights = (
            model_converter.convert_to_hf({mcore_name: weights}, layer_index_preprocessed=True, **kwargs) or {}
        )
        hf_named_weights.extend([(name, weight) for name, weight in converted_weights.items()])

    if ep_group is None or dist.get_world_size(ep_group) == 1:
        return hf_named_weights

    names = [name for name, _ in hf_named_weights]
    # TODO: use cpu but not communicate
    ep_group_size = dist.get_world_size(ep_group)
    all_names = [None for _ in range(dist.get_world_size(ep_group))]
    dist.all_gather_object(all_names, names, group=ep_group)
    handles = []
    all_named_weights = []
    for i, (name, weight) in enumerate(hf_named_weights):
        gathered_weights = [torch.empty_like(weight) for _ in range(ep_group_size)]
        handles.append(dist.all_gather(gathered_weights, weight.contiguous(), group=ep_group, async_op=True))
        for rank, gathered_weight in enumerate(gathered_weights):
            ep_name = all_names[rank][i]
            all_named_weights.append((ep_name, gathered_weight))

    for handle in handles:
        handle.wait()
    return all_named_weights


def _gather_hf_weights(
    model_converter: ModelConverter,
    named_weights: list[tuple[str, torch.Tensor]],
    buffer_size: Optional[int] = None,
    **kwargs,
):
    mca_config = model_converter.mca_config
    other_weights_with_info = []
    expert_weights_with_info = []
    for mcore_name, weight in named_weights:
        if model_converter.dist_converter.is_expert_parallel_weight(mcore_name):
            expert_weights_with_info.append((mcore_name, weight))
        else:
            other_weights_with_info.append((mcore_name, weight))

    def _process_and_yield_weights(weights_info, group=None, ep_group=None):
        # TODO: skip tp dup weights gather
        waiting_weights, waiting_weights_size = [], 0
        group_size = 1 if group is None else dist.get_world_size(group)
        group_size *= 1 if ep_group is None else dist.get_world_size(ep_group)
        for mcore_name, weight in weights_info:
            weight_size = weight.numel() * weight.element_size() * group_size
            if buffer_size is not None and waiting_weights_size + weight_size > buffer_size:
                yield gather_and_convert_weights(waiting_weights, model_converter, group, ep_group, **kwargs)
                waiting_weights, waiting_weights_size = [], 0
            waiting_weights.append((mcore_name, weight))
            waiting_weights_size += weight_size

        if waiting_weights:
            yield gather_and_convert_weights(waiting_weights, model_converter, group, ep_group, **kwargs)

    ep_group = None
    if mca_config.expert_model_parallel_size is not None and mca_config.expert_model_parallel_size > 1:
        ep_group = mpu.get_expert_model_parallel_group()

    yield from _process_and_yield_weights(expert_weights_with_info, mpu.get_expert_tensor_parallel_group(), ep_group)
    yield from _process_and_yield_weights(other_weights_with_info, mpu.get_tensor_model_parallel_group())


def _iter_vp_stage_named_weights(
    models: list[McaGPTModel], model_converter: ModelConverter, adapter_name: str | None = None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield (mcore_name, tensor) pairs across all virtual pipeline stages.

    adapter_name=None: export base model weights only (LoRA keys filtered out,
        `.base_layer.` prefix stripped to restore canonical mcore names).
    adapter_name="<name>": export only that adapter's LoRA delta weights.
    Non-PeftModel: export full state dict as-is.

    NOTE: Semantic difference from upstream PEFT — upstream's
    ``get_peft_model_state_dict(adapter_name=None)`` exports the *default*
    adapter's LoRA state, whereas here ``adapter_name=None`` means "export
    base weights only, with LoRA keys stripped".  Callers must pass an
    explicit adapter name to get LoRA delta weights.
    """
    for vp_stage, model in enumerate(models):
        if is_peft_available() and isinstance(model, PeftModel):
            # adapter_name=None: export base weights only, stripping PEFT wrapper
            # naming (`.base_layer.`) so ModelConverter sees canonical mcore names.
            if adapter_name is None:
                full_state_dict = model.state_dict_for_save_checkpoint()
                mcore_state_dict = {}
                for name, weight in full_state_dict.items():
                    if ".lora_" in name or ".lora_embedding_" in name or ".lora_magnitude_vector" in name:
                        continue
                    # LoRA wrappers expose base tensors as "...base_layer.<tensor>"; converter expects
                    # the original tensor names without the wrapper hop.
                    normalized_name = name.replace(".base_layer.", ".")
                    # Fail fast if stripping ".base_layer." produces a key that already exists,
                    # which would silently overwrite an unrelated weight.
                    if normalized_name in mcore_state_dict:
                        raise ValueError(
                            f"base_layer name collision: '{name}' normalizes to '{normalized_name}' "
                            f"which already exists in state dict"
                        )
                    mcore_state_dict[normalized_name] = weight
            else:
                mcore_state_dict = get_peft_model_state_dict(
                    model, model.state_dict_for_save_checkpoint(), adapter_name=adapter_name
                )
        else:
            mcore_state_dict = model.state_dict_for_save_checkpoint()
        for mcore_name, weight in sorted(mcore_state_dict.items()):
            if mcore_name.endswith("_extra_state"):
                continue
            mcore_name = model_converter.dist_converter.preprocess_layer_index(mcore_name, vp_stage=vp_stage)
            yield mcore_name, weight


def gather_pp_stage_hf_weights(
    models: list[McaGPTModel], buffer_size: int, adapter_name: str | None = None, **kwargs
):
    # gather tp&ep weights, not including pipeline parallel
    if not mpu.model_parallel_is_initialized():
        raise RuntimeError("Model parallelism must be initialized before save as hf inflight.")

    # Resolve lora_rank from peft_config if not already provided by caller
    # (gather_all_hf_weights already resolves rank and passes it in **kwargs).
    if "lora_rank" not in kwargs:
        lora_rank = _resolve_lora_rank(models[0], adapter_name)
        if lora_rank is not None:
            kwargs["lora_rank"] = lora_rank

    model_config = models[0].config
    model_converter = ModelConverter(model_config, to_hf=True, efficient_mode=True)
    yield from _gather_hf_weights(
        model_converter,
        list(_iter_vp_stage_named_weights(models, model_converter, adapter_name=adapter_name)),
        buffer_size,
        **kwargs,
    )


def gather_weights_meta_cross_pp(models: list[McaGPTModel], adapter_name: str | None = None):
    if not mpu.model_parallel_is_initialized():
        raise RuntimeError("Model parallelism must be initialized before save as hf inflight.")
    model_config = models[0].config
    if model_config.pipeline_model_parallel_size <= 1:
        return None
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    model_converter = ModelConverter(model_config, to_hf=True, efficient_mode=True)
    named_weights_meta = []
    for mcore_name, weight in _iter_vp_stage_named_weights(models, model_converter, adapter_name=adapter_name):
        weight_size = weight.numel() * weight.element_size()
        if model_converter.dist_converter.is_expert_parallel_weight(mcore_name):
            weight_size *= model_config.expert_model_parallel_size * model_config.expert_tensor_parallel_size
        else:
            weight_size *= model_config.tensor_model_parallel_size
        named_weights_meta.append(
            {
                "name": mcore_name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "pp_stage": pp_rank,
                "size": weight_size,
            }
        )
    all_named_weights_meta = [None for _ in range(model_config.pipeline_model_parallel_size)]
    dist.all_gather_object(all_named_weights_meta, named_weights_meta, group=mpu.get_pipeline_model_parallel_group())
    all_named_weights_meta = sorted(
        [meta for metas in all_named_weights_meta for meta in metas], key=lambda x: x["name"]
    )
    expert_weights_meta = []
    other_weights_meta = []
    for meta in all_named_weights_meta:
        if model_converter.dist_converter.is_expert_parallel_weight(meta["name"]):
            expert_weights_meta.append(meta)
        else:
            other_weights_meta.append(meta)
    return expert_weights_meta + other_weights_meta


def _resolve_lora_rank(model: McaGPTModel, adapter_name: str | None) -> int | None:
    """Resolve LoRA rank from model.peft_config, the single source of truth.

    Works for both PeftModel instances and project-specific wrappers that carry peft_config
    without subclassing PeftModel.  Fails fast when an explicit adapter_name is requested
    but not found in the config.
    """
    peft_configs = getattr(model, "peft_config", None)
    if not isinstance(peft_configs, dict) or not peft_configs:
        return None

    if adapter_name is None:
        # Best-effort: use any adapter's rank for converter ops during base-weight export.
        peft_cfg = next(iter(peft_configs.values()))
    else:
        peft_cfg = peft_configs.get(adapter_name)
        if peft_cfg is None:
            raise RuntimeError(f"Missing peft_config for adapter {adapter_name!r}")

    lora_rank = getattr(peft_cfg, "r", None)
    return int(lora_rank) if lora_rank is not None else None


def gather_all_hf_weights(
    models: list[McaGPTModel], buffer_size: int, weights_meta: Optional[list[dict]], adapter_name: str | None = None
):
    """Gather weights across all parallelism dimensions (TP, EP, PP) and convert to HF naming.

    Yields batches of (hf_name, tensor) pairs, bounded by buffer_size.

    Without PP (pp_size <= 1): delegates to gather_pp_stage_hf_weights for TP/EP gather only.
    With PP: broadcasts each weight from its owning PP stage to all ranks, then gathers TP/EP.
    Weights are processed in buffered batches to limit peak memory.

    weights_meta: pre-computed list from gather_weights_meta_cross_pp(), each entry is
        {"name": str, "shape": list, "dtype": str, "pp_stage": int, "size": int}.
        Only needed when pp_size > 1; ignored otherwise.
    """
    if not mpu.model_parallel_is_initialized():
        raise RuntimeError("Model parallelism must be initialized before save as hf inflight.")

    kwargs: dict = {}
    lora_rank = _resolve_lora_rank(models[0], adapter_name)
    if lora_rank is not None:
        kwargs["lora_rank"] = lora_rank

    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(
            "gather_all_hf_weights: adapter=%r lora_rank=%s model_cls=%s",
            adapter_name,
            lora_rank,
            type(models[0]).__name__,
        )

    pp_size = models[0].config.pipeline_model_parallel_size
    if pp_size <= 1:
        yield from gather_pp_stage_hf_weights(models, buffer_size, adapter_name=adapter_name, **kwargs)
        return

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    model_converter = ModelConverter(
        models[0].config, pipeline_model_parallel_rank=pp_rank, to_hf=True, efficient_mode=True
    )
    cur_stage_state_dict = {
        mcore_name: weight
        for mcore_name, weight in _iter_vp_stage_named_weights(models, model_converter, adapter_name=adapter_name)
    }

    def _gather_batch_params(named_weights_with_stage: list[tuple[str, torch.Tensor, int]]):
        # named_weights_with_stage: list of tuples, each tuple is (mcore_name, weight, pp_stage)
        named_weights, handles = [], []
        for mcore_name, weight, pp_stage in named_weights_with_stage:
            named_weights.append((mcore_name, weight))
            handles.append(
                dist.broadcast(
                    weight, group=mpu.get_pipeline_model_parallel_group(), async_op=True, group_src=pp_stage
                )
            )
        for handle in handles:
            handle.wait()
        yield from _gather_hf_weights(model_converter, named_weights, **kwargs)

    waiting_weights, waiting_weights_size = [], 0
    for weight_meta in weights_meta:
        weight_size = weight_meta["size"]
        if waiting_weights_size + weight_size > buffer_size and waiting_weights:
            yield from _gather_batch_params(waiting_weights)
            waiting_weights, waiting_weights_size = [], 0
        if weight_meta["pp_stage"] == pp_rank:
            weight = cur_stage_state_dict[weight_meta["name"]]
        else:
            weight = torch.empty(weight_meta["shape"], dtype=weight_meta["dtype"], device=current_platform.device_type)
        waiting_weights.append((weight_meta["name"], weight, weight_meta["pp_stage"]))
        waiting_weights_size += weight_size
    if waiting_weights:
        yield from _gather_batch_params(waiting_weights)


class MegatronWeightUpdater:
    def __init__(
        self,
        pipeline_config: PPOConfig,
        worker_config: WorkerConfig,
        model_update_name: str,
        models_unwrapped,
        infer_cluster: Cluster,
    ):
        self.pipeline_config = pipeline_config
        self.worker_config = worker_config
        self.model_update_name = model_update_name
        self.models_unwrapped = models_unwrapped
        self.model_update_infer_workers = infer_cluster.workers
        self._model_update_buffer_size = (
            pipeline_config.model_update_buffer_size_mb * 1024 * 1024
        )  # Convert MB to bytes
        # Uses `adapters` (not legacy `lora_target`) because __post_init__ normalizes
        # single-LoRA configs into the `adapters` dict, so this covers both cases.
        self.is_lora = self.worker_config.model_args.adapters is not None
        self.infer_worker_config = infer_cluster.worker_config
        self.infer_cluster = infer_cluster
        self.is_colocated = is_actor_infer_overlapping_with_any_cluster(
            infer_cluster.worker_config, actor_train=worker_config
        )
        self._broadcast_workers = None

        # Colocated mode attributes
        self._infer_parallel_cpu_group = None
        self._co_infer_worker = None
        self._buffer_num = None

        # Separated mode attributes
        self.model_update_group_name = None
        self._model_update_locker = None

        if self.is_colocated:
            self._setup_colocated_model_update()
        else:
            self._setup_separated_model_update()

    def model_update(self, adapters_to_update: list[str] | None = None):
        if self.is_colocated:
            return self._colocated_model_update(adapters_to_update=adapters_to_update)
        return self._separated_model_update(adapters_to_update=adapters_to_update)

    def _setup_colocated_model_update(self):
        logger.info(f"RANK {dist.get_rank()} Setup colocated model update")
        infer_worker_devices_num = self.infer_worker_config.num_gpus_per_worker
        train_world_size = dist.get_world_size()

        device_start_diff = min(self.worker_config.device_mapping) - min(self.infer_worker_config.device_mapping)
        device_end_diff = max(self.worker_config.device_mapping) - max(self.infer_worker_config.device_mapping)

        assert device_start_diff % infer_worker_devices_num == 0
        assert device_end_diff % infer_worker_devices_num == 0

        for start_rank in range(0, train_world_size, infer_worker_devices_num):
            end_rank = start_rank + infer_worker_devices_num
            assert end_rank <= train_world_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._infer_parallel_cpu_group = new_group
        infer_worker_idx = (dist.get_rank() + device_start_diff) // infer_worker_devices_num
        self._co_infer_worker = None
        if 0 <= infer_worker_idx < len(self.model_update_infer_workers):
            self._co_infer_worker = self.model_update_infer_workers[infer_worker_idx]

        # rank0 broadcast to mismatch workers
        if dist.get_rank() == 0 and (device_start_diff > 0 or device_end_diff < 0):
            self._broadcast_workers = []
            if device_start_diff > 0:
                self._broadcast_workers.extend(
                    self.model_update_infer_workers[: device_start_diff // infer_worker_devices_num]
                )
            if device_end_diff < 0:
                self._broadcast_workers.extend(
                    self.model_update_infer_workers[device_end_diff // infer_worker_devices_num :]
                )
            self._setup_broadcast_group()

        # weights_meta is computed per-adapter inside _gather_and_distribute_weights()
        # so that metadata names match the adapter-specific state dict keys.

    def _setup_separated_model_update(self):
        self._model_update_locker = Locker.options(
            name="model_update_locker", get_if_exists=True, namespace=RAY_NAMESPACE
        ).remote()
        if not (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        ):
            return

        self._broadcast_workers = self.model_update_infer_workers
        self._setup_broadcast_group()

    def _setup_broadcast_group(self):
        if not self._broadcast_workers:
            return

        ep_rank = 0
        if (
            self.models_unwrapped[0].config.num_moe_experts is not None
            and self.models_unwrapped[0].config.num_moe_experts > 1
        ):
            ep_rank = mpu.get_expert_model_parallel_rank()
        model_update_group_name = f"{self.model_update_name}_pp{mpu.get_pipeline_model_parallel_rank()}_ep{ep_rank}"
        self.model_update_group_name = model_update_group_name

        num_gpus_per_infer_worker = self.infer_worker_config.num_gpus_per_worker
        infer_device_num = num_gpus_per_infer_worker * len(self._broadcast_workers)
        master_address, master_port = get_node_ip(), collect_free_port()

        refs = [
            infer_worker.setup_collective_group.remote(
                master_address=master_address,
                master_port=master_port,
                group_name=self.model_update_group_name,
                rank_offset=i * num_gpus_per_infer_worker + 1,
                world_size=infer_device_num + 1,
            )
            for i, infer_worker in enumerate(self._broadcast_workers)
        ]
        collective.init_collective_group(
            infer_device_num + 1,
            0,
            group_name=self.model_update_group_name,
            master_addr=master_address,
            master_port=master_port,
        )
        ray.get(refs)

        logger.info(f"Init weights update group {model_update_group_name}")

    def _broadcast_to_infer_workers(self, hf_named_weights) -> list[ray.ObjectRef]:
        if not self._broadcast_workers:
            return []
        refs = [
            worker.broadcast_parameter.remote(
                group_name=self.model_update_group_name,
                names=[n for n, _ in hf_named_weights],
                dtypes=[w.dtype for _, w in hf_named_weights],
                shapes=[w.shape for _, w in hf_named_weights],
                is_lora=self.is_lora,
            )
            for worker in self._broadcast_workers
        ]
        handles = []
        for _, weight in hf_named_weights:
            handles.append(
                collective.broadcast(tensor=weight, src_rank=0, group_name=self.model_update_group_name, async_op=True)
            )
        for handle in handles:
            handle.wait()
        return refs

    def _colocated_model_update(self, *, adapters_to_update: list[str] | None = None):
        """Transfer weights to colocated inference workers via CUDA IPC.

        LoRA mode: loops over each adapter, transfers only LoRA delta weights (not base model),
        then calls add_lora() to register the adapter in the inference engine.
        Base mode: transfers full model weights in a single pass.
        """
        co_infer_rank = dist.get_rank(self._infer_parallel_cpu_group)
        if self.is_lora:
            peft_configs = self.models_unwrapped[0].peft_config
            selected = set(adapters_to_update) if adapters_to_update is not None else None
            for adapter_name, peft_config in peft_configs.items():
                if selected is not None and adapter_name not in selected:
                    continue
                self._gather_and_distribute_weights(adapter_name)
                # Register adapter on all infer workers (colocated + broadcast).
                # BLOCKING: upstream was fire-and-forget which races with inference requests.
                # Fix vs upstream: upstream only registered on _co_infer_worker.
                # Pure colocated: add_lora on _co_infer_worker only.
                # Partial overlap: add_lora on both _co_infer_worker and _broadcast_workers.
                # Pure separated: handled by _separated_model_update() on _broadcast_workers.
                add_lora_refs: list[ray.ObjectRef] = []
                if co_infer_rank == 0 and self._co_infer_worker is not None:
                    add_lora_refs.append(
                        self._co_infer_worker.add_lora.remote(
                            adapter_name=adapter_name, peft_config=asdict(peft_config)
                        )
                    )
                if dist.get_rank() == 0 and self._broadcast_workers:
                    add_lora_refs.extend(
                        w.add_lora.remote(adapter_name=adapter_name, peft_config=asdict(peft_config))
                        for w in self._broadcast_workers
                    )
                if add_lora_refs:
                    ray.get(add_lora_refs)
        else:
            self._gather_and_distribute_weights(None)
        return {}

    def _gather_and_distribute_weights(self, adapter_name: str | None = None):
        """Gather HF-format weights from this PP stage and push them to colocated inference workers.

        Converts Megatron-Core weights to HF naming in buffered batches, serializes each batch,
        and distributes to inference workers via gather (colocated) or broadcast (remote).

        adapter_name: which adapter's weights to gather. None means base weights only
            (LoRA keys stripped); a string means that adapter's LoRA delta weights.
        """
        refs = []
        infer_parallel_size = dist.get_world_size(self._infer_parallel_cpu_group)
        co_infer_rank = dist.get_rank(self._infer_parallel_cpu_group)
        # Compute weights_meta with the actual adapter_name so metadata names match
        # the state dict keys used by gather_all_hf_weights (base vs LoRA names).
        weights_meta = gather_weights_meta_cross_pp(self.models_unwrapped, adapter_name=adapter_name)
        for hf_named_weights in gather_all_hf_weights(
            self.models_unwrapped,
            buffer_size=self._model_update_buffer_size,
            weights_meta=weights_meta,
            adapter_name=adapter_name,
        ):
            if self._co_infer_worker is not None:
                serialized_tensors = serialize_named_weights(
                    hf_named_weights, infer_strategy=self.infer_worker_config.strategy_args.strategy_name
                )
                infer_parallel_tensors = [None] * infer_parallel_size if co_infer_rank == 0 else None
                dist.gather_object(
                    serialized_tensors, infer_parallel_tensors, group_dst=0, group=self._infer_parallel_cpu_group
                )

            if refs:
                ray.get(refs)
                refs = []
            if co_infer_rank == 0 and self._co_infer_worker is not None:
                refs.append(
                    self._co_infer_worker.update_parameter_in_bucket.remote(
                        infer_parallel_tensors, is_lora=self.is_lora
                    )
                )
            if self._broadcast_workers:
                refs.extend(self._broadcast_to_infer_workers(hf_named_weights))

        if refs:
            ray.get(refs)

    def _separated_model_update(self, *, adapters_to_update: list[str] | None = None):
        """Broadcast weights from train workers to remote (non-colocated) infer workers.

        Gathers HF-format weights from this PP stage in buffered batches and broadcasts
        each batch under a distributed lock to avoid conflicts with inference requests.

        LoRA mode: iterates over each adapter, broadcasts LoRA adapter weights, then registers
        the adapter on infer workers via add_lora().
        Base mode: broadcasts full model weights in a single pass.
        """
        if not mpu.get_expert_data_parallel_rank() == 0:
            return {}

        logger.info(f"start broadcast model update {self.model_update_name}")
        if self.is_lora:
            peft_configs = self.models_unwrapped[0].peft_config
            selected = set(adapters_to_update) if adapters_to_update is not None else None
            for adapter_name, peft_config in peft_configs.items():
                if selected is not None and adapter_name not in selected:
                    continue
                logger.info(f"model_update: broadcasting adapter={adapter_name!r}")
                for hf_named_weights in gather_pp_stage_hf_weights(
                    self.models_unwrapped,
                    buffer_size=self._model_update_buffer_size,
                    adapter_name=adapter_name,
                ):
                    if not self._broadcast_workers:
                        continue
                    self._broadcast_bucket_under_lock(hf_named_weights)
                # After broadcasting LoRA tensors, register the adapter on all infer workers.
                if self._broadcast_workers:
                    logger.info(f"model_update: registering adapter={adapter_name!r} on infer workers")
                    ray.get(
                        [
                            w.add_lora.remote(adapter_name=adapter_name, peft_config=asdict(peft_config))
                            for w in self._broadcast_workers
                        ]
                    )
                    logger.info(f"model_update: adapter={adapter_name!r} registration complete")
        else:
            for hf_named_weights in gather_pp_stage_hf_weights(
                self.models_unwrapped, buffer_size=self._model_update_buffer_size
            ):
                if not self._broadcast_workers:
                    continue
                self._broadcast_bucket_under_lock(hf_named_weights)
        return {}

    def _broadcast_bucket_under_lock(self, hf_named_weights: list[tuple[str, torch.Tensor]]) -> None:
        """Acquire model_update lock, broadcast one bucket to infer workers, then release."""
        while not ray.get(self._model_update_locker.acquire.remote()):
            time.sleep(0.1)
        try:
            refs = self._broadcast_to_infer_workers(hf_named_weights)
            ray.get(refs)
        finally:
            ray.get(self._model_update_locker.release.remote())
