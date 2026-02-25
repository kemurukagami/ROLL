import gc
import hashlib
import json
import time
from collections import OrderedDict
from typing import Iterable, Tuple

import torch
import vllm
from packaging.version import Version

from roll.platforms import current_platform
from roll.third_party.vllm.vllm_utils import TensorLoRARequest, patch_vllm_lora_manager
from roll.utils.collective import collective
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.functionals import get_dist_info_from_comm_plan
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import monkey_patch_torch_reductions, named_tensors_from_bucket

logger = get_logger()


class TensorLoraManager:
    def __init__(self):
        self.lora_params = OrderedDict()
        self._lora_names: dict[str, int] = {}  # Track adapter_name -> lora_int_id for routing lookups.

    def get_lora_id(self, adapter_name: str) -> int | None:
        # Return None when adapter has not been registered on this worker yet.
        return self._lora_names.get(adapter_name, None)

    def add_weight(self, name: str, weight: torch.Tensor):
        self.lora_params[name] = weight

    def build_request(self, adapter_name: str, peft_config: dict) -> TensorLoRARequest:
        """
        Generate a unique LoRA ID based on adapter name + PEFT config so every
        rank computes the same id for the same adapter registration.
        """
        # Use a stable hash key (adapter + config only). Do NOT include call-order counters,
        # otherwise different registration order across workers yields inconsistent adapter ids.
        peft_config_for_hash = dict(peft_config)
        peft_config_for_hash["adapter_name"] = adapter_name
        peft_config_str = json.dumps(peft_config_for_hash, sort_keys=True)
        hash_obj = hashlib.sha256(peft_config_str.encode("utf-8"))
        hex_dig = hash_obj.hexdigest()
        lora_int_id = int(hex_dig, 16) % 0x7FFFFFFF
        self._lora_names[adapter_name] = lora_int_id

        lora_request = TensorLoRARequest(
            lora_name=adapter_name,
            lora_int_id=lora_int_id,
            lora_path="dummy_lora_path",
            peft_config=peft_config_for_hash,
            lora_tensors=self.lora_params,
        )
        del self.lora_params
        self.lora_params = OrderedDict()
        return lora_request


class WorkerBase:
    def custom_init_worker(self, *args, **kwargs):
        self.weight_loaded: bool = True
        self.kv_cache_loaded: bool = True
        self.buffers = None
        self.buffer_cache = None
        self.tensor_lora_manager = TensorLoraManager()

    # Use custom prefix because worker_extension_cls can not have conflicting method names with vllm worker.
    def custom_add_lora(self, adapter_name: str, peft_config: dict) -> bool:
        # Build request with adapter name so routing can map name -> id consistently.
        lora_request = self.tensor_lora_manager.build_request(adapter_name, peft_config)
        lora_int_id = lora_request.lora_int_id
        staged_count = len(lora_request.lora_tensors) if lora_request.lora_tensors else 0
        # Diagnostic: check if adapter is still in vLLM's Python registry. After offload_states(level=2),
        # the registry is cleared, so in_vllm_cache=True here means the adapter was registered without
        # an intervening sleep (e.g. back-to-back add_lora calls). The cached GPU tensors are valid here.
        lora_manager = getattr(getattr(self, "model_runner", None), "lora_manager", None)
        in_vllm_cache = (
            lora_int_id in lora_manager.list_adapters()
            if lora_manager is not None and callable(getattr(lora_manager, "list_adapters", None))
            else None
        )
        logger.info(
            "[vllm][add_lora] enter adapter=%s int_id=%s staged_tensors=%s in_vllm_cache=%s weight_loaded=%s",
            adapter_name, lora_int_id, staged_count, in_vllm_cache, self.weight_loaded,
        )
        self.reload_model()
        add_lora = getattr(getattr(self, "model_runner", None), "add_lora", None)
        if not callable(add_lora):
            raise NotImplementedError(
                "vLLM worker does not expose model_runner.add_lora; "
                "ensure the configured vLLM version supports runtime LoRA registration."
            )
        try:
            ok = add_lora(lora_request)
        except Exception as exc:
            # Roll back local mapping so we do not keep a phantom adapter id.
            self.tensor_lora_manager._lora_names.pop(adapter_name, None)
            logger.error(
                "[vllm][add_lora] FAILED adapter=%s int_id=%s in_vllm_cache=%s exc=%s",
                adapter_name, lora_int_id, in_vllm_cache, exc,
            )
            raise
        if ok is False:
            # Roll back local mapping so verification sees only successfully-added adapters.
            self.tensor_lora_manager._lora_names.pop(adapter_name, None)
            logger.error(
                "[vllm][add_lora] returned_False adapter=%s int_id=%s in_vllm_cache=%s",
                adapter_name, lora_int_id, in_vllm_cache,
            )
            raise RuntimeError(f"vLLM add_lora returned False for adapter={adapter_name!r}")
        logger.info(
            "[vllm][add_lora] ok adapter=%s int_id=%s in_vllm_cache=%s",
            adapter_name, lora_int_id, in_vllm_cache,
        )
        return True

    def custom_list_loras(self) -> list[int]:
        # Query runtime vLLM LoRA state instead of tensor_lora_manager._lora_names.
        # This allows strategy-side visibility checks to detect slots that were evicted from GPU state.
        lora_manager = getattr(getattr(self, "model_runner", None), "lora_manager", None)
        if lora_manager is None:
            return []
        list_adapters = getattr(lora_manager, "list_adapters", None)
        if not callable(list_adapters):
            return []
        raw = list_adapters()
        if isinstance(raw, dict):
            raw = list(raw.keys())
        lora_ids = []
        for item in raw:
            if isinstance(item, int):
                lora_ids.append(item)
                continue
            # Some vLLM versions may return adapter names/ids as strings.
            # Resolve names through local adapter_name->id map to keep readiness checks accurate.
            if isinstance(item, str):
                if item.isdigit():
                    lora_ids.append(int(item))
                    continue
                mapped_id = self.tensor_lora_manager.get_lora_id(item)
                if isinstance(mapped_id, int):
                    lora_ids.append(mapped_id)
                continue
            lora_int_id = getattr(item, "lora_int_id", None)
            if isinstance(lora_int_id, int):
                lora_ids.append(lora_int_id)
        return sorted(set(lora_ids))

    def custom_get_lora_id(self, adapter_name: str) -> int | None:
        # Strategy uses this to resolve adapter name into vLLM integer adapter id.
        return self.tensor_lora_manager.get_lora_id(adapter_name)

    def reload_model(self):
        if not self.weight_loaded:
            self.wake_up(["weights"])
            self.weight_loaded = True
            # [debug] Stage 3: model structure just allocated on GPU by wake_up.
            # With the streaming approach in broadcast_parameter, no receive buffers exist yet,
            # so device_used = baseline + model_weights only.
            _free3, _total3 = torch.cuda.mem_get_info()
            logger.info(
                f"[debug][wake_up_done] "
                f"device_used={(_total3 - _free3) / 1024**3:.3f}GB "
                f"allocated={torch.cuda.memory_allocated() / 1024**3:.3f}GB"
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Before updating parameters, reinitialize the previously released model.
        self.reload_model()
        if vllm.__version__ < "0.8.5":
            from roll.third_party.vllm.vllm_utils import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(self.model_runner.model)
        # Root cause: vLLM's _create_lora_modules() permanently replaces all LoRA target modules
        # with wrapper objects at LoRAModelManager init time (e.g. gate_up_proj becomes
        # gate_up_proj.base_layer). AutoWeightsLoader skips the root module and directly calls
        # each child's load_weights (e.g. Qwen2Model.load_weights). That child builds its own
        # params_dict from self.named_parameters() and applies stacked_params_mapping
        # (gate_proj -> gate_up_proj), producing a fused key that no longer exists -> KeyError.
        # Fix: patch named_parameters() on every submodule that has its own load_weights
        # (those are the ones AutoWeightsLoader calls directly). Each alias maps the unwrapped
        # name to the same tensor as the base_layer counterpart.
        model = self.model_runner.model
        params_dict = dict(model.named_parameters(remove_duplicate=False))
        lora_active = any(".base_layer." in k for k in params_dict)
        if not lora_active:
            model.load_weights(weights=weights)
            return
        # Collect submodules (not root) that have their own load_weights — AutoWeightsLoader
        # calls these directly. Build per-submodule aliases stripping ".base_layer.".
        patches: dict = {}
        for submod_name, submod in model.named_modules():
            if submod is model:
                continue  # AutoWeightsLoader skips root to avoid infinite recursion
            if not callable(getattr(submod, "load_weights", None)):
                continue
            sub_params = dict(submod.named_parameters(remove_duplicate=False))
            if not any(".base_layer." in k for k in sub_params):
                continue
            sub_aliases = {
                k.replace(".base_layer.", "."): v
                for k, v in sub_params.items()
                if ".base_layer." in k and k.replace(".base_layer.", ".") not in sub_params
            }
            orig = submod.named_parameters

            # Closure captures the correct orig and sub_aliases for each submodule.
            def _make_aliased(orig_fn, aliased_dict):
                def _aliased(*args, **kwargs):
                    yield from orig_fn(*args, **kwargs)
                    yield from aliased_dict.items()
                return _aliased

            submod.named_parameters = _make_aliased(orig, sub_aliases)
            patches[submod_name] = (submod, orig)
        try:
            model.load_weights(weights=weights)
        finally:
            for _, (submod, orig) in patches.items():
                submod.named_parameters = orig

    def load_states(self):
        self.reload_model()
        if not self.kv_cache_loaded:
            self.wake_up(["kv_cache"])
            self.kv_cache_loaded = True
        if vllm.__version__ < "0.8.5" and self.buffers is not None:
            # https://github.com/vllm-project/vllm/issues/16564
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self.buffers:
                    buffer.data.copy_(self.buffers[name].data)
            self.buffers = None

    def offload_states(self, level):
        assert (self.weight_loaded and self.kv_cache_loaded) or (not self.weight_loaded and not self.kv_cache_loaded)
        if not self.weight_loaded:
            logger.info("[vllm][offload] already offloaded, skip (level=%s)", level)
            # Clear staged LoRA tensors even when model weights are already offloaded.
            # These tensors are sync staging buffers, not persistent model state.
            if getattr(self, "tensor_lora_manager", None) is not None and self.tensor_lora_manager.lora_params:
                staged_count = len(self.tensor_lora_manager.lora_params)
                self.tensor_lora_manager.lora_params = OrderedDict()
                logger.info("[vllm][offload] cleared staged LoRA tensors while already-offloaded: count=%s", staged_count)
            return
        _desc = "destroy weights+KV" if level == 2 else "swap weights to CPU, discard KV"
        logger.info("[vllm][offload] sleep(level=%s) start: %s", level, _desc)
        if vllm.__version__ < "0.8.5" and level == 2:
            # https://github.com/vllm-project/vllm/issues/16564
            model = self.model_runner.model
            self.buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
        self.sleep(level)
        self.weight_loaded = False
        self.kv_cache_loaded = False
        if hasattr(self, "recv_manager"):
            self.recv_manager.clear()
        # Drop staged LoRA tensors so repeated selective-sync cycles do not accumulate GPU buffers.
        # Adapter registration ids stay in tensor_lora_manager._lora_names for routing.
        if getattr(self, "tensor_lora_manager", None) is not None and self.tensor_lora_manager.lora_params:
            staged_count = len(self.tensor_lora_manager.lora_params)
            self.tensor_lora_manager.lora_params = OrderedDict()
            logger.info("[vllm][offload] cleared staged LoRA tensors: count=%s", staged_count)
        # sleep(level=2) frees ALL GPU memory including LoRA tensors, but vLLM's Python-side LoRA cache
        # (LRUCacheWorkerLoRAManager) still holds the adapter entries pointing at the now-freed GPU memory.
        # On the next add_lora call, vLLM would take the else-branch (adapter "in cache") and skip
        # reloading LoRA tensors to GPU → using freed memory during generate → CUDA error / process crash.
        # Fix: evict all registered adapters from vLLM's Python cache here, so the next add_lora always
        # takes the fresh-load path. This also ensures newly trained LoRA weights are always applied.
        if (
            level == 2
            and getattr(self, "tensor_lora_manager", None) is not None
            and self.tensor_lora_manager._lora_names
        ):
            lora_manager = getattr(getattr(self, "model_runner", None), "lora_manager", None)
            remove_adapter = getattr(lora_manager, "remove_adapter", None) if lora_manager is not None else None
            evicted = 0
            if callable(remove_adapter):
                for int_id in self.tensor_lora_manager._lora_names.values():
                    remove_adapter(int_id)
                    evicted += 1
            self.tensor_lora_manager._lora_names = {}
            logger.info("[vllm][offload] cleared adapter id map and evicted vllm cache: count=%s", evicted)
        gc.collect()
        current_platform.empty_cache()
        logger.info("[vllm][offload] sleep(level=%s) done: GPU memory %s", level, "fully freed" if level == 2 else "weights on CPU, KV discarded")

    def setup_collective_group(self, *args, **kwargs):
        # Dynamic comm_plan based group setup (selective model-update style).
        if "comm_plan" in kwargs:
            comm_plan = kwargs["comm_plan"]
            backend = kwargs["backend"]
            rank_in_cluster = int(kwargs["rank_in_cluster"])
            timeout_s = kwargs.get("timeout_s", None)

            group_rank, comm_plan_args = get_dist_info_from_comm_plan(
                comm_plan, rank_in_cluster=rank_in_cluster, rank_in_worker=int(self.rank)
            )
            if group_rank is None:
                logger.info(
                    f"[schedrl][vllm][collective] setup_skip "
                    f"rank_in_cluster={rank_in_cluster} rank_in_worker={int(self.rank)}"
                )
                return

            group_name = comm_plan_args["group_name"]
            master_address = comm_plan_args["master_addr"]
            master_port = comm_plan_args["master_port"]
            world_size = int(len(comm_plan_args["tgt_devices"]) + 1)
            logger.info(
                f"[schedrl][vllm][collective] setup_enter group_name={group_name} "
                f"rank={group_rank} world_size={world_size} master={master_address}:{master_port} "
                f"timeout_s={timeout_s}"
            )
            collective.init_collective_group(
                world_size,
                rank=int(group_rank),
                backend=backend,
                group_name=group_name,
                master_addr=master_address,
                master_port=master_port,
                timeout_s=timeout_s,
            )
            collective.allreduce(torch.zeros(1, device=current_platform.device_type), group_name=group_name)
            logger.info(
                f"[schedrl][vllm][collective] setup_exit group_name={group_name} "
                f"rank={group_rank} world_size={world_size}"
            )
            return

        # Legacy / persistent broadcast group style.
        if len(args) < 6:
            raise TypeError(
                "setup_collective_group expects either comm_plan kwargs or "
                "(master_address, master_port, rank_offset, world_size, group_name, backend, timeout_s=?)."
            )
        master_address, master_port, rank_offset, world_size, group_name, backend = args[:6]
        timeout_s = kwargs.get("timeout_s", None)
        group_rank = int(self.rank) + int(rank_offset)
        logger.info(
            f"[schedrl][vllm][collective] setup_enter group_name={group_name} "
            f"rank={group_rank} world_size={world_size} master={master_address}:{master_port} "
            f"rank_offset={rank_offset} timeout_s={timeout_s}"
        )
        collective.init_collective_group(
            int(world_size),
            rank=group_rank,
            backend=backend,
            group_name=group_name,
            master_addr=master_address,
            master_port=int(master_port),
            timeout_s=timeout_s,
        )
        logger.info(
            f"[schedrl][vllm][collective] setup_exit group_name={group_name} "
            f"rank={group_rank} world_size={world_size}"
        )

    def destroy_collective_group(self, group_name: str):
        logger.info(f"[schedrl][vllm][collective] destroy_enter group_name={group_name}")
        collective.destroy_collective_group(group_name)
        logger.info(f"[schedrl][vllm][collective] destroy_exit group_name={group_name}")

    def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        # [debug] Stage 1: log GPU memory before any receive buffer is allocated.
        # If another process still has model weights loaded, device_used will be much higher
        # than the expected idle baseline (~3.5 GiB for 6 idle processes on this test config).
        _free_bytes, _total_bytes = torch.cuda.mem_get_info()
        _device_used_gb = (_total_bytes - _free_bytes) / 1024**3
        _alloc_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(
            f"[schedrl][vllm][broadcast] enter group_name={group_name} "
            f"num_tensors={len(names)} is_lora={int(bool(is_lora))} "
            f"[debug] device_used={_device_used_gb:.3f}GB allocated={_alloc_gb:.3f}GB "
            f"device_total={_total_bytes / 1024**3:.3f}GB"
        )

        if is_lora:
            # LoRA tensors are small: keep async batch pattern so NCCL can pipeline transfers.
            weights_and_handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handle = collective.broadcast(tensor=weight, src_rank=0, group_name=group_name, async_op=True)
                weights_and_handles.append((name, weight, handle))
            for name, weight, handle in weights_and_handles:
                handle.wait()
                self.tensor_lora_manager.add_weight(name, weight)
            logger.info(f"[schedrl][vllm][broadcast] exit group_name={group_name} mode=lora")
            return

        # Base weights: reload model FIRST, then stream one tensor at a time via a generator.
        # Peak memory = model_weights + one_tensor_buffer (not model + ALL buffers simultaneously).
        # Passing a generator to load_weights means LoRA patch logic runs ONCE for all tensors
        # (O(1) named_modules scan), vs calling load_weights 290 times (O(290) scans).
        self.reload_model()

        def _streaming_weights_gen():
            for _name, _dtype, _shape in zip(names, dtypes, shapes):
                _target_dtype = _dtype if isinstance(_dtype, torch.dtype) else getattr(torch, _dtype)
                _buf = torch.empty(_shape, dtype=_target_dtype, device=self.device)
                # Blocking broadcast: receive this tensor before allocating the next buffer.
                collective.broadcast(tensor=_buf, src_rank=0, group_name=group_name, async_op=False)
                yield _name, _buf
                del _buf  # free buffer before allocating the next one

        # load_weights calls reload_model() internally; no-op since weight_loaded=True after
        # the reload_model() call above.
        self.load_weights(weights=_streaming_weights_gen())

        # [debug] Stage 4: all tensors loaded; peak (model + one_buffer) has already passed.
        _free4, _total4 = torch.cuda.mem_get_info()
        logger.info(
            f"[debug][broadcast_load_done] group_name={group_name} "
            f"device_used={(_total4 - _free4) / 1024**3:.3f}GB "
            f"allocated={torch.cuda.memory_allocated() / 1024**3:.3f}GB"
        )
        logger.info(f"[schedrl][vllm][broadcast] exit group_name={group_name} mode=weights")

    def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        monkey_patch_torch_reductions()
        bucket_with_meta = MultiprocessingSerializer.deserialize(serialized_named_tensors[self.rank])
        # Support both formats:
        # - {"bucket": <torch.Tensor>, "tensors_meta": ...}  (legacy / CUDA-IPC path)
        # - {"bucket_bytes": <bytes>, "tensors_meta": ...}  (SchedRL CPU-cache safe path)
        if "bucket" not in bucket_with_meta:
            bucket_bytes = bucket_with_meta.get("bucket_bytes")
            if bucket_bytes is None:
                raise RuntimeError("update_parameter_in_bucket missing 'bucket' or 'bucket_bytes'")
            bucket_with_meta["bucket"] = torch.frombuffer(memoryview(bucket_bytes), dtype=torch.int8).to(
                device=self.device
            ).contiguous()
            # Avoid passing unexpected kwargs into named_tensors_from_bucket.
            bucket_with_meta.pop("bucket_bytes", None)
        else:
            bucket = bucket_with_meta["bucket"]
            if not getattr(bucket, "is_cuda", False):
                bucket_with_meta["bucket"] = bucket.to(device=self.device).contiguous()
            bucket_with_meta.pop("bucket_bytes", None)
        named_params = list(named_tensors_from_bucket(**bucket_with_meta))
        if is_lora:
            for name, weight in named_params:
                self.tensor_lora_manager.add_weight(name, weight)
            return
        self.load_weights([(name, weight) for name, weight in named_params])

    def process_weights_after_loading(self):
        if (Version("0.11.0") == Version(vllm.__version__) or
                Version("0.11.1rc1") == Version(vllm.__version__) or
                Version("0.11.1rc2.dev0+gc3a722fcb.d20251021") == Version(vllm.__version__)):
            from vllm.model_executor.model_loader.utils import process_weights_after_loading,set_default_torch_dtype
            device_config = self.device_config
            load_config = self.vllm_config.load_config
            load_device = (device_config.device if load_config.device is None else load_config.device)
            target_device = torch.device(load_device)
            with set_default_torch_dtype(self.model_config.dtype):
                process_weights_after_loading(self.model_runner.model,self.model_config,target_device)


class WorkerV1(WorkerBase):
    def custom_init_worker(self, *args, **kwargs):
        super().custom_init_worker(*args, **kwargs)
        patch_vllm_lora_manager()

    # custom_add_lora is inherited from WorkerBase so all worker variants share adapter-name logic.
