import gc
import hashlib
import json
import pickle
import time
from collections import OrderedDict
from typing import Iterable, List, Optional, Tuple

import torch
import vllm
from packaging.version import Version

from roll.platforms import current_platform
from roll.third_party.vllm.vllm_utils import TensorLoRARequest, patch_vllm_lora_manager
from roll.utils.collective import collective
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.functionals import get_dist_info_from_comm_plan
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import compute_weight_stats, monkey_patch_torch_reductions, named_tensors_from_bucket

logger = get_logger()


class TensorLoraManager:
    """Manages LoRA adapter staging and confirmed registration for vLLM workers.

    Two concerns:
    (a) Staging: collects incoming tensor weights (add_weight) before they are passed
        to vLLM for ingestion via build_request.
    (b) Tracking: maintains a confirmed adapter_name -> lora_int_id map (_lora_names)
        so routing and readiness checks can look up integer adapter ids by name.
        An entry in _lora_names means vLLM has confirmed the adapter is loaded on GPU;
        it is never set speculatively.
    """

    def __init__(self):
        self.lora_params = OrderedDict()
        self.add_lora_count = 0
        self._lora_names: dict[str, int] = {}  # Track adapter_name -> lora_int_id for routing lookups.
        # Preserve raw received tensors (HF-format) per adapter for post-sync verification.
        # Populated in build_request() before lora_params is cleared; survives until next sync.
        self._staged_weights: dict[str, OrderedDict] = {}

    def get_lora_id(self, adapter_name: str) -> int | None:
        """Return the vLLM integer adapter id for adapter_name, or None if not yet registered.

        Returns None when the adapter has not been confirmed as loaded by vLLM on this worker.
        Callers should treat None as "not ready" and retry or skip the operation.
        """
        # Return None when adapter has not been registered on this worker yet.
        return self._lora_names.get(adapter_name, None)

    def register(self, adapter_name: str, lora_int_id: int) -> None:
        """Record a confirmed adapter registration.

        Must be called only after vLLM's add_lora succeeds.
        Invariant: an entry in _lora_names means the adapter is actually loaded in vLLM on GPU.
        Violation (calling before vLLM confirms) would cause routing to route to an unloaded adapter.
        """
        # Called only after vLLM confirms the adapter is loaded successfully.
        # Invariant: entry in _lora_names ↔ adapter successfully registered in vLLM.
        self._lora_names[adapter_name] = lora_int_id

    def add_weight(self, name: str, weight: torch.Tensor):
        self.lora_params[name] = weight

    def build_request(self, adapter_name: str, peft_config: dict) -> TensorLoRARequest:
        """Build a TensorLoRARequest from staged weights and return it.

        Computes a stable lora_int_id from the adapter name + PEFT config so every
        TP-rank worker produces the same integer id for the same adapter, regardless
        of registration order.  The old design used a call-order counter, which caused
        different TP ranks to compute different ids when adapters were registered in
        different orders — leading to NCCL group membership mismatches.

        Does NOT update _lora_names.  Registration is intentionally deferred to
        register(), which is called by custom_add_lora only after vLLM confirms success.
        This keeps _lora_names as a strictly confirmed-state map.

        Consumes and resets self.lora_params after building the request.
        """
        self.add_lora_count += 1
        peft_config["add_lora_count"] = self.add_lora_count
        # Use a stable hash key (adapter + config only). Do NOT include call-order counters,
        # otherwise different registration order across workers yields inconsistent adapter ids.
        # Exclude add_lora_count from hash — it increments per call, producing different int_ids
        # for the same adapter across sync cycles and causing vLLM LRU eviction mismatches.
        peft_config_for_hash = {k: v for k, v in peft_config.items() if k != "add_lora_count"}
        peft_config_for_hash["adapter_name"] = adapter_name
        peft_config_str = json.dumps(peft_config_for_hash, sort_keys=True)
        hash_obj = hashlib.sha256(peft_config_str.encode("utf-8"))
        hex_dig = hash_obj.hexdigest()
        lora_int_id = int(hex_dig, 16) % 0x7FFFFFFF
        # Do NOT set _lora_names here — registration is recorded by register() only after
        # vLLM confirms the adapter loaded successfully in custom_add_lora.

        lora_request = TensorLoRARequest(
            lora_name=adapter_name,
            lora_int_id=lora_int_id,
            lora_path="dummy_lora_path",
            peft_config=peft_config_for_hash,
            lora_tensors=self.lora_params,
        )
        # Preserve raw received tensors for post-sync verification before clearing.
        # These are the same HF-format tensors the sender produced, so stats comparison
        # against sender stats is valid (same format, no vLLM transformation applied yet).
        self._staged_weights[adapter_name] = self.lora_params
        # Normal-path cleanup: transfer ownership of staged tensors to lora_request, then
        # reset lora_params immediately.  lora_request is a local in custom_add_lora; once
        # vLLM's add_lora() copies the tensors into GPU memory and the function returns,
        # lora_request goes out of scope and Python GC frees the staging buffers.
        # No separate cleanup step is needed on the happy path.
        self.lora_params = OrderedDict()
        return lora_request


class WorkerBase:
    """Mixin that extends vLLM's WorkerExtensionCls with RLix-specific lifecycle methods.

    All methods use the "custom_" prefix to avoid name conflicts with vLLM's own worker
    methods.  WorkerV1 (and future V2) subclass this to inherit the full implementation;
    they only override what differs between engine versions.

    Key responsibilities:
    - LoRA adapter registration and lifecycle (custom_add_lora, custom_list_loras,
      custom_get_lora_id).
    - GPU memory lifecycle: reload_model, load_states, offload_states.
    - Parameter broadcast and bucket update for model-weight synchronisation.
    - NCCL collective group management for model updates.
    """

    def custom_init_worker(self, *args, **kwargs):
        self.weight_loaded: bool = True
        self.kv_cache_loaded: bool = True
        self.buffers = None
        self.buffer_cache = None
        self.tensor_lora_manager = TensorLoraManager()

    # Use custom prefix because worker_extension_cls can not have conflicting method names with vllm worker.
    def custom_add_lora(
        self,
        adapter_name: str,
        peft_config: dict,
        *,
        lora_local_ranks: Optional[List[int]] = None,
        wake_after_add: bool = True,
    ) -> bool:
        """Register a LoRA adapter with vLLM on this worker.

        Pre-condition: staged LoRA tensors have already been delivered via add_weight calls.
        Post-condition: adapter is loaded in vLLM and tensor_lora_manager._lora_names[adapter_name]
        is set only on success.

        Why conditional wake-up here:
        LoRA tensors are allocated outside the cumem "weights" pool.  If we only called
        reload_model() (which wakes weights only), the KV cache would remain uninitialised.
        A subsequent load_states_partial call that tries wake_up(["kv_cache"]) on a GPU
        that is already near-full with model weights + LoRA tensors would OOM.
        For multi-adapter updates:
          - non-final adapters call reload_model() to keep broadcast memory low
          - final adapter calls load_states() to initialize KV cache before rollout
        We avoid follow-up strategy RPC verification after this call to prevent
        reentrancy stalls.

        Registration is deferred to after vLLM confirms success so _lora_names only ever
        holds adapters that are actually resident on GPU.
        """
        # Partial-overlap support: skip registration on ranks not in the mask.
        if lora_local_ranks is not None and self.rank not in lora_local_ranks:
            return True  # match existing True return convention for non-participating ranks

        # Build request with adapter name so routing can map name -> id consistently.
        lora_request = self.tensor_lora_manager.build_request(adapter_name, peft_config)
        lora_int_id = lora_request.lora_int_id
        staged_count = len(lora_request.lora_tensors) if lora_request.lora_tensors else 0
        # Diagnostic: check if adapter is still in vLLM's Python registry. After offload_states() at
        # either sleep level, the registry is cleared, so in_vllm_cache=True here means the adapter was
        # registered without an intervening sleep (e.g. back-to-back add_lora calls). GPU tensors are valid here.
        lora_manager = getattr(getattr(self, "model_runner", None), "lora_manager", None)
        in_vllm_cache = (
            lora_int_id in lora_manager.list_adapters()
            if lora_manager is not None and callable(getattr(lora_manager, "list_adapters", None))
            else None
        )
        logger.info(
            "[vllm][add_lora] enter adapter=%s int_id=%s staged_tensors=%s in_vllm_cache=%s weight_loaded=%s wake_after_add=%s",
            adapter_name, lora_int_id, staged_count, in_vllm_cache, self.weight_loaded, wake_after_add,
        )
        # Ensure weights are resident before add_lora. Final adapter also wakes KV cache.
        if wake_after_add:
            self.load_states()
        else:
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
            logger.error(
                "[vllm][add_lora] FAILED adapter=%s int_id=%s in_vllm_cache=%s exc=%s",
                adapter_name, lora_int_id, in_vllm_cache, exc,
            )
            raise
        if ok is False:
            logger.error(
                "[vllm][add_lora] returned_False adapter=%s int_id=%s in_vllm_cache=%s",
                adapter_name, lora_int_id, in_vllm_cache,
            )
            raise RuntimeError(f"vLLM add_lora returned False for adapter={adapter_name!r}")
        # vLLM confirmed success — record the registration now so _lora_names only ever
        # contains adapters that are actually loaded in vLLM.
        self.tensor_lora_manager.register(adapter_name, lora_request.lora_int_id)
        logger.info(
            "[vllm][add_lora] ok adapter=%s int_id=%s in_vllm_cache=%s",
            adapter_name, lora_int_id, in_vllm_cache,
        )
        return True

    def custom_list_loras(self) -> list[int]:
        """Return the sorted list of vLLM integer adapter ids currently loaded on this worker.

        Queries the live vLLM LoRA manager directly rather than tensor_lora_manager._lora_names,
        because _lora_names is a local Python map that is cleared on sleep().  Querying vLLM at
        runtime detects evicted slots that the Python map might still show after partial failures.

        Normalises heterogeneous return types across vLLM versions:
        - dict  → keys are adapter ids
        - list[int]  → used directly
        - list[str]  → numeric strings cast to int; name strings resolved via _lora_names
        - list[object with lora_int_id attr]  → attribute extracted

        Returns an empty list when no LoRA manager is present (LoRA not enabled).
        """
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
        """Return the vLLM integer adapter id for adapter_name, or None if not yet registered.

        Provides a stable public API on the worker so strategy code does not need to reach into
        tensor_lora_manager directly.  Returns None when the adapter has not been confirmed loaded.
        """
        # Strategy uses this to resolve adapter name into vLLM integer adapter id.
        return self.tensor_lora_manager.get_lora_id(adapter_name)

    def custom_verify_model(self, expected_stats: dict) -> dict:
        """Compute weight stats from this TP rank and return them for strategy-level aggregation.

        Base model: reads live GPU parameters from model_runner.model.named_parameters().
        End-to-end — these are the actual tensors used for inference. Stats are computed
        in-place using .sum(dtype=float32) — no fp32 copy is allocated, only a scalar.
        When LoRA modules are active, named_parameters() returns base weights only; LoRA
        delta tensors are plain torch.Tensors (not nn.Parameters) stored in
        lora_a_stacked/lora_b_stacked GPU buffers, so they do NOT appear in named_parameters().

        LoRA: reads raw received tensors from tensor_lora_manager._staged_weights (transport+
        delivery verification — same HF-format as sender, before vLLM's _load_adapter
        transformation). Identical across all TP ranks.

        Also performs a LoRA presence check: verifies every adapter in _lora_names exists in
        vLLM's live lora_manager.list_adapters().

        Returns per-rank stats dict for strategy-level TP aggregation (base) and comparison (LoRA).
        """
        result: dict = {}

        # LoRA presence check: every registered adapter must be in vLLM's live manager.
        # Direct attribute access — model_runner.lora_manager is always present on vLLM
        # workers when LoRA is active (which is the only case where _lora_names is non-empty).
        if self.tensor_lora_manager._lora_names:
            live_ids = set(self.model_runner.lora_manager.list_adapters())
            for adapter_name, expected_id in self.tensor_lora_manager._lora_names.items():
                    if expected_id not in live_ids:
                        raise RuntimeError(
                            f"verify_model: adapter {adapter_name!r} (int_id={expected_id}) "
                            f"not in vLLM live adapters {sorted(live_ids)}"
                        )

        # Base model stats: live GPU parameters (TP-sharded per rank).
        # remove_duplicate=True (default) so tied weights (e.g. embed_tokens/lm_head when
        # tie_word_embeddings=True) are counted once, matching the sender's gather_all_hf_weights.
        if "base" in expected_stats:
            model = self.model_runner.model
            base_stats = compute_weight_stats(model.named_parameters())
            result["base"] = base_stats

        # LoRA stats: raw received tensors (identical across TP ranks).
        if "lora" in expected_stats:
            result["lora"] = {}
            for adapter_name in expected_stats["lora"]:
                staged = self.tensor_lora_manager._staged_weights.get(adapter_name)
                if staged is None:
                    raise RuntimeError(
                        f"verify_model: no staged weights for adapter {adapter_name!r}; "
                        f"available={sorted(self.tensor_lora_manager._staged_weights.keys())}"
                    )
                adapter_stats = compute_weight_stats(staged.items())
                result["lora"][adapter_name] = adapter_stats

        logger.info("[vllm][verify_model] rank=%s stats_keys=%s", self.rank, sorted(result.keys()))
        return result

    def reload_model(self):
        """Allocate the GPU weight memory pool — does NOT update parameter values.

        Calls wake_up(["weights"]) to restore the CuMem "weights" pool back to GPU.
        After this returns, weight tensors are addressable on GPU but their values are
        whatever was there before sleep (restored from CPU at level=1, or re-initialized
        at level=2). No new parameter values are written here.

        To write new trainer weights into the restored pool, call load_weights() next.
        For a full wake-up (weights + KV cache), use load_states() instead.

        Idempotent: guarded by weight_loaded flag, so repeated calls are no-ops.

        The [debug][wake_up_done] log is a Stage 3 breadcrumb for memory profiling:
        at this point no receive buffers exist yet (streaming approach), so
        device_used = baseline + model_weights only.
        """
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
        """Overwrite in-GPU parameter values with the trainer's latest weights.

        This is the second step of model update, after reload_model() allocates the
        weight memory pool. reload_model() makes weight tensors addressable on GPU;
        load_weights() makes them correct by copying the trainer's new values in.

        Accepts a generator of (param_name, tensor) pairs so tensors arrive one at a
        time (streaming), keeping peak GPU memory low during broadcast.

        LoRA alias patch:
        When LoRA is active, vLLM wraps every target module at init time
        (e.g. gate_up_proj → gate_up_proj.base_layer). AutoWeightsLoader then looks
        for the original fused key (gate_up_proj) which no longer exists → KeyError.
        Fix: temporarily monkey-patch named_parameters() on affected submodules to
        also yield the unwrapped alias, then restore the original after load_weights.
        """
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
            # Build aliases stripping ".base_layer." — fail fast if both the
            # wrapped key and its canonical form exist in the same submodule.
            sub_aliases = {}
            for param_name, param_value in sub_params.items():
                if ".base_layer." not in param_name:
                    continue
                canonical = param_name.replace(".base_layer.", ".")
                if canonical in sub_params:
                    raise ValueError(
                        f"base_layer alias collision: both '{param_name}' and '{canonical}' "
                        f"exist in submodule parameters"
                    )
                sub_aliases[canonical] = param_value
            orig = submod.named_parameters

            # _make_aliased is a factory to avoid the classic Python late-binding closure bug.
            # Without it, a plain lambda/def inside the loop would capture `orig` and `sub_aliases`
            # by reference, so all patches would use the values from the last loop iteration.
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
        """Fully wake up this worker: model weights + KV cache.

        Idempotent: each sub-step is guarded by its own flag (weight_loaded, kv_cache_loaded).
        Use this instead of reload_model() when LoRA adapters will be registered immediately
        after, to avoid a later wake_up(["kv_cache"]) on a near-full GPU (OOM risk).
        """
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

    def offload_states(self, level: int):
        """Sleep this worker to free GPU memory, evicting LoRA state as part of the teardown.

        level=1: swap model weights to CPU, discard KV cache and LoRA tensors.
        level=2: destroy everything (weights, KV cache, LoRA tensors).

        LoRA eviction rationale:
        LoRA tensors use the default CuMem tag (not the "weights" tag), so sleep() at either
        level discards their GPU memory.  However, vLLM's Python-side LRUCacheWorkerLoRAManager
        still holds entries pointing at the now-freed GPU memory.  On the next add_lora call,
        vLLM finds the adapter "in cache" and skips reloading, then accesses the freed memory →
        CUDA error or silent corruption.
        Fix: always evict stale vLLM adapter registrations here so the next add_lora always
        takes the fresh-load path and applies the latest trained LoRA weights.

        Assert invariant: weight_loaded and kv_cache_loaded must be in sync — either both
        True (fully awake) or both False (already offloaded).  A mixed state indicates a bug.
        """
        assert (self.weight_loaded and self.kv_cache_loaded) or (not self.weight_loaded and not self.kv_cache_loaded)
        if not self.weight_loaded:
            logger.info("[vllm][offload] already offloaded, skip (level=%s)", level)
            # Safety-net cleanup: staged tensors survive only if staging happened but
            # custom_add_lora was never called (e.g. error mid-cycle, aborted training step).
            # On the normal path, build_request() already transferred ownership to a local
            # lora_request that goes out of scope after add_lora() returns, freeing the
            # tensors then.  This block handles the abnormal path to prevent GPU leaks.
            if getattr(self, "tensor_lora_manager", None) is not None and self.tensor_lora_manager.lora_params:
                staged_count = len(self.tensor_lora_manager.lora_params)
                self.tensor_lora_manager.lora_params = OrderedDict()
                logger.info("[vllm][offload] cleared staged LoRA tensors while already-offloaded: count=%s", staged_count)
            return
        # LoRA tensors use the default CuMem tag, not the "weights" tag, so sleep(level=1) discards them too.
        _desc = "destroy weights+KV+LoRA" if level == 2 else "swap weights to CPU, discard KV+LoRA"
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
        if getattr(self, "tensor_lora_manager", None) is not None and self.tensor_lora_manager.lora_params:
            staged_count = len(self.tensor_lora_manager.lora_params)
            self.tensor_lora_manager.lora_params = OrderedDict()
            logger.info("[vllm][offload] cleared staged LoRA tensors: count=%s", staged_count)
        # LoRA tensors use the default CuMem tag, not the "weights" tag.
        # sleep(level=1) therefore discards LoRA GPU memory just like level=2 does.
        # vLLM's Python-side LoRA cache (LRUCacheWorkerLoRAManager) still holds entries pointing at
        # now-freed GPU memory after either sleep level. On the next add_lora call, vLLM would take the
        # else-branch (adapter "in cache") and skip reloading → using freed memory → CUDA error / crash.
        # Fix: always evict stale vLLM adapter registrations after any sleep level, so the next add_lora
        # always takes the fresh-load path and newly trained LoRA weights are applied every cycle.
        if (
            getattr(self, "tensor_lora_manager", None) is not None
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
        logger.info("[vllm][offload] sleep(level=%s) done: GPU memory %s", level, "fully freed" if level == 2 else "weights on CPU, KV+LoRA discarded")

    def setup_collective_group(self, *args, **kwargs):
        """Initialise an NCCL collective group for model-weight broadcasting.

        Supports two call styles:

        1. comm_plan style (RLix selective model-update):
           Keyword args: comm_plan, backend, rank_in_cluster, timeout_s (optional).
           Calls get_dist_info_from_comm_plan to resolve which NCCL group this worker
           belongs to.  If group_rank is None, this worker is not part of the update
           group and the call returns immediately (skip — not an error).
           Ends with a dummy allreduce barrier to verify NCCL connectivity before any
           broadcast, catching misconfigured groups early.

        2. Legacy positional style (persistent broadcast group):
           Positional args: master_address, master_port, rank_offset, world_size,
           group_name, backend.  Optional kwarg: timeout_s.
           All workers are expected to participate.

        master_port is always cast to int to prevent type mismatch errors in collective init.
        """
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
                    f"[rlix][vllm][collective] setup_skip "
                    f"rank_in_cluster={rank_in_cluster} rank_in_worker={int(self.rank)}"
                )
                return

            group_name = comm_plan_args["group_name"]
            master_address = comm_plan_args["master_addr"]
            master_port = comm_plan_args["master_port"]
            world_size = int(len(comm_plan_args["tgt_devices"]) + 1)
            logger.info(
                f"[rlix][vllm][collective] setup_enter group_name={group_name} "
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
            # Dummy allreduce barrier: verifies NCCL connectivity immediately after init.
            # Detects misconfigured groups (wrong world_size, wrong ranks) before any real broadcast.
            collective.allreduce(torch.zeros(1, device=current_platform.device_type), group_name=group_name)
            logger.info(
                f"[rlix][vllm][collective] setup_exit group_name={group_name} "
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
            f"[rlix][vllm][collective] setup_enter group_name={group_name} "
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
            f"[rlix][vllm][collective] setup_exit group_name={group_name} "
            f"rank={group_rank} world_size={world_size}"
        )

    def destroy_collective_group(self, group_name: str):
        """Tear down an NCCL collective group and release its resources.

        Call after each model-update cycle completes to free NCCL communicator handles.
        A new group will be created on the next setup_collective_group call.

        Guard: partial-overlap IPC local ranks never called setup_collective_group, so
        collective.is_group_exist() returns False for them — skip destroy silently to
        avoid a KeyError in collective.destroy_collective_group (collective.py:65).
        """
        if not collective.is_group_exist(group_name):
            logger.info(
                f"[rlix][vllm][collective] destroy_skip_not_joined group_name={group_name} rank={self.rank}"
            )
            return
        logger.info(f"[rlix][vllm][collective] destroy_enter group_name={group_name}")
        collective.destroy_collective_group(group_name)
        logger.info(f"[rlix][vllm][collective] destroy_exit group_name={group_name}")

    def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False, *, broadcast_local_ranks=None):
        """Receive broadcasted tensors from rank 0. Base weights are written to GPU immediately;
        LoRA tensors are staged in tensor_lora_manager for later add_lora registration.

        is_lora=False (base model weights):
          Overwrites the model's in-GPU weight tensors directly, one at a time via a streaming
          generator. reload_model() is called first to ensure the weight memory pool exists,
          then each tensor is received and written in-place before the next buffer is allocated.
          Peak memory = model_weights + one_tensor_buffer.

        is_lora=True (LoRA adapter weights):
          Does NOT write to the model. Received tensors are staged in tensor_lora_manager
          and only applied to the vLLM engine later when custom_add_lora is called.
          LoRA tensors are small so all receives are issued async in a batch to let NCCL
          pipeline the transfers.
        """
        # [debug] Stage 1: log GPU memory before any receive buffer is allocated.
        # If another process still has model weights loaded, device_used will be much higher
        # than the expected idle baseline (~3.5 GiB for 6 idle processes on this test config).
        _free_bytes, _total_bytes = torch.cuda.mem_get_info()
        _device_used_gb = (_total_bytes - _free_bytes) / 1024**3
        _alloc_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(
            f"[rlix][vllm][broadcast] enter group_name={group_name} "
            f"num_tensors={len(names)} is_lora={int(bool(is_lora))} "
            f"[debug] device_used={_device_used_gb:.3f}GB allocated={_alloc_gb:.3f}GB "
            f"device_total={_total_bytes / 1024**3:.3f}GB"
        )

        # Partial-overlap support: ranks not in the mask never joined the NCCL group; skip early.
        if broadcast_local_ranks is not None and self.rank not in broadcast_local_ranks:
            return

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
            logger.info(f"[rlix][vllm][broadcast] exit group_name={group_name} mode=lora")
            return

        # Base weights: reload model FIRST, then stream one tensor at a time via a generator.
        # Peak memory = model_weights + one_tensor_buffer (not model + ALL buffers simultaneously).
        # Passing a generator to load_weights means LoRA patch logic runs ONCE for all tensors
        # (O(1) named_modules scan), vs calling load_weights 290 times (O(290) scans).
        self.reload_model()

        def _streaming_weights_gen():
            # One buffer at a time: allocate → blocking broadcast (wait for data) → yield →
            # del _buf before the loop advances to the next tensor.  This keeps peak memory at
            # model_weights + one_buffer rather than model_weights + all_buffers.
            for _name, _dtype, _shape in zip(names, dtypes, shapes):
                _target_dtype = _dtype if isinstance(_dtype, torch.dtype) else getattr(torch, _dtype)
                _buf = torch.empty(_shape, dtype=_target_dtype, device=self.device)
                # Blocking broadcast: receive this tensor before allocating the next buffer.
                collective.broadcast(tensor=_buf, src_rank=0, group_name=group_name, async_op=False)
                yield _name, _buf
                # Each parameter has a different shape (embedding, attention, MLP, bias, ...),
                # so the buffer cannot be reused — a new torch.empty is required each iteration.
                # del here ensures the old GPU block is returned to the CUDA caching allocator
                # before the next torch.empty runs.  Without it, both tensors would be alive
                # simultaneously at the loop boundary → peak = 2 buffers instead of 1.
                del _buf

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
        logger.info(f"[rlix][vllm][broadcast] exit group_name={group_name} mode=weights")

    def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False, *, ipc_local_ranks=None):
        """Deserialise a packed parameter bucket and apply it to the model or stage for LoRA.

        Counterpart to broadcast_parameter: same base/LoRA split, but tensors arrive
        pre-packed in a serialized bucket (CUDA-IPC or CPU-bytes) instead of via NCCL broadcast.

        is_lora=False (base model weights):
          Calls load_weights() to overwrite in-GPU parameter values with the unpacked tensors.
          No explicit reload_model() here — load_weights() handles that internally.

        is_lora=True (LoRA adapter weights):
          Stages each unpacked tensor in tensor_lora_manager.add_weight(), same as
          broadcast_parameter's LoRA path. Applied to vLLM later via custom_add_lora.

        The bucket is serialised as {"bucket": <torch.Tensor>, "tensors_meta": ...}
        via either CUDA IPC (ForkingPickler, default) or CPU byte serialization
        (standard pickle, model_update_transport="cpu_pickle"). pickle.loads() handles
        both formats — the rebuild functions are resolved by name during unpickling
        regardless of which pickler created the stream.

        named_params is materialised with list() because named_tensors_from_bucket returns a
        generator and generators can only be consumed once.
        """
        # Partial-overlap support: broadcast-only ranks receive weights via NCCL instead;
        # returning early here prevents double-application of the same weights.
        if ipc_local_ranks is not None and self.rank not in ipc_local_ranks:
            return
        # monkey_patch_torch_reductions is needed for CUDA IPC payloads (ensures GPU UUID
        # mapping during rebuild_cuda_tensor). Harmless for CPU pickle payloads.
        monkey_patch_torch_reductions()
        bucket_with_meta = pickle.loads(serialized_named_tensors[self.rank])
        bucket = bucket_with_meta["bucket"]
        # Some transport/offload paths deliver a CPU tensor; upload to GPU before slicing.
        if not getattr(bucket, "is_cuda", False):
            bucket = bucket.to(device=self.device).contiguous()
        named_params = list(named_tensors_from_bucket(bucket=bucket, tensors_meta=bucket_with_meta["tensors_meta"]))
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
    """vLLM V1 engine worker variant.

    The only V1-specific behaviour is calling patch_vllm_lora_manager() at init time.
    That patch fixes vLLM's LRUCacheWorkerLoRAManager so evicted adapter entries are
    properly removed from the Python-side cache, preventing stale-pointer CUDA errors on
    the next add_lora call after a sleep cycle.

    All other logic (LoRA registration, weight broadcasting, collective group management,
    offload/reload lifecycle) is inherited from WorkerBase.
    """

    def custom_init_worker(self, *args, **kwargs):
        super().custom_init_worker(*args, **kwargs)
        patch_vllm_lora_manager()

    # custom_add_lora is inherited from WorkerBase so all worker variants share adapter-name logic.
