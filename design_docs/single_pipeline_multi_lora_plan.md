# Plan: Port Multi-LoRA Standalone Pipeline to ROLL_schedrl

## Context
Port `AgenticMultiLoraPipeline` from `ROLL_multi_lora` into `ROLL_schedrl` so it runs
end-to-end as a standalone (non-SchedRL) pipeline. Strategy: selective copy of exactly
the LoRA-specific code blocks, not whole files (except one genuinely new file).

**Internal routing key migration**: `domain` is removed as a LoRA routing fallback.
Multi-adapter LoRA paths require `non_tensor_batch["lora_name"]` strictly; single-adapter
paths auto-fill `lora_name` if absent (via `ensure_lora_name_in_batch`). **Breaking
change for RLVR multi-LoRA callers that currently set only `domain`** — those paths must
update to inject `lora_name` before deployment. The agentic pipeline is fully safe: env
managers (Changes 4–8) inject `lora_name`, never `domain`.

Source baseline: `external/ROLL_multi_lora` current HEAD.
All edits are in: `external/ROLL_schedrl/`

---

## Files Touched (16 total, ordered by dependency)

| # | File (relative to `external/ROLL_schedrl/`) | Change |
|---|-----|--------|
| 1 | `roll/utils/lora_routing.py` | Add public `get_lora_name_array`; remove `domain` fallback from private helper; add `ensure_lora_name_in_batch` |
| 2 | `roll/configs/model_args.py` | Add `adapter_name` to `LoraArguments`; add 2 formal fields + full normalization block to `ModelArguments` |
| 3 | `roll/distributed/strategy/vllm_strategy.py` | Add module-level helper; add 7 methods; update `add_lora` signature; replace 2 routing blocks |
| 4–8 | `roll/pipeline/agentic/env_manager/{traj,step,step_concat,vl_traj,agent_native}_env_manager.py` | Add `lora_name` injection in `format_messages` + `formulate_rollouts` + `create_placeholder_rollout`; fix numpy import for step_concat |
| 9 | `roll/schedrl_adapter/multi_lora_pipeline.py` | Fix trained-adapter detection |
| 10 | `roll/pipeline/agentic/agentic_multi_lora_pipeline.py` | **New file** – whole-file copy + 2 revisions |
| 11 | `examples/qwen2.5-0.5B-agentic/n-agent_train_sokoban_multi_lora_async.yaml` | **New file** – adapted YAML (filename matches source `_async` suffix) |
| 12 | `roll/distributed/strategy/megatron_strategy.py` | Update LoRA docstrings: `domain` → `lora_name` |
| 13 | `roll/pipeline/base_worker.py` | Add `lora_name` auto-fill guard + `_broadcast_non_tensor_batch`; add `get_lora_id`/`list_loras`/`wait_loras_ready` wrappers; update docstring |
| 14 | `roll/pipeline/sft/sft_worker.py` | Add `lora_name` auto-fill guard + `_broadcast_non_tensor_batch`; update docstring |
| 15 | `roll/third_party/vllm/async_llm.py` | Add `get_lora_id` and `list_loras` async methods |
| 16 | `roll/third_party/vllm/worker.py` | Update `TensorLoraManager` to track adapter-name→ID; add `custom_get_lora_id`/`custom_list_loras` to `WorkerBase`; update `custom_add_lora` signature; remove `WorkerV1.custom_add_lora` (inherit from base) |

---

## Change 1 – `roll/utils/lora_routing.py`

Three edits to this file:

### 1a – Add public `get_lora_name_array` (strict lora_name-only)

Copy verbatim from `ROLL_multi_lora/roll/utils/lora_routing.py` function `get_lora_name_array`:
```python
def get_lora_name_array(non_tensor_batch: Mapping[str, Any]) -> np.ndarray:
    """Return lora_name array; requires non_tensor_batch["lora_name"] (no domain fallback)."""
    if "lora_name" not in non_tensor_batch:
        raise RuntimeError(
            'Missing `non_tensor_batch["lora_name"]` (required for multi-LoRA routing). '
            f"Available keys={sorted(non_tensor_batch.keys())}"
        )
    val = non_tensor_batch["lora_name"]
    if not isinstance(val, np.ndarray) or val.dtype != object:
        raise TypeError(
            f'Expected `non_tensor_batch["lora_name"]` to be np.ndarray(dtype=object), '
            f"got {type(val)} dtype={getattr(val, 'dtype', None)}"
        )
    return val
```

### 1b – Remove domain fallback from private `_get_lora_name_array`

**Remove** the `domain`-first loop body and replace with a direct `lora_name` check:

```python
# Before:
def _get_lora_name_array(non_tensor_batch: Mapping[str, Any]) -> np.ndarray:
    """... Checks ``domain`` first ..."""
    for key in ("domain", "lora_name"):
        if key in non_tensor_batch:
            ...
    raise RuntimeError('Missing `non_tensor_batch["domain"]` (or "lora_name") ...')

# After:
def _get_lora_name_array(non_tensor_batch: Mapping[str, Any]) -> np.ndarray:
    """Return per-sample lora_name array. Requires non_tensor_batch["lora_name"]."""
    return get_lora_name_array(non_tensor_batch)
```

This makes `_get_lora_name_array` a thin wrapper that delegates to the public strict version.
Any code calling `resolve_microbatch_lora_name` now requires `lora_name` key (no domain fallback).

### 1c – Add `ensure_lora_name_in_batch` helper (auto-fill policy)

Add this new function after `get_lora_name_array`. It implements the single-adapter
auto-fill policy for legacy producers that don't inject `lora_name`:

```python
def ensure_lora_name_in_batch(
    non_tensor_batch: dict,
    *,
    adapters: Mapping[str, Any] | None,
    batch_size: int | None = None,
) -> None:
    """Ensure non_tensor_batch["lora_name"] is set. Auto-fills for single-adapter configs.

    Policy:
    - If "lora_name" already present: no-op (validation happens at routing time).
    - If absent and adapters is None or empty: no-op (non-LoRA mode).
    - If absent and exactly one adapter: auto-fill with that adapter's key.
      batch_size inferred from existing dict values; callers may pass batch_size
      explicitly when non_tensor_batch may be empty.
    - If absent and multiple adapters: fail fast (producer must inject lora_name).
    """
    if "lora_name" in non_tensor_batch:
        return
    if not adapters:
        return
    if len(adapters) == 1:
        only_key = next(iter(adapters.keys()))
        # Infer batch size: use caller-supplied hint first; then first array in dict.
        if batch_size is None:
            if not non_tensor_batch:
                # Empty batch metadata and no size hint — fail fast loud.
                raise RuntimeError(
                    "ensure_lora_name_in_batch: cannot auto-fill lora_name in single-adapter "
                    "mode with empty non_tensor_batch and no batch_size provided. "
                    "Pass batch_size= from the tensor batch, or inject lora_name explicitly."
                )
            batch_size = len(next(iter(non_tensor_batch.values())))
        non_tensor_batch["lora_name"] = np.full(batch_size, only_key, dtype=object)
        return
    raise RuntimeError(
        "Missing non_tensor_batch['lora_name'] in multi-adapter mode. "
        f"Configured adapters: {sorted(adapters.keys())}. "
        "Producers must inject lora_name (e.g., via env_manager.format_messages)."
    )
```

`np` is already imported at the module level.

### 1d – Update module docstring

Replace the existing module docstring (lines 1–11):
```python
"""LoRA routing utilities for multi-LoRA microbatch dispatch.

The canonical routing key is ``non_tensor_batch["lora_name"]``.
Multi-adapter callers must inject this key before calling routing functions.
Single-adapter callers may rely on ``ensure_lora_name_in_batch`` auto-fill
(applied at vllm_strategy and worker boundaries before routing is reached).
"""
```

**Migration note**: After Change 1, `get_lora_name_array` / `resolve_microbatch_lora_name`
are strict — `domain`-only batches raise immediately. In single-adapter mode,
`ensure_lora_name_in_batch` (Change 1c) auto-fills `lora_name` before routing is reached,
so legacy single-adapter callers continue to work. Existing RLVR **multi-adapter** callers
that currently set only `domain` must inject `lora_name` before deploying to production.

---

## Change 2 – `roll/configs/model_args.py`

Three edits:

### 2a – Add `adapter_name` field to `LoraArguments`

ROLL_schedrl's `LoraArguments` is missing this field. Add before `additional_target`:
```python
adapter_name: str = field(
    default="default",
    metadata={"help": "The name of the adapter to be injected."},
)
```

### 2b – Add two formal fields to `ModelArguments`

Add after the existing fields, before `__post_init__`:
```python
# Track whether legacy lora_rank/lora_target fields were used (set in __post_init__).
_legacy_lora_fields_used: bool = field(default=False, repr=False)
# Map raw YAML adapter keys → canonical normalized keys (set in __post_init__).
adapter_name_map: dict[str, str] = field(default_factory=dict, init=False)
```

### 2c – Add normalization block to `ModelArguments.__post_init__`

Add import at top of file:
```python
from roll.utils.lora_routing import normalize_domain
```

Inside `__post_init__`, after the existing top-level field processing, add this block:

```python
# Part 1: Convert legacy single-LoRA fields (lora_rank/lora_target) to adapters dict.
# Ensures is_lora = (adapters is not None) works for both old and new configs.
if self.adapters is None and self.lora_rank is not None and self.lora_target is not None:
    self.adapters = {
        "default": LoraArguments(
            adapter_name="default",
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_target=self.lora_target,
        )
    }
    self._legacy_lora_fields_used = True

# Part 2: Normalize adapter keys to canonical lowercase; fail fast on name collisions.
# Collision suffixing (foo_2) is intentionally NOT used: suffixed adapters are unreachable
# via normalize_domain(tag), causing silent routing failures. Fail fast instead.
if self.adapters is not None:
    normalized_adapters: dict[str, LoraArguments] = {}
    raw_to_final: dict[str, str] = {}
    seen_bases: set[str] = set()
    for raw_adapter_name, adapter_config in self.adapters.items():
        base = normalize_domain(raw_adapter_name)
        if base in seen_bases:
            raise RuntimeError(
                f"Adapter name collision: '{raw_adapter_name}' normalizes to '{base}' "
                "which conflicts with an earlier adapter. Use distinct adapter names."
            )
        seen_bases.add(base)
        adapter_config.adapter_name = base
        # Part 3: Per-adapter field processing (lora_alpha default, lora_target split).
        if adapter_config.lora_alpha is None or adapter_config.lora_alpha <= 0:
            adapter_config.lora_alpha = adapter_config.lora_rank * 2
        if adapter_config.lora_target is not None and not any(
            c in adapter_config.lora_target for c in ["*", "$", "|", "("]
        ):
            adapter_config.lora_target = split_arg(adapter_config.lora_target)
        adapter_config.additional_target = split_arg(adapter_config.additional_target)
        normalized_adapters[base] = adapter_config
        raw_to_final[str(raw_adapter_name)] = base
    self.adapters = normalized_adapters
    self.adapter_name_map = raw_to_final
```

Source for Part 1 (legacy conversion): `ROLL_multi_lora/roll/configs/model_args.py` lines 147–157.
Source for Part 3 (field processing): `ROLL_multi_lora/roll/configs/model_args.py` lines 169–176.

**Migration note for collision fail-fast**: Configs with adapter names that normalize to
the same base (e.g., `foo` and `Foo`) will now raise at startup. Users must rename adapters
before upgrading. This is intentional: the previous suffix behavior (`foo_2`) silently
created unreachable adapters via tag-based routing.

---

## Change 3 – `roll/distributed/strategy/vllm_strategy.py`

### 3a – Add import

```python
from roll.utils.lora_routing import get_lora_name_array, resolve_microbatch_lora_name, ensure_lora_name_in_batch
```

### 3b – Fix `is_lora` and `max_loras` in `initialize` method

ROLL_schedrl's `initialize` directly sets `enable_prefix_caching` and `max_num_batched_tokens`
in `vllm_config.update(...)` at the top (no `has_*` guards). ROLL_multi_lora introduces `has_*`
boolean guards to avoid overriding user-set values. When copying the LoRA block, ALSO add the
three `has_*` definitions immediately after `vllm_config = copy.deepcopy(...)` (or at the start
of the method, before the existing `vllm_config.update(...)` block):

```python
has_enable_prefix_caching = "enable_prefix_caching" in vllm_config
has_enable_chunked_prefill = "enable_chunked_prefill" in vllm_config
has_max_num_batched_tokens = "max_num_batched_tokens" in vllm_config
```

These `has_*` booleans are referenced by the LoRA block below and MUST be defined first.

**Remove** (current single-LoRA block, identified by `lora_target is not None` check):
```python
self.is_lora = self.worker_config.model_args.lora_target is not None
if self.is_lora:
    lora_kwargs = {
        "enable_lora": True,
        "max_loras": 1,
        "max_lora_rank": self.worker_config.model_args.lora_rank,
    }
    vllm_config.update(lora_kwargs)
    vllm_config["load_format"] = "auto"
```

**Replace with** (copy verbatim from ROLL_multi_lora `initialize` LoRA block):
```python
self._vllm_max_loras = int(vllm_config.get("max_loras") or 0) if "max_loras" in vllm_config else None
self.is_lora = self.worker_config.model_args.adapters is not None
if self.is_lora:
    if not has_enable_prefix_caching:
        vllm_config["enable_prefix_caching"] = False
    if not has_enable_chunked_prefill:
        vllm_config["enable_chunked_prefill"] = False
    if not has_max_num_batched_tokens:
        max_model_len = int(vllm_config.get("max_model_len") or 0)
        vllm_config["max_num_batched_tokens"] = max(8192, max_model_len)
    max_loras_cfg = int(vllm_config.get("max_loras", 0) or 0)
    lora_kwargs = {
        "enable_lora": True,
        "max_loras": max(max_loras_cfg, len(self.worker_config.model_args.adapters) + 1),
        "max_lora_rank": max(a.lora_rank for a in self.worker_config.model_args.adapters.values()),
    }
    vllm_config.update(lora_kwargs)
    vllm_config["load_format"] = "auto"

if self.is_lora and vllm_config.get("load_format") == "dummy":
    raise RuntimeError(
        "vLLM LoRA mode requires real base model weights; got load_format='dummy'. "
        "Set vllm strategy_config.load_format='auto' or disable LoRA."
    )

# Adapter-ID APIs (get_lora_id, list_loras) are only available on the V1 engine path.
# Fail fast here rather than at runtime routing/verification.
if self.is_lora:
    vllm_use_v1 = int(os.environ.get("VLLM_USE_V1", "1"))
    if vllm_use_v1 != 1:
        raise RuntimeError(
            "LoRA mode in ROLL_schedrl requires VLLM_USE_V1=1. "
            "Non-v1 engine path does not expose adapter-id APIs required by multi-LoRA routing."
        )
```

**Why safe for legacy configs**: Change 2 converts `lora_rank/lora_target` to
`adapters={"default":...}` in `__post_init__`. So `adapters is not None` is True, and
`max_loras=max(0,1+1)=2`, `max_lora_rank=legacy_rank` — correct for single-adapter.

### 3c – Add missing helpers and methods

**Add module-level function BEFORE the class definition** (copy verbatim from
ROLL_multi_lora vllm_strategy.py function `_normalize_lora_int_ids_loaded`, which is
defined BEFORE `class VllmStrategy`):
```python
def _normalize_lora_int_ids_loaded(value) -> list[int]:
    # vLLM list_loras may return flat [id,...] or nested [[id,...],...] across ranks.
    if not isinstance(value, list) or not value:
        return []
    if isinstance(value[0], list):
        flat: list[int] = []
        for sub in value:
            if not isinstance(sub, list):
                continue
            for item in sub:
                if isinstance(item, int):
                    flat.append(item)
        return sorted(set(flat))
    return [item for item in value if isinstance(item, int)]
```

**Add to VllmStrategy class** (copy verbatim from ROLL_multi_lora, in this order):

1. `@staticmethod _should_debug_lora_routing()` — reads `ROLL_DEBUG_LORA_ROUTING` env var.
   Source: static method `_should_debug_lora_routing` in ROLL_multi_lora VllmStrategy.

2. `_log_lora_routing_context(self, *, where, input_ids, attention_mask, non_tensor_batch)` —
   debug helper; calls `_should_debug_lora_routing()`.
   Source: method `_log_lora_routing_context` in ROLL_multi_lora VllmStrategy.

3. `list_loras(self)` — wraps `model.list_loras()` via `_normalize_lora_int_ids_loaded`.
   Source: method `list_loras` in ROLL_multi_lora VllmStrategy.

4. `wait_loras_ready(self, adapter_names, timeout_s)` — polls until all adapters loaded.
   Source: method `wait_loras_ready` in ROLL_multi_lora VllmStrategy.

5. `get_lora_id(self, adapter_name)` — calls `model.get_lora_id`; normalizes list result.
   Source: method `get_lora_id` in ROLL_multi_lora VllmStrategy.

6. `_wait_for_lora_visible(self, *, adapter, lora_int_id, where)` — polls `list_loras`
   until the id appears; raises after 3 retries.
   Source: method `_wait_for_lora_visible` in ROLL_multi_lora VllmStrategy.

**Update existing `add_lora`** (currently `async def add_lora(self, peft_config)`):
```python
async def add_lora(self, adapter_name: str = "default", peft_config: dict = None):
    # Backward-compatible: FSDP2 single-LoRA path calls add_lora(peft_config=...) with no adapter_name.
    # Multi-LoRA via FSDP2 model_update is NOT supported; guard below catches it.
    if peft_config is None:
        raise RuntimeError("add_lora: peft_config must not be None")
    adapters = self.worker_config.model_args.adapters or {}
    if adapter_name not in adapters:
        raise RuntimeError(
            f"add_lora: unknown adapter_name={adapter_name!r}. "
            f"Valid adapters: {sorted(adapters.keys())}"
        )
    if adapter_name == "default" and len(adapters) > 1:
        raise RuntimeError(
            "add_lora called with adapter_name='default' in multi-LoRA mode. "
            "FSDP2 model_update path does not support multi-LoRA. "
            f"Configured adapters: {list(adapters.keys())}"
        )
    # Body copied verbatim from ROLL_multi_lora VllmStrategy.add_lora
    existing = await self.get_lora_id(adapter_name)
    if existing is not None:
        loaded = _normalize_lora_int_ids_loaded(await self.model.list_loras())
        if existing not in loaded:
            await self._wait_for_lora_visible(
                adapter=adapter_name,
                lora_int_id=existing,
                where="vllm_strategy.add_lora:existing_not_visible",
            )
        return
    peft_config["target_modules"] = sorted(adapters[adapter_name].lora_target)
    await self.model.add_lora(adapter_name, peft_config)
    lora_int_id = await self.get_lora_id(adapter_name)
    if lora_int_id is None:
        raise RuntimeError(f"LoRA adapter registration did not produce an id: adapter={adapter_name!r}")
    loaded = _normalize_lora_int_ids_loaded(await self.model.list_loras())
    if lora_int_id not in loaded:
        await self._wait_for_lora_visible(
            adapter=adapter_name,
            lora_int_id=lora_int_id,
            where="vllm_strategy.add_lora:not_visible_after_add",
        )
        # _wait_for_lora_visible either returns (adapter visible) or raises (timed out).
        # If we reach here, adapter became visible — done. Do NOT fall through to raise.
        return
```

**FSDP2 backward compat**: `fsdp2/model_update.py` calls `worker.add_lora.remote(peft_config=...)`.
With new signature: `adapter_name` defaults to `"default"`. Guard: `len(adapters)==1` for
single-LoRA → guard does NOT fire. No changes to `fsdp2/model_update.py`.

### 3d – Replace LoRA block in `_generate_standard`

Locate function `_generate_standard`. **Remove** the dummy single-lora block (identified by
`lora_request = LoRARequest(..., lora_path="dummy_lora_path")`).

**Insert `ensure_lora_name_in_batch` call** immediately before the LoRA routing block
(before the `if self.is_lora:` block being copied):
```python
# Auto-fill lora_name for single-adapter legacy producers; fail-fast for multi-adapter missing.
# NOTE: _generate_standard uses `batch.non_tensor_batch`, not a bare `non_tensor_batch` local.
# Pass batch_size from tensor batch so auto-fill works even when non_tensor_batch is empty.
if self.is_lora:
    ensure_lora_name_in_batch(
        batch.non_tensor_batch,
        adapters=self.worker_config.model_args.adapters,
        batch_size=batch.batch["input_ids"].size(0),
    )
```

**Replace with** the per-prompt routing block from ROLL_multi_lora function
`_generate_standard`. Uses `get_lora_name_array`, `_log_lora_routing_context`,
`_normalize_lora_int_ids_loaded`, `get_lora_id`. Copy verbatim.

### 3e – Replace LoRA block in `generate_request`

Locate function `generate_request`. **Remove** the dummy single-lora block (same
`lora_path="dummy_lora_path"` pattern).

**Insert `ensure_lora_name_in_batch` call** immediately before the LoRA routing block:
```python
# Pass batch_size so auto-fill works even when non_tensor_batch is empty.
if self.is_lora:
    ensure_lora_name_in_batch(
        data.non_tensor_batch,
        adapters=self.worker_config.model_args.adapters,
        batch_size=data.batch["input_ids"].size(0),
    )
```

**Replace ONLY the LoRA routing block** from ROLL_multi_lora function `generate_request`.
The LoRA block starts at `lora_request = None` / `if self.is_lora:` (ROLL_multi_lora line ~565).
Uses `resolve_microbatch_lora_name`, `get_lora_id`, `_normalize_lora_int_ids_loaded`,
`_log_lora_routing_context`, `_wait_for_lora_visible`. Copy verbatim.

**Critical: do NOT copy the vocab validation block** (ROLL_multi_lora lines ~524–564) that
precedes the LoRA block in ROLL_multi_lora's `generate_request`. That block references
`self._allowed_token_ids` (direct attribute access) and `self._model_vocab_size` — neither
is initialized in ROLL_schedrl's `VllmStrategy.__init__`. Copying it verbatim causes an
`AttributeError` (`_allowed_token_ids`) or a guaranteed `RuntimeError` (`_model_vocab_size`
is None and the code raises on that). Only replace the dummy LoRA block; leave the rest of
ROLL_schedrl's `generate_request` function body unchanged.

**Also: do NOT copy any logging context** that references `_vllm_max_num_batched_tokens` or
`_vllm_max_num_seqs` from ROLL_multi_lora — those attributes are initialized in ROLL_multi_lora's
`initialize` but not in ROLL_schedrl's.

After Change 1, `resolve_microbatch_lora_name` in ROLL_schedrl calls `_get_lora_name_array`
which now delegates to `get_lora_name_array` (strict lora_name-only). The copied LoRA block
is therefore strict by default — no additional precondition needed.

---

## Changes 4–8 – Env managers (5 files)

Each file gets two sets of changes: injection in `format_messages` (inference) and
injection in `formulate_rollouts` (training). Both paths must carry `lora_name`.

### Imports

**For all 5 files** — add to existing imports:
```python
from roll.utils.lora_routing import normalize_domain
```

**For `step_concat_env_manager.py` only** — also add (file has NO numpy import currently):
```python
import numpy as np
```

### format_messages injection

**Inject block** immediately before `return lm_input` in `format_messages`.

`DataProto.non_tensor_batch` defaults to `{}` (not `None`), so no `None` guard is needed.

```python
# Inject lora_name so vLLM routes each request to the correct adapter.
if self.pipeline_config.actor_infer.model_args.adapters is not None:
    adapters = self.pipeline_config.actor_infer.model_args.adapters
    if len(adapters) == 1:
        # Single adapter: inject the sole adapter key directly; no tag validation.
        # Tags like "SimpleSokoban" won't match adapter "default", so avoid validation.
        lm_input.non_tensor_batch["lora_name"] = np.array(
            [next(iter(adapters.keys()))], dtype=object
        )
    else:
        # Multi-adapter: validate tag → adapter name; fail fast on unknown tag.
        normalized = normalize_domain(self.rollout_cache.tag)
        valid_adapters = set(adapters.keys())
        if normalized not in valid_adapters:
            raise RuntimeError(
                f"Env tag {self.rollout_cache.tag!r} normalizes to {normalized!r} "
                f"which is not in configured adapters: {sorted(valid_adapters)}"
            )
        lm_input.non_tensor_batch["lora_name"] = np.array([normalized], dtype=object)
```

`np` is already imported in traj/vl_traj/agent_native. Import added above for step_concat.

**Anchor per file — insert in `format_messages` before its final `return lm_input`:**

| File | Note |
|------|------|
| `traj_env_manager.py` | Multiple `return lm_input` exist; insert only in `format_messages` |
| `step_env_manager.py` | Standard injection (non_tensor_batch defaults to `{}`) |
| `step_concat_env_manager.py` | Standard injection; numpy import also added |
| `vl_traj_env_manager.py` | Multiple `return lm_input` exist; insert only in `format_messages` |
| `agent_native_env_manager.py` | Standard injection |

### formulate_rollouts injection

Training batches are assembled in `formulate_rollouts`. Each env manager sets `tags` but
NOT `lora_name` in `non_tensor_batch`. The training path (`train_step_lora`) requires
`lora_name`. Inject alongside `tags` in each file:

**`step_env_manager.py`** — `formulate_rollouts` creates `DataProto` with a
`non_tensor_batch` dict at line ~114. Insert this block immediately before the
`DataProto(...)` call, then use `_lora_name` in the dict.

Same single-vs-multi split as `format_messages`:
```python
# Compute lora_name to inject alongside tags in training batch.
if self.pipeline_config.actor_train.model_args.adapters is not None:
    adapters = self.pipeline_config.actor_train.model_args.adapters
    if len(adapters) == 1:
        # Single adapter: use the sole adapter key; no tag validation.
        _lora_name = next(iter(adapters.keys()))
    else:
        # Multi-adapter: validate tag → adapter.
        _lora_name = normalize_domain(self.rollout_cache.tag)
        _valid = set(adapters.keys())
        if _lora_name not in _valid:
            raise RuntimeError(
                f"Env tag {self.rollout_cache.tag!r} normalizes to {_lora_name!r} "
                f"which is not in configured adapters: {sorted(_valid)}"
            )
else:
    _lora_name = self.rollout_cache.tag
# Then include _lora_name in the non_tensor_batch dict:
non_tensor_batch={..., "tags": ..., "lora_name": np.array([_lora_name], dtype=object), ...}
```

**`traj_env_manager.py`** — `formulate_rollouts` calls `lm_input.non_tensor_batch.update({...})`
at line ~410. Apply the same inline block before the `.update()` call:
```python
# (Same _lora_name computation block as step_env_manager.py above)
lm_input.non_tensor_batch.update({..., "lora_name": np.array([_lora_name], dtype=object)})
```

**`vl_traj_env_manager.py`** — same pattern as `traj_env_manager.py` (`.update()` path)

**`agent_native_env_manager.py`** — same inline block as `step_env_manager.py` (dict constructor)

**`step_concat_env_manager.py`** — inherits `formulate_rollouts` from `StepEnvManager`;
no change needed here (covered by the `step_env_manager.py` fix).

### create_placeholder_rollout injection (agent_native only)

Only `agent_native_env_manager.py` has `create_placeholder_rollout` (line ~437). This
failure-mode path builds its own `non_tensor_batch` dict (line ~465) with `tags` but no
`lora_name`. It must also inject `lora_name` to avoid routing failures on failure rollouts.

Use this exact placement (two-step sequence, not inline control flow inside dict literal):

```python
# Step 1: compute _lora_name BEFORE constructing non_tensor_batch.
if self.pipeline_config.actor_train.model_args.adapters is not None:
    adapters = self.pipeline_config.actor_train.model_args.adapters
    if len(adapters) == 1:
        _lora_name = next(iter(adapters.keys()))
    else:
        _lora_name = normalize_domain(self.env_config['tag'])
        _valid = set(adapters.keys())
        if _lora_name not in _valid:
            raise RuntimeError(
                f"Env tag {self.env_config['tag']!r} normalizes to {_lora_name!r} "
                f"which is not in configured adapters: {sorted(_valid)}"
            )
else:
    _lora_name = self.env_config['tag']

# Step 2: include the computed value in dict construction.
lm_input.non_tensor_batch = {
    ...,
    "tags": np.array([self.env_config['tag']], dtype=object),
    "lora_name": np.array([_lora_name], dtype=object),
    ...,
}
```


---

## Change 9 – `roll/schedrl_adapter/multi_lora_pipeline.py`

**Targeted fix** – trained-adapter detection inside `run()`.

`domain` here is overloaded-as-adapter (maps through `self._tag_to_adapter`) — this is the
adapter-resolution context that must change to `lora_name`. (Dataset `domain` in schedulers
is a different concept and stays unchanged.)

Locate and **remove** this pattern (uses `domain` as adapter key; env_managers never set it;
also references `adapters` variable undefined in `run()` scope):
```python
domain_tags = set(batch.non_tensor_batch.get("domain", []))
trained_adapters = list(dict.fromkeys(
    self._tag_to_adapter[tag]
    for tag in domain_tags
    if tag in self._tag_to_adapter
))
```

**Replace with** (fail-fast on missing or unrecognized `lora_name` — no silent no-op):
```python
# lora_name values are canonical adapter names (injected by env_manager via normalize_domain).
# Fail fast: missing lora_name or no recognized adapters is a contract violation.
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
```

`np` is NOT needed here (direct key access; no empty-array default).

---

## Change 10 – New file `roll/pipeline/agentic/agentic_multi_lora_pipeline.py`

**Whole-file copy** from `ROLL_multi_lora` — this file does not exist in ROLL_schedrl.
Then two revisions:

**Revision A** – Harden `partial_gpu_mode` to hardcoded invariant.

Locate the `partial_gpu_mode` guard inside `__init__` (not `initialize_pipeline`):
```python
# Original (from ROLL_multi_lora, inside __init__):
if not self.pipeline_config.partial_gpu_mode:
    raise RuntimeError(
        "AgenticMultiLoraPipeline requires partial_gpu_mode=true. ..."
    )
self.partial_gpu_mode = self._validate_partial_gpu_config()
```

Replace with (validate only if explicitly set to False, otherwise default to True):
```python
# Hardcoded constraint: partial_gpu_mode must be true.
# Only validate if the config attribute exists and was explicitly set to False.
if hasattr(self.pipeline_config, "partial_gpu_mode") and self.pipeline_config.partial_gpu_mode is False:
    raise RuntimeError(
        "AgenticMultiLoraPipeline: partial_gpu_mode must be true (hardcoded constraint)."
    )
self.partial_gpu_mode = self._validate_partial_gpu_config()
```

`sleep_level` check is already correct (defaults to `1` if absent, raises otherwise).

**Revision B** – Add comment on normalization contract in `run()`:
```python
# Adapter keys in model_args.adapters are canonical lowercase (normalized in __post_init__).
tag_to_adapter = {tag: normalize_domain(tag) for tag in self.rollout_schedulers.keys()}
```

---

## Change 11 – New YAML `examples/qwen2.5-0.5B-agentic/n-agent_train_sokoban_multi_lora_async.yaml`

Adapted from ROLL_multi_lora source YAML. Key differences:

| Field | Source YAML | Target YAML |
|---|---|---|
| `lora_naming` block | present | **removed** |
| Adapter keys | `SimpleSokoban`, `LargerSokoban` | **unchanged** (normalized in `__post_init__`) |
| `tags` | `[SimpleSokoban, LargerSokoban]` | **unchanged** (normalized at runtime) |
| `sleep_level` | absent | **absent** (hardcoded) |
| `partial_gpu_mode` | absent | **absent** (hardcoded) |
| `_NEBULA_USER_ID` | present | **removed** |
| `ROLL_DEBUG_LORA_ROUTING` | present | kept |
| `pipeline_cls` | `...AgenticMultiLoraPipeline` | same |

---

## Change 12 – `roll/distributed/strategy/megatron_strategy.py` (docstring)

**Docstring-only change** in `train_step_lora` and `inner_forward_step`. After Change 1,
only `lora_name` is valid — `domain` is no longer a LoRA routing key.

Locate the docstring block that says:
```
"""Adapter routing uses ``non_tensor_batch["domain"]`` (ROLL_schedrl
convention) or ``non_tensor_batch["lora_name"]`` as fallback."""
```

Replace with:
```
"""Adapter routing requires ``non_tensor_batch["lora_name"]`` (canonical key).
The legacy ``domain`` fallback has been removed; producers must inject ``lora_name``."""
```

Apply the same update to `inner_forward_step` if it contains similar wording.

**Scope note on `domain` in schedulers**: The scheduler files
(`async_generate_scheduler.py:460`, `generate_scheduler.py:1226`,
`user_defined_rollout_loop.py:37`) read `domain` for **dataset routing** (which reward
function to call, which domain's data) — an entirely different concept from LoRA adapter
routing. These callers never call `_get_lora_name_array` or `resolve_microbatch_lora_name`.
Change 1 does NOT affect them. No changes needed to scheduler files.

## Change 13 – `roll/pipeline/base_worker.py` (guard + docstring)

Two edits to `train_step_lora`:

**Add import** at top of file:
```python
from roll.utils.lora_routing import ensure_lora_name_in_batch
```

**Docstring update** (change `domain` → `lora_name`):
```python
# Before:
"""Multi-LoRA training step.
Routes per-adapter microbatches via ``non_tensor_batch["domain"]`` to ..."""

# After:
"""Multi-LoRA training step.
Routes per-adapter microbatches via ``non_tensor_batch["lora_name"]`` to ..."""
```

**Add auto-fill guard** as the first executable line of the method body, before `data.to(...)`:
```python
# Auto-fill lora_name for single-adapter legacy producers; fail fast for multi-adapter missing.
# DataProto.non_tensor_batch defaults to {} so no None init needed.
# Pass batch_size from tensor batch so auto-fill works even when non_tensor_batch is empty.
_bs = data.batch.batch_size[0] if data.batch is not None else None
ensure_lora_name_in_batch(
    data.non_tensor_batch,
    adapters=self.worker_config.model_args.adapters,
    batch_size=_bs,
)
# Ensure lora_name is broadcast to all Megatron ranks (no-op for non-Megatron strategies).
# DataProto.meta_info defaults to {} but guard for explicit None to be safe.
if self.worker_config.model_args.adapters is not None:
    if data.meta_info is None:
        data.meta_info = {}
    data.meta_info["_broadcast_non_tensor_batch"] = True
```

**Also add these 3 worker wrapper methods** (copy the `add_lora` wrapper pattern at line ~484):
```python
async def get_lora_id(self, adapter_name: str):
    """Delegate to VllmStrategy.get_lora_id; called by multi_lora_pipeline verify step."""
    return await self.strategy.get_lora_id(adapter_name)

async def list_loras(self):
    """Delegate to VllmStrategy.list_loras; called by multi_lora_pipeline verify step."""
    return await self.strategy.list_loras()

async def wait_loras_ready(self, adapter_names: list[str], timeout_s: float):
    """Delegate to VllmStrategy.wait_loras_ready; called by multi_lora_pipeline verify step."""
    await self.strategy.wait_loras_ready(adapter_names, timeout_s=timeout_s)
```

Do NOT change any other `_broadcast_non_tensor_batch` logic beyond this addition.

## Change 14 – `roll/pipeline/sft/sft_worker.py` (guard + docstring)

Two edits to `train_step_lora`:

**Add import** at top of file:
```python
from roll.utils.lora_routing import ensure_lora_name_in_batch
```

**Docstring update** (change `domain` → `lora_name`):
```python
# Before:
"""... The microbatch must carry ``non_tensor_batch["domain"]`` (or
``"lora_name"``) to identify which adapter owns the batch."""

# After:
"""... The microbatch must carry ``non_tensor_batch["lora_name"]``
to identify which adapter owns the batch."""
```

**Add auto-fill guard** immediately after `if data.meta_info is None:` block and before
the `data = self.strategy.get_data_input(data)` call:
```python
# Auto-fill lora_name for single-adapter legacy producers; fail fast for multi-adapter missing.
# DataProto.non_tensor_batch defaults to {} so no None init needed.
# Pass batch_size from tensor batch so auto-fill works even when non_tensor_batch is empty.
_bs = data.batch.batch_size[0] if data.batch is not None else None
ensure_lora_name_in_batch(
    data.non_tensor_batch,
    adapters=self.worker_config.model_args.adapters,
    batch_size=_bs,
)
# Ensure lora_name is broadcast to all Megatron ranks (no-op for non-Megatron strategies).
# DataProto.meta_info defaults to {} but guard for explicit None to be safe.
if self.worker_config.model_args.adapters is not None:
    if data.meta_info is None:
        data.meta_info = {}
    data.meta_info["_broadcast_non_tensor_batch"] = True
```

Do NOT change any other `_broadcast_non_tensor_batch` logic beyond this addition.

---

## Change 15 – `roll/third_party/vllm/async_llm.py`

**Add 2 methods** after the existing `add_lora` method. Copy verbatim from
`ROLL_multi_lora/roll/third_party/vllm/async_llm.py` (lines 74–78):

```python
async def get_lora_id(self, *args, **kwargs):
    return await self.engine_core.collective_rpc_async(method="custom_get_lora_id", args=args, kwargs=kwargs)

async def list_loras(self) -> list[int]:
    return await self.engine_core.collective_rpc_async(method="custom_list_loras")
```

These wrap the worker-level `custom_get_lora_id` / `custom_list_loras` methods added in Change 16.

---

## Change 16 – `roll/third_party/vllm/worker.py`

Four edits in dependency order:

### 16a – `TensorLoraManager.__init__`: add `_lora_names` tracking dict

Add `self._lora_names: dict[str, int] = {}` after existing fields:
```python
def __init__(self):
    self.lora_params = OrderedDict()
    self.add_lora_count = 0
    self._lora_names: dict[str, int] = {}  # adapter_name → lora_int_id
```

### 16b – `TensorLoraManager`: add `get_lora_id` method

Insert after `__init__`:
```python
def get_lora_id(self, adapter_name: str) -> int | None:
    """Return registered lora_int_id for adapter_name, or None if not registered."""
    return self._lora_names.get(adapter_name, None)
```

### 16c – `TensorLoraManager.build_request`: update signature + ID tracking

**Old signature**: `build_request(self, peft_config: dict) -> TensorLoRARequest`
**New signature**: `build_request(self, adapter_name: str, peft_config: dict) -> TensorLoRARequest`

Changes inside method:
- Include `adapter_name` in hash to distinguish adapters: add `peft_config["adapter_name"] = adapter_name` before `peft_config_str`
- Use `lora_name=adapter_name` in `TensorLoRARequest(...)` (not the old `f"{lora_int_id}"`)
- Track: `self._lora_names[adapter_name] = lora_int_id` before building the request object

Full updated body:
```python
def build_request(self, adapter_name: str, peft_config: dict) -> TensorLoRARequest:
    """Generate a unique LoRA ID based on adapter name + PEFT config."""
    self.add_lora_count += 1
    peft_config["adapter_name"] = adapter_name       # include adapter_name in hash
    peft_config["add_lora_count"] = self.add_lora_count
    peft_config_str = json.dumps(peft_config, sort_keys=True)
    hash_obj = hashlib.sha256(peft_config_str.encode("utf-8"))
    hex_dig = hash_obj.hexdigest()
    lora_int_id = int(hex_dig, 16) % 0x7FFFFFFF
    self._lora_names[adapter_name] = lora_int_id     # track name → id

    lora_request = TensorLoRARequest(
        lora_name=adapter_name,                      # use adapter_name, not str(id)
        lora_int_id=lora_int_id,
        lora_path="dummy_lora_path",
        peft_config=peft_config,
        lora_tensors=self.lora_params,
    )
    del self.lora_params
    self.lora_params = OrderedDict()
    return lora_request
```

### 16d – `WorkerBase`: add 3 methods; update `custom_add_lora` (from `WorkerV1` → `WorkerBase`)

**Move** full `custom_add_lora` implementation from `WorkerV1` to `WorkerBase` with updated
adapter-name-aware signature (copy body from ROLL_multi_lora `WorkerBase.custom_add_lora`):
```python
def custom_add_lora(self, adapter_name: str, peft_config: dict) -> bool:
    """Register a LoRA adapter by name. Called via collective_rpc_async."""
    lora_request = self.tensor_lora_manager.build_request(adapter_name, peft_config)
    self.reload_model()
    add_lora = getattr(getattr(self, "model_runner", None), "add_lora", None)
    if not callable(add_lora):
        raise NotImplementedError(
            "vLLM worker does not expose model_runner.add_lora; "
            "ensure the configured vLLM version supports runtime LoRA registration."
        )
    try:
        ok = add_lora(lora_request)
    except Exception:
        self.tensor_lora_manager._lora_names.pop(adapter_name, None)
        raise
    if ok is False:
        self.tensor_lora_manager._lora_names.pop(adapter_name, None)
        raise RuntimeError(f"vLLM add_lora returned False for adapter={adapter_name!r}")
    return True

def custom_list_loras(self) -> list[int]:
    """Return lora_int_ids for all registered adapters."""
    return sorted(set(self.tensor_lora_manager._lora_names.values()))

def custom_get_lora_id(self, adapter_name: str) -> int | None:
    """Return lora_int_id for adapter_name, or None if not registered."""
    return self.tensor_lora_manager.get_lora_id(adapter_name)
```

### 16e – `WorkerV1`: remove `custom_add_lora` override (inherit from `WorkerBase`)

**Remove** the existing `WorkerV1.custom_add_lora` method:
```python
# REMOVE THIS:
def custom_add_lora(self, peft_config) -> bool:
    lora_request = self.tensor_lora_manager.build_request(peft_config)
    super().reload_model()
    return self.model_runner.add_lora(lora_request)
```

`WorkerV1` now inherits `custom_add_lora(adapter_name, peft_config)` from `WorkerBase`.
`WorkerV1.custom_init_worker` already calls `patch_vllm_lora_manager()` — no change there.

---

## Normalization Contract

**Multi-adapter case (e.g. tags: [SimpleSokoban, LargerSokoban]):**
```
YAML: adapters: {SimpleSokoban: ..., LargerSokoban: ...}
         ↓  ModelArguments.__post_init__  (Change 2)
Config:  adapters.keys() = {"simplesokoban", "largersokoban"}

env_manager.format_messages  (Changes 4–8, multi-adapter branch):
    normalize_domain("SimpleSokoban") → "simplesokoban"  ∈ valid_adapters ✓
    lora_name = "simplesokoban"
non_tensor_batch["lora_name"] = np.array(["simplesokoban"], dtype=object)
         ↓  vllm_strategy._generate_standard  (Change 3d)
         get_lora_name_array → per-prompt LoRARequest(lora_name="simplesokoban") ✓
         ↓  vllm_strategy.generate_request  (Change 3e)
         resolve_microbatch_lora_name → strict lora_name ✓
vLLM routes to "simplesokoban" LoRA adapter
```

**Single-adapter case (e.g. legacy lora_rank + tag SimpleSokoban):**
```
YAML: lora_rank=8, lora_target=q_proj → adapters: {"default": ...}  (Change 2)
Config:  adapters.keys() = {"default"}

env_manager.format_messages  (Changes 4–8, single-adapter branch):
    lora_name = "default"  (sole adapter key, no tag normalization)
non_tensor_batch["lora_name"] = np.array(["default"], dtype=object)
         ↓  vllm_strategy routing: get_lora_name_array → LoRARequest(lora_name="default") ✓
vLLM routes to "default" LoRA adapter  (no regression for legacy single-LoRA configs)
```

---

## Verification

**Static checks (run from repo root):**
```bash
# 1. Public get_lora_name_array and ensure_lora_name_in_batch exist
grep "^def get_lora_name_array\|^def ensure_lora_name_in_batch" \
    external/ROLL_schedrl/roll/utils/lora_routing.py

# 2. Domain fallback removed from _get_lora_name_array
grep -A5 "def _get_lora_name_array" external/ROLL_schedrl/roll/utils/lora_routing.py
# Expected: no "domain" key reference in the body

# 3. vllm_strategy uses adapters-based is_lora
grep "adapters is not None" external/ROLL_schedrl/roll/distributed/strategy/vllm_strategy.py

# 4. module-level _normalize_lora_int_ids_loaded defined before class
grep -n "_normalize_lora_int_ids_loaded\|^class VllmStrategy" \
    external/ROLL_schedrl/roll/distributed/strategy/vllm_strategy.py
# Expected: _normalize_lora_int_ids_loaded line# < class VllmStrategy line#

# 5. No lora_naming/ensure_lora_name in agentic pipeline
grep -r "lora_naming\|ensure_lora_name" external/ROLL_schedrl/roll/pipeline/agentic/

# 6. vLLM plumbing: get_lora_id and list_loras in async_llm; custom_* in worker
grep "def get_lora_id\|def list_loras" external/ROLL_schedrl/roll/third_party/vllm/async_llm.py
grep "def custom_get_lora_id\|def custom_list_loras\|def custom_add_lora" \
    external/ROLL_schedrl/roll/third_party/vllm/worker.py
# Expected: all 3 present; custom_add_lora signature includes adapter_name

# 7. base_worker has get_lora_id, list_loras, wait_loras_ready wrappers
grep "def get_lora_id\|def list_loras\|def wait_loras_ready" \
    external/ROLL_schedrl/roll/pipeline/base_worker.py
# Expected: all 3 present

# 8. TensorLoraManager tracks _lora_names; no WorkerV1.custom_add_lora override
grep "_lora_names" external/ROLL_schedrl/roll/third_party/vllm/worker.py
grep "class WorkerV1" -A 20 external/ROLL_schedrl/roll/third_party/vllm/worker.py
# Expected: _lora_names present; WorkerV1 has no custom_add_lora
```

**Runtime smoke (cd external/ROLL_schedrl first):**
```bash
# 1. New imports resolve
python -c "
from roll.utils.lora_routing import get_lora_name_array, resolve_microbatch_lora_name, normalize_domain
from roll.pipeline.agentic.agentic_multi_lora_pipeline import AgenticMultiLoraPipeline
print('imports ok')
"

# 2. adapter_name field exists in LoraArguments
python -c "
import dataclasses
from roll.configs.model_args import LoraArguments
names = [f.name for f in dataclasses.fields(LoraArguments)]
assert 'adapter_name' in names, f'adapter_name missing: {names}'
print('LoraArguments.adapter_name ok')
"

# 3. Legacy single-LoRA config converts to adapters
python -c "
from roll.configs.model_args import ModelArguments
m = ModelArguments(model_name_or_path='x', lora_rank=8, lora_target='q_proj,v_proj')
assert m.adapters is not None, 'Legacy lora_rank/lora_target not converted to adapters'
assert 'default' in m.adapters, f'Expected default adapter: {list(m.adapters.keys())}'
assert m._legacy_lora_fields_used, 'Expected _legacy_lora_fields_used=True'
print('Legacy single-LoRA conversion ok')
"

# 4. Multi-adapter normalization ok; collision raises
python -c "
from roll.configs.model_args import ModelArguments, LoraArguments
m = ModelArguments(
    model_name_or_path='x',
    adapters={'SimpleSokoban': LoraArguments(lora_rank=8, lora_target='q_proj'),
              'LargerSokoban': LoraArguments(lora_rank=8, lora_target='q_proj')}
)
assert set(m.adapters.keys()) == {'simplesokoban', 'largersokoban'}
assert m.adapter_name_map == {'SimpleSokoban': 'simplesokoban', 'LargerSokoban': 'largersokoban'}
print('Multi-adapter normalization ok')
try:
    ModelArguments(model_name_or_path='x',
        adapters={'foo': LoraArguments(lora_rank=8, lora_target='q_proj'),
                  'FOO': LoraArguments(lora_rank=8, lora_target='q_proj')})
    assert False, 'Expected RuntimeError on collision'
except RuntimeError:
    print('Collision fail-fast ok')
"

# 5. strict lora_name routing: domain key is no longer accepted
python -c "
import numpy as np
from roll.utils.lora_routing import get_lora_name_array, resolve_microbatch_lora_name

# Positive: lora_name present
batch_ok = {'lora_name': np.array(['simplesokoban'], dtype=object)}
arr = get_lora_name_array(batch_ok)
assert arr[0] == 'simplesokoban'

# Negative: domain only (no lora_name) must raise
batch_domain_only = {'domain': np.array(['simplesokoban'], dtype=object)}
try:
    get_lora_name_array(batch_domain_only)
    assert False, 'Expected RuntimeError for domain-only batch'
except RuntimeError:
    pass
try:
    resolve_microbatch_lora_name(batch_domain_only)
    assert False, 'Expected RuntimeError for domain-only batch in resolve_microbatch'
except RuntimeError:
    pass
print('Strict lora_name routing ok (domain-only raises)')
"

# 6. add_lora backward-compat signature
python -c "
import inspect
from roll.distributed.strategy.vllm_strategy import VllmStrategy
sig = inspect.signature(VllmStrategy.add_lora)
params = dict(sig.parameters)
assert params['adapter_name'].default == 'default'
assert params['peft_config'].default is None
print('add_lora backward-compat signature ok')
"
```

**Key runtime signals to confirm during actual training:**
1. `actor_train.model_args.adapters.keys()` are lowercase after config init.
2. `non_tensor_batch["lora_name"]` present after each `format_messages` call.
3. vLLM `is_lora=True` and `max_loras >= 3` when 2 adapters configured.
4. `train_step_lora` microbatches have `lora_name` key set.
5. SchedRL control-plane `trained_adapters` is non-empty after first training step.

**Scope boundary checks (static):**
```bash
# generate_request LoRA block does NOT reference _allowed_token_ids or _model_vocab_size
grep "_allowed_token_ids\|_model_vocab_size" \
    external/ROLL_schedrl/roll/distributed/strategy/vllm_strategy.py
# Expected: zero matches (these attrs are not initialized in ROLL_schedrl VllmStrategy.__init__)

# train_step_lora guards are present in both worker files
grep -A5 "train_step_lora" \
    external/ROLL_schedrl/roll/pipeline/base_worker.py \
    external/ROLL_schedrl/roll/pipeline/sft/sft_worker.py | grep "lora_name"
# Expected: matches showing the fail-fast guard in each file
```

---

## Post-Smoke Fix Updates (2026-02-22)

The following fixes were applied after initial porting to make the smoke test pass for:
`examples/qwen2.5-0.5B-agentic/n-agent_train_sokoban_multi_lora_async.yaml`

### 1) vLLM KV-cache startup safety

File:
- `external/ROLL_schedrl/examples/qwen2.5-0.5B-agentic/n-agent_train_sokoban_multi_lora_async.yaml`

Change:
- `actor_infer.strategy_args.strategy_config.gpu_memory_utilization` changed from `0.65` to `0.8`.

Reason:
- Prevents vLLM startup failure (`No available memory for the cache blocks`) in the tested 2-worker async setup.

### 2) GroupQueueManager actor-name collision fix

File:
- `external/ROLL_schedrl/roll/distributed/scheduler/rollout_scheduler.py`

Change:
- Group queue actor name now includes env manager name:
  - with pipeline id: `..._group_queue_manager_{env_name}_{mode}`
  - without pipeline id: `GroupQueueManager-{env_name}-{mode}`

Reason:
- Multiple per-tag train rollout schedulers were creating the same actor name and failing on duplicate registration.

### 3) Missing RolloutScheduler wrapper APIs for partial-GPU flow

File:
- `external/ROLL_schedrl/roll/distributed/scheduler/rollout_scheduler.py`

Changes:
- Added delegating async methods:
  - `resume()`
  - `get_inflight_counts(dp_ranks)`
  - `get_offload_ranks_for_target_gpus(target_gpus)`
  - `offload_dp_ranks(dp_ranks)`

Reason:
- `AgenticMultiLoraPipeline` calls these methods on rollout schedulers during shrink/expand; missing methods caused `ActorHandle` attribute errors.

### 4) Missing RequestScheduler methods used by shrink/expand barrier

File:
- `external/ROLL_schedrl/roll/distributed/scheduler/generate_scheduler.py`

Changes:
- Added:
  - `get_inflight_counts(dp_ranks)`
  - `get_offload_ranks_for_target_gpus(target_gpus)`
  - `offload_dp_ranks(dp_ranks)`

Reason:
- Enables explicit drain barrier + one-time offload flow used by multi-scheduler partial-GPU mode.

### 5) Train/infer correction metadata fix (`train_infer_is_weight`)

File:
- `external/ROLL_schedrl/roll/pipeline/agentic/agentic_multi_lora_pipeline.py`

Changes:
- Set `batch.meta_info["loss_mask_keys"] = ["response_mask"]` before `_prepare_batch`.
- Added train/infer correction call in `_prepare_batch`:
  - `apply_train_infer_correction_to_batch(...)`
  - passes `update_mask_keys=batch.meta_info["loss_mask_keys"]`
  - merges returned correction metrics.

Reason:
- Fixed runtime failures:
  - `AssertionError: Please set loss_mask_keys in meta info`
  - `KeyError: train_infer_is_weight`

### 6) Smoke test execution result

Command:
```bash
cd /workspace/SchedRL/external/ROLL_schedrl
PYTHONPATH=/workspace/SchedRL/external/ROLL_schedrl /venv/main/bin/python \
  examples/start_agentic_pipeline.py \
  --config_path qwen2.5-0.5B-agentic \
  --config_name n-agent_train_sokoban_multi_lora_async
```

Result:
- Completed with exit code `0`
- Log contains `pipeline complete!`
