# Plan: Simplify vllm_strategy.py relative to commit 777dad6

## Context

`roll/distributed/strategy/vllm_strategy.py` grew from ~320 lines (commit `777dad6`) to ~1121 lines
after multi-LoRA routing was added. Several additions are dead/unused code or over-engineered.
Goal: remove ~100 lines without changing observable behavior.

**Not changed:** `_wait_for_lora_visible` — its 3-retry / exponential-backoff logic is intentional
for the add_lora race condition and must stay. Whether vLLM's internal `custom_add_lora` RPC is
synchronous w.r.t. `list_loras` visibility is unverified; the retry loop is the safety net.
`wait_loras_ready` (Change 4) can be fail-fast precisely *because* `add_lora` already went through
this retry loop before returning.

## File to modify

`roll/distributed/strategy/vllm_strategy.py`

---

## Change 1 — Remove dead null-check for `lora_request` (5 lines)

**Location:** lines 641–645, inside `generate_request`, inside `if self.is_lora:` block.

`lora_request` is unconditionally assigned `LoRARequest(...)` at line 635. The check at line 641
can never be true. Remove:
```python
            if lora_request is None:
                raise RuntimeError(
                    "Expected non-null lora_request for vLLM request (is_lora=True), but got None. "
                    "This indicates a LoRA routing bug."
                )
```

---

## Change 2 — Remove `ROLL_VLLM_DISABLE_LORA_REQUEST` env var + `lora_request_enabled` (8 lines)

**Location:** lines 582–588, inside `generate_request`, inside `if self.is_lora:` block.

This env var "disables" LoRA routing but immediately raises `RuntimeError` when LoRA is enabled —
a trap with no valid use case. `lora_request_enabled` is written to `data.meta_info` but never
read anywhere externally. Remove:
```python
            # Safety check: allow disabling LoRA request passing for debugging
            lora_request_enabled = os.environ.get("ROLL_VLLM_DISABLE_LORA_REQUEST", "0") != "1"
            data.meta_info["lora_request_enabled"] = lora_request_enabled
            if not lora_request_enabled:
                raise RuntimeError(
                    "LoRA routing is enabled (is_lora=True) but ROLL_VLLM_DISABLE_LORA_REQUEST=1 disables passing "
                    "LoRARequest into vLLM. Unset ROLL_VLLM_DISABLE_LORA_REQUEST to ensure rollouts use adapters."
                )
```

---

## Change 3 — Remove `_should_debug_lora_routing()` + `_log_lora_routing_context()` + 5 call sites (~75 lines)

**Delete both methods** at lines 80–146.

**Remove the `_log_lora_routing_context(...)` call at each of the 5 call sites** (keep the
surrounding `raise` / `raise RuntimeError` / `logger.error` statements):

| Site | Location | Pattern |
|------|----------|---------|
| A | `_generate_standard` — `get_lora_name_array_failed` catch | `except: _log(...); raise` → `except: raise` |
| B | `_generate_standard` — length-mismatch block | `_log(...); logger.error(...); raise RuntimeError(...)` → remove only the `_log(...)` call |
| C | `generate_request` — `resolve_microbatch_lora_name_failed` catch | `except: _log(...); raise` → `except: raise` |
| D | `generate_request` — `lora_id_missing` block | `_log(...); raise RuntimeError(...)` → remove only the `_log(...)` call |
| E | `generate_request` — `lora_id_not_loaded` block (line ~621) | `_log(...); await _wait_for_lora_visible(...)` → remove only the `_log(...)` call |

**Note on site E (redundancy):** After removing the `_log_lora_routing_context` call at site E,
the pattern becomes: inline `list_loras` check (lines 619–620) → `_wait_for_lora_visible` which
also calls `list_loras`. The double call is harmless; leave it for now.

---

## Change 4 — Simplify `wait_loras_ready` to fail-fast (~35 lines → ~15 lines)

**Location:** lines 926–961.

**Verified call chain** (traced through source):
- `model_update_lora_subset` → `model_update_group.model_update()` → `megatron_strategy.selective_sync_active_cache`
  calls `worker.add_lora.remote(...)` wrapped in `ray.get()` — blocking until `add_lora` completes on every target worker.
- `VllmStrategy.add_lora` calls `_wait_for_lora_visible` before returning, which retries up to 3×
  to confirm the adapter is visible in `list_loras()`.
- Back in `_initial_model_update` / the training loop, `self.actor_infer.load_states()` is called next.
  `VllmStrategy.load_states` only calls `reset_prefix_cache()` when `is_model_in_gpu=True` (set by
  `add_lora`), so it does **not** unload adapters.
- Then `_verify_lora_model_update` → `wait_loras_ready`.

**Conclusion:** By the time `wait_loras_ready` runs, all adapters were confirmed visible before
`add_lora` returned (via `_wait_for_lora_visible`), and `load_states()` does not disturb them.
The polling loop is redundant. A single snapshot check is correct and sufficient.

Secondary reason: polling loops with `asyncio.sleep` violate CLAUDE.md "No retry logic".

**Replace the method body with:**
```python
    async def wait_loras_ready(self, adapter_names: list[str], timeout_s: float = 30.0) -> None:
        """Assert all named LoRA adapters are currently loaded; fail fast if any are missing.

        Args:
            adapter_names: Adapter names to verify. Empty list is a no-op.
            timeout_s: Unused — kept for API compatibility with existing callers.
        """
        if not adapter_names:
            return
        loaded = await self.list_loras()
        missing: list[tuple[str, int | None]] = []
        for adapter_name in adapter_names:
            lora_int_id = await self.get_lora_id(adapter_name)
            if lora_int_id is None or lora_int_id not in loaded:
                missing.append((adapter_name, lora_int_id))
        if missing:
            raise RuntimeError(
                f"LoRA adapters not ready: missing={missing!r} loaded_sample={loaded[:16]!r}"
            )
```

External callers (`base_worker.py:594`, `agentic_multi_lora_pipeline.py:245`) pass both
`adapter_names` and `timeout_s` kwargs — both are still accepted; `timeout_s` is now unused.

---

## Change 5 — Fix stale comment in `add_lora` (1 line)

**Location:** line 909, inside `add_lora`, after the `_wait_for_lora_visible` call.

Current comment: `# _wait_for_lora_visible returns only when adapter is visible or raises on timeout.`

`_wait_for_lora_visible` has no timeout parameter — it retries a fixed 3 times with exponential
backoff. The word "timeout" is inaccurate.

**Replace with:**
```python
            # _wait_for_lora_visible retries up to 3 times; raises if still not visible.
```

---

## Summary

| Change | Lines removed |
|--------|--------------|
| 1. Dead null-check | −5 |
| 2. ROLL_VLLM_DISABLE_LORA_REQUEST | −8 |
| 3. Debug helpers + 5 call sites | −75 |
| 4. `wait_loras_ready` polling | −21 |
| 5. Stale comment | 0 (edit) |
| **Total** | **~−109 lines** |

---

## Change 6 — Improve `setup_collective_group` comments to explain *why* two styles exist

**Location:** lines 587–671 in `vllm_strategy.py`.

**Problem:** The current section header and docstring describe *what* each style's parameters are,
but not *why* two styles exist — the fundamental difference in rank-assignment model is unexplained.
A reader doesn't understand why comm_plan doesn't need `master_address`/`master_port`/`rank_offset`
or why the new style can skip non-participating workers.

**Replace the section header block (lines 587–601) with:**
```python
    # =====================================================================
    # Collective Communication Group Management
    # =====================================================================
    # Two call styles exist because they solve different weight-sync problems:
    #
    # Style 1 — comm_plan (multi-LoRA / partial-GPU selective sync):
    #   Used when only a *subset* of inference workers should receive a weight
    #   broadcast (e.g. only the GPUs serving adapter A, not those serving B).
    #   The caller builds a comm_plan dict mapping cluster-rank → connection
    #   details (master_addr, master_port, group_name, participant list).
    #   Each vLLM worker looks up its own rank_in_cluster in the plan; if absent
    #   it silently skips group creation. master_address / master_port / world_size
    #   are NOT passed separately because they are encoded per-rank inside the plan.
    #   Built by ModelUpdateService; used for INV-4-safe selective adapter sync.
    #
    # Style 2 — legacy positional args (base model / all-rank broadcast):
    #   Used when ALL inference workers participate in the same group.
    #   Caller computes master_address, master_port, world_size, group_name
    #   upfront and passes them identically to every worker. rank_offset converts
    #   local intra-worker rank to group rank. No per-worker lookup needed because
    #   every worker always joins.
    # =====================================================================
```

**Replace the docstring (lines 604–637) with:**
```python
        """Create a NCCL process group for trainer→inference weight synchronization.

        Two calling styles are supported — choose based on whether all workers
        participate or only a subset:

        **Style 1: comm_plan (selective sync, multi-LoRA / partial-GPU)**
            Pass ``comm_plan`` as a kwarg. The plan is a dict built by
            ``ModelUpdateService`` that encodes per-rank connection info
            (master_addr, master_port, group_name, participant list).
            Each vLLM GPU worker resolves its own role by looking up
            ``rank_in_cluster`` (= ``self.worker.rank``, the DP rank) in the
            plan. Workers whose rank is absent skip group creation silently,
            enabling INV-4-safe per-adapter selective broadcasts.

            Required kwargs: ``comm_plan``
            Optional kwargs: ``backend``, ``timeout_s``

        **Style 2: legacy positional args (all-rank broadcast)**
            Pass connection details as kwargs: ``master_address``, ``master_port``,
            ``rank_offset``, ``world_size``, ``group_name``. Every worker joins
            the same group; rank is ``rank_offset + local_rank``. Used for
            single-LoRA or full-model broadcasts where no worker should be skipped.

            Required kwargs: ``master_address``, ``master_port``, ``rank_offset``,
                             ``world_size``, ``group_name``
            Optional kwargs: ``backend``, ``timeout_s``

        Raises:
            TypeError: If neither style's required arguments are present.
        """
```

**No logic changes** — only the header comment block and docstring are modified.

---

## Verification

```bash
cd external/ROLL_rlix

# 1. Confirm removed names are gone
grep -rn "_log_lora_routing_context\|_should_debug_lora_routing\|ROLL_VLLM_DISABLE_LORA_REQUEST\|lora_request_enabled\|ROLL_DEBUG_LORA_ROUTING\|ROLL_DEBUG_PUNICA" --include="*.py"

# 2. Lint + type checks
make precommit
```
