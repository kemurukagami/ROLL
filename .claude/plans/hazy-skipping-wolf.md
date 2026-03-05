# Plan: Update stale comments/docstrings in scheduler + pipeline files

## Context

Compared to commit `777dad6180a32e278802f4775eeb9d821511f648`, eight scheduler/pipeline
files have new or rewritten methods whose docstrings are missing, thin, or describe
the old `target_gpus` signature. This plan brings them up to date.

## Files

- `roll/distributed/scheduler/generate_scheduler.py` (sections 1–6 below)
- `roll/distributed/scheduler/storage.py` (section 7)
- `roll/distributed/scheduler/rollout_scheduler.py` (section 7)
- `roll/distributed/scheduler/resource_manager.py` (section 7)
- `roll/pipeline/agentic/agentic_pipeline.py` (section 7)
- `roll/pipeline/agentic/agentic_multi_lora_pipeline.py` (section 7)
- `roll/distributed/scheduler/initialize.py` (no new public methods — skip)
- `roll/distributed/scheduler/log_monitor.py` (no new public methods — skip)

---

## Changes

### 1. `GlobalCounter` class (line 609) — add class docstring

No docstring exists. Add:
```python
"""Monotonically increasing counter as a Ray actor.

Used to assign unique global IDs across distributed workers without coordination cost.
get_value() returns the current counter and increments it atomically (single-actor
execution guarantees no races).
"""
```

### 2. `_validate_dp_ranks_input` (line 1838) — add docstring

No docstring exists. Add:
```python
"""Validate and normalize a dp_ranks list input.

Checks: non-empty list[int], each value in [0, world_size), no duplicates.
Returns a normalized list of plain ints (coerces numpy ints etc.).

Args:
    dp_ranks: Candidate DP ranks to validate.
    mode: Label used in error messages ("shrink" or "expand").

Returns:
    Normalized list[int] with duplicates rejected.

Raises:
    ValueError: If list is empty, values out of range, or contains duplicates.
    TypeError: If any element is not an int.
"""
```

### 3. `shrink_workers` docstring (lines 1853-1889) — fix stale steps and args

Steps 1-3 still describe the old GPU-ID-based flow. Args still say `target_gpus`.
Replace the docstring body:

**Old steps:**
```
1. Validates target_gpus input
2. Calculates DP ranks to offload based on GPU overlap
3. Validates calculated ranks against active state
4. Atomically (under routing_lock): ...
```

**New steps:**
```
1. Validates dp_ranks input (type, range, duplicates)
2. If skip_offload=True: filters to only currently-active ranks (idempotent no-op
   if all ranks already inactive)
3. If skip_offload=False: validates ranks are active (strict check)
4. Atomically (under routing_lock):
   - Rebalances routing: aborts in-flight requests on shrinking workers and drains
     their queues (abort RPCs and drain also run under routing_lock — see FIXME
     comment in _rebalance_on_shrink for G02-RULE-26.2)
5. If skip_offload=False: offloads model states from shrinking workers to CPU
6. Returns metrics for monitoring
```

**Args — replace:**
```
target_gpus: GPU IDs to free (e.g., [4, 5, 6, 7] to free second half of 8 GPUs)
```
**With:**
```
dp_ranks: DP ranks to deactivate/offload.
skip_offload: If True, skip physical model offload and treat already-inactive
    ranks as a no-op. Use when another coupled scheduler will handle the offload,
    or during init-time shrink where ranks are not yet loaded.
```

**Raises — replace `target_gpus invalid` with `dp_ranks invalid`.**

**Example — update:**
```python
# Full shrink with offload
result = await scheduler.shrink_workers([2, 3])
# Returns: {"aborted": 10, "remapped": 5, "shrink_duration_ms": 2340.5, "offload_ranks": [2, 3]}

# Routing-only shrink (another scheduler handles offload)
result = await scheduler.shrink_workers([2, 3], skip_offload=True)
```

**Side Effects — add:**
```
- Serialized under _op_lock (prevents concurrent shrink/expand)
- If skip_offload=True and ranks already inactive: returns zero-metrics immediately
```

### 4. `expand_workers` docstring (lines 1930-1971) — fix stale steps and args

Same pattern as shrink_workers. Steps 1-2 still describe old GPU-based calculation.
Args still say `target_gpus`. DO_TIME_SHARING path not mentioned.

**Old steps 1-2:**
```
1. Validates target_gpus input
2. Calculates DP ranks to restore based on GPU overlap
```

**New steps 1-2:**
```
1. Validates dp_ranks input (type, range, duplicates)
2. If skip_load=True: filters to only currently-inactive ranks (no-op if all
   already active). Skips model loading; only updates routing state.
```

**Args — replace `target_gpus` with:**
```
dp_ranks: DP ranks to restore to active set.
skip_load: If True, skip model loading (use when model_update already synced
    weights). In DO_TIME_SHARING mode (when skip_load=False), triggers selective
    model weight sync via ModelUpdateService before loading vLLM states.
```

**Raises — replace `target_gpus invalid` with `dp_ranks invalid`.**

**Example — update to use dp_ranks directly:**
```python
result = await scheduler.expand_workers([2, 3])
# Returns: {"aborted": 3, "remapped": 3, "expand_duration_ms": 1850.2, "load_ranks": [2, 3]}
```

**Side Effects — add:**
```
- Serialized under _op_lock (prevents concurrent shrink/expand)
- If skip_load=True and ranks already active: returns zero-metrics immediately
- In DO_TIME_SHARING mode: syncs selected worker weights via ModelUpdateService
  before loading vLLM states (avoids holding KV cache during weight sync)
```

### 5. `_rebalance_on_expand` docstring (lines 1635-1670) — fix stale algorithm notes

Two implementation notes are now wrong:

**Remove:**
```
- Check at line 1146 (if not dp_rank in old_active_dp_ranks) is redundant
  since dp_rank_to_src_ranks already contains only old workers, but kept as defensive guard
- If all workers exhausted before reaching target, loop may cycle indefinitely
  (no explicit check for empty state, but pop(0) will eventually empty all lists)
```

**Replace with:**
```
- Round-robin uses a while loop with empty_streak detection (not cycle()) to
  terminate cleanly when all worker lists are exhausted before the abort target
- Calls self.resume() automatically when expanding from zero active ranks
  (was_empty check), unblocking suspended generate_one_request() callers
```

Also fix the algorithm step:
- "3. Round-robin iterate over old workers using cycle()" →
  "3. Round-robin iterate over old workers using while loop with empty-streak guard"

### 6. `_rebalance_on_shrink` (private `_rebalance_on_shrink` method, ~line 1529)

Docstring says "RuntimeError: If shrink operation fails" but doesn't document the
shrink-to-zero behavior or rollback of `need_suspend`.

Add to docstring:
```
Side Effects:
    - Sets need_suspend=True and clears suspend_notifier if shrinking to zero
      active ranks (blocks future generate_one_request() until expansion).
    - On exception: rolls back active_dp_ranks and need_suspend, re-sets
      suspend_notifier to unblock waiters.
    - See FIXME (G02-RULE-26.2) in this method for known locking constraints on
      abort RPCs under routing_lock.
```

(Do not add a second FIXME for G02-RULE-26.2 — one already exists in the code.)

---

### 7. Additional files changed since `777dad6`

#### `roll/distributed/scheduler/storage.py`

Four new methods have no docstrings: `try_put`, `delete`, `delete_prefix`, `delete_port_claims`.
Add one-line docstrings describing: what key/prefix means, what the return value is,
and (for `delete_port_claims`) what `pipeline_id` scopes.

#### `roll/distributed/scheduler/rollout_scheduler.py`

- `shrink_sampler(dp_ranks, skip_offload)` and `expand_sampler(dp_ranks, skip_load)` — public
  Ray-remote API; document that they delegate to `RequestScheduler.shrink_workers` /
  `expand_workers` and that `dp_ranks` replaces the old `target_gpus` parameter.
- `shutdown(timeout)` — document the timeout semantics and that it cancels in-flight tasks.
- `resume()` — document that it unblocks a suspended sampler (delegates to `RequestScheduler.resume`).
- Batch tracker helpers (`put`, `_resolve_num_return_sequences`, `_estimate_total_required`,
  `_mark_new_batch`, `_compute_progress`, `_maybe_emit_progress`) are private; add one-line
  docstrings only where the name is not self-explanatory (e.g. `_estimate_total_required`
  should note it accounts for `num_return_sequences`).

#### `roll/distributed/scheduler/resource_manager.py`

- `get_state()` — already has docstring `"""Return serializable state for proxy construction."""`, OK.
- `get_or_create_roll_resource_manager_actor(num_gpus_per_node)` — has docstring, OK.
- `ResourceManagerProxy` class and its methods (`nodes_placement_group`,
  `allocate_placement_group`) — add class-level docstring explaining it is a
  synchronous drop-in backed by a shared Ray actor, and why (cross-process access).

#### `roll/pipeline/agentic/agentic_pipeline.py`

- Module-level `target_gpus_to_dp_ranks_to_remove` / `target_gpus_to_dp_ranks_to_add`
  already have docstrings. OK.
- Private `_target_gpus_to_dp_ranks_to_remove` / `_target_gpus_to_dp_ranks_to_add` on the
  pipeline class have no docstrings — add one-liners noting they delegate to the module-level
  functions with `self._infer_device_mapping`.

#### `roll/pipeline/agentic/agentic_multi_lora_pipeline.py`

- `is_lora_training(pipeline_config)` — has a docstring stub `""" """`, fill it in:
  explain what condition makes it return True.
- `_verify_lora_model_update` and `_initial_model_update` — already have docstrings, OK.
- Add an inline comment above the sequential expand block (recently changed) explaining
  that the first scheduler must complete its load before others update routing.

---

## Verification

- `cd external/ROLL_rlix && make precommit` — linting/style passes
- `grep -n "target_gpus" roll/distributed/scheduler/generate_scheduler.py` —
  should only match the method name `_validate_target_gpus` (still legitimately present),
  not appear inside any docstring or comment body
- `grep -rn "target_gpus" roll/distributed/scheduler/rollout_scheduler.py roll/pipeline/agentic/` —
  should return zero results (all references migrated to `dp_ranks`)
