# Plan: Simplify New GroupQueueManager + Coordinator Progress Code

## Context

Code review of the two-level reporting implementation. Assessing simplification candidates
from `vast-rolling-flame.md` and the new `GroupQueueManager` code.

---

## Coordinator (`rlix/pipeline/coordinator.py`)

### KEEP: max_concurrency + _progress_lock

`resize_infer` holds `_resize_sync_lock` for seconds (Ray.get blocking). With max_concurrency=1,
progress reports queue behind resize — rlix scheduler sees stale data during every expand/shrink.
`COORDINATOR_MAX_CONCURRENCY=4` lets progress calls run concurrently with resize calls (different
locks). `_progress_lock` guards `_scheduler_reports` and bucket state against two concurrent
progress calls. **Keep both.**

### KEEP: coordinator bucket deduplication (_coord_progress_last_bucket)

The proposal to remove it claims "GQM bucket == coordinator bucket" — this is wrong. GQM computes
`percent_completed` for its own stream (e.g., train=20%). Coordinator computes it from the
aggregate (e.g., train 20% + val 0% → ~10%). Different values, different thresholds. Removing
the coordinator check would cause every individual-stream 2% tick to trigger a scheduler call
(N× more calls). **Keep it.**

### REMOVE: step-based eviction (_coord_current_step + clear())

**Current (lines 270–274):**
```python
current_step = metrics.get("current_train_step")
if current_step is not None and current_step != self._coord_current_step:
    self._scheduler_reports.clear()
    self._coord_current_step = current_step
    self._coord_progress_last_bucket = None  # Force emit on first report of new step
```

Why remove:
- `_scheduler_reports[scheduler_key] = report` already overwrites stale entries (last-write-wins).
  Train step N overwrites train step N-1 (same key `train:__fft__`). Val likewise.
- The `clear()` creates a race window: after train triggers clear, val's entry is missing until
  val's next report. Aggregate `total_required` is temporarily understated (val's target gone).
- The stale LoRA problem it tries to solve is rare; natural overwrite handles train/val correctly.

**Fix:** Remove `_coord_current_step` field and the 5-line eviction block from `__init__` and
`report_progress_from_scheduler`. Also remove the mention of it from the docstring.

---

## GroupQueueManager (`rollout_scheduler.py`)

### DONE: self.config = config

Already added at line 373. Fixes latent AttributeError in `_resolve_num_return_sequences`
fallback path.

### Apply: move ProgressReport import to module level

**Current (line 534, inside `_maybe_emit_progress`):**
```python
from rlix.protocol.types import ProgressReport
```

`COORDINATOR_ACTOR_NAME_PREFIX` from same module is already at top-level. No reason for lazy import.

**Fix:**
```python
# line 25 — extend existing import:
from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX, ProgressReport
```
Remove the in-method `from rlix.protocol.types import ProgressReport`.

### Apply: remove duplicate percent_completed computation

**Current (lines 517 and 541):**
```python
percent_completed = float(collected) / float(max(total_required, 1))   # line 517
...
percent_completed=float(collected) / float(max(total_required, 1)),    # line 541 — duplicate
```

**Fix:** `percent_completed=percent_completed,` on line 541.

### Apply: remove redundant `collected >= total_required` condition

**Current (lines 521–526):**
```python
should_emit = (
    bucket != self._progress_last_bucket
    or remaining == 0
    or collected >= total_required   # redundant: remaining=max(total_required-collected,0)
    or self._progress_new_batch
)
```

`remaining == 0` iff `collected >= total_required` (from line 500 definition). **Remove** the
`or collected >= total_required` line.

### Apply: simplify oldest_ts loop with min() generator

**Current (lines 493–498):**
```python
oldest_ts: Optional[float] = None
for group_queue in self.group_queue.values():
    for group in group_queue.groups.values():
        if len(group.rollouts) < self.group_size:
            if oldest_ts is None or group.created_at < oldest_ts:
                oldest_ts = group.created_at
```

**Fix:**
```python
oldest_ts: Optional[float] = min(
    (group.created_at
     for gq in self.group_queue.values()
     for group in gq.groups.values()
     if len(group.rollouts) < self.group_size),
    default=None,
)
```

---

## Pipeline Namespace Deduplication (separate but applies now)

`f"pipeline_{pipeline_id}_NS"` appears in 4 places with no shared definition.
`full_finetune_pipeline.py` already has a comment flagging this drift risk.

**Fix:** Add a public function to `rlix/protocol/types.py` (after the constants block):

```python
def get_pipeline_namespace(pipeline_id: str) -> str:
    """Canonical Ray namespace for a per-pipeline coordinator actor."""
    return f"pipeline_{pipeline_id}_NS"
```

Update all 4 call sites to import and use it:

- `rlix/pipeline/coordinator.py` — remove `_get_pipeline_namespace`, import from types
- `rlix/pipeline/full_finetune_pipeline.py:87` — replace inline string, remove drift comment
- `rlix/scheduler/scheduler.py:1194` — replace method body with `return get_pipeline_namespace(pipeline_id)`
- `external/ROLL_rlix/roll/distributed/scheduler/rollout_scheduler.py:390` — replace inline string

(`ROLL_rlix` already imports from `rlix.protocol.types` so no new cross-repo dependency.)

---

## Files

- `external/ROLL_rlix/roll/distributed/scheduler/rollout_scheduler.py`
- `rlix/pipeline/coordinator.py`
- `rlix/protocol/types.py`
- `rlix/pipeline/full_finetune_pipeline.py`
- `rlix/scheduler/scheduler.py`

---

## Train vs Val `remaining` Calculation — Assessment

**Question:** Should train and val calculate `remaining` differently?

**Answer: No — same formula is correct for both.**

Key facts from `agentic_config.py __post_init__` (lines 238–244):
- `num_return_sequences` is forced to 1 for **all** env managers (train, val, actor_infer).
- So `_resolve_num_return_sequences()` always returns 1 for both modes.

Result:
- Train: `total_required = rollout_batch_size * 1 = rollout_batch_size`
- Val:   `total_required = val_batch_size * 1 = val_batch_size`
- Both:  `remaining = max(total_required - collected, 0)`

`self.rollout_batch_size` is already set correctly (line 406 for train, line 410 for val),
so the formula is the same but with the right batch size — no special-casing needed.

**Val between steps:** Val `remaining=0` (done) persists in `_scheduler_reports` until val
sends its `new_batch=True` report for the next step. During this window, coordinator sees
val as complete (0 remaining), which is correct — val has no pending demand until its next batch.

**No code change needed for this finding.**

---

## Verification

`make precommit` from `external/ROLL_rlix/`.
