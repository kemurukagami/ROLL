# Fix P1: G02-RULE-26.2 — Unbounded `routing_lock` Hold

## Context

`shrink_workers` acquires `routing_lock` then calls `rebalance_on_shrink`, which internally:
1. Does async abort RPCs (`await asyncio.gather(*abort_futures)`)
2. Polls a drain loop (`while True: await asyncio.sleep(3)`)

Both happen **while `routing_lock` is held**. Every concurrent `generate_one_request` call blocks on the lock for up to 30 s. The same issue exists in `_rebalance_on_expand` (abort RPCs under lock, no drain loop).

**Goal:** hold `routing_lock` only for synchronous state mutation; move all async I/O outside.

---

## Critical Files

- `roll/distributed/scheduler/generate_scheduler.py`
  - `RequestScheduler._rebalance_on_shrink` (lines ~1529–1599)
  - `RequestScheduler.rebalance_on_shrink` (lines ~1494–1527, timeout wrapper)
  - `RequestScheduler._rebalance_on_expand` (lines ~1634–1754)
  - `RequestScheduler.rebalance_on_expand` (lines ~1601–1632, timeout wrapper)
  - `RequestScheduler.shrink_workers` (lines ~1889–1927)
  - `RequestScheduler.expand_workers` (lines ~1929–2037)

---

## Implementation Plan

### Step 1 — Make `_rebalance_on_shrink` synchronous (no awaits)

Split the method into two parts:

**Keep inside `_rebalance_on_shrink` (sync, under `routing_lock`):**
- Update `active_dp_ranks` (remove shrink ranks)
- Set `need_suspend` / clear `suspend_notifier` if shrink-to-zero
- Snapshot `running_requests[dp_rank]` for each shrink rank → build `abort_by_dp_rank: Dict[int, List[str]]`
- Snapshot `src_rank2_dp_rank` entries pointing to shrink ranks → build `src_ranks_to_remap: Set[int]`
- Return `(abort_by_dp_rank, src_ranks_to_remap, total_aborted)` instead of awaiting
- Keep the existing rollback logic in the `except` block (it is sync)

**Remove from `_rebalance_on_shrink`:**
- `await asyncio.gather(*abort_futures)` — move to caller
- `while True: await asyncio.sleep(3)` drain loop — move to caller
- `self._clear_src_rank_mappings(src_ranks_to_remap)` — move to caller (after drain)

Rename signature to make intent clear:
```python
def _shrink_routing_state(self, shrink_dp_ranks: List[int]) -> Tuple[Dict[int, List[str]], Set[int], int]:
    """Mutate routing state for shrink. Caller holds routing_lock. Returns abort plan."""
```

Drop the `rebalance_on_shrink` timeout wrapper — the timeout moves to `shrink_workers` level.

### Step 2 — Restructure `shrink_workers` to do I/O outside the lock

```python
async with self._op_lock:
    start_time = time.time()
    offload_ranks = self._validate_dp_ranks_input(dp_ranks, mode="shrink")
    # ... existing skip_offload idempotence filter ...

    # Phase A: fast state mutation only — held briefly
    old_active_ranks = self.active_dp_ranks.copy()
    old_need_suspend = self.need_suspend
    async with self.routing_lock:
        abort_by_dp_rank, src_ranks_to_remap, total_aborted = self._shrink_routing_state(offload_ranks)

    # Phase B: async I/O outside lock
    try:
        abort_futures = [
            self.infer_cluster.workers[dp_rank].abort_requests.remote(request_ids)
            for dp_rank, request_ids in abort_by_dp_rank.items()
            if request_ids
        ]
        await asyncio.gather(*abort_futures)

        # Drain: wait for in-flight completions outside lock
        deadline = time.time() + 30.0
        while True:
            remain = sum(len(self.running_requests[r]) for r in offload_ranks)
            if remain == 0:
                break
            if time.time() >= deadline:
                raise RuntimeError(f"shrink drain timed out after 30s, {remain} requests still running")
            logger.info(f"Shrink: draining {remain} remaining requests on {offload_ranks}")
            await asyncio.sleep(3)

        # Phase C: brief lock re-acquire to clear stale src_rank mappings
        async with self.routing_lock:
            self._clear_src_rank_mappings(src_ranks_to_remap)

    except Exception as e:
        # Rollback routing state under lock
        async with self.routing_lock:
            self.active_dp_ranks = old_active_ranks
            self.need_suspend = old_need_suspend
            if not self.need_suspend:
                self.suspend_notifier.set()
        raise RuntimeError(f"Shrink failed: {e}") from e

    if not bool(skip_offload):
        offload_refs = self.infer_cluster.offload_states_partial(...)
        await asyncio.gather(...)

    return {"aborted": total_aborted, "remapped": len(src_ranks_to_remap), ...}
```

### Step 3 — Apply same split to `_rebalance_on_expand` / `expand_workers`

`_rebalance_on_expand` also does `await asyncio.gather(*abort_futures)` under `routing_lock` (no drain loop, but same lock-hold problem).

Apply same pattern:
- Rename `_rebalance_on_expand` → `_expand_routing_state` (sync, returns `abort_by_dp_rank, total_aborted`)
- `expand_workers` awaits abort futures **after** releasing `routing_lock`
- Drop the `rebalance_on_expand` timeout wrapper; timeout handled at `expand_workers` level

```python
# In expand_workers, after loading:
async with self.routing_lock:
    abort_by_dp_rank, total_aborted = self._expand_routing_state(load_ranks)

abort_futures = [...]
await asyncio.gather(*abort_futures)  # outside lock
```

Note: expand has no drain loop, so Phase C (re-lock for cleanup) is not needed.

### Step 4 — Remove now-unused timeout wrappers

`rebalance_on_shrink` and `rebalance_on_expand` (the public wrappers with `asyncio.wait_for`) can be removed entirely — they were only called from `shrink_workers`/`expand_workers`, and the 30-second deadline now lives in the drain loop in `shrink_workers`.

---

## Correctness Notes

- After Phase A (`routing_lock` released), new `generate_one_request` calls will NOT route to shrinking ranks because `active_dp_ranks` was already updated under the lock. Any pre-existing in-flight requests on those ranks are handled by the drain loop.
- The `src_rank2_dp_rank` stale entries are safe between Phase A and Phase C: `generate_one_request` already lazily evicts stale entries pointing to inactive ranks (line ~1346–1348).
- The rollback in Phase B re-acquires `routing_lock` briefly — this is safe since no other shrink/expand can run concurrently (`_op_lock` is held).

---

## Verification

Run the existing scheduler unit tests:
```bash
cd external/ROLL_rlix && make test -k "scheduler"
```

Manual check: confirm `routing_lock` hold duration drops by inspecting log timestamps between "Shrink: waiting..." entries and the next "dispatch generate_request" log in `generate_one_request`.
