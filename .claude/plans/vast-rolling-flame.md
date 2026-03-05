# Plan: Eliminate Duplicated Pipeline Namespace Format String

## Context

`f"pipeline_{pipeline_id}_NS"` appears in 4 places with no shared canonical definition:
- `rlix/pipeline/coordinator.py:24` — private `_get_pipeline_namespace` (canonical source)
- `rlix/pipeline/full_finetune_pipeline.py:87` — inlined with comment "mirrors coordinator.py"
- `rlix/scheduler/scheduler.py:1194` — reimplemented as an actor method
- `external/ROLL_rlix/roll/distributed/scheduler/rollout_scheduler.py:390` — our new code (from nifty-strolling-tiger plan)

Any namespace renaming requires 4 coordinated edits. `full_finetune_pipeline.py` already has a comment flagging this drift risk.

## Fix

### 1. Add public function to `rlix/protocol/types.py`

```python
def get_pipeline_namespace(pipeline_id: str) -> str:
    """Canonical Ray namespace for a per-pipeline coordinator actor."""
    return f"pipeline_{pipeline_id}_NS"
```

Place it after the constants block (after line 17).

### 2. Update all 4 call sites

**`rlix/pipeline/coordinator.py`** — replace private function with import:
```python
# remove
def _get_pipeline_namespace(pipeline_id: str) -> str:
    return f"pipeline_{pipeline_id}_NS"

# add to imports
from rlix.protocol.types import ..., get_pipeline_namespace
```

**`rlix/pipeline/full_finetune_pipeline.py:86-87`** — replace inline string:
```python
# before
# Namespace convention mirrors coordinator.py:_get_pipeline_namespace().
namespace = f"pipeline_{self._pipeline_id}_NS"

# after
namespace = get_pipeline_namespace(self._pipeline_id)
```

**`rlix/scheduler/scheduler.py:1194`** — replace method body:
```python
async def get_pipeline_namespace(self, *, pipeline_id: str) -> str:
    return get_pipeline_namespace(pipeline_id)
```
(import `get_pipeline_namespace` from `rlix.protocol.types`)

**`external/ROLL_rlix/roll/distributed/scheduler/rollout_scheduler.py:390`** — replace inline:
```python
# before
coordinator_namespace = f"pipeline_{self.pipeline_id}_NS"

# after
from rlix.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX, get_pipeline_namespace
coordinator_namespace = get_pipeline_namespace(self.pipeline_id)
```

## Files to Change

- `rlix/protocol/types.py` — add `get_pipeline_namespace`
- `rlix/pipeline/coordinator.py` — remove private fn, import from types
- `rlix/pipeline/full_finetune_pipeline.py` — use imported fn
- `rlix/scheduler/scheduler.py` — use imported fn
- `external/ROLL_rlix/roll/distributed/scheduler/rollout_scheduler.py` — use imported fn

## Verification

```bash
python3 -c "from rlix.protocol.types import get_pipeline_namespace; assert get_pipeline_namespace('p1') == 'pipeline_p1_NS'"
grep -rn "pipeline_.*_NS" rlix/ external/ROLL_rlix/roll/  # should return 0 inline occurrences
```
