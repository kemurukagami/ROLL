# Plan: Remove duplicate methods from RollResourceManagerProxy

## Context
`RollResourceManagerProxy` (resource_manager.py:223) duplicates two methods that are
already defined identically on `ResourceManager`. Since the proxy's `__init__` sets the
same instance attributes (`node2pg`, `num_nodes`, `gpu_per_node`, etc.) that the parent
methods read, inheriting is safe and removes ~50 lines of duplicate logic.

## File to modify
`roll/distributed/scheduler/resource_manager.py`

## Change

### 1. Inherit from ResourceManager
```python
# before
class RollResourceManagerProxy:

# after
class RollResourceManagerProxy(ResourceManager):
```

### 2. Remove `nodes_placement_group` (lines 245-246)
Inherited from `ResourceManager` — identical body `return self.node2pg[node_rank]`.

### 3. Remove `allocate_placement_group` (lines 248-296)
Inherited from `ResourceManager` — identical logic. The comment block explaining the
async-safe motivation can be moved to the class docstring or `__init__` instead.

### 4. Keep `destroy_placement_group` override (lines 298-302)
This intentionally overrides the parent to raise `NotImplementedError`, so it stays.

### 5. Keep `__init__` as-is
Does not call `super().__init__()` (correct — avoids Ray cluster discovery).
Python allows inheriting methods without calling the parent constructor as long as
the required instance attributes are set, which `__init__` already does.

## Result
~50 lines removed. Proxy stays a valid drop-in for `ResourceManager` callers.
No behavior change.

## Verification
Run: `cd external/ROLL_rlix && make precommit`
Check: no import errors, mypy passes on the file.
