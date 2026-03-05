"""Cluster-wide key-value store for multi-pipeline coordination.

This module provides SharedStorage, a Ray actor that persists across pipeline lifecycles
and enables coordination between ephemeral pipelines in time-sharing mode.

Usage:
- Port allocation claims: MASTER_ADDR_PORT:<addr>:<port> -> pipeline_id
- Checkpoint path caching: model_path:<model_name> -> local_path

# TODO(rlix): Delegate SharedStorage to rlix orchestrator similar to ResourceManager pattern.
# Currently rlix directly calls delete_port_claims/delete_prefix on this actor.
# Consider creating SharedStorageProxy and moving actor lifecycle management to rlix,
# with actor name/namespace defined in rlix.protocol.types (like ROLL_RESOURCE_MANAGER_ACTOR_NAME).
# This would:
# - Remove hardcoded "SHARED_STORAGE_ACTOR" / "global_storage_namespace" from rlix/orchestrator.py
# - Allow rlix to control storage lifecycle independent of ROLL
# - Enable per-tenant storage isolation if needed
"""
import ray

from roll.utils.logging import get_logger

logger = get_logger()


@ray.remote
class SharedStorage:
    """Cluster-wide key-value store shared across all ROLL pipelines.

    In time-sharing mode, this actor is shared across all ROLL pipelines and used for:
    - Port allocation claims (MASTER_ADDR_PORT:* keys) to prevent port collisions
    - Model checkpoint path caching to avoid redundant downloads

    The actor persists for the lifetime of the Ray cluster, enabling coordination
    between ephemeral pipelines that come and go.
    """

    def __init__(self):
        self._storage = {}

    def put(self, key, data):
        """Store data under the given key, overwriting any existing value.

        Args:
            key: The storage key (typically a string identifier).
            data: The value to store (will be placed in Ray object store).

        Note:
            This overwrites existing values silently. Use try_put() for atomic put-if-absent.
        """
        ref = ray.put(data)
        self._storage[key] = ref

    def try_put(self, key, data) -> bool:
        """Atomically store data only if the key does not already exist.

        This is used for lock-free resource claiming (e.g., port allocation).
        Multiple pipelines can race to claim a port; only one succeeds.

        Args:
            key: The storage key to claim.
            data: The value to store (typically pipeline_id for port claims).

        Returns:
            True if the key was successfully claimed (did not exist).
            False if the key already exists (claim failed).
        """
        if key in self._storage:
            return False
        ref = ray.put(data)
        self._storage[key] = ref
        return True

    def get(self, key):
        """Retrieve data stored under the given key.

        Args:
            key: The storage key to look up.

        Returns:
            The stored value, or None if the key does not exist.
        """
        ref = self._storage.get(key)
        if ref is None:
            logger.warning(f"{key} is not found in storage")
            return None
        return ray.get(ref)

    def delete(self, key) -> None:
        """Remove a single key from storage.

        Args:
            key: The storage key to delete.

        Note:
            Silently ignores keys that don't exist.
        """
        self._storage.pop(key, None)

    def delete_prefix(self, prefix: str) -> int:
        """Delete all keys that start with the given prefix.

        Used for bulk cleanup when a pipeline terminates, removing all its
        associated entries from shared storage.

        Args:
            prefix: The key prefix to match (e.g., "pipeline_123:").

        Returns:
            The number of keys deleted.

        Raises:
            ValueError: If prefix is not a string.
        """
        if not isinstance(prefix, str):
            raise ValueError(f"prefix must be str, got {type(prefix).__name__}")
        keys = [k for k in self._storage.keys() if isinstance(k, str) and k.startswith(prefix)]
        for k in keys:
            self._storage.pop(k, None)
        return len(keys)

    def delete_port_claims(self, pipeline_id: str) -> int:
        """Release all port claims owned by a specific pipeline.

        When a pipeline terminates, this removes all MASTER_ADDR_PORT:* entries
        that were claimed by that pipeline, allowing the ports to be reused.

        Args:
            pipeline_id: The ID of the pipeline whose port claims should be released.

        Returns:
            The number of port claims deleted.

        Raises:
            ValueError: If pipeline_id is not a non-empty string.
        """
        if not isinstance(pipeline_id, str) or pipeline_id == "":
            raise ValueError("pipeline_id must be non-empty str")
        deleted = 0
        for key in list(self._storage.keys()):
            if not isinstance(key, str) or not key.startswith("MASTER_ADDR_PORT:"):
                continue
            ref = self._storage.get(key)
            if ref is None:
                continue
            value = ray.get(ref)
            if value != pipeline_id:
                continue
            self._storage.pop(key, None)
            deleted += 1
        return deleted
