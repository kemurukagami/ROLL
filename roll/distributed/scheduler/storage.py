import ray

from roll.utils.logging import get_logger

logger = get_logger()


@ray.remote
class SharedStorage:

    def __init__(self):
        self._storage = {}

    def put(self, key, data):
        ref = ray.put(data)
        self._storage[key] = ref

    def try_put(self, key, data) -> bool:
        if key in self._storage:
            return False
        ref = ray.put(data)
        self._storage[key] = ref
        return True

    def get(self, key):
        ref = self._storage.get(key)
        if ref is None:
            logger.warning(f"{key} is not found in storage")
            return None
        return ray.get(ref)

    def delete(self, key) -> None:
        self._storage.pop(key, None)

    def delete_prefix(self, prefix: str) -> int:
        if not isinstance(prefix, str):
            raise ValueError(f"prefix must be str, got {type(prefix).__name__}")
        keys = [k for k in self._storage.keys() if isinstance(k, str) and k.startswith(prefix)]
        for k in keys:
            self._storage.pop(k, None)
        return len(keys)

    def delete_port_claims(self, pipeline_id: str) -> int:
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
