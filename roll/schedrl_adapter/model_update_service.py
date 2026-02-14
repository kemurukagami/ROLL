from __future__ import annotations

import uuid
from typing import Any, List

import ray

from roll.distributed.executor.cluster import Cluster
from roll.utils.logging import get_logger

logger = get_logger()


@ray.remote
class ModelUpdateService:
    """Per-pipeline service for selective sync on expand (ENG-123 Phase 4).

    Contract:
    - Scheduler-side trigger only: no promotion forwarding, no validation, no coalescing.
    - Calls into sender-side sync, which serializes via sender cache_lock.
    """

    def __init__(self, *, pipeline_id: str, src_cluster: Cluster, tgt_cluster: Cluster):
        if not isinstance(pipeline_id, str) or pipeline_id == "":
            raise ValueError("pipeline_id must be non-empty str")
        self.pipeline_id = pipeline_id
        self.src_cluster: Any = src_cluster
        self.tgt_cluster: Any = tgt_cluster

        self._sync_nonce = uuid.uuid4().hex[:8]

    def sync_selected_workers(self, tgt_dp_ranks: List[int]) -> None:
        tgt_dp_ranks = sorted(set(int(r) for r in tgt_dp_ranks))
        if not tgt_dp_ranks:
            raise ValueError("tgt_dp_ranks must be non-empty")

        infer_world_size = int(self.tgt_cluster.world_size)
        invalid = [r for r in tgt_dp_ranks if r < 0 or r >= infer_world_size]
        if invalid:
            raise ValueError(f"Invalid tgt_dp_ranks={invalid}; infer_world_size={infer_world_size}")

        tgt_device_mapping = getattr(self.tgt_cluster.worker_config, "device_mapping", None)
        tgt_num_gpus_per_worker = getattr(self.tgt_cluster.worker_config, "num_gpus_per_worker", None)

        if not tgt_device_mapping:
            raise RuntimeError("tgt_cluster device_mapping is empty; selective sync requires GPU infer workers")

        if not isinstance(tgt_num_gpus_per_worker, int) or int(tgt_num_gpus_per_worker) <= 0:
            raise RuntimeError("tgt_cluster.worker_config.num_gpus_per_worker must be positive int")

        tgt_device_mapping = [int(x) for x in tgt_device_mapping]

        sync_id = f"selective_sync/{self.pipeline_id}/{self._sync_nonce}/{uuid.uuid4().hex[:8]}"
        logger.info(
            f"[ModelUpdateService] sync_selected_workers_enter pipeline_id={self.pipeline_id} "
            f"sync_id={sync_id} tgt_dp_ranks={tgt_dp_ranks}"
        )

        refs = [
            worker.selective_sync_active_cache.remote(
                sync_id=sync_id,
                tgt_dp_ranks=tgt_dp_ranks,
                tgt_workers=self.tgt_cluster.workers,
                tgt_device_mapping=tgt_device_mapping,
                tgt_num_gpus_per_worker=int(tgt_num_gpus_per_worker),
            )
            for worker in self.src_cluster.workers
        ]
        ray.get(refs)

        logger.info(
            f"[ModelUpdateService] sync_selected_workers_exit pipeline_id={self.pipeline_id} sync_id={sync_id}"
        )
