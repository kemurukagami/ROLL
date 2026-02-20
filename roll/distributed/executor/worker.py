import logging
import os
import socket
import asyncio
import inspect
import threading
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import ray

from roll.configs.worker_config import WorkerConfig
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.storage import SharedStorage
from roll.utils.checkpoint_manager import download_model
from roll.utils.constants import GLOBAL_STORAGE_NAMESPACE, RAY_NAMESPACE, STORAGE_NAME
from roll.utils.context_managers import state_offload_manger
from roll.utils.logging import get_logger
from roll.utils.network_utils import collect_free_port, get_node_ip
from roll.utils.offload_states import OffloadStateType
from roll.utils.offload_nccl import monkey_patch_torch_dist

from roll.platforms import current_platform


@dataclass
class RankInfo:
    world_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1

    rank: int = 0
    tp_rank: int = 0
    dp_rank: int = 0
    pp_rank: int = 0
    cp_rank: int = 0

    @property
    def is_pipeline_last_stage(self):
        return self.pp_rank == (self.pp_size - 1)


class Worker:

    def __init__(self, worker_config: WorkerConfig):
        if worker_config.offload_nccl:
            monkey_patch_torch_dist()
        self.worker_config = worker_config
        self.pipeline_config = None
        self.worker_name = os.environ.get("WORKER_NAME", None)
        self.cluster_name = os.environ.get("CLUSTER_NAME", None)
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.pipeline_id = os.environ.get("PIPELINE_ID") or None
        self.shared_storage = SharedStorage.options(
            name=STORAGE_NAME, get_if_exists=True, namespace=GLOBAL_STORAGE_NAMESPACE
        ).remote()

        if self.rank == 0:
            master_addr = self.get_node_ip()
            master_port = str(self.get_free_port())
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = int(os.environ["MASTER_PORT"])
        if self.pipeline_id is None and os.environ.get("ROLL_RAY_NAMESPACE"):
            raise RuntimeError("PIPELINE_ID must be set when ROLL_RAY_NAMESPACE is set (multi-pipeline mode)")
        rendezvous_key = f"{self.pipeline_id}:{self.cluster_name}" if self.pipeline_id else self.cluster_name
        self.shared_storage.put.remote(rendezvous_key, {"MASTER_ADDR": self.master_addr, "MASTER_PORT": self.master_port})
        # NOTE: 自定义Worker时根据需要配置rank_info
        self.rank_info = RankInfo(
            world_size=self.world_size,
            rank=self.rank,
            dp_rank=self.rank,
            dp_size=self.world_size,
        )
        self.thread_executor: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(max_workers=5)
        self._logger = None

    def __repr__(self):
        return f"{type(self).__name__}({self.worker_name})"

    @property
    def logger(self) -> logging.Logger:
        """
        在ray.Actor内要使用自定义的logger, 避免ray context造成的logger不一致
        """
        self._logger = get_logger()
        return self._logger

    @staticmethod
    def get_node_ip():
        return get_node_ip()

    @staticmethod
    def get_free_port():
        shared_storage = SharedStorage.options(
            name=STORAGE_NAME, get_if_exists=True, namespace=GLOBAL_STORAGE_NAMESPACE
        ).remote()
        pipeline_id = os.environ.get("PIPELINE_ID") or None
        master_addr = Worker.get_node_ip()
        max_retry_count = int(os.environ.get("MAX_PORT_RETRY_COUNT", 1000))
        retry_count = 0
        master_port = collect_free_port()
        while retry_count < max_retry_count:
            master_addr_port_key = f"MASTER_ADDR_PORT:{master_addr}:{master_port}"
            claimed = ray.get(shared_storage.try_put.remote(master_addr_port_key, pipeline_id if pipeline_id else True))
            if claimed:
                break
            master_port = collect_free_port()
            retry_count += 1
        if retry_count >= max_retry_count:
            raise RuntimeError(f"Can not allocate unique MASTER_PORT on {master_addr}.")
        return master_port

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port

    def _get_strategy_load_state(self) -> Optional[bool]:
        """Check if strategy model is loaded in GPU.

        Handles multiple strategy implementations:
        - vLLM: strategy.is_model_in_gpu (direct attribute)
        - SGLang: strategy.model.is_model_in_gpu (nested attribute)
        - Others: None (not trackable)

        Returns:
            True if loaded, False if offloaded, None if not trackable
        """
        if getattr(self, "strategy", None) is None:
            return None

        # Try direct attribute (vLLM pattern)
        is_loaded = getattr(self.strategy, 'is_model_in_gpu', None)

        # Try nested attribute (SGLang pattern)
        if is_loaded is None and hasattr(self.strategy, 'model'):
            is_loaded = getattr(self.strategy.model, 'is_model_in_gpu', None)

        return is_loaded

    @staticmethod
    def get_visible_gpus():
        return current_platform.get_visible_gpus()

    def get_devices_info(self):
        devices_info = [
            dict(rank=rank, node_rank=pg["node_rank"], gpu_rank=pg["gpu_rank"])
            for rank, pg in enumerate(self.worker_config.resource_placement_groups)
        ]
        return devices_info

    def get_rank_info(self):
        return self.rank_info

    def initialize(self, pipeline_config, *args, **kwargs):
        self.pipeline_config = pipeline_config

        model_name = self.worker_config.model_args.model_name_or_path
        if model_name:
            self.worker_config.model_args.model_name_or_path = download_model(model_name)

        if self.pipeline_config.resume_from_checkpoint:
            self.logger.info(f"resume_from_checkpoint: {self.pipeline_config.resume_from_checkpoint}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_states(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.load_states()
        else:
            self.logger.warning("worker has not strategy")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def process_weights_after_loading(self):
        if getattr(self, "strategy", None) is not None:
            result = self.strategy.process_weights_after_loading()
            if inspect.isawaitable(result):
                await result
        

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_states(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.offload_states()
        else:
            self.logger.warning("worker has not strategy")

    def broadcast_parameter(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.broadcast_parameter(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def setup_model_update(self, *args, **kwargs):
        self.strategy.setup_model_update(*args, **kwargs)

    def setup_collective_group(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self._maybe_await(self.strategy.setup_collective_group(*args, **kwargs))
        else:
            self.logger.warning("worker has not strategy")

    def teardown_collective_groups(self, model_update_name: str, group_names: List[str]) -> None:
        if getattr(self, "strategy", None) is None:
            self.logger.warning("worker has not strategy")
            return
        teardown = getattr(self.strategy, "teardown_collective_groups", None)
        if callable(teardown):
            self._maybe_await(teardown(model_update_name, group_names))
            return
        # Backward compatibility: destroy groups one by one if teardown is not implemented.
        destroy = getattr(self.strategy, "destroy_collective_group", None)
        if callable(destroy):
            for name in group_names:
                self._maybe_await(destroy(name))
            return
        raise RuntimeError(f"{type(self.strategy).__name__} does not support teardown_collective_groups")

    def destroy_collective_group(self, group_name: str) -> None:
        if getattr(self, "strategy", None) is None:
            self.logger.warning("worker has not strategy")
            return
        destroy = getattr(self.strategy, "destroy_collective_group", None)
        if callable(destroy):
            self._maybe_await(destroy(group_name))
            return
        # Fail fast: we cannot safely infer model_update_name for bookkeeping cleanup.
        # Call teardown_collective_groups(model_update_name=..., group_names=...) when that context exists.
        raise RuntimeError(
            f"{type(self.strategy).__name__} does not support destroy_collective_group; "
            "use teardown_collective_groups(model_update_name=..., group_names=...) instead."
        )

    @staticmethod
    def _maybe_await(result: Any) -> Any:
        if not inspect.isawaitable(result):
            return result

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if not loop.is_running():
            return loop.run_until_complete(result)

        out: Dict[str, Any] = {}
        err: Dict[str, BaseException] = {}

        def runner():
            try:
                out["value"] = asyncio.run(result)
            except BaseException as exc:
                err["exc"] = exc

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        t.join()
        if err:
            raise err["exc"]
        return out.get("value")

    def setup_p2p_collective_group(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.setup_p2p_collective_group(*args, **kwargs)
        else:
            self.logger.warning("worker does not have a strategy")

    def start_model_update(self, *args, **kwargs):
        metrics = {}
        if getattr(self, "strategy", None) is not None:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/model_update",
                load_kwargs={"include": [OffloadStateType.model_params]},
            ):
                exec_metrics: Dict = self.strategy.model_update(*args, **kwargs)
            metric_prefix = f"time/{self.cluster_name}/model_update"
            metrics.update({f"{metric_prefix}/{k}": v for k, v in exec_metrics.items()})
        else:
            self.logger.warning("worker has not strategy")

        output = DataProto(meta_info={"metrics": metrics})
        return output

    def model_update_set_read_done_handle(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.model_update_set_read_done_handle(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def update_parameter_in_bucket(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.update_parameter_in_bucket(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    def build_latest_bucket_cache(
        self, checkpoint_version: int, global_step: int, adapter_name: str | None = None
    ) -> None:
        """
        Build a sender-side CPU bucket cache for selective sync under SchedRL.

        This is a thin wrapper around the strategy implementation. Fail fast if unsupported.
        """
        if getattr(self, "strategy", None) is None:
            raise RuntimeError("worker has no strategy")
        fn = getattr(self.strategy, "_build_latest_bucket_cache", None)
        if not callable(fn):
            raise RuntimeError(f"{type(self.strategy).__name__} does not support build_latest_bucket_cache")
        fn(checkpoint_version=int(checkpoint_version), global_step=int(global_step), adapter_name=adapter_name)

    def promote_active_checkpoint(self, checkpoint_version: int, global_step: int) -> None:
        if getattr(self, "strategy", None) is None:
            raise RuntimeError("worker has no strategy")
        promote = getattr(self.strategy, "promote_active_checkpoint", None)
        if not callable(promote):
            raise RuntimeError(f"{type(self.strategy).__name__} does not support promote_active_checkpoint")
        promote(checkpoint_version=int(checkpoint_version), global_step=int(global_step))

    def promote_active_adapter_checkpoint(
        self, adapter_name: str, checkpoint_version: int, global_step: int
    ) -> None:
        """Promote a per-adapter cache version as active. Thin wrapper around strategy implementation."""
        if getattr(self, "strategy", None) is None:
            raise RuntimeError("worker has no strategy")
        fn = getattr(self.strategy, "promote_active_adapter_checkpoint", None)
        if not callable(fn):
            raise RuntimeError(f"{type(self.strategy).__name__} does not support promote_active_adapter_checkpoint")
        fn(str(adapter_name), int(checkpoint_version), int(global_step))

    def selective_sync_active_cache(
        self,
        *,
        sync_id: str,
        tgt_dp_ranks,
        tgt_workers,
        tgt_device_mapping,
        tgt_num_gpus_per_worker: int,
        model_update_name: str | None = None,
        comm_plan=None,
        is_leader: bool = False,
        adapters_to_sync: list[str] | None = None,
    ) -> None:
        if getattr(self, "strategy", None) is None:
            raise RuntimeError("worker has no strategy")
        fn = getattr(self.strategy, "selective_sync_active_cache", None)
        if not callable(fn):
            raise RuntimeError(f"{type(self.strategy).__name__} does not support selective_sync_active_cache")
        self.logger.info(
            "[schedrl][selective_sync] worker_call_enter "
            f"sync_id={sync_id} tgt_dp_ranks={list(tgt_dp_ranks)} "
            f"tgt_num_gpus_per_worker={tgt_num_gpus_per_worker}"
        )
        fn(
            sync_id=str(sync_id),
            tgt_dp_ranks=tgt_dp_ranks,
            tgt_workers=tgt_workers,
            tgt_device_mapping=tgt_device_mapping,
            tgt_num_gpus_per_worker=int(tgt_num_gpus_per_worker),
            model_update_name=model_update_name,
            comm_plan=comm_plan,
            is_leader=bool(is_leader),
            adapters_to_sync=adapters_to_sync,
        )
        self.logger.info(f"[schedrl][selective_sync] worker_call_exit sync_id={sync_id}")

    def add_lora(self, *args, **kwargs):
        if getattr(self, "strategy", None) is not None:
            self.strategy.add_lora(*args, **kwargs)
        else:
            self.logger.warning("worker has not strategy")

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE)
    def get_metrics(self, metric_names: Optional[List[str]] = None) -> DataProto:
        """
        Get performance metrics from the strategy layer.

        Args:
            metric_names: Optional list of specific metric names to filter

        Returns:
            Dictionary of metric names to aggregated values
        """
        if getattr(self, "strategy", None) is not None:
            metrics = self.strategy.get_metrics(metric_names=metric_names)
        else:
            metrics = {}
        return DataProto(meta_info={"metrics": metrics})
