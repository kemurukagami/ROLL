import ray

from roll.configs.base_config import PPOConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.functionals import reduce_metrics_list
from roll.utils.logging import get_logger

logger = get_logger()


def _aggregate_sender_stats(stats_list: list[dict]) -> dict:
    """Aggregate weight stats across PP stages (sum-of-sums, max-of-maxes, min-of-mins).

    Each entry in stats_list comes from one PP-stage reporter. For colocated path
    only one worker reports, so no aggregation is needed. For separated path, one
    reporter per PP stage — their stats must be combined.
    """
    result: dict = {}

    # Aggregate base stats.
    base_entries = [entry["base"] for entry in stats_list if "base" in entry]
    if base_entries:
        result["base"] = {
            "sum": sum(entry["sum"] for entry in base_entries),
            "max": max(entry["max"] for entry in base_entries),
            "min": min(entry["min"] for entry in base_entries),
        }

    # Aggregate per-adapter LoRA stats.
    all_adapter_names: set[str] = set()
    for entry in stats_list:
        if "lora" in entry:
            all_adapter_names.update(entry["lora"].keys())
    if all_adapter_names:
        lora_result: dict = {}
        for adapter_name in sorted(all_adapter_names):
            adapter_entries = [
                entry["lora"][adapter_name]
                for entry in stats_list
                if "lora" in entry and adapter_name in entry["lora"]
            ]
            if adapter_entries:
                lora_result[adapter_name] = {
                    "sum": sum(entry["sum"] for entry in adapter_entries),
                    "max": max(entry["max"] for entry in adapter_entries),
                    "min": min(entry["min"] for entry in adapter_entries),
                }
        if lora_result:
            result["lora"] = lora_result

    return result


class ModelUpdateGroup:
    def __init__(self, src_cluster: Cluster, tgt_cluster: Cluster, pipeline_config: PPOConfig, frequency=1):
        self.src_cluster = src_cluster
        self.tgt_cluster = tgt_cluster
        self.frequency = frequency
        self.pipeline_config = pipeline_config
        self.model_update_name = f"model_update/{self.src_cluster.cluster_name}_2_{self.tgt_cluster.cluster_name}"
        train_devices = set(src_cluster.worker_config.device_mapping or [])
        infer_devices = set(tgt_cluster.worker_config.device_mapping or [])

        assert (max(train_devices) - min(train_devices)) == (len(train_devices) - 1), f"{train_devices=} must be continuous"
        assert (max(infer_devices) - min(infer_devices)) == (len(infer_devices) - 1), f"{infer_devices=} must be continuous"

        ray.get(
            [
                train_worker.setup_model_update.remote(
                    infer_cluster=self.tgt_cluster, model_update_name=self.model_update_name
                )
                for train_worker in self.src_cluster.workers
            ]
        )

    def model_update(self, step=None, adapters_to_update: set[str] | None = None):
        if step % self.frequency != 0:
            return {}

        kwargs = {"model_update_name": self.model_update_name}
        if adapters_to_update is not None:
            kwargs["adapters_to_update"] = sorted(adapters_to_update)

        dataprotos: list[DataProto] = ray.get(
            [
                train_worker.start_model_update.remote(**kwargs)
                for train_worker in self.src_cluster.workers
            ]
        )

        # Extract weight_stats separately before reduce_metrics_list (which would
        # corrupt nested dicts via np.mean). Only non-empty stats from canonical
        # reporter workers are included.
        sender_stats_list = [
            dataproto.meta_info["weight_stats"]
            for dataproto in dataprotos
            if dataproto.meta_info.get("weight_stats")
        ]
        if sender_stats_list:
            aggregated_stats = _aggregate_sender_stats(sender_stats_list)
            if aggregated_stats:
                # Fire verify_model on all target infer workers and wait (fail-fast).
                verify_refs = [
                    infer_worker.verify_model.remote(expected_stats=aggregated_stats)
                    for infer_worker in self.tgt_cluster.workers
                ]
                ray.get(verify_refs)
                logger.info(
                    "[ModelUpdateGroup] verify_model ok tgt_workers=%d stats_keys=%s",
                    len(verify_refs), sorted(aggregated_stats.keys()),
                )

        return reduce_metrics_list([dataproto.meta_info["metrics"] for dataproto in dataprotos])
