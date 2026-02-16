import dataclasses
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import os
import ray
from ray.util.placement_group import PlacementGroup

from roll.platforms import current_platform
from roll.utils.ray_utils import get_visible_gpus, get_node_rank


class ResourceManager:
    def __init__(self, num_gpus_per_node, num_nodes):
        """
            The ResourceManager centrally manages the required GPU/CPU resources,
            facilitating Ray to deploy Actors on specified GPU devices.
        """
        # NOTE: Some environments can expose Ray "GPU" resources even when `torch.cuda.is_available()`
        # is False inside the driver/worker process (e.g., CUDA init failure or restricted device access).
        # For GPU placement groups we must use Ray's built-in "GPU" resource key to ensure bundles are schedulable.
        ray_device_key = current_platform.ray_device_key
        device_control_env_var = getattr(current_platform, "device_control_env_var", None)
        if int(num_gpus_per_node or 0) > 0:
            ray_device_key = "GPU"
            if not device_control_env_var:
                device_control_env_var = "CUDA_VISIBLE_DEVICES"

        available_resources = ray.available_resources()
        available_gpu = available_resources.get(ray_device_key, 0)

        nodes_maybe_used = []
        ray_nodes = ray.nodes()
        for node in ray_nodes:
            resource = node["Resources"]
            node_gpu_num = int(resource.get(ray_device_key, 0))
            if node_gpu_num >= num_gpus_per_node:
                nodes_maybe_used.append(node)
        nodes_maybe_used = sorted(nodes_maybe_used, key=lambda n: n["Resources"]["CPU"])

        ray_num_nodes = len(nodes_maybe_used)
        if num_nodes is None:
            num_nodes = ray_num_nodes

        assert num_nodes <= ray_num_nodes, (f"The Ray clusters(ray_num_nodes: {ray_num_nodes}) cannot meet the "
                                            f"required number of nodes (`num_nodes`{num_nodes}).")
        self.num_nodes = num_nodes
        self.gpu_per_node = num_gpus_per_node
        self.num_gpus = self.gpu_per_node * self.num_nodes
        self._pipeline_id = os.environ.get("PIPELINE_ID") or None
        self._pg_name_prefix = f"schedrl_pg:{self._pipeline_id}:" if self._pipeline_id else None

        if self.gpu_per_node > 0:
            assert self.num_gpus <= available_gpu, f"num_gpus {self.num_gpus} > available_gpu {available_gpu}"
            bundles = []
            for i in range(self.num_nodes):
                node = nodes_maybe_used[i]
                node_cpu = int(node["Resources"]["CPU"])
                bundles.append({ray_device_key: self.gpu_per_node, "CPU": max(node_cpu / 2, 1)})

            self.placement_groups = [
                ray.util.placement_group([bundle], name=f"{self._pg_name_prefix}{i}" if self._pg_name_prefix else None)
                for i, bundle in enumerate(bundles)
            ]
            ray.get([pg.ready() for pg in self.placement_groups])
            gpu_ranks = ray.get([
                get_visible_gpus.options(
                    placement_group=pg,
                    **({"num_gpus": self.gpu_per_node} if self.gpu_per_node > 0 else {})
                ).remote(device_control_env_var)
                for pg in self.placement_groups
            ])
            print(f"gpu ranks: {gpu_ranks}")
            self.node_ranks = ray.get(
                [get_node_rank.options(placement_group=pg).remote() for pg in self.placement_groups])
            if self.node_ranks.count(0) > 1:
                # NODE_RANK environment variable is not set in the cluster, so a default value is used for NODE_RANK.
                self.node_ranks = list(range(len(self.placement_groups)))

            self.gpu_ranks = [int(gpu_rank[0]) for gpu_rank in gpu_ranks]
            self.node2pg: Dict[int, PlacementGroup] = {}
            for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
                self.node2pg[node_rank] = placement_group
            print(f"node2pg: {self.node2pg}")
        else:
            assert self.num_nodes == 1
            node = nodes_maybe_used[0]
            node_cpu = int(node["Resources"]["CPU"])
            bundles = [{"CPU": node_cpu}] * self.num_nodes
            self.placement_groups = [
                ray.util.placement_group([bundle], name=f"{self._pg_name_prefix}cpu:{i}" if self._pg_name_prefix else None)
                for i, bundle in enumerate(bundles)
            ]
            ray.get([pg.ready() for pg in self.placement_groups])
            self.node_ranks = [0]
            self.node2pg: Dict[int, PlacementGroup] = {}
            for node_rank, placement_group in zip(self.node_ranks, self.placement_groups):
                self.node2pg[node_rank] = placement_group

    def nodes_placement_group(self, node_rank) -> PlacementGroup:
        """
        mesh table是 m×n，获取第node_rank nodel上gpu_rank的PlacementGroup，用于把ray.Actor部署到指定的GPU上
        """
        return self.node2pg[node_rank]

    def destroy_placement_group(self):
        [ray.util.remove_placement_group(pg) for pg in self.placement_groups]

    def allocate_placement_group(self, world_size, device_mapping: List[int] = None) -> List[List[Dict]]:
        """
            Allocate resources according to device_mapping (numbered by GPU RANK)
            - GPUs: Specify required GPU indices via device_mapping
            - CPUs: Specify via world_size

            Return Type: List[List[Dict]]
              Dict Keys:
                - node_rank
                - gpu_rank
                - placement_group
              List[Dict]: Represents GPUs allocated to a worker and access to placement groups
              Example: If num_gpus_per_worker=8, then len(List[Dict])=8

            A Worker is defined as a group of resource owners (can span multiple machines) that can independently use allocated resources to execute computation operations.
        """
        allocated_pg = []
        ray_address = f"{ray.get_runtime_context().gcs_address}"
        if device_mapping:
            num_gpus_per_worker = len(device_mapping) // world_size
            grouped_ranks = [
                list(device_mapping[i : i + num_gpus_per_worker])
                for i in range(0, len(device_mapping), num_gpus_per_worker)
            ]
            for group in grouped_ranks:
                pg_list = []
                for rank in group:
                    node_rank = rank // self.gpu_per_node
                    gpu_rank = rank % self.gpu_per_node

                    assert node_rank < self.num_nodes, (f"device_mapping used gpus are more than "
                                                        f"num_nodes×num_gpus_per_node={self.num_nodes}×{self.gpu_per_node}")

                    pg = self.nodes_placement_group(node_rank)
                    pg_list.append(
                        dict(node_rank=node_rank, gpu_rank=gpu_rank, placement_group=pg, ray_address=ray_address)
                    )
                allocated_pg.append(pg_list)
        else:
            # Try to spread the CPU workers across various nodes to avoid the out-of-memory (OOM) situation caused
            # by the concentration of CPU workers in one place and the resulting peak memory usage.
            for rank in range(world_size):
                node_rank = rank % self.num_nodes
                allocated_pg.append(
                    [
                        dict(
                            node_rank=node_rank,
                            gpu_rank=None,
                            placement_group=self.nodes_placement_group(node_rank),
                            ray_address=ray_address,
                        )
                    ]
                )

        assert len(allocated_pg) == world_size

        return allocated_pg
