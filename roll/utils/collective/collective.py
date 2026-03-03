from datetime import timedelta
from typing import Optional, Union

from torch._C._distributed_c10d import ReduceOp
from torch.distributed import Backend
import torch.distributed as dist

from roll.platforms import current_platform
from roll.utils.collective.pg_utils import init_custom_process_group
from roll.utils.logging import get_logger

logger = get_logger()


class GroupManager:

    def __init__(self):
        """
        ProcessGroup manager backed by torch.distributed.
        ref: https://github.com/ray-project/ray/blob/master/python/ray/util/collective/collective.py
        """
        self._name_group_map = {}
        # Reverse map: group object → name. Needed for backend inspection via get_group_backend().
        self._group_name_map = {}

    def create_collective_group(
        self,
        backend,
        world_size,
        rank,
        master_addr: str,
        master_port: int,
        group_name,
        global_ranks=None,
        timeout_s: Optional[float] = None,
    ):
        # Convert seconds to timedelta; None keeps the PyTorch default (1800s).
        # Configurable timeout lets callers tune for slow cross-node initializations.
        timeout = None if timeout_s is None else timedelta(seconds=float(timeout_s))
        group = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=timeout,
            world_size=world_size,
            rank=rank,
            group_name=group_name,
            global_ranks=global_ranks
        )
        self._name_group_map[group_name] = group
        self._group_name_map[group] = group_name
        return group

    def is_group_exist(self, group_name):
        return group_name in self._name_group_map

    def get_group_by_name(self, group_name):
        """Get the collective group handle by its name."""
        if not self.is_group_exist(group_name):
            # Fail fast: returning None here caused silent hangs in downstream collective ops.
            raise KeyError("The group '{}' is not initialized.".format(group_name))
        return self._name_group_map[group_name]

    def destroy_collective_group(self, group_name):
        """Group destructor."""
        if not self.is_group_exist(group_name):
            raise KeyError("The group '{}' does not exist.".format(group_name))

        # release the collective group resource
        g = self._name_group_map[group_name]
        try:
            dist.destroy_process_group(g)
        except Exception as e:
            # Wrap with group name so callers can identify which group failed.
            raise RuntimeError(f"Failed to destroy process group: group_name={group_name}") from e
        # clean up the dicts
        del self._group_name_map[g]
        del self._name_group_map[group_name]


_group_mgr = GroupManager()


def init_collective_group(
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    backend: Union[str, Backend] = current_platform.communication_backend,
    group_name: str = "default",
    global_ranks: Optional[list] = None,
    # Per-group timeout (seconds). None uses PyTorch's default (1800s).
    # Set explicitly for groups that span slow cross-node links.
    timeout_s: Optional[float] = None,
):
    global _group_mgr
    if not group_name:
        raise ValueError("group_name '{}' needs to be a string.".format(group_name))

    if _group_mgr.is_group_exist(group_name):
        raise RuntimeError("Trying to initialize a group twice.")

    assert world_size > 0
    assert rank >= 0
    assert rank < world_size
    logger.info(
        "[rlix][collective] init_enter "
        f"group_name={group_name} backend={backend} rank={rank}/{world_size} master={master_addr}:{master_port} "
        f"timeout_s={timeout_s}"
    )
    _group_mgr.create_collective_group(
        backend,
        world_size,
        rank,
        master_addr,
        master_port,
        group_name,
        global_ranks=global_ranks,
        timeout_s=timeout_s,
    )
    logger.info(f"[rlix][collective] init_exit group_name={group_name} rank={rank}/{world_size}")


def allreduce(tensor, group_name: str = "default", op=ReduceOp.SUM):
    global _group_mgr
    dist.all_reduce(tensor, op=op, group=_group_mgr.get_group_by_name(group_name))


def broadcast(tensor, src_rank: int = 0, group_name: str = "default", async_op=False):
    global _group_mgr
    return dist.broadcast(tensor, src=src_rank, group=_group_mgr.get_group_by_name(group_name), async_op=async_op)

def barrier(group_name):
    global _group_mgr
    dist.barrier(group=_group_mgr.get_group_by_name(group_name), device_ids=[0])

def all_gather_object(object_list, obj, group_name):
    global _group_mgr
    dist.all_gather_object(object_list, obj, group=_group_mgr.get_group_by_name(group_name))

def broadcast_object_list(object_list, src=None, group_name="default", device=None, group_src=None):
    global _group_mgr
    assert (src is not None and group_src is None) or (src is None and group_src is not None),\
        ("Either src or group_src must be set, but they cannot be set simultaneously.")
    dist.broadcast_object_list(object_list, src=src, group_src=group_src, group=_group_mgr.get_group_by_name(group_name))


def destroy_collective_group(group_name: str) -> None:
    global _group_mgr
    _group_mgr.destroy_collective_group(group_name)


def get_group_backend(group_name: str):
    # Expose backend lookup for callers that need to branch on CPU/GPU transport behavior.
    global _group_mgr
    group = _group_mgr.get_group_by_name(group_name)
    return dist.get_backend(group)
