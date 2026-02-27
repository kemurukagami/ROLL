import asyncio
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray._private import profiling
from ray.runtime_env import RuntimeEnv
from tqdm import tqdm

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import RequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, EnvMonitorConfig
from roll.distributed.scheduler.rollout_mock_mixin import RolloutMockMixin
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.functionals import append_to_dict
from roll.utils.import_utils import safe_import_class
from roll.utils.constants import RAY_NAMESPACE, schedrl_env_vars
from roll.utils.logging import get_logger

logger = get_logger()


class EnvActivityMonitor:
    """Environment activity monitor for tracking and detecting hung envs."""

    def __init__(self, config: EnvMonitorConfig, group_queue_dict: Dict[int, 'GroupQueue']):
        """
        Args:
            config: EnvMonitorConfig object
            group_queue_dict: Reference to GroupQueue dict for checking episode status
        """
        self.group_queue_dict = group_queue_dict
        self.enable = config.enable

        # Configuration parameters
        self.monitor_interval = config.monitor_interval  # seconds
        self.hung_timeout = config.hung_timeout  # seconds (default: 1 hour)

        # Tracking data structures - Dual-timestamp approach
        # Track when env starts processing an episode
        # Key: ((group_id, env_id), episode_id) -> Value: timestamp
        self.env_episode_start: Dict[Tuple[Tuple[int, int], int], float] = {}

        # Track when env submits episode rollout
        # Key: ((group_id, env_id), episode_id) -> Value: timestamp
        self.env_episode_submit: Dict[Tuple[Tuple[int, int], int], float] = {}

        # Track each env's current episode (for cleanup)
        # Key: (group_id, env_id) -> Value: episode_id
        self.env_current_episode: Dict[Tuple[int, int], int] = {}

        # Monitor task
        self.monitor_task: Optional[asyncio.Task] = None

    def record_episode_start(self, group_id: int, env_id: int, episode_id: int):
        """
        Record when env starts processing a new episode.
        Called from GroupQueue.get_episode_id() when an episode is assigned to an env.

        Args:
            group_id: Group ID
            env_id: Environment ID
            episode_id: Episode ID assigned to this env
        """
        if not self.enable:
            return

        env_key = (group_id, env_id)
        episode_key = ((group_id, env_id), episode_id)

        # Automatic cleanup: Remove old episode records for this env
        old_episode_id = self.env_current_episode.get(env_key)
        if old_episode_id is not None and old_episode_id != episode_id:
            old_episode_key = ((group_id, env_id), old_episode_id)
            self.env_episode_start.pop(old_episode_key, None)
            self.env_episode_submit.pop(old_episode_key, None)

        # Record new episode start time
        self.env_episode_start[episode_key] = time.time()
        self.env_current_episode[env_key] = episode_id

    def record_activity(self, group_id: int, env_id: int, episode_id: int, rollout: Optional[DataProto]):
        """
        Record env activity when submitting a rollout.
        Called from GroupQueueManager.put() when env submits rollout.

        Args:
            group_id: Group ID
            env_id: Environment ID
            episode_id: Episode ID
            rollout: Rollout data (None means env is exiting)
        """
        if not self.enable:
            return

        env_key = (group_id, env_id)
        episode_key = ((group_id, env_id), episode_id)

        if rollout is None:
            # Env calls put(..., None) to signal exit, remove all tracking
            self.env_episode_start.pop(episode_key, None)
            self.env_episode_submit.pop(episode_key, None)
            self.env_current_episode.pop(env_key, None)
            return

        # Normal rollout submission, record submit time
        self.env_episode_submit[episode_key] = time.time()

    def start_monitoring(self):
        """Start background monitoring task."""
        if not self.enable or self.monitor_task is not None:
            return

        self.monitor_task = asyncio.create_task(self._monitor_loop())

    def stop_monitoring(self):
        """Stop background monitoring task."""
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None

    def cleanup_episode(self, group_id: int, episode_id: int):
        """
        Clean up monitoring data for completed episode.
        Note: With dual-timestamp tracking, cleanup is mostly automatic in record_episode_start().
        This method is kept for compatibility but has minimal work to do.
        """
        if not self.enable:
            return

        # No cleanup needed - dual-timestamp approach handles cleanup automatically
        # when new episodes start via record_episode_start()
        pass

    async def _monitor_loop(self):
        """Background monitoring task that periodically detects hung envs and logs."""
        while True:
            try:
                await asyncio.sleep(self.monitor_interval)
                self.check_and_log_hung_envs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[EnvMonitor] Monitor loop error: {e}")

    def check_and_log_hung_envs(self):
        """
        Detect and log hung envs using dual-timestamp tracking.

        Detection Logic:
        - For each env with a start time recorded:
          - Check if current episode has a submit time
          - If no submit time and (now - start_time) > hung_timeout:
            → Report as hung
          - If submit time exists:
            → Env has completed, don't report (even if timestamp is old)
        """
        now = time.time()
        hung_envs_by_group = {}  # group_id -> list of hung env info

        # Iterate over all episode start records
        for episode_key, start_time in self.env_episode_start.items():
            (group_id, env_id), episode_id = episode_key

            # Check if this episode has been submitted
            submit_time = self.env_episode_submit.get(episode_key)

            if submit_time is None:
                # Env started but hasn't submitted (still processing)
                inactive_time = now - start_time

                if inactive_time > self.hung_timeout:
                    # Report as hung
                    if group_id not in hung_envs_by_group:
                        hung_envs_by_group[group_id] = []

                    hung_envs_by_group[group_id].append({
                        "env_id": env_id,
                        "episode_id": episode_id,
                        "inactive_seconds": int(inactive_time),
                    })
            # else: Episode submitted, env is waiting for next episode (normal)

        # Output logs
        if hung_envs_by_group:
            for group_id, hung_envs in hung_envs_by_group.items():
                hung_env_ids = [e["env_id"] for e in hung_envs]
                logger.warning(
                    f"[EnvMonitor] Group {group_id}: Detected {len(hung_envs)} hung envs: {hung_env_ids}"
                )
                for env_info in hung_envs[:5]:  # Only log details for first 5
                    logger.warning(
                        f"[EnvMonitor]   - env_id={env_info['env_id']}, "
                        f"episode_id={env_info['episode_id']}, "
                        f"inactive_for={env_info['inactive_seconds']}s"
                    )
                if len(hung_envs) > 5:
                    logger.warning(f"[EnvMonitor]   ... and {len(hung_envs) - 5} more")


@dataclass
class GroupData:
    group_id: int
    episode_id: int
    create_step: int
    created_at: float
    rollouts: List[DataProto] = field(default_factory=list)
    running_rollouts: int = 0 

class GroupQueue:
    def __init__(
        self,
        group_id,
        progress_bar: tqdm,
        group_size,
        group_size_redundancy,
        max_traj_per_env,
        async_generation_ratio,
        group_filter,
        env_monitor: Optional['EnvActivityMonitor'] = None,
    ):
        self.group_id = group_id
        self.progress_bar = progress_bar

        self.group_size = group_size
        self.group_size_redundancy = group_size_redundancy
        self.max_traj_per_env = max_traj_per_env
        self.async_generation_ratio = async_generation_ratio
        self.group_filter = group_filter
        self.group_filter_count = 0
        self.env_monitor = env_monitor

        self.current_step = None
        self.next_episode_id = 0
        self.groups: Dict[int, GroupData] = {}

        self.progress = asyncio.Event()
        self.complete = asyncio.Event()

        self.quit = False

    def clear(self):
        self.current_step = None
        self.next_episode_id = 0
        self.groups.clear()

        self.progress = asyncio.Event()
        self.complete = asyncio.Event()

    def shutdown(self):
        self.quit = True
        self.groups.clear()
        self.progress.set()

    def advance_group(self, create_step):
        assert not self.quit
        self.groups[self.next_episode_id] = GroupData(
            group_id=self.group_id,
            episode_id=self.next_episode_id,
            create_step=create_step,
            created_at=time.time(),
        )
        self.next_episode_id += 1

    def _advance_step(self, create_step):
        if self.max_traj_per_env is None:
            return
        for _ in range(self.max_traj_per_env):
            self.advance_group(create_step)

    def advance_step(self, step):
        if self.current_step is None:
            # first time into advance_step, generate extra groups for async training
            for _ in range(self.async_generation_ratio):
                self._advance_step(step)
        else:
            # remove outdated groups for async training
            expired_episodes = []
            for episode_id, group in self.groups.items():
                if step - group.create_step > self.async_generation_ratio:
                    expired_episodes.append(episode_id)
            for episode_id in expired_episodes:
                self.groups.pop(episode_id)
                if self.env_monitor:
                    self.env_monitor.cleanup_episode(self.group_id, episode_id)

        self.current_step = step
        self._advance_step(step)
        self.progress.set()

    async def get_episode_id(self, env_id: Optional[int] = None) -> Optional[int]:
        """
        Get the next episode_id for an env to process.

        Args:
            env_id: Environment ID requesting work (None for backward compatibility)

        Returns:
            episode_id to process, or None if shutting down
        """
        while not self.quit:
            # iterate over groups in order
            for episode_id, group in self.groups.items():
                if group.running_rollouts < self.group_size + self.group_size_redundancy:
                    group.running_rollouts += 1

                    # Record episode start for hang detection
                    if self.env_monitor and env_id is not None:
                        self.env_monitor.record_episode_start(self.group_id, env_id, episode_id)

                    return episode_id
            if self.max_traj_per_env is None:
                while self.current_step is None:
                    self.progress.clear()
                    await self.progress.wait()
                self.advance_group(self.current_step)
                continue
            else:
                self.progress.clear()
                await self.progress.wait()
        return None

    def put(self, episode_id, start_step, rollout) -> Dict[str, Any]:
        if episode_id not in self.groups:  # ignore rollouts from outdated episode
            return {"status": "ignored", "non_null_added": 0}
        group = self.groups[episode_id]
        assert start_step >= group.create_step, f"{start_step=} {group.create_step=}"
        group.rollouts.append(rollout)
        if len(group.rollouts) == self.group_size:
            if all(rollout is None for rollout in group.rollouts):
                logger.info(f"GroupQueue: group {self.group_id} exit")
                self.complete.set()
                return {"status": "exit", "non_null_added": 0}
            elif self.group_filter.filter(group_id=self.group_id, episode_id=episode_id, group=group.rollouts):
                logger.info(f"filter rollout group {group.group_id} episode {group.episode_id}")
                self.group_filter_count += 1
                self.groups.pop(episode_id)
                if self.env_monitor:
                    self.env_monitor.cleanup_episode(self.group_id, episode_id)
                self.advance_group(create_step=self.current_step)
                return {"status": "filtered", "non_null_added": 0 if rollout is None else 1}
            else:
                self.complete.set()
                self.progress_bar.update(self.group_size)
                return {"status": "completed", "non_null_added": 0 if rollout is None else 1}
        return {"status": "partial", "non_null_added": 0 if rollout is None else 1}

    async def get(self) -> GroupData:
        while True:
            while not self.groups:
                self.complete.clear()
                await self.complete.wait()
            episode_id = next(iter(self.groups)) # must consume the first group (smallest episode_id)
            group = self.groups[episode_id]
            if len(group.rollouts) >= self.group_size:
                self.groups.pop(episode_id)
                if self.env_monitor:
                    self.env_monitor.cleanup_episode(self.group_id, episode_id)
                return group
            self.complete.clear()
            await self.complete.wait()

@ray.remote
class GroupQueueManager:
    def __init__(self, config, env_manager_config: EnvManagerConfig, mode):
        self.mode = mode
        self.env_manager_config = env_manager_config
        self.group_size = self.env_manager_config.group_size
        self.progress_bar = tqdm(desc=f"{self.mode} rollout progress(total trajectory)", mininterval=self.env_manager_config.max_traj_per_env)
        self.pending_gets = set()
        self.rollout_complete = {}

        self.pipeline_id = os.environ.get("PIPELINE_ID") or None
        self._schedrl_enabled = os.environ.get("SCHEDRL_CONTROL_PLANE", "") == "schedrl" and self.mode == "train"
        self.adapter_id = self.env_manager_config.tags[0] if getattr(self.env_manager_config, "tags", None) else None
        self._schedrl_scheduler = None
        if self._schedrl_enabled:
            if not self.pipeline_id:
                raise RuntimeError("SCHEDRL_CONTROL_PLANE=schedrl requires PIPELINE_ID to be set")
            try:
                self._schedrl_scheduler = ray.get_actor("schedrl:scheduler", namespace="schedrl")
            except Exception as e:
                # Expectation: the central schedrl scheduler actor ('schedrl:scheduler')
                # must already be created before GroupQueueManager is instantiated.
                # Fail loudly with a clear message to aid debugging of startup ordering.
                raise RuntimeError(
                    "Failed to resolve schedrl:scheduler in namespace 'schedrl'. "
                    "GroupQueueManager expects the central scheduler actor to be present before startup; "
                    "ensure the orchestrator created it earlier or that startup ordering is correct."
                ) from e

        group_filter_cls = safe_import_class(env_manager_config.group_filter_cls)
        assert group_filter_cls
        self.group_filter = group_filter_cls(config, env_manager_config, mode)

        if self.mode == "train":
            self.async_generation_ratio = config.async_generation_ratio
            self.max_traj_per_env = env_manager_config.max_traj_per_env if config.rollout_batch_size > 0 else None
            self.rollout_batch_size = int(config.rollout_batch_size)
        else:
            self.async_generation_ratio = 0
            self.max_traj_per_env = env_manager_config.max_traj_per_env if config.val_batch_size > 0 else None
            self.rollout_batch_size = int(config.val_batch_size)

        # Initialize env activity monitor first (before creating GroupQueues)
        self.group_queue: Dict[int, GroupQueue] = {}
        self.env_monitor = EnvActivityMonitor(
            config=config.env_monitor,
            group_queue_dict=self.group_queue
        )

        # Create GroupQueues with env_monitor reference
        for rank, rank_env_configs in env_manager_config.env_configs.items():
            for env_id, env_config in rank_env_configs.items():
                group_id = env_config["group_id"]
                if group_id not in self.group_queue:
                    self.group_queue[group_id] = GroupQueue(
                        group_id=group_id,
                        progress_bar=self.progress_bar,
                        group_size=env_manager_config.group_size,
                        group_size_redundancy=env_manager_config.group_size_redundancy,
                        max_traj_per_env=self.max_traj_per_env,
                        async_generation_ratio=self.async_generation_ratio,
                        group_filter=self.group_filter,
                        env_monitor=self.env_monitor,
                    )

        # Start monitoring after all GroupQueues are created
        if config.env_monitor.enable:
            self.env_monitor.start_monitoring()

        # for debug
        self.total = 0
        self.waiting = 0

        # Progress tracking (SchedRL only; fork parity).
        self._progress_last_bucket: Optional[int] = None
        self._progress_new_batch = False
        self._progress_total_required_estimated = self._estimate_total_required()
        self._progress_collected_estimated = 0
        self._progress_episode_non_null: Dict[Tuple[int, int], int] = {}
        if self._schedrl_enabled:
            self._mark_new_batch()
            self._maybe_emit_progress(current_train_step=None)

    def _resolve_num_return_sequences(self) -> int:
        # SchedRL progress should be expressed in "trajectory units" that match the rollout batch contract.
        #
        # In ROLL's request scheduler, the effective number of finished samples required for a "batch" is
        # `batch_size * num_return_sequences` (not scaled by async_generation_ratio).
        raw = None
        generating_args = getattr(self.env_manager_config, "generating_args", None)
        if generating_args is not None:
            raw = getattr(generating_args, "num_return_sequences", None)
        if raw is None:
            actor_infer = getattr(self.config, "actor_infer", None)
            generating_args = getattr(actor_infer, "generating_args", None) if actor_infer is not None else None
            if generating_args is not None:
                raw = getattr(generating_args, "num_return_sequences", None)

        n = 1 if raw is None else int(raw)
        if n <= 0:
            raise RuntimeError(f"Invalid num_return_sequences={raw!r}; expected > 0")
        return n

    def _estimate_total_required(self) -> int:
        if self.max_traj_per_env is None:
            return 0
        # Denominator for progress is the per-step rollout batch target (trajectory units).
        # It must not depend on async_generation_ratio (async controls overlap/pausing, not the required sample count).
        num_return_sequences = self._resolve_num_return_sequences()
        return int(self.rollout_batch_size) * int(num_return_sequences)

    def _mark_new_batch(self) -> None:
        self._progress_total_required_estimated = self._estimate_total_required()
        self._progress_new_batch = True

    def _compute_progress(self) -> Tuple[int, int, int, Optional[float]]:
        if self.max_traj_per_env is None:
            # Unbounded mode: do not report progress in Phase 3.
            return 0, 0, 0, None

        total_required = self._progress_total_required_estimated
        collected = min(self._progress_collected_estimated, total_required)

        oldest_ts: Optional[float] = None
        for group_queue in self.group_queue.values():
            for group in group_queue.groups.values():
                if len(group.rollouts) < self.group_size:
                    if oldest_ts is None or group.created_at < oldest_ts:
                        oldest_ts = group.created_at

        remaining = max(total_required - collected, 0)
        return total_required, collected, remaining, oldest_ts

    def _maybe_emit_progress(self, *, current_train_step: Optional[int]) -> None:
        if not self._schedrl_enabled:
            return
        if self.max_traj_per_env is None:
            return
        if self._schedrl_scheduler is None:
            raise RuntimeError("SCHEDRL progress enabled but schedrl:scheduler handle is missing")
        if not self.pipeline_id:
            raise RuntimeError("SCHEDRL progress enabled but PIPELINE_ID is missing")

        total_required, collected, remaining, oldest_ts = self._compute_progress()
        if total_required <= 0:
            return

        percent_completed = float(collected) / float(max(total_required, 1))
        # 2% buckets (0..50). Bucket 0 means 0% completed, bucket 50 means 100% completed.
        bucket = math.floor(percent_completed * 50)

        should_emit = (
            bucket != self._progress_last_bucket
            or remaining == 0
            or collected >= total_required
            or self._progress_new_batch
        )
        if not should_emit:
            return

        emitted_for_new_batch = self._progress_new_batch
        self._progress_last_bucket = bucket
        self._progress_new_batch = False

        from schedrl.protocol.types import ProgressReport

        report = ProgressReport(
            pipeline_id=str(self.pipeline_id),
            queued_trajectories=0,
            inflight_trajectories=0,
            step_target_trajectories=int(total_required),
            percent_completed=float(collected) / float(max(total_required, 1)),
            oldest_unfinished_creation_ts=oldest_ts,
            fifo_timestamp=time.time(),
            metrics={
                "mode": self.mode,
                "remaining": int(remaining),
                "bucket": int(bucket),
                "new_batch": bool(emitted_for_new_batch),
                "current_train_step": current_train_step,
                "adapter_id": self.adapter_id,
            },
        )
        self._schedrl_scheduler.report_progress.remote(report)

    def collect_metrics(self):
        group_filter_count = 0
        for group_queue in self.group_queue.values():
            group_filter_count += group_queue.group_filter_count
            group_queue.group_filter_count = 0
        return {"scheduler/group_filter_count": group_filter_count}

    def clear(self):
        self.rollout_complete = {}
        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()
        for group_queue in self.group_queue.values():
            group_queue.clear()
        if self.max_traj_per_env is not None:
            self._progress_collected_estimated = 0
            self._progress_episode_non_null.clear()
        self._mark_new_batch()
        self._maybe_emit_progress(current_train_step=None)

    def advance_step(self, step):
        for group_queue in self.group_queue.values():
            group_queue.advance_step(step)
        if self.max_traj_per_env is not None:
            self._progress_collected_estimated = 0
            self._progress_episode_non_null.clear()
        self._mark_new_batch()
        self._maybe_emit_progress(current_train_step=int(step) if step is not None else None)

    async def get_episode_id(self, group_id, env_id=None):
        """
        Get the next episode ID for an environment.

        Args:
            group_id: Group ID
            env_id: Environment ID (for hang detection tracking)

        Returns:
            episode_id to process
        """
        assert group_id in self.group_queue
        return await self.group_queue[group_id].get_episode_id(env_id)

    def shutdown(self):
        # Stop monitoring task
        self.env_monitor.stop_monitoring()

        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()
        for group_queue in self.group_queue.values():
            group_queue.shutdown()

    def put(self, group_id, episode_id, start_step, rollout: DataProto, env_id=None):
        """
        Put rollout data to queue.

        Args:
            group_id: Group ID
            episode_id: Episode ID
            start_step: Starting step
            rollout: Rollout data (can be None for final submission)
            env_id: Environment ID (optional, for monitoring)

        Backward compatibility:
        - Old calls: put(group_id, episode_id, start_step, rollout) - env_id defaults to None
        - New calls: put(group_id, episode_id, start_step, rollout, env_id) - enables monitoring
        """
        assert group_id in self.group_queue

        # Record env activity only if env_id is provided
        if env_id is not None:
            self.env_monitor.record_activity(group_id, env_id, episode_id, rollout)

        self.waiting += 1
        put_result = self.group_queue[group_id].put(episode_id, start_step, rollout)
        if self.max_traj_per_env is not None:
            status = str(put_result.get("status", ""))
            non_null_added = int(put_result.get("non_null_added", 0))
            episode_key = (group_id, episode_id)

            if non_null_added:
                self._progress_episode_non_null[episode_key] = self._progress_episode_non_null.get(episode_key, 0) + 1
                self._progress_collected_estimated += non_null_added

            if status in {"filtered", "exit"}:
                rolled_back = self._progress_episode_non_null.pop(episode_key, 0)
                self._progress_collected_estimated = max(self._progress_collected_estimated - rolled_back, 0)
            elif status == "completed":
                self._progress_episode_non_null.pop(episode_key, None)
        self.waiting -= 1
        self.total += 1
        self._maybe_emit_progress(current_train_step=int(start_step) if start_step is not None else None)

    async def get_batch(self, batch_size, current_step) -> List[DataProto]:
        """
        return completed rollouts group by group_id with least start_step
        """
        # TODO: No need to get from every group queue, instead we can reuse 
        # a group queue as long as there are enough rollouts to avoid tail-latency?
        # But this will cause im-balance in episode_id.

        # When batch_size < 0, iterate until exit run_rollout_loop immediately.
        ret: List[DataProto] = []
        progress_bar = tqdm(desc=f"{self.mode} rollout get_batch progress(trajectory)", mininterval=self.group_size)
        while batch_size < 0 or len(ret) < batch_size:

            if len(self.rollout_complete) == len(self.group_queue):
                break

            async def wait_a_episode():
                # Only wait for new episode when there are no pending GroupQueue.get,
                # this way we can avoid starvation of some env.
                if not self.pending_gets:
                    pending = set(
                        [
                            asyncio.create_task(self.group_queue[group_id].get(), name=str(group_id))
                            for group_id in self.group_queue if str(group_id) not in self.rollout_complete
                        ]
                    )
                else:
                    pending = self.pending_gets
                    self.pending_gets = set()

                while pending and (batch_size < 0 or len(ret) < batch_size):

                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    while done and (batch_size < 0 or len(ret) < batch_size):
                        d = done.pop()
                        try:
                            group = await d
                        except asyncio.CancelledError:
                            # Best-effort cleanup: cancellation is expected during clear()/shutdown().
                            continue
                        except Exception as e:
                            # Fail-fast: clean up any outstanding tasks and surface the error.
                            for t in pending:
                                t.cancel()
                            raise RuntimeError(f"GroupQueue.get() task failed (group_id={d.get_name()!r})") from e
                        group_rollout = group.rollouts
                        self.total -= len(group_rollout)

                        group_rollout = [rollout for rollout in group_rollout if rollout is not None]
                        if len(group_rollout) == 0:
                            self.rollout_complete[d.get_name()] = True
                            continue

                        if current_step - group.create_step > self.async_generation_ratio:
                            logger.info(f"ignore rollout, current_step({current_step}) - create_step({group.create_step}) "
                                        f"exceed async_generation_ratio({self.async_generation_ratio}) "
                                        f"{group.group_id=} {group.episode_id=}")
                            continue

                        group_rollout = group_rollout[:self.group_size]
                        ret.extend(group_rollout)
                        progress_bar.update(len(group_rollout))

                    assert batch_size < 0 or (done and len(ret) >= batch_size) or (not done and len(ret) <= batch_size), f"{batch_size=}, {len(ret)=}, {done=}"
                    if done:
                        self.pending_gets.update(done)
                self.pending_gets.update(pending)

            await wait_a_episode()
        get_batch_return_start_time = time.time()
        for d in ret:
            d.meta_info["get_batch_return_start_time"] = get_batch_return_start_time
        return ret

class RolloutScheduler(RolloutMockMixin):
    """
    Usage:
        # User should control load_states/offload_states in pipeline by themselves.
        actor_infer
        train_rollout_scheduler = RolloutScheduler(actor_infer)
        val_rollout_scheduler = RolloutScheduler(actor_infer)
        while True:
            ray.get(train_rollout_scheduler.suspend.remote())
            model_update()
            if val:
                ray.get(val_rollout_scheduler.get_batch.remote())
            ray.get(train_rollout_scheduler.get_batch.remote())
            rollout()
        ray.get(train_rollout_scheduler.shutdown.remote())
    """
    async def __init__(self, config, env_manager_config: EnvManagerConfig, resource_manager, infer_cluster, mode, request_scheduler=None, collator=None):
        # NOTE: This actor is async. Avoid blocking calls in __init__ and log each construction phase
        # to pinpoint startup stalls (e.g., placement group allocation, child actor creation).
        self.logger = logger
        self.logger.info(f"[RolloutScheduler] __init__ enter mode={mode}")
        self.config = config
        self.env_manager_config = env_manager_config
        self.resource_manager = resource_manager
        self.infer_cluster = infer_cluster
        self.mode = mode
        self.pipeline_id = os.environ.get("PIPELINE_ID") or None
        if self.pipeline_id is None and os.environ.get("ROLL_RAY_NAMESPACE"):
            raise RuntimeError("PIPELINE_ID must be set when ROLL_RAY_NAMESPACE is set (multi-pipeline mode)")

        env_num = self.env_manager_config.world_size * self.env_manager_config.max_env_num_per_worker

        # Ray creates separate worker processes for these control-plane actors (queue + request scheduler).
        # In this environment we hit OS thread limits during import-time TorchInductor initialization inside
        # those workers. Disable torch.compile / inductor compile workers and cap common thread pools.
        env_vars = {
                "TORCH_COMPILE_DISABLE": "1",
                # TorchInductor async compile uses a subprocess pool when compile_threads > 1.
                # In this environment that can fail with EAGAIN (fork/pthread_create) and crash Ray workers.
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                # Reduce Ray core worker RPC thread footprint (helps avoid hitting OS thread limits).
                "RAY_num_server_call_thread": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
        }
        # Ensure per-pipeline env vars are visible in these control-plane actor processes in SchedRL mode.
        env_vars.update(schedrl_env_vars())
        runtime_env = RuntimeEnv(env_vars=env_vars)

        self.logger.info(f"[RolloutScheduler] creating GroupQueueManager mode={self.mode}")
        self.env_output_queue = GroupQueueManager.options(
            name=(
                # Include env-manager name so multiple train schedulers (one per tag) do not collide on actor name.
                f"{self.pipeline_id}_group_queue_manager_{self.env_manager_config.name}_{mode}"
                if self.pipeline_id
                else f"GroupQueueManager-{self.env_manager_config.name}-{mode}"
            ),
            namespace=RAY_NAMESPACE,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False),
            runtime_env=runtime_env,
            max_concurrency=env_num + 1,  # reserve extra one for get_batch
        ).remote(
            self.config,
            self.env_manager_config,
            mode
        )
        self.logger.info(f"[RolloutScheduler] created GroupQueueManager mode={self.mode}")

        if request_scheduler is not None:
            self.generate_scheduler = request_scheduler
            self.logger.info(f"[RolloutScheduler] using SHARED RequestScheduler mode={self.mode}")
        else:
            self.logger.info(f"[RolloutScheduler] creating RequestScheduler mode={self.mode}")
            self.generate_scheduler = RequestScheduler.options(
                    name=(
                        f"{self.pipeline_id}_request_scheduler_{mode}"
                        if self.pipeline_id
                        else f"RequestScheduler-{self.env_manager_config.name}-{mode}"
                    ),
                    namespace=RAY_NAMESPACE,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=ray.get_runtime_context().get_node_id(),
                        soft=False,
                    ),
                    runtime_env=runtime_env,
                    max_concurrency=env_num + 1,  # reserve extra one for suspend/resume
                ).remote(infer_cluster=self.infer_cluster, pipeline_config=config, resource_manager=self.resource_manager)
            self.logger.info(f"[RolloutScheduler] created RequestScheduler mode={self.mode}")

        self.logger.info(f"[RolloutScheduler] creating env Cluster mode={self.mode} name={self.env_manager_config.name}")
        self.es_manager: Any = Cluster(
            name=self.env_manager_config.name,
            worker_cls=self.env_manager_config.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.env_manager_config,
            resolve_topology=False,
        )
        self.logger.info(f"[RolloutScheduler] created env Cluster mode={self.mode} name={self.env_manager_config.name}")
        # Do not block with ray.get() inside this async actor's constructor.
        # We kick off initialization on env workers and await it in get_batch().
        self.logger.info(f"[RolloutScheduler] submitting env initialize mode={self.mode}")
        self._es_initialize_refs = self.es_manager.initialize(
            pipeline_config=self.config,
            generate_scheduler=self.generate_scheduler,
            output_queue=self.env_output_queue,
            collator=collator,
            mode=self.mode,
            blocking=False,
        )
        self._es_initialized = False
        self.logger.info(
            f"[RolloutScheduler] submitted env initialize mode={self.mode} "
            f"num_refs={len(self._es_initialize_refs) if self._es_initialize_refs else 0}"
        )

        self.rollout_task = None

        # Partial GPU mode state atomicity
        self.mode_switch_lock = asyncio.Lock()  # Prevent concurrent shrink/expand

        # Initialize rollout mock mechanism from mixin
        self._init_rollout_mock()
        self.logger.info(f"[RolloutScheduler] __init__ exit mode={self.mode}")

    async def shutdown(self, timeout: float = 10.0):
        if self.rollout_task is None:
            return

        async def _do_shutdown():
            await asyncio.gather(*self.es_manager.stop(blocking=False))
            await self.env_output_queue.shutdown.remote()
            await self.generate_scheduler.abort_request.remote()
            await self.rollout_task

        try:
            await asyncio.wait_for(_do_shutdown(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"shutdown timed out after {timeout}s, force-skipping")
            if self.rollout_task is not None and not self.rollout_task.done():
                self.rollout_task.cancel()
        self.rollout_task = None

    async def suspend(self):
        await self.generate_scheduler.suspend.remote()

    async def resume(self):
        # Delegate resume so partial-GPU pipeline can re-enable request dispatch after expand.
        await self.generate_scheduler.resume.remote()

    async def get_inflight_counts(self, dp_ranks: List[int]) -> Dict[int, int]:
        # Delegate to RequestScheduler so caller observes in-flight state from routing owner.
        return await self.generate_scheduler.get_inflight_counts.remote(dp_ranks)

    async def get_offload_ranks_for_target_gpus(self, target_gpus: List[int]) -> List[int]:
        # Delegate rank-mapping logic to RequestScheduler for consistency with shrink/expand semantics.
        return await self.generate_scheduler.get_offload_ranks_for_target_gpus.remote(target_gpus)

    async def offload_dp_ranks(self, dp_ranks: List[int]) -> Dict[str, Any]:
        # Delegate physical offload to RequestScheduler to keep model-state transitions centralized.
        return await self.generate_scheduler.offload_dp_ranks.remote(dp_ranks)

    async def _run_rollout_loop(self, seed):
        self.logger.info(f"[RolloutScheduler] start _run_rollout_loop seed={seed} mode={self.mode}")
        await asyncio.gather(*self.es_manager.run_rollout_loop(seed, blocking=False))

    async def _get_batch(self, batch_size, global_step):
        return await self.env_output_queue.get_batch.remote(batch_size, global_step)

    async def get_batch(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]
        self.logger.info(f"[RolloutScheduler] get_batch enter mode={self.mode} global_step={global_step} batch_size={batch_size}")

        # MOCK MODE: Load pre-recorded data, skip rollout (from mixin)
        if self._should_load_mock(global_step):
            return await self._load_mock_batch(global_step)

        if not self._es_initialized:
            self.logger.info(f"[RolloutScheduler] awaiting env worker initialize mode={self.mode}")
            init_refs = self._es_initialize_refs or []
            await asyncio.gather(*init_refs)
            self._es_initialized = True
            self.logger.info(f"[RolloutScheduler] env worker initialize done mode={self.mode}")

        # start env manager
        if self.rollout_task is None:
            seed = random.randint(0, 1000000) if self.mode == "train" else self.config.seed
            self.rollout_task = asyncio.create_task(self._run_rollout_loop(seed))
            self.logger.info(f"[RolloutScheduler] created rollout_task seed={seed} mode={self.mode}")

        self.logger.info(f"[RolloutScheduler] update_step start mode={self.mode} global_step={global_step}")
        await asyncio.gather(*self.es_manager.update_step(global_step, blocking=False))
        self.logger.info(f"[RolloutScheduler] update_step done mode={self.mode} global_step={global_step}")

        self.logger.info(f"[RolloutScheduler] advance_step start mode={self.mode} global_step={global_step}")
        await self.env_output_queue.advance_step.remote(global_step)
        self.logger.info(f"[RolloutScheduler] advance_step done mode={self.mode} global_step={global_step}")
        if os.environ.get("SCHEDRL_CONTROL_PLANE", "") != "schedrl":
            await self.generate_scheduler.resume.remote()

        get_task = asyncio.create_task(self._get_batch(batch_size, global_step))
        self.logger.info(f"[RolloutScheduler] wait for env_output_queue.get_batch mode={self.mode} global_step={global_step}")
        wait_timeout_s = float(os.environ.get("ROLL_ROLLOUT_GET_BATCH_TIMEOUT_S", "1800"))
        done, _ = await asyncio.wait(
            {get_task, self.rollout_task},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=wait_timeout_s,
        )
        if not done:
            raise RuntimeError(
                f"[RolloutScheduler] get_batch timed out after {wait_timeout_s}s "
                f"(mode={self.mode}, global_step={global_step}, batch_size={batch_size}). "
                f"Likely stuck: env rollout loop not producing rollouts, or GroupQueueManager waiting for episodes."
            )
        if self.rollout_task.done() and self.rollout_task.exception() is not None:
            await self.rollout_task
        data_batch = await get_task
        self.logger.info(
            f"[RolloutScheduler] env_output_queue.get_batch returned mode={self.mode} "
            f"global_step={global_step} items={len(data_batch) if data_batch else 0}"
        )
        if batch_size <= 0:
            await self.rollout_task
            self.rollout_task = None
            await self.env_output_queue.clear.remote()

        if len(data_batch) == 0:
            return None

        metrics = {}
        get_batch_return_start_time = None
        for d_item in data_batch:
            get_batch_return_start_time = d_item.meta_info.pop("get_batch_return_start_time", None)
            append_to_dict(metrics, d_item.meta_info["metrics"])
        if get_batch_return_start_time is not None:
            metrics["time/get_batch_cost_gqm"] = time.time() - get_batch_return_start_time
        metrics.update(await self.env_output_queue.collect_metrics.remote())
        batch = DataProto.concat(data_batch)
        batch.meta_info["metrics"] = metrics
        batch.meta_info["get_batch_return_start_time"] = time.time()

        # DUMP MODE: Save merged batch (from mixin)
        await self._maybe_dump_batch(batch, global_step)

        return batch

    async def shrink_sampler(self, dp_ranks: List[int], skip_offload: bool = False) -> Dict[str, Any]:
        """Thin wrapper: Delegate shrink operation to RequestScheduler.

        v4.6 ARCHITECTURAL CHANGE: RolloutScheduler no longer performs validation,
        calculation, or state management. All worker lifecycle operations are now
        owned by RequestScheduler for atomic execution under routing_lock.

        Args:
            dp_ranks: DP ranks to offload / deactivate for routing
            skip_offload: If True, skip physical offload (use when another coupled scheduler already offloaded).

        Returns:
            Dict with metrics from RequestScheduler.shrink_workers():
                - "shrink_duration_ms": Total shrink operation time
                - "offload_ranks": DP ranks offloaded
                - "aborted": Number of requests aborted
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "rollout_scheduler_duration_ms": Timing from RolloutScheduler perspective

        Raises:
            RuntimeError: If shrink_workers() fails (propagated from RequestScheduler)

        Side Effects:
            - Calls RequestScheduler.shrink_workers() which performs:
              * Validation, calculation, rebalancing, state offload atomically
              * All operations protected by routing_lock

        Example:
            # Shrink before training to free actor_train GPUs
            metrics = await rollout_scheduler.shrink_sampler.remote([4, 5, 6, 7])
            # RequestScheduler handles: validation → calculation → rebalance → offload
        """
        start_time = time.time()

        # Delegate complete shrink operation to RequestScheduler (atomic under routing_lock)
        result = await self.generate_scheduler.shrink_workers.remote(dp_ranks, skip_offload=bool(skip_offload))

        # Add timing from RolloutScheduler perspective
        result["rollout_scheduler_duration_ms"] = (time.time() - start_time) * 1000

        return result

    async def expand_sampler(self, dp_ranks: List[int], skip_load: bool = False) -> Dict[str, Any]:
        """Thin wrapper: Delegate expand operation to RequestScheduler.

        v4.6 ARCHITECTURAL CHANGE: RolloutScheduler no longer performs validation,
        calculation, or state management. All worker lifecycle operations are now
        owned by RequestScheduler for atomic execution under routing_lock.

        Args:
            dp_ranks: DP ranks to load / activate for routing
            skip_load: If True, skip model loading (use when model_update already loaded states).
                      This only updates active_dp_ranks to restore routing state.

        Returns:
            Dict with metrics from RequestScheduler.expand_workers():
                - "expand_duration_ms": Total expand operation time
                - "load_ranks": DP ranks reloaded
                - "aborted": Number of requests aborted (proportional rebalancing)
                - "remapped": Number of src_ranks remapped (same as aborted)
                - "rollout_scheduler_duration_ms": Timing from RolloutScheduler perspective

        Raises:
            RuntimeError: If expand_workers() fails (propagated from RequestScheduler)

        Side Effects:
            - Calls RequestScheduler.expand_workers() which performs:
              * Validation, calculation, state loading (unless skip_load=True), routing updates atomically
              * All operations protected by routing_lock

        Example:
            # Expand after training to restore actor_train GPUs
            metrics = await rollout_scheduler.expand_sampler.remote([4, 5, 6, 7])
            # RequestScheduler handles: validation → calculation → load → rebalance

            # After model_update already loaded states, just restore routing:
            metrics = await rollout_scheduler.expand_sampler.remote([4, 5, 6, 7], skip_load=True)
        """
        start_time = time.time()

        # Delegate complete expand operation to RequestScheduler (atomic under routing_lock)
        result = await self.generate_scheduler.expand_workers.remote(dp_ranks, skip_load)

        # Add timing from RolloutScheduler perspective
        result["rollout_scheduler_duration_ms"] = (time.time() - start_time) * 1000

        return result

    async def get_active_dp_ranks(self) -> Set[int]:
        """Return the current active DP ranks from the underlying RequestScheduler.

        Used for state verification after initialization shrink operations.

        # FIXME: remove this method and have all callers look up RequestScheduler directly
        # via ray.get_actor(f"RequestScheduler-{pipeline_id}", namespace=RAY_NAMESPACE)
        # and call get_active_dp_ranks() on it. The RolloutScheduler indirection adds
        # an unnecessary hop and obscures which actor owns the authoritative state.
        """
        return await self.generate_scheduler.get_active_dp_ranks.remote()
    def get_generate_scheduler_name(self) -> str:
        """Return the name of the RequestScheduler actor (for verification)."""
        # Note: self.generate_scheduler is an ActorHandle, but we want the name it was created with.
        # However, we can't easily get the name from the handle itself in a clean way across Ray versions.
        # But we can get it from the internal _actor_name if available, or just return the handle representation.
        # For simplicity in this specific verification, we'll return the name we expect if it's a shared actor.
        # Actually, let's just return the actor handle's task name or similar if possible, 
        # but better to just return the name we stored.
        return getattr(self.generate_scheduler, "_actor_name", "unknown")

