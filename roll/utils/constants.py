import enum
import logging
import os


_SCHEDRL_CONTROL_PLANE = os.environ.get("SCHEDRL_CONTROL_PLANE", "")
if _SCHEDRL_CONTROL_PLANE == "schedrl":
    ray_namespace = os.environ.get("ROLL_RAY_NAMESPACE")
    if not ray_namespace:
        raise RuntimeError("SCHEDRL_CONTROL_PLANE=schedrl requires ROLL_RAY_NAMESPACE to be set before importing roll.*")
    pipeline_id = os.environ.get("PIPELINE_ID")
    if not pipeline_id:
        raise RuntimeError("SCHEDRL_CONTROL_PLANE=schedrl requires PIPELINE_ID to be set before importing roll.*")

RAY_NAMESPACE = os.environ.get("ROLL_RAY_NAMESPACE", "roll")
GLOBAL_STORAGE_NAMESPACE = "global_storage_namespace"
STORAGE_NAME = "SHARED_STORAGE_ACTOR"
GENERATE_SCHEDULER_NAME = "GENERATE_SCHEDULER_ACTOR"
REWARD_SCHEDULER_NAME = "REWARD_SCHEDULER_ACTOR"

BARRIER_NAME = "BARRIER_ACTOR_NAME"

CHECKPOINT_MANAGER_NAME = "CHECKPOINT_MANAGER_ACTOR"

SCHEDULER_NAME = "scheduler.pt"
OPTIMIZER_NAME = "optimizer.pt"
DIST_OPTIMIZER_DIR = "dist_optimizer"
RNG_STATE_DIR = "rng_state"

CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "roll")

IGNORE_INDEX = -100


def schedrl_env_vars() -> dict[str, str]:
    """Env vars that must be present in all per-pipeline Ray actor processes in SchedRL mode.

    Use this when creating child actors from within a pipeline actor; Ray does not reliably
    inherit runtime_env env vars from parent actors.
    """
    if os.environ.get("SCHEDRL_CONTROL_PLANE", "") != "schedrl":
        return {}
    # In SchedRL mode, roll.* import already validated these exist; keep them explicit here too.
    pipeline_id = os.environ.get("PIPELINE_ID")
    ray_namespace = os.environ.get("ROLL_RAY_NAMESPACE")
    if not pipeline_id:
        raise RuntimeError("SCHEDRL_CONTROL_PLANE=schedrl requires PIPELINE_ID to be set")
    if not ray_namespace:
        raise RuntimeError("SCHEDRL_CONTROL_PLANE=schedrl requires ROLL_RAY_NAMESPACE to be set")
    grpc_pool_size = os.environ.get("RAY_grpc_server_thread_pool_size", "4")
    omp_threads = os.environ.get("OMP_NUM_THREADS", "1")
    logging.getLogger(__name__).info(
        "[schedrl_env_vars] pid=%d RAY_grpc_server_thread_pool_size=%s OMP_NUM_THREADS=%s",
        os.getpid(),
        grpc_pool_size,
        omp_threads,
    )
    return {
        "PIPELINE_ID": pipeline_id,
        "ROLL_RAY_NAMESPACE": ray_namespace,
        "SCHEDRL_CONTROL_PLANE": "schedrl",
        "SCHEDRL_LIBRARY_MODE": os.environ.get("SCHEDRL_LIBRARY_MODE", "1"),
        # Keep imports working when Ray workers start outside the repo root.
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        # Limit math library threads per actor to avoid hitting container pids.max.
        "OMP_NUM_THREADS": omp_threads,
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
        # Limit gRPC sync thread pool per actor to avoid hitting container pids.max.
        # Default is 32; 4 is sufficient for RL pipeline actor communication throughput.
        "RAY_grpc_server_thread_pool_size": grpc_pool_size,
    }


class GenerateStopReason(enum.Enum):
    FINISH = enum.auto()
    ABORT = enum.auto()
    MAX_LENGTH = enum.auto()
    NO_SYSTEM_PROMPT = enum.auto()
    
    
class EpisodeStopReason(enum.Enum):
    FINISH = "finish"   
    MAX_LENGTH = "max_length"         
    MAX_STEPS = "max_steps" 
    ABORT = "abort"     
    ENV_RESET_FAILED = "env_reset_failed" 
    SANDBOX_INIT_FAILED = "sandbox_init_failed" 
    ENV_TIMEOUT = "env_timeout"   
    LLM_GENERATE_FAILED = "llm_generate_failed" 
    UNKNOWN = "unknown"
    NO_SYSTEM_PROMPT = "no_system_prompt"
    EVAL_GT = "eval_gt"
