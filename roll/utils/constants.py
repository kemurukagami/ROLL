import enum
import logging
import os


# Validate required env vars at import time so that misconfigured Ray workers
# crash immediately with a clear message rather than failing deep in actor init.
_RLIX_CONTROL_PLANE = os.environ.get("RLIX_CONTROL_PLANE", "")
if _RLIX_CONTROL_PLANE == "rlix":
    ray_namespace = os.environ.get("ROLL_RAY_NAMESPACE")
    if not ray_namespace:
        raise RuntimeError("RLIX_CONTROL_PLANE=rlix requires ROLL_RAY_NAMESPACE to be set before importing roll.*")
    pipeline_id = os.environ.get("PIPELINE_ID")
    if not pipeline_id:
        raise RuntimeError("RLIX_CONTROL_PLANE=rlix requires PIPELINE_ID to be set before importing roll.*")

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


def rlix_env_vars() -> dict[str, str]:
    """Env vars that must be present in all per-pipeline Ray actor processes in RLix mode.

    Use this when creating child actors from within a pipeline actor; Ray does not reliably
    inherit runtime_env env vars from parent actors.
    
    Only propagates env vars that are explicitly set; no defaults are applied here.
    Defaults should be configured by the orchestrator or container environment.
    """
    if os.environ.get("RLIX_CONTROL_PLANE", "") != "rlix":
        return {}
    # In RLix mode, roll.* import already validated these exist; keep them explicit here too.
    pipeline_id = os.environ.get("PIPELINE_ID")
    ray_namespace = os.environ.get("ROLL_RAY_NAMESPACE")
    if not pipeline_id:
        raise RuntimeError("RLIX_CONTROL_PLANE=rlix requires PIPELINE_ID to be set")
    if not ray_namespace:
        raise RuntimeError("RLIX_CONTROL_PLANE=rlix requires ROLL_RAY_NAMESPACE to be set")
    
    env_vars: dict[str, str] = {
        "PIPELINE_ID": pipeline_id,
        "ROLL_RAY_NAMESPACE": ray_namespace,
        "RLIX_CONTROL_PLANE": "rlix",
    }
    
    # Propagate PYTHONPATH if set (for imports when Ray workers start outside repo root).
    if pythonpath := os.environ.get("PYTHONPATH"):
        env_vars["PYTHONPATH"] = pythonpath
    
    # Propagate thread-limiting vars only if explicitly set (to avoid PID limits in containers).
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
                "RAY_grpc_server_thread_pool_size"):
        if value := os.environ.get(var):
            env_vars[var] = value
    
    logging.getLogger(__name__).info(
        "[rlix_env_vars] pid=%d propagating %d env vars",
        os.getpid(),
        len(env_vars),
    )
    return env_vars


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
