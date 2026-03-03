import os

from .platform import Platform
from ..utils.logging import get_logger


logger = get_logger()


class CpuPlatform(Platform):
    """Platform for nodes without GPU/NPU accelerators (e.g., scheduler/coordinator nodes).
    
    In RLix mode (time-sharing), CPU actors spawn GPU workers on other nodes and need to
    configure GPU visibility for those child workers. The device_control_env_var and
    ray_experimental_noset attributes are only applied in this mode via update_env_vars_for_visible_devices.
    
    In standalone mode, CPU actors don't spawn GPU workers, so these attributes are unused.
    """
    device_name: str = "CPU"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    ray_device_key: str = "CPU"
    communication_backend: str = "gloo"

    # GPU visibility attributes: only needed in RLix mode where CPU actors spawn GPU workers.
    # In standalone mode, these are None and update_env_vars_for_visible_devices() early-exits.
    if os.environ.get("RLIX_CONTROL_PLANE", "") == "rlix":
        device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
        ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    else:
        device_control_env_var: str = None
        ray_experimental_noset: str = None

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        return

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        env_vars = {
            # This is a following temporiary fix for starvation of plasma lock at
            # https://github.com/ray-project/ray/pull/16408#issuecomment-861056024.
            # When the system is overloaded (rpc queueing) and can not pull Object from remote in a short period
            # (e.g. DynamicSampliningScheduler.report_response using ray.get inside Threaded Actor), the minimum
            # 1000ms batch timeout can still starve others (e.g. Release in callback of PinObjectIDs, reported here
            # https://github.com/ray-project/ray/pull/16402#issuecomment-861222140), which in turn, will exacerbates
            # queuing of rpc.
            # So we set a small timeout for PullObjectsAndGetFromPlasmaStore to avoid holding store_client lock
            # too long.
            "RAY_get_check_signal_interval_milliseconds": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "RAY_CGRAPH_get_timeout": '600',
        }
        return env_vars

    @classmethod
    def get_vllm_run_time_env_vars(cls, gpu_rank: str) -> dict:
        env_vars = {
            "PYTORCH_CUDA_ALLOC_CONF": "",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        }
        return env_vars

    @classmethod
    def apply_ulysses_patch(cls) -> None:
        return
