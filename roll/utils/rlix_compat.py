"""Optional rlix compatibility layer."""

try:
    from rlix.protocol.types import (
        RLIX_NAMESPACE,
        COORDINATOR_ACTOR_NAME_PREFIX,
        ROLL_RESOURCE_MANAGER_ACTOR_NAME,
        get_pipeline_namespace,
        ProgressReport,
    )
    RLIX_AVAILABLE = True
except ImportError:
    RLIX_AVAILABLE = False
    RLIX_NAMESPACE: str = "rlix"
    COORDINATOR_ACTOR_NAME_PREFIX: str = "rlix:coordinator:"
    ROLL_RESOURCE_MANAGER_ACTOR_NAME: str = "rlix:roll_resource_manager"

    def get_pipeline_namespace(pipeline_id: str) -> str:
        return f"pipeline_{pipeline_id}_NS"

    ProgressReport = None  # type: ignore[misc,assignment]