from __future__ import annotations
import os


def _get_env_timeout_s(var_name: str, default_s: float) -> float:
    """Read a timeout in seconds from an env var; fall back to default_s if unset or invalid."""
    # Copied verbatim from multi_lora_pipeline.py:55-64; no logic change.
    raw = os.environ.get(var_name)
    if raw is None:
        return default_s
    try:
        val = float(raw)
    except ValueError:
        return default_s
    return val if val > 0 else default_s
