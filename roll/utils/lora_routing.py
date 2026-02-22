"""LoRA routing utilities for multi-LoRA microbatch dispatch.

The canonical routing key is ``non_tensor_batch["lora_name"]``.
Multi-adapter callers must inject this key before routing.
Single-adapter callers can use ``ensure_lora_name_in_batch`` to auto-fill.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_INVALID_ADAPTER_CHARS = re.compile(r"[^a-z0-9._-]+")
_MULTI_UNDERSCORES = re.compile(r"_+")


def normalize_domain(domain: str) -> str:
    domain = domain.strip().lower()
    domain = _INVALID_ADAPTER_CHARS.sub("_", domain)
    domain = _MULTI_UNDERSCORES.sub("_", domain).strip("_")
    if not domain:
        raise ValueError("normalize_domain() produced an empty adapter name")
    return domain


@dataclass(frozen=True)
class LoraNameRouting:
    raw_lora_name: str
    lora_name: str


def _require_str(val: Any, *, where: str) -> str:
    if not isinstance(val, str):
        raise TypeError(f"Expected str for {where}, got {type(val)}")
    return val


def get_lora_name_array(non_tensor_batch: Mapping[str, Any]) -> np.ndarray:
    """Return the per-sample LoRA name array from ``non_tensor_batch["lora_name"]``."""
    if "lora_name" not in non_tensor_batch:
        raise RuntimeError(
            'Missing `non_tensor_batch["lora_name"]` (required for multi-LoRA routing). '
            f"Available keys={sorted(non_tensor_batch.keys())}"
        )
    lora_name = non_tensor_batch["lora_name"]
    if not isinstance(lora_name, np.ndarray) or lora_name.dtype != object:
        raise TypeError(
            f'Expected `non_tensor_batch["lora_name"]` to be np.ndarray(dtype=object), '
            f"got {type(lora_name)} dtype={getattr(lora_name, 'dtype', None)} "
            f"shape={getattr(lora_name, 'shape', None)}"
        )
    return lora_name


def ensure_lora_name_in_batch(
    non_tensor_batch: dict,
    *,
    adapters: Mapping[str, Any] | None,
    batch_size: int | None = None,
) -> None:
    """Ensure ``non_tensor_batch["lora_name"]`` exists using strict single-vs-multi policy."""
    if "lora_name" in non_tensor_batch:
        return
    if not adapters:
        return
    if len(adapters) == 1:
        only_key = next(iter(adapters.keys()))
        # Keep this strict: infer shape or fail so callers fix producer contract early.
        if batch_size is None:
            if not non_tensor_batch:
                raise RuntimeError(
                    "ensure_lora_name_in_batch: cannot auto-fill lora_name in single-adapter "
                    "mode with empty non_tensor_batch and no batch_size provided."
                )
            batch_size = len(next(iter(non_tensor_batch.values())))
        non_tensor_batch["lora_name"] = np.full(batch_size, only_key, dtype=object)
        return
    raise RuntimeError(
        "Missing non_tensor_batch['lora_name'] in multi-adapter mode. "
        f"Configured adapters: {sorted(adapters.keys())}. "
        "Producers must inject lora_name."
    )


def _get_lora_name_array(non_tensor_batch: Mapping[str, Any]) -> np.ndarray:
    """Return per-sample LoRA name array. Requires ``non_tensor_batch['lora_name']``."""
    return get_lora_name_array(non_tensor_batch)


def resolve_microbatch_lora_name(non_tensor_batch: Mapping[str, Any]) -> LoraNameRouting:
    """Resolve the adapter name for a homogeneous microbatch.

    The microbatch must consist entirely of samples for a single adapter;
    mixing adapters within one microbatch raises RuntimeError.
    """
    lora_arr = _get_lora_name_array(non_tensor_batch)
    if lora_arr.size == 0:
        raise RuntimeError('Empty adapter name array in non_tensor_batch.')
    raw_lora_names = [_require_str(d, where='adapter name item') for d in lora_arr.tolist()]
    unique = sorted(set(raw_lora_names))
    if len(unique) != 1:
        raise RuntimeError(f"Microbatch must be adapter-homogeneous; got adapter names={unique}")
    raw_lora_name = unique[0]
    normalized = normalize_domain(raw_lora_name)
    if normalized != raw_lora_name:
        raise RuntimeError(
            f"Invalid adapter name={raw_lora_name!r}: expected normalized form {normalized!r}. "
            "Adapter names must be lowercase alphanumeric with dots, hyphens, or underscores."
        )
    return LoraNameRouting(raw_lora_name=raw_lora_name, lora_name=raw_lora_name)
