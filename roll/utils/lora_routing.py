"""LoRA routing utilities for multi-LoRA microbatch dispatch.

Ported from ROLL_multi_lora with one key adaptation:
  ROLL_schedrl uses ``non_tensor_batch["domain"]`` as the routing key
  (consistent with the existing SchedRL pipeline conventions), while
  ROLL_multi_lora uses ``non_tensor_batch["lora_name"]``.

``resolve_microbatch_lora_name`` therefore checks ``domain`` first and
falls back to ``lora_name`` so that tests or pipelines which use either
convention are both supported.
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


def _get_lora_name_array(non_tensor_batch: Mapping[str, Any]) -> np.ndarray:
    """Return the per-sample lora/domain name array.

    Checks ``domain`` first (ROLL_schedrl convention), then falls back to
    ``lora_name`` (ROLL_multi_lora convention).
    """
    for key in ("domain", "lora_name"):
        if key in non_tensor_batch:
            val = non_tensor_batch[key]
            if not isinstance(val, np.ndarray) or val.dtype != object:
                raise TypeError(
                    f'Expected `non_tensor_batch["{key}"]` to be np.ndarray(dtype=object), '
                    f"got {type(val)} dtype={getattr(val, 'dtype', None)} "
                    f"shape={getattr(val, 'shape', None)}"
                )
            return val
    raise RuntimeError(
        'Missing `non_tensor_batch["domain"]` (or "lora_name") required for multi-LoRA routing. '
        f"Available keys={sorted(non_tensor_batch.keys())}"
    )


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
