"""LoRA routing utilities for multi-LoRA microbatch dispatch.

Routing contract
----------------
Every batch that reaches a multi-LoRA worker must carry a per-sample adapter
name in ``non_tensor_batch["lora_name"]`` as an ``np.ndarray(dtype=object)``.

- **Multi-adapter producers** (schedulers, env managers) must inject this key
  and normalize adapter names via ``normalize_domain()`` before dispatch.
- **Single-adapter producers** may call ``ensure_lora_name_in_batch()`` to
  auto-fill the array; it raises if the batch is multi-adapter and the key is
  missing.
- **Workers** call ``resolve_microbatch_lora_name()`` to assert homogeneity
  (all samples in the microbatch belong to the same adapter) and retrieve the
  routing key.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_INVALID_ADAPTER_CHARS = re.compile(r"[^a-z0-9._-]+")
_MULTI_UNDERSCORES = re.compile(r"_+")


def normalize_domain(domain: str) -> str:
    """Canonicalize an adapter name to a lowercase slug.

    All routing lookups compare normalized names, so callers that use different
    capitalizations or separators (e.g. "Math/v2", "math_v2") resolve to the
    same key.  Raises ``ValueError`` on an empty result so a bad name never
    silently maps to the wrong adapter.

    Examples::

        normalize_domain("Math/v2")   -> "math_v2"
        normalize_domain("  GPT  ")   -> "gpt"
        normalize_domain("a--b__c")   -> "a-b_c"  # consecutive separators collapsed
    """
    # Lowercase and strip surrounding whitespace first.
    domain = domain.strip().lower()
    # Replace any char outside [a-z0-9._-] with underscore.
    domain = _INVALID_ADAPTER_CHARS.sub("_", domain)
    # Collapse consecutive underscores and remove leading/trailing ones.
    domain = _MULTI_UNDERSCORES.sub("_", domain).strip("_")
    if not domain:
        raise ValueError("normalize_domain() produced an empty adapter name")
    return domain


@dataclass(frozen=True)
class LoraNameRouting:
    """Resolved adapter name for one microbatch.

    Keeping both raw and normalized names allows callers to log the original
    spelling for debugging while using the normalized form for registry lookups.
    """

    raw_lora_name: str  # name as it appeared in non_tensor_batch["lora_name"]
    lora_name: str      # normalized slug used for adapter registry lookups


def _require_str(val: Any, *, where: str) -> str:
    # numpy array items may come back as numpy.str_ rather than Python str;
    # reject early so routing comparisons don't silently use the wrong type.
    if not isinstance(val, str):
        raise TypeError(f"Expected str for {where}, got {type(val)}")
    return val


def get_lora_name_array(non_tensor_batch: Mapping[str, Any]) -> np.ndarray:
    """Return the per-sample LoRA name array from ``non_tensor_batch["lora_name"]``.

    Raises ``RuntimeError`` if the key is absent, ``TypeError`` if the value is
    not an ``np.ndarray(dtype=object)``.
    """
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
    """Ensure ``non_tensor_batch["lora_name"]`` exists, enforcing single-vs-multi policy.

    - If the key already exists: no-op.
    - If ``adapters`` is empty or None: no-op (non-LoRA path).
    - If exactly one adapter is configured: auto-fill the array with that adapter's name.
      ``batch_size`` is inferred from another batch key when not provided.
    - If multiple adapters are configured: raise ``RuntimeError`` — producers must inject
      ``lora_name`` explicitly; there is no safe default to choose.
    """
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


def resolve_microbatch_lora_name(non_tensor_batch: Mapping[str, Any]) -> LoraNameRouting:
    """Resolve the adapter name for a homogeneous microbatch.

    Asserts that every sample in the microbatch belongs to the same adapter;
    raises ``RuntimeError`` if adapters are mixed or the name is not normalized.

    Workers call this immediately before dispatching to an adapter-specific
    forward pass.  Producers are responsible for splitting mixed batches before
    this point.
    """
    lora_arr = get_lora_name_array(non_tensor_batch)
    if lora_arr.size == 0:
        raise RuntimeError('Empty adapter name array in non_tensor_batch.')
    raw_lora_names = [_require_str(d, where='adapter name item') for d in lora_arr.tolist()]
    unique = sorted(set(raw_lora_names))
    if len(unique) != 1:
        raise RuntimeError(f"Microbatch must be adapter-homogeneous; got adapter names={unique}")
    raw_lora_name = unique[0]
    normalized = normalize_domain(raw_lora_name)
    # Names in the batch must already be normalized so registry lookups are exact.
    # Producers (schedulers, env managers) must call normalize_domain() before
    # writing lora_name into non_tensor_batch; catch violations here early.
    if normalized != raw_lora_name:
        raise RuntimeError(
            f"Invalid adapter name={raw_lora_name!r}: expected normalized form {normalized!r}. "
            "Adapter names must be lowercase alphanumeric with dots, hyphens, or underscores."
        )
    return LoraNameRouting(raw_lora_name=raw_lora_name, lora_name=raw_lora_name)
