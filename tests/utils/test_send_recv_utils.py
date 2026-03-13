"""Tests for cpu_serialize weight transfer in send_recv_utils.

Covers bucket pack/unpack, full cpu_serialize serialize/deserialize round-trips,
and end-to-end scenarios with realistic model weights. All tests are CPU-only
since the cpu_serialize path is specifically designed for CPU serialization.

The benchmark test uses Ray object store (ray.put/ray.get) to faithfully
measure cross-process transport cost, matching the production flow where
serialized bytes travel through Ray's object store (/dev/shm) between
training and inference workers.
"""

import io
import time
from typing import Optional

import pytest
import ray
import torch

from roll.utils.send_recv_utils import (
    _bucket_named_tensors,
    named_tensors_from_bucket,
    serialize_named_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_named_weights(
    shapes: list[tuple[str, tuple[int, ...]]],
    dtype: torch.dtype = torch.float32,
) -> list[tuple[str, torch.Tensor]]:
    """Create deterministic named tensors via torch.arange for reproducibility."""
    named_weights: list[tuple[str, torch.Tensor]] = []
    for name, shape in shapes:
        numel = 1
        for dim in shape:
            numel *= dim
        # arange in float32 then cast — avoids dtype issues with arange on bf16
        tensor = torch.arange(numel, dtype=torch.float32).reshape(shape).to(dtype)
        named_weights.append((name, tensor))
    return named_weights


def _deserialize_cpu_serialize(
    serialized: bytes,
    target_device: str = "cpu",
) -> list[tuple[str, torch.Tensor]]:
    """Mirror the vLLM worker deserialization path for cpu_serialize payloads.

    Steps: torch.load -> move bucket to device -> named_tensors_from_bucket.
    This matches the logic in worker.py for cpu_serialize transport (torch.save format).
    """
    payload = torch.load(io.BytesIO(serialized), weights_only=True)
    bucket = payload["bucket"].to(target_device)
    tensors_meta = payload["tensors_meta"]
    return named_tensors_from_bucket(bucket, tensors_meta)


def _assert_named_weights_equal(
    actual: list[tuple[str, torch.Tensor]],
    expected: list[tuple[str, torch.Tensor]],
    msg: Optional[str] = None,
) -> None:
    """Assert exact match on name, shape, dtype, and values."""
    prefix = f"{msg}: " if msg else ""
    assert len(actual) == len(expected), (
        f"{prefix}length mismatch: {len(actual)} vs {len(expected)}"
    )
    for idx, ((actual_name, actual_tensor), (expected_name, expected_tensor)) in enumerate(
        zip(actual, expected)
    ):
        assert actual_name == expected_name, (
            f"{prefix}name mismatch at index {idx}: {actual_name!r} vs {expected_name!r}"
        )
        assert actual_tensor.shape == expected_tensor.shape, (
            f"{prefix}shape mismatch for {actual_name}: {actual_tensor.shape} vs {expected_tensor.shape}"
        )
        assert actual_tensor.dtype == expected_tensor.dtype, (
            f"{prefix}dtype mismatch for {actual_name}: {actual_tensor.dtype} vs {expected_tensor.dtype}"
        )
        assert torch.equal(actual_tensor, expected_tensor), (
            f"{prefix}value mismatch for {actual_name}"
        )


# ---------------------------------------------------------------------------
# TestBucketRoundTrip — unit tests for _bucket_named_tensors / named_tensors_from_bucket
# ---------------------------------------------------------------------------


class TestBucketRoundTrip:
    """Unit tests for bucket pack (_bucket_named_tensors) and unpack (named_tensors_from_bucket)."""

    def test_single_tensor(self) -> None:
        """One (4,3) float32 tensor survives bucket round-trip."""
        weights = _make_named_weights([("layer.weight", (4, 3))])
        bucket, meta = _bucket_named_tensors(weights)
        reconstructed = named_tensors_from_bucket(bucket, meta)
        _assert_named_weights_equal(reconstructed, weights)

    def test_multiple_tensors(self) -> None:
        """Three tensors with different shapes survive bucket round-trip."""
        shapes = [
            ("layer0.weight", (4, 3)),
            ("layer1.bias", (8,)),
            ("layer2.weight", (2, 5, 3)),
        ]
        weights = _make_named_weights(shapes)
        bucket, meta = _bucket_named_tensors(weights)
        reconstructed = named_tensors_from_bucket(bucket, meta)
        _assert_named_weights_equal(reconstructed, weights)

    def test_preserves_dtype_bfloat16(self) -> None:
        """bfloat16 dtype is preserved through bucket round-trip."""
        weights = _make_named_weights([("bf16.weight", (4, 3))], dtype=torch.bfloat16)
        bucket, meta = _bucket_named_tensors(weights)
        reconstructed = named_tensors_from_bucket(bucket, meta)
        _assert_named_weights_equal(reconstructed, weights)

    def test_preserves_dtype_float16(self) -> None:
        """float16 dtype is preserved through bucket round-trip."""
        weights = _make_named_weights([("fp16.weight", (4, 3))], dtype=torch.float16)
        bucket, meta = _bucket_named_tensors(weights)
        reconstructed = named_tensors_from_bucket(bucket, meta)
        _assert_named_weights_equal(reconstructed, weights)

    def test_empty_raises(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="Cannot create empty tensor bucket"):
            _bucket_named_tensors([])

    def test_scalar_shaped_tensor(self) -> None:
        """(1,) shaped tensor (edge case) survives bucket round-trip."""
        weights = _make_named_weights([("scalar.param", (1,))])
        bucket, meta = _bucket_named_tensors(weights)
        reconstructed = named_tensors_from_bucket(bucket, meta)
        _assert_named_weights_equal(reconstructed, weights)

    def test_large_tensor(self) -> None:
        """(1024, 512) tensor survives bucket round-trip."""
        weights = _make_named_weights([("large.weight", (1024, 512))])
        bucket, meta = _bucket_named_tensors(weights)
        reconstructed = named_tensors_from_bucket(bucket, meta)
        _assert_named_weights_equal(reconstructed, weights)


# ---------------------------------------------------------------------------
# TestCpuSerializeSerialize — unit tests for serialize_named_weights with cpu_serialize
# ---------------------------------------------------------------------------


class TestCpuSerializeSerialize:
    """Unit tests for full cpu_serialize serialize -> deserialize round-trip."""

    def test_roundtrip_single_tensor(self) -> None:
        """Single tensor survives cpu_serialize serialize/deserialize."""
        weights = _make_named_weights([("layer.weight", (4, 3))])
        serialized = serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
        reconstructed = _deserialize_cpu_serialize(serialized)
        _assert_named_weights_equal(reconstructed, weights)

    def test_roundtrip_multiple_tensors(self) -> None:
        """Multiple tensors survive cpu_serialize serialize/deserialize."""
        shapes = [
            ("model.embed.weight", (16, 8)),
            ("model.layer.weight", (8, 8)),
            ("model.head.bias", (16,)),
        ]
        weights = _make_named_weights(shapes)
        serialized = serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
        reconstructed = _deserialize_cpu_serialize(serialized)
        _assert_named_weights_equal(reconstructed, weights)

    def test_roundtrip_bfloat16(self) -> None:
        """bfloat16 tensors survive cpu_serialize round-trip."""
        weights = _make_named_weights([("bf16.weight", (4, 3))], dtype=torch.bfloat16)
        serialized = serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
        reconstructed = _deserialize_cpu_serialize(serialized)
        _assert_named_weights_equal(reconstructed, weights)

    def test_roundtrip_float16(self) -> None:
        """float16 tensors survive cpu_serialize round-trip."""
        weights = _make_named_weights([("fp16.weight", (4, 3))], dtype=torch.float16)
        serialized = serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
        reconstructed = _deserialize_cpu_serialize(serialized)
        _assert_named_weights_equal(reconstructed, weights)

    def test_roundtrip_large_multi_layer(self) -> None:
        """4 layers with realistic shapes survive cpu_serialize round-trip."""
        shapes = [
            ("model.layers.0.self_attn.q_proj.weight", (512, 512)),
            ("model.layers.0.self_attn.k_proj.weight", (128, 512)),
            ("model.layers.0.mlp.gate_proj.weight", (1024, 512)),
            ("model.layers.0.mlp.down_proj.weight", (512, 1024)),
        ]
        weights = _make_named_weights(shapes, dtype=torch.bfloat16)
        serialized = serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
        reconstructed = _deserialize_cpu_serialize(serialized)
        _assert_named_weights_equal(reconstructed, weights)

    def test_payload_is_bytes(self) -> None:
        """serialize_named_weights with cpu_serialize returns bytes."""
        weights = _make_named_weights([("layer.weight", (4, 3))])
        serialized = serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
        assert isinstance(serialized, bytes)

    def test_payload_contains_cpu_tensor(self) -> None:
        """Deserialized bucket tensor resides on CPU."""
        weights = _make_named_weights([("layer.weight", (4, 3))])
        serialized = serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
        # cpu_serialize now uses torch.save format
        payload = torch.load(io.BytesIO(serialized), weights_only=True)
        bucket = payload["bucket"]
        assert bucket.device == torch.device("cpu")

    def test_invalid_transport_raises(self) -> None:
        """Unknown transport raises ValueError."""
        weights = _make_named_weights([("layer.weight", (4, 3))])
        with pytest.raises(ValueError, match="Unsupported model_update_transport"):
            serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="unknown_transport")


# ---------------------------------------------------------------------------
# TestCpuSerializeEndToEnd — realistic end-to-end scenarios
# ---------------------------------------------------------------------------


class TestCpuSerializeEndToEnd:
    """End-to-end tests simulating realistic weight transfer scenarios."""

    def test_multi_rank_independent_payloads(self) -> None:
        """Serialize same weights N times (simulating N ranks), deserialize each independently."""
        shapes = [
            ("model.embed.weight", (32, 16)),
            ("model.layer.weight", (16, 16)),
            ("model.head.weight", (32, 16)),
        ]
        original_weights = _make_named_weights(shapes, dtype=torch.bfloat16)
        num_ranks = 4

        # Simulate per-rank serialization (each rank gets its own copy)
        serialized_list = [
            serialize_named_weights(original_weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
            for _rank in range(num_ranks)
        ]

        # Each rank deserializes independently and must match original
        for rank_idx in range(num_ranks):
            reconstructed = _deserialize_cpu_serialize(serialized_list[rank_idx])
            _assert_named_weights_equal(reconstructed, original_weights, msg=f"rank {rank_idx}")

    def test_batched_weight_updates(self) -> None:
        """Split weights into batches, serialize/deserialize each, combine and verify."""
        all_shapes = [
            ("model.layers.0.weight", (64, 32)),
            ("model.layers.1.weight", (64, 32)),
            ("model.layers.2.weight", (64, 32)),
            ("model.layers.3.weight", (64, 32)),
        ]
        all_weights = _make_named_weights(all_shapes, dtype=torch.bfloat16)

        # Split into two batches (simulating buffer-size-bounded transfer)
        batch_size = 2
        batches = [all_weights[start:start + batch_size] for start in range(0, len(all_weights), batch_size)]

        # Serialize and deserialize each batch, collect results
        combined: list[tuple[str, torch.Tensor]] = []
        for batch in batches:
            serialized = serialize_named_weights(batch, infer_strategy="vllm", model_update_transport="cpu_serialize")
            reconstructed = _deserialize_cpu_serialize(serialized)
            combined.extend(reconstructed)

        # Combined results must match original
        _assert_named_weights_equal(combined, all_weights)

    def test_deserialized_bucket_is_contiguous(self) -> None:
        """Deserialized bucket tensor is contiguous in memory."""
        weights = _make_named_weights([("layer.weight", (32, 16))], dtype=torch.bfloat16)
        serialized = serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
        # cpu_serialize now uses torch.save format
        payload = torch.load(io.BytesIO(serialized), weights_only=True)
        bucket = payload["bucket"]
        assert bucket.is_contiguous(), "Deserialized bucket must be contiguous"

    def test_lora_adapter_weights(self) -> None:
        """LoRA-style weight names with low-rank shapes survive round-trip."""
        # Typical LoRA adapter naming and shapes (rank=8)
        lora_rank = 8
        hidden_dim = 512
        shapes = [
            ("base_model.model.layers.0.self_attn.q_proj.lora_A.weight", (lora_rank, hidden_dim)),
            ("base_model.model.layers.0.self_attn.q_proj.lora_B.weight", (hidden_dim, lora_rank)),
            ("base_model.model.layers.0.self_attn.v_proj.lora_A.weight", (lora_rank, hidden_dim)),
            ("base_model.model.layers.0.self_attn.v_proj.lora_B.weight", (hidden_dim, lora_rank)),
        ]
        weights = _make_named_weights(shapes, dtype=torch.bfloat16)
        serialized = serialize_named_weights(weights, infer_strategy="vllm", model_update_transport="cpu_serialize")
        reconstructed = _deserialize_cpu_serialize(serialized)
        _assert_named_weights_equal(reconstructed, weights, msg="lora adapter")

    @pytest.mark.slow
    def test_full_model_state_dict_roundtrip(self) -> None:
        """Init Qwen/Qwen2.5-1.5B-Instruct with dummy weights, serialize all, verify exact match.

        Uses from_config (random init) to skip multi-GB download — correctness
        test only needs matching shapes/dtypes, not pretrained values.
        Skipped if transformers is not installed.
        """
        transformers = pytest.importorskip("transformers")

        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        model = transformers.AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16)

        # Convert state_dict to list of (name, tensor) pairs
        original_weights = list(model.state_dict().items())
        assert len(original_weights) > 0, "Model state dict should not be empty"

        # Serialize via cpu_serialize and deserialize
        serialized = serialize_named_weights(
            original_weights, infer_strategy="vllm", model_update_transport="cpu_serialize"
        )
        reconstructed = _deserialize_cpu_serialize(serialized)

        _assert_named_weights_equal(reconstructed, original_weights, msg="full model state dict")


# ---------------------------------------------------------------------------
# Benchmark — compare cuda_ipc vs cpu_serialize end-to-end with real model
#
# Uses Ray object store (ray.put/ray.get) for cross-process transport,
# matching the production flow: serialize_named_weights() → .remote() →
# Ray object store (/dev/shm) → pickle.loads() on inference worker.
# ---------------------------------------------------------------------------


@ray.remote
def _ray_cpu_serialize_deserialize(serialized: bytes) -> float:
    """Deserialize cpu_serialize payload in a Ray worker — mirrors production worker.py:761-766.

    In production, serialized bytes travel through Ray's object store (/dev/shm)
    before reaching the inference worker. Calling this via .remote() replicates
    that exact transfer path, including the object store transport cost.
    """
    import io
    import time

    import torch

    from roll.utils.send_recv_utils import named_tensors_from_bucket

    deserialize_start = time.perf_counter()
    # torch.load matches production worker.py deserialization (torch.save format)
    payload = torch.load(io.BytesIO(serialized), weights_only=True)
    bucket = payload["bucket"]
    # Unpack tensors from bucket (same as production worker.py:766)
    _ = list(named_tensors_from_bucket(bucket=bucket, tensors_meta=payload["tensors_meta"]))
    return time.perf_counter() - deserialize_start


@ray.remote(num_gpus=1)
def _ray_cuda_ipc_deserialize(serialized: bytes) -> float:
    """Deserialize cuda_ipc payload in a Ray worker — mirrors production worker.py:758-766.

    CUDA IPC handles are cross-process only, so deserialization must happen in a
    separate process. Ray worker runs on a different process with its own GPU context.
    """
    import time

    from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
    from roll.utils.send_recv_utils import monkey_patch_torch_reductions

    # Production path: monkey patch must be applied before deserialization (worker.py:760)
    monkey_patch_torch_reductions()

    deserialize_start = time.perf_counter()
    payload = MultiprocessingSerializer.deserialize(serialized)
    # Force materialization of the GPU tensor
    _ = payload["bucket"].shape
    return time.perf_counter() - deserialize_start


@pytest.mark.slow
def test_benchmark_cuda_ipc_vs_cpu_serialize() -> None:
    """Benchmark serialize + transport + deserialize for both transports.

    Both transports use Ray object store for cross-process transfer, matching the
    production flow where .remote() implicitly puts the payload into Ray's object
    store (/dev/shm shared memory). This captures the transport cost that dominates
    cpu_serialize (~1.2 GB bytes blob) vs cuda_ipc (~few KB IPC handles).

    If CUDA is not available, only the cpu_serialize path is benchmarked and a warning is printed.
    Requires transformers.
    """
    transformers = pytest.importorskip("transformers")

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("\nWARNING: CUDA not available — only benchmarking cpu_serialize (skipping cuda_ipc)")

    # Use dummy weights (random init from config) to skip multi-GB download.
    # Benchmark measures transport performance, not weight correctness.
    benchmark_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    config = transformers.AutoConfig.from_pretrained(benchmark_model_name)
    model = transformers.AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16)

    # Extract state dict and free model to reclaim memory for bucket allocation
    weights_cpu = list(model.state_dict().items())
    del model
    if cuda_available:
        torch.cuda.empty_cache()

    # Ray must be initialized for cross-process transport via object store
    num_gpus = torch.cuda.device_count() if cuda_available else 0
    ray.init(num_cpus=2, num_gpus=num_gpus, ignore_reinit_error=True)
    try:
        _run_benchmark(weights_cpu, cuda_available=cuda_available, model_name=benchmark_model_name)
    finally:
        ray.shutdown()


def _run_benchmark(
    weights_cpu: list[tuple[str, torch.Tensor]], *, cuda_available: bool, model_name: str
) -> None:
    """Run the actual benchmark after Ray is initialized.

    Separated from the test function to keep ray.init/shutdown in a clean try/finally.
    When cuda_available=False, only the cpu_serialize path is benchmarked.

    Args:
        weights_cpu: Pre-extracted state dict entries on CPU. Model must be deleted
            before calling this to free GPU memory for bucket allocation.
        cuda_available: Whether CUDA is available for cuda_ipc benchmarking.
        model_name: Model identifier for benchmark output header.
    """
    total_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in weights_cpu)
    total_mb = total_bytes / (1024 * 1024)

    NUM_WARMUP_ROUNDS = 2
    NUM_BENCHMARK_ROUNDS = 5

    # Move weights to GPU if available — production path starts with GPU tensors.
    # Both cpu_serialize and cuda_ipc serialize from GPU weights in production.
    weights_gpu: list[tuple[str, torch.Tensor]] | None = None
    if cuda_available:
        try:
            weights_gpu = [(name, tensor.cuda()) for name, tensor in weights_cpu]
        except torch.cuda.OutOfMemoryError:
            print(
                "\nWARNING: Not enough GPU memory to hold weights — "
                "benchmarking cpu_serialize with CPU weights only (pinned memory path not exercised)"
            )

    # ------ cpu_serialize: serialize (pinned GPU→CPU + torch.save) + transport via Ray + deserialize ------
    # Production: weights on GPU → serialize_named_weights does pinned GPU→CPU + torch.save.
    # Falls back to CPU weights if GPU memory is insufficient.
    cpu_serialize_weights = weights_gpu if weights_gpu is not None else weights_cpu

    # Warmup: run full serialize + ray.put + remote deserialize cycle
    for _warmup in range(NUM_WARMUP_ROUNDS):
        serialized = serialize_named_weights(
            cpu_serialize_weights, infer_strategy="vllm", model_update_transport="cpu_serialize"
        )
        serialized_ref = ray.put(serialized)
        ray.get(_ray_cpu_serialize_deserialize.remote(serialized_ref))

    cpu_serialize_serialize_times: list[float] = []
    cpu_serialize_transport_times: list[float] = []
    cpu_serialize_deserialize_times: list[float] = []
    cpu_serialize_payload_bytes = 0
    for _round in range(NUM_BENCHMARK_ROUNDS):
        if cuda_available:
            torch.cuda.synchronize()

        # Serialize: pinned GPU→CPU copy + torch.save (or torch.save only if CPU weights)
        start = time.perf_counter()
        serialized = serialize_named_weights(
            cpu_serialize_weights, infer_strategy="vllm", model_update_transport="cpu_serialize"
        )
        cpu_serialize_serialize_times.append(time.perf_counter() - start)
        cpu_serialize_payload_bytes = len(serialized)

        # Transport: put bytes into Ray object store (/dev/shm)
        start = time.perf_counter()
        serialized_ref = ray.put(serialized)
        cpu_serialize_transport_times.append(time.perf_counter() - start)

        # Deserialize: Ray worker receives from object store + torch.load
        deserialize_elapsed = ray.get(_ray_cpu_serialize_deserialize.remote(serialized_ref))
        cpu_serialize_deserialize_times.append(deserialize_elapsed)

    # ------ cuda_ipc: requires GPU weights (already moved above) ------
    cuda_ipc_serialize_times: list[float] = []
    cuda_ipc_transport_times: list[float] = []
    cuda_ipc_deserialize_times: list[float] = []
    cuda_ipc_payload_bytes = 0

    if weights_gpu is not None:
        from roll.utils.send_recv_utils import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()

        # Warmup: serialize + transport + remote deserialize
        for _warmup in range(NUM_WARMUP_ROUNDS):
            serialized = serialize_named_weights(
                weights_gpu, infer_strategy="vllm", model_update_transport="cuda_ipc"
            )
            serialized_ref = ray.put(serialized)
            ray.get(_ray_cuda_ipc_deserialize.remote(serialized_ref))

        for _round in range(NUM_BENCHMARK_ROUNDS):
            torch.cuda.synchronize()

            # Serialize: ForkingPickler with cudaIpcGetMemHandle
            start = time.perf_counter()
            serialized = serialize_named_weights(
                weights_gpu, infer_strategy="vllm", model_update_transport="cuda_ipc"
            )
            cuda_ipc_serialize_times.append(time.perf_counter() - start)
            cuda_ipc_payload_bytes = len(serialized)

            # Transport: put serialized IPC handles into Ray object store
            start = time.perf_counter()
            serialized_ref = ray.put(serialized)
            cuda_ipc_transport_times.append(time.perf_counter() - start)

            # Deserialize: Ray worker (with GPU) receives and reconstructs via IPC handles
            deserialize_elapsed = ray.get(_ray_cuda_ipc_deserialize.remote(serialized_ref))
            cuda_ipc_deserialize_times.append(deserialize_elapsed)

    # ------ Print results ------
    _print_benchmark_results(
        model_name=model_name,
        num_weights=len(weights_cpu),
        total_mb=total_mb,
        num_benchmark_rounds=NUM_BENCHMARK_ROUNDS,
        num_warmup_rounds=NUM_WARMUP_ROUNDS,
        cpu_serialize_serialize_times=cpu_serialize_serialize_times,
        cpu_serialize_transport_times=cpu_serialize_transport_times,
        cpu_serialize_deserialize_times=cpu_serialize_deserialize_times,
        cpu_serialize_payload_bytes=cpu_serialize_payload_bytes,
        cuda_ipc_serialize_times=cuda_ipc_serialize_times,
        cuda_ipc_transport_times=cuda_ipc_transport_times,
        cuda_ipc_deserialize_times=cuda_ipc_deserialize_times,
        cuda_ipc_payload_bytes=cuda_ipc_payload_bytes,
    )


def _median(values: list[float]) -> float:
    """Return the median of a list of floats."""
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    return sorted_values[mid]


def _print_benchmark_results(
    *,
    model_name: str,
    num_weights: int,
    total_mb: float,
    num_benchmark_rounds: int,
    num_warmup_rounds: int,
    cpu_serialize_serialize_times: list[float],
    cpu_serialize_transport_times: list[float],
    cpu_serialize_deserialize_times: list[float],
    cpu_serialize_payload_bytes: int,
    cuda_ipc_serialize_times: list[float],
    cuda_ipc_transport_times: list[float],
    cuda_ipc_deserialize_times: list[float],
    cuda_ipc_payload_bytes: int,
) -> None:
    """Print formatted benchmark comparison of cpu_serialize vs cuda_ipc.

    When cuda_ipc lists are empty (no CUDA available), only cpu_serialize results are printed.
    """
    cpu_ser = _median(cpu_serialize_serialize_times)
    cpu_trans = _median(cpu_serialize_transport_times)
    cpu_de = _median(cpu_serialize_deserialize_times)
    cpu_total = cpu_ser + cpu_trans + cpu_de
    cpu_payload_mb = cpu_serialize_payload_bytes / (1024 * 1024)

    has_cuda_ipc = len(cuda_ipc_serialize_times) > 0

    print(f"\n{'=' * 95}")
    print(f"Benchmark: {model_name} ({total_mb:.1f} MB, {num_weights} tensors)")
    print(f"Rounds: {num_benchmark_rounds} (median), Warmup: {num_warmup_rounds}")
    print(f"Transport: Ray object store (ray.put/ray.get), matching production .remote() path")
    print(f"{'-' * 95}")
    print(
        f"{'Transport':<12} {'Payload (MB)':>13} {'Serialize (ms)':>15} "
        f"{'Transport (ms)':>15} {'Deserialize (ms)':>17} {'Total (ms)':>11}"
    )
    print(f"{'-' * 95}")
    print(
        f"{'cpu_serialize':<12} {cpu_payload_mb:>13.1f} {cpu_ser * 1000:>15.2f} "
        f"{cpu_trans * 1000:>15.2f} {cpu_de * 1000:>17.2f} {cpu_total * 1000:>11.2f}"
    )

    if has_cuda_ipc:
        ipc_ser = _median(cuda_ipc_serialize_times)
        ipc_trans = _median(cuda_ipc_transport_times)
        ipc_de = _median(cuda_ipc_deserialize_times)
        ipc_total = ipc_ser + ipc_trans + ipc_de
        ipc_payload_mb = cuda_ipc_payload_bytes / (1024 * 1024)

        print(
            f"{'cuda_ipc':<12} {ipc_payload_mb:>13.3f} {ipc_ser * 1000:>15.2f} "
            f"{ipc_trans * 1000:>15.2f} {ipc_de * 1000:>17.2f} {ipc_total * 1000:>11.2f}"
        )
        print(f"{'-' * 95}")
        speedup = cpu_total / ipc_total if ipc_total > 0 else float("inf")
        print(f"cuda_ipc speedup: {speedup:.2f}x")
    else:
        print(f"{'-' * 95}")
        print("cuda_ipc: SKIPPED (CUDA not available)")

    print(f"{'=' * 95}")
