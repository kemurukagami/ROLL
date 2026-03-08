"""
Integration tests: isolated single-LoRA step equivalence (sequential clusters).

Strategy
--------
Run the two clusters **sequentially** on the *same* GPU set so GPU requirements are
halved compared to running them in parallel.

Phase 1 — isolated cluster (multi-LoRA, ROLL_rlix ported strategy):
  - Register all adapters under ``is_lora_optimizer_isolated=True``.
  - For each adapter in turn, run ``train_step_lora`` for *n_steps* steps.
  - Record the scalar loss returned at every step.
  - Teardown.

Phase 2 — reference clusters (upstream single-LoRA, standard megatron_train):
  - For each adapter, create a **fresh** single-adapter cluster on the *same* GPUs.
  - Restore the adapter's initial weights (saved before Phase 1).
  - Run ``train_step`` for the same *n_steps* steps with the same token tensors.
  - Record the scalar loss at every step.
  - Teardown.

Assertion
---------
  loss[adapter][step] from Phase 1 == loss[adapter][step] from Phase 2
  (``torch.testing.assert_close(rtol=1e-5, atol=1e-6)`` on every scalar).

Test matrix
-----------
| TC | dp | tp | pp | Adapters | GPUs needed   |
|----|----|----|----|----------|---------------|
|  1 |  1 |  1 |  1 | a, b     | 1 (dp*tp*pp)  |
|  2 |  2 |  1 |  1 | a, b, c  | 2 (dp*tp*pp)  |
|  3 |  1 |  2 |  1 | a, b, c  | 2 (dp*tp*pp)  |
|  4 |  2 |  2 |  1 | a, b, c  | 4 (dp*tp*pp)  |
|  5 |  1 |  1 |  2 | a, b, c  | 2 (dp*tp*pp)  |
|  6 |  1 |  2 |  2 | a, b, c  | 4 (dp*tp*pp)  |
|  7 |  2 |  1 |  2 | a, b, c  | 4 (dp*tp*pp)  |

Determinism contract
--------------------
For the two-phase sequential design to produce numerically identical losses, ALL
stochastic operations must be either eliminated or seeded identically across phases.
The test enforces this via four mechanisms:

1. ``lora_dropout=0.0``
   LoRA adapter layers have no dropout, removing the primary source of
   LoRA-specific stochasticity.

2. ``model_config_kwargs={"attention_dropout": 0.0, "hidden_dropout": 0.0}``
   The frozen base model's dropout layers (attention and hidden) affect the
   activations that flow back through LoRA parameters.  Even though base weights
   are frozen, non-zero dropout yields different activation patterns across phases
   (because the global RNG state advances during Phase 1 adapter_a training and
   is NOT at the same position when Phase 2 reference for adapter_b starts from a
   fresh seed).  Setting both to 0.0 via ``model_config_kwargs`` eliminates this
   dependence on RNG state entirely.
   NOTE: Qwen2.5-0.5B-Instruct already ships with attention_dropout=0.0, so this
   is defensive rather than corrective for that model, but is required for safety.

3. ``is_offload_optimizer_states_in_train_step=False`` in microbatch meta_info
   Prevents asynchronous CPU↔GPU optimizer-state offload between steps, which
   could introduce timing-dependent numerical differences.

4. ``pipeline_config.seed=42`` (same for all clusters)
   Megatron uses this seed to initialise its per-rank RNG tracker.  Both clusters
   are seeded identically so any remaining RNG-dependent operation (e.g., Megatron
   TP dropout, weight init) starts from the same state.

Phase 1 dependencies (must be ported into ROLL_rlix before tests pass):
  - ``MegatronTrainStrategy.train_step_lora``  with ``is_lora_optimizer_isolated=True``
  - ``Worker.train_step_lora``
  - ``Worker.{get_lora_tensors, set_lora_tensors, copy_lora_params}``
"""
import os
import random
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch

from roll.configs.model_args import LoraArguments, ModelArguments
from roll.configs.training_args import TrainingArguments
from roll.configs.worker_config import StrategyArguments, WorkerConfig
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.distributed.scheduler.storage import SharedStorage
from roll.utils.constants import RAY_NAMESPACE, STORAGE_NAME

# Worker name shared between the two phases so loss key extraction is uniform.
_WORKER_NAME = "sft_train"

# ---- Determinism: zero out ALL base-model dropout (see module docstring §2) ----
# These kwargs are forwarded to the Hugging Face / Megatron model config so that
# attention softmax dropout and hidden-state FF dropout are disabled for every
# cluster in both phases.  This ensures forward-pass activations are deterministic
# regardless of the global PyTorch RNG state.
_ZERO_DROPOUT_MODEL_CONFIG_KWARGS: dict = {
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,

}
_LORA_TARGETS = "all-linear,all-router"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _unique_cluster_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _ensure_shared_storage() -> None:
    try:
        SharedStorage.options(name=STORAGE_NAME, get_if_exists=True, namespace=RAY_NAMESPACE).remote()
    except Exception:
        SharedStorage.options(name=STORAGE_NAME, namespace=RAY_NAMESPACE).remote()


def _ray_init() -> None:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(namespace=RAY_NAMESPACE, ignore_reinit_error=True, log_to_driver=False)
    _ensure_shared_storage()


def _seed_driver(seed: int = 42) -> None:
    """Seed the driver-process RNG.

    Ray worker processes are seeded via ``pipeline_config.seed``; this seeds the
    driver-side Python/NumPy/Torch state for any host-side random operations
    (e.g. generating token sequences).  Call before each cluster creation phase
    so both phases start from the same host-side RNG position.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_pipeline_config(*, seed: int = 42, sequence_length: int = 64) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        max_grad_norm=1.0,
        sequence_length=sequence_length,
        resume_from_checkpoint=False,
        model_update_buffer_size_mb=256,
        is_actor_infer_colocated=False,
    )


def _download_model(model_id: str) -> str:
    """Download model from Hugging Face and return the local snapshot path."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id)


def _system_envs() -> dict:
    root = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join([str(root), str(root / "mcore_adapter" / "src")])
    return {"PYTHONPATH": pythonpath}


def _isolated_worker_config(
    *,
    adapter_names: list[str],
    model_dir: str,
    dp: int,
    tp: int,
    pp: int = 1,
    gradient_accumulation_steps: int = 1,
) -> WorkerConfig:
    """WorkerConfig for the isolated multi-LoRA cluster.

    Determinism:
    - ``lora_dropout=0.0``   — no randomness in LoRA layers.
    - ``model_config_kwargs`` — zeros attention & hidden dropout in the base model
      so frozen base-model activations are deterministic regardless of RNG state.
    """
    adapters = {
        name: LoraArguments(lora_rank=8, lora_alpha=16, lora_dropout=0.0, lora_target=_LORA_TARGETS)
        for name in adapter_names
    }
    return WorkerConfig(
        name=_WORKER_NAME,
        worker_cls="roll.pipeline.sft.sft_worker.SFTWorker",
        model_args=ModelArguments(
            model_name_or_path=model_dir,
            dtype="bf16",
            adapters=adapters,
            model_config_kwargs=_ZERO_DROPOUT_MODEL_CONFIG_KWARGS,
        ),
        training_args=TrainingArguments(
            max_steps=999,           # effectively unlimited; we drive steps externally
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=1e-4,
            weight_decay=0.0,
        ),
        strategy_args=StrategyArguments(
            strategy_name="megatron_train",
            strategy_config={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": 1,
                "context_parallel_size": 1,
                "overlap_p2p_comm": False,
                "use_distributed_optimizer": False,   # required by isolated prototype
                "is_lora_optimizer_isolated": True,
            },
        ),
        device_mapping=f"list(range(0, {dp * tp * pp}))",
        infer_batch_size=1,
        system_envs=_system_envs(),
    )


def _reference_worker_config(
    *,
    adapter_name: str,
    model_dir: str,
    dp: int,
    tp: int,
    pp: int = 1,
    gradient_accumulation_steps: int = 1,
) -> WorkerConfig:
    """WorkerConfig for an upstream single-LoRA reference cluster.

    Uses the *same* GPU set as the isolated cluster (sequential execution).

    Determinism: applies the same ``model_config_kwargs`` and ``lora_dropout=0.0``
    as the isolated cluster so both phases are identically dropout-free.
    """
    adapters = {
        adapter_name: LoraArguments(lora_rank=8, lora_alpha=16, lora_dropout=0.0, lora_target=_LORA_TARGETS)
    }
    return WorkerConfig(
        name=_WORKER_NAME,
        worker_cls="roll.pipeline.sft.sft_worker.SFTWorker",
        model_args=ModelArguments(
            model_name_or_path=model_dir,
            dtype="bf16",
            adapters=adapters,
            model_config_kwargs=_ZERO_DROPOUT_MODEL_CONFIG_KWARGS,
        ),
        training_args=TrainingArguments(
            max_steps=999,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=1e-4,
            weight_decay=0.0,
        ),
        strategy_args=StrategyArguments(
            strategy_name="megatron_train",
            strategy_config={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": 1,
                "context_parallel_size": 1,
                "overlap_p2p_comm": False,
                "use_distributed_optimizer": False,
            },
        ),
        device_mapping=f"list(range(0, {dp * tp * pp}))",
        infer_batch_size=1,
        system_envs=_system_envs(),
    )


def _make_microbatch(input_ids: torch.Tensor, adapter_name: str, global_step: int) -> DataProto:
    """Build a single-row DataProto microbatch routed to *adapter_name*.

    Determinism: ``is_offload_optimizer_states_in_train_step=False`` disables the
    async CPU↔GPU optimizer-state offload that happens between steps.  In
    ``isolated`` mode the optimizer states are always kept resident anyway, but
    setting this on the reference cluster prevents any timing-dependent numerical
    differences from asynchronous offload.
    """
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    mb = DataProto.from_single_dict(
        {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    )
    mb.non_tensor_batch["lora_name"] = np.array([adapter_name] * input_ids.shape[0], dtype=object)
    mb.meta_info = {
        "lora_name": adapter_name,
        "global_step": global_step,
        "_broadcast_non_tensor_batch": True,
        # Disable async optimizer-state offload to remove a potential source of
        # timing-dependent numerical non-determinism between the two phases.
        "is_offload_optimizer_states_in_train_step": False,
        "loss_mask_keys": ["labels"],
    }
    return mb


def _extract_loss(result: DataProto) -> float:
    """Extract the scalar loss from a train_step / train_step_lora DataProto result.

    Checks both ``{worker_name}/loss`` (upstream convention) and
    ``{worker_name}/loss@sum`` (ROLL_rlix convention).
    """
    metrics: dict = result.meta_info.get("metrics", {}) if result.meta_info else {}
    for key in (f"{_WORKER_NAME}/loss", f"{_WORKER_NAME}/loss@sum"):
        if key in metrics:
            val = metrics[key]
            # val may be a tensor or a list of tensors (append_to_dict accumulates into lists)
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, torch.Tensor):
                return float(val.mean().item())
            return float(val)
    available = list(metrics.keys())
    raise KeyError(
        f"Expected loss key '{_WORKER_NAME}/loss' (or '/loss@sum') in metrics but got: {available}. "
        "Check that the SFTWorker's loss_func emits the expected key."
    )


def _shutdown(cluster: Cluster) -> None:
    try:
        cluster.execute_all_sync("shutdown")
    except Exception:
        pass
    for worker in getattr(cluster, "workers", []):
        try:
            ray.kill(worker, no_restart=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Core test logic, shared by all 4 test cases
# ---------------------------------------------------------------------------

def _run_equivalence_test(
    *,
    adapter_names: list[str],
    dp: int,
    tp: int,
    pp: int = 1,
    model_dir: str,
    resource_manager: ResourceManager,
    pipeline_config: SimpleNamespace,
    n_steps: int = 3,
    seed: int = 42,
    phase1_order: str = "sequential",
) -> None:
    """
    Phase 1: isolated multi-LoRA cluster
    ----------------------------------------
    1. Create cluster (all adapters, ``is_lora_optimizer_isolated=True``).
    2. Seed all adapters with identical initial weights (copy from first).
    3. Save those initial weights for Phase 2 reference clusters.
    4. Train all adapters for *n_steps* steps under one of two orderings
       controlled by *phase1_order*:

       ``"sequential"`` — train every step of adapter A before touching adapter B:

           for adapter in adapter_names:
               for step in range(n_steps):
                   train_step_lora(adapter, step)

       ``"interleaved"`` — round-robin across adapters, one step at a time:

           for step in range(n_steps):
               for adapter in adapter_names:
                   train_step_lora(adapter, step)

       Both orderings must produce the *same* per-adapter per-step loss because
       ``isolated`` mode isolates each adapter's optimizer state so that one
       adapter's step does NOT affect any other adapter's weight or momentum.
    5. Teardown cluster.

    Phase 2: upstream single-LoRA reference clusters (sequential, same GPUs)
    --------------------------------------------------------------------------
    6. For each adapter:
       a. Create a **fresh** single-adapter cluster on the *same* GPUs.
       b. Restore this adapter's initial weights (saved in step 3).
       c. Run ``train_step`` for *n_steps* steps with the same token tensors
          and the same ``pipeline_config.seed``.
       d. Collect per-step loss.
       e. Teardown cluster.

    Assertion
    ---------
    For every (adapter, step) pair:
      isolated_loss[adapter][step] == reference_loss[adapter][step].

    Determinism
    -----------
    - ``lora_dropout=0.0`` in both WorkerConfigs.
    - ``model_config_kwargs`` forces ``attention_dropout=0.0`` and
      ``hidden_dropout=0.0`` so frozen base-model activations are RNG-independent.
    - ``is_offload_optimizer_states_in_train_step=False`` in every microbatch.
    - Driver-side RNG is reset via ``_seed_driver(seed)`` before both phases.
    - Both clusters use the same ``pipeline_config.seed`` (worker-side Megatron RNG).
    """
    debug_trace = os.environ.get("RLIX_DEBUG_ISOLATED_LORA", "") not in ("", "0", "false", "False")

    # Fixed token sequences, one per step (different steps → different data,
    # making the multi-step comparison more discriminating).
    # These are generated with a deterministic formula so they don't depend on
    # host-side RNG state (same tensors across phases).
    # Replicate batch across dp-ranks so dispatch_dp_mp_dispatch_first can chunk
    # the batch evenly (batch_size must be >= dp).  Each dp rank receives an
    # identical row so the per-rank loss equals the single-rank reference loss.
    # Megatron PP with non-interleaved schedule needs >=2 microbatches in practice.
    # Keep GA=1 for non-PP tests, and GA=2 for PP tests to avoid PP stalls.
    ga_steps = 2 if pp > 1 else 1
    token_width = int(pipeline_config.sequence_length) if pp > 1 else 8
    step_input_ids: list[torch.Tensor] = [
        torch.tensor(
            [[((step * 7 + i) % 29) + 1 for i in range(token_width)]] * (dp * ga_steps),
            dtype=torch.long,
        )
        for step in range(n_steps)
    ]

    # -----------------------------------------------------------------------
    # Phase 1: isolated cluster
    # Reset driver-side RNG so host-side tensor construction is reproducible.
    # -----------------------------------------------------------------------
    _seed_driver(seed)
    pa_cfg = _isolated_worker_config(
        adapter_names=adapter_names,
        model_dir=model_dir,
        dp=dp,
        tp=tp,
        pp=pp,
        gradient_accumulation_steps=ga_steps,
    )
    pa_cluster = Cluster(
        name=_unique_cluster_name("multi_lora_isolated"),
        worker_cls=pa_cfg.worker_cls,
        resource_manager=resource_manager,
        worker_config=pa_cfg,
    )
    pa_cluster.initialize(pipeline_config=pipeline_config, blocking=True)

    # Ensure all adapters start from identical weights (copy from first).
    first = adapter_names[0]
    for other in adapter_names[1:]:
        pa_cluster.copy_lora_params(src_adapter=first, dst_adapter=other)
    # For non-PP runs, normalize DP rank drift at init.
    # PP runs shard LoRA tensors by stage, so rank-0 tensors cannot be broadcast
    # to every rank.
    if pp == 1:
        for name in adapter_names:
            pa_cluster.set_lora_tensors(name, pa_cluster.get_lora_tensors(name)[0])

    init_weights: dict[str, dict[str, torch.Tensor]] | None = None
    if pp == 1:
        init_weights = {
            name: pa_cluster.get_lora_tensors(name)[0]
            for name in adapter_names
        }

    # Train all adapters for n_steps steps under the requested ordering.
    isolated_losses: dict[str, list[float]] = {name: [] for name in adapter_names}
    isolated_lora_trace: dict[str, list[dict[str, torch.Tensor]]] = {
        name: [] for name in adapter_names
    }

    if phase1_order == "sequential":
        # All steps for adapter A, then all steps for adapter B, ...
        # Mirrors the simplest RLix scheduling policy.
        for name in adapter_names:
            for step in range(n_steps):
                mb = _make_microbatch(step_input_ids[step], name, global_step=step)
                result = pa_cluster.train_step_lora(mb)
                isolated_losses[name].append(_extract_loss(result))
                if debug_trace:
                    isolated_lora_trace[name].append(pa_cluster.get_lora_tensors(name)[0])

    elif phase1_order == "interleaved":
        # Round-robin: one step per adapter per outer iteration.
        # Verifies that interleaving does NOT corrupt any adapter's loss
        # trajectory — the key correctness claim of isolated optimizer
        # isolation.  Each adapter has its own step counter so global_step
        # is per-adapter, matching what the reference cluster sees.
        adapter_step: dict[str, int] = {name: 0 for name in adapter_names}
        for _outer in range(n_steps):
            for name in adapter_names:
                s = adapter_step[name]
                mb = _make_microbatch(step_input_ids[s], name, global_step=s)
                result = pa_cluster.train_step_lora(mb)
                isolated_losses[name].append(_extract_loss(result))
                if debug_trace:
                    isolated_lora_trace[name].append(pa_cluster.get_lora_tensors(name)[0])
                adapter_step[name] += 1

    else:
        raise ValueError(
            f"Unknown phase1_order={phase1_order!r}; expected 'sequential' or 'interleaved'"
        )

    _shutdown(pa_cluster)

    # -----------------------------------------------------------------------
    # Phase 2: upstream single-LoRA reference clusters (sequential, same GPUs)
    # Reset driver-side RNG to the same state as before Phase 1 so any
    # driver-side random ops are identical.
    # -----------------------------------------------------------------------
    _seed_driver(seed)
    reference_losses: dict[str, list[float]] = {}
    reference_lora_trace: dict[str, list[dict[str, torch.Tensor]]] = {
        name: [] for name in adapter_names
    }

    for name in adapter_names:
        ref_cfg = _reference_worker_config(
            adapter_name=name,
            model_dir=model_dir,
            dp=dp,
            tp=tp,
            pp=pp,
            gradient_accumulation_steps=ga_steps,
        )
        ref_cluster = Cluster(
            name=_unique_cluster_name(f"ref_{name}"),
            worker_cls=ref_cfg.worker_cls,
            resource_manager=resource_manager,
            worker_config=ref_cfg,
        )
        ref_cluster.initialize(pipeline_config=pipeline_config, blocking=True)

        # Restore initial weights from Phase 1 so both runs start identically.
        # PP runs keep LoRA tensors sharded by stage; this helper applies one
        # tensor dict to all ranks, so only restore in non-PP mode.
        if init_weights is not None:
            ref_cluster.set_lora_tensors(name, init_weights[name])

        step_losses: list[float] = []
        for step in range(n_steps):
            mb = _make_microbatch(step_input_ids[step], name, global_step=step)
            result = ref_cluster.train_step(mb)
            step_losses.append(_extract_loss(result))
            if debug_trace:
                reference_lora_trace[name].append(ref_cluster.get_lora_tensors(name)[0])

        _shutdown(ref_cluster)
        reference_losses[name] = step_losses

    if debug_trace:
        # Lightweight diff report to bisect divergence between isolated and reference runs.
        for name in adapter_names:
            if init_weights is None:
                continue
            init_tensors = init_weights[name]
            for step in range(n_steps):
                pa_tensors = isolated_lora_trace[name][step]
                ref_tensors = reference_lora_trace[name][step]
                max_diff = 0.0
                max_key = None
                max_pa_delta = 0.0
                max_ref_delta = 0.0
                for k, pa_v in pa_tensors.items():
                    ref_v = ref_tensors.get(k)
                    if ref_v is None:
                        raise KeyError(f"[debug] Missing tensor {k!r} in reference trace for {name!r}")
                    d = (pa_v.float() - ref_v.float()).abs().max().item()
                    if d > max_diff:
                        max_diff = d
                        max_key = k
                    init_v = init_tensors.get(k)
                    if init_v is None:
                        raise KeyError(f"[debug] Missing tensor {k!r} in init trace for {name!r}")
                    pa_d = (pa_v.float() - init_v.float()).abs().max().item()
                    ref_d = (ref_v.float() - init_v.float()).abs().max().item()
                    if pa_d > max_pa_delta:
                        max_pa_delta = pa_d
                    if ref_d > max_ref_delta:
                        max_ref_delta = ref_d
                print(f"[debug] adapter={name} step={step} max_lora_param_abs_diff={max_diff:.6e} key={max_key}")
                print(f"[debug] adapter={name} step={step} max_abs_delta_vs_init: isolated={max_pa_delta:.6e} reference={max_ref_delta:.6e}")

    # -----------------------------------------------------------------------
    # Assert: isolated loss == reference loss at every (adapter, step)
    # -----------------------------------------------------------------------
    for name in adapter_names:
        pa_losses = isolated_losses[name]
        ref_losses = reference_losses[name]
        assert len(pa_losses) == len(ref_losses) == n_steps, (
            f"[adapter={name}] Unexpected step count: pa={len(pa_losses)}, ref={len(ref_losses)}"
        )
        for step, (pa_loss, ref_loss) in enumerate(zip(pa_losses, ref_losses)):
            pa_t = torch.tensor(pa_loss)
            ref_t = torch.tensor(ref_loss)
            torch.testing.assert_close(
                pa_t,
                ref_t,
                rtol=1e-5,
                atol=1e-6,
                msg=(
                    f"Loss mismatch at adapter={name!r} step={step} "
                    f"[dp={dp}, tp={tp}, pp={pp}]: "
                    f"isolated={pa_loss:.8f}, reference={ref_loss:.8f}"
                ),
            )


# ---------------------------------------------------------------------------
# TC-1: dp=1, tp=1, adapters=[a, b]  — needs 1 GPU
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    torch.cuda.device_count() < 1,
    reason="TC-1 requires >= 1 CUDA device (dp=1, tp=1).",
)
def test_tc1_isolated_single_lora_step_dp1_tp1():
    """
    TC-1  dp=1, tp=1, adapters=[a, b], n_steps=3.

    Exercises both Phase-1 orderings against the same single-LoRA reference:
    - ``sequential``: all steps for adapter_a, then all steps for adapter_b.
    - ``interleaved``: step 0 → [a, b], step 1 → [a, b], step 2 → [a, b].

    Both must produce losses matching the reference at every (adapter, step).
    GPU budget: 1 (clusters run sequentially on the same GPU).
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = _download_model(model_id)

    os.environ.setdefault("roll_RPC_TIMEOUT", "600")
    _ray_init()

    dp, tp = 1, 1
    resource_manager = ResourceManager(num_nodes=1, num_gpus_per_node=torch.cuda.device_count())
    pipeline_config = _make_pipeline_config(seed=42, sequence_length=64)

    for order in ("sequential", "interleaved"):
        _run_equivalence_test(
            adapter_names=["adapter_a", "adapter_b"],
            dp=dp,
            tp=tp,
            model_dir=model_dir,
            resource_manager=resource_manager,
            pipeline_config=pipeline_config,
            n_steps=3,
            phase1_order=order,
        )


# ---------------------------------------------------------------------------
# TC-2: dp=2, tp=1, adapters=[a, b, c]  — needs 2 GPUs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="TC-2 requires >= 2 CUDA devices (dp=2, tp=1).",
)
def test_tc2_isolated_single_lora_step_dp2_tp1():
    """
    TC-2  dp=2, tp=1, adapters=[a, b, c], n_steps=3.

    Exercises both Phase-1 orderings under data parallelism (dp=2).
    GPU budget: 2 (clusters run sequentially).
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = _download_model(model_id)

    os.environ.setdefault("roll_RPC_TIMEOUT", "600")
    _ray_init()

    dp, tp = 2, 1
    resource_manager = ResourceManager(num_nodes=1, num_gpus_per_node=torch.cuda.device_count())
    pipeline_config = _make_pipeline_config(seed=42, sequence_length=64)

    for order in ("sequential", "interleaved"):
        _run_equivalence_test(
            adapter_names=["adapter_a", "adapter_b", "adapter_c"],
            dp=dp,
            tp=tp,
            model_dir=model_dir,
            resource_manager=resource_manager,
            pipeline_config=pipeline_config,
            n_steps=3,
            phase1_order=order,
        )


# ---------------------------------------------------------------------------
# TC-3: dp=1, tp=2, adapters=[a, b, c]  — needs 2 GPUs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="TC-3 requires >= 2 CUDA devices (dp=1, tp=2).",
)
def test_tc3_isolated_single_lora_step_dp1_tp2():
    """
    TC-3  dp=1, tp=2, adapters=[a, b, c], n_steps=3.

    Exercises both Phase-1 orderings under tensor parallelism (tp=2).
    GPU budget: 2 (clusters run sequentially).
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = _download_model(model_id)

    os.environ.setdefault("roll_RPC_TIMEOUT", "600")
    _ray_init()

    dp, tp = 1, 2
    resource_manager = ResourceManager(num_nodes=1, num_gpus_per_node=torch.cuda.device_count())
    pipeline_config = _make_pipeline_config(seed=42, sequence_length=64)

    for order in ("sequential", "interleaved"):
        _run_equivalence_test(
            adapter_names=["adapter_a", "adapter_b", "adapter_c"],
            dp=dp,
            tp=tp,
            model_dir=model_dir,
            resource_manager=resource_manager,
            pipeline_config=pipeline_config,
            n_steps=3,
            phase1_order=order,
        )


# ---------------------------------------------------------------------------
# TC-4: dp=2, tp=2, adapters=[a, b, c]  — needs 4 GPUs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="TC-4 requires >= 4 CUDA devices (dp=2, tp=2).",
)
def test_tc4_isolated_single_lora_step_dp2_tp2():
    """
    TC-4  dp=2, tp=2, adapters=[a, b, c], n_steps=3.

    Exercises both Phase-1 orderings under combined data + tensor parallelism.
    GPU budget: 4 (clusters run sequentially).
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = _download_model(model_id)

    os.environ.setdefault("roll_RPC_TIMEOUT", "600")
    _ray_init()

    dp, tp = 2, 2
    resource_manager = ResourceManager(num_nodes=1, num_gpus_per_node=torch.cuda.device_count())
    pipeline_config = _make_pipeline_config(seed=42, sequence_length=64)

    for order in ("sequential", "interleaved"):
        _run_equivalence_test(
            adapter_names=["adapter_a", "adapter_b", "adapter_c"],
            dp=dp,
            tp=tp,
            pp=1,
            model_dir=model_dir,
            resource_manager=resource_manager,
            pipeline_config=pipeline_config,
            n_steps=3,
            phase1_order=order,
        )


# ---------------------------------------------------------------------------
# TC-5: dp=1, tp=1, pp=2, adapters=[a, b, c]  — needs 2 GPUs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="TC-5 requires >= 2 CUDA devices (dp=1, tp=1, pp=2).",
)
def test_tc5_isolated_single_lora_step_dp1_tp1_pp2():
    """
    TC-5  dp=1, tp=1, pp=2, adapters=[a, b, c], n_steps=1.

    Exercises both Phase-1 orderings under pipeline parallelism.
    GPU budget: 2 (clusters run sequentially).
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = _download_model(model_id)

    os.environ.setdefault("roll_RPC_TIMEOUT", "600")
    _ray_init()

    dp, tp, pp = 1, 1, 2
    resource_manager = ResourceManager(num_nodes=1, num_gpus_per_node=torch.cuda.device_count())
    pipeline_config = _make_pipeline_config(seed=42, sequence_length=64)

    for order in ("sequential", "interleaved"):
        _run_equivalence_test(
            adapter_names=["adapter_a", "adapter_b", "adapter_c"],
            dp=dp,
            tp=tp,
            pp=pp,
            model_dir=model_dir,
            resource_manager=resource_manager,
            pipeline_config=pipeline_config,
            n_steps=3,
            phase1_order=order,
        )


# ---------------------------------------------------------------------------
# TC-6: dp=1, tp=2, pp=2, adapters=[a, b, c]  — needs 4 GPUs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="TC-6 requires >= 4 CUDA devices (dp=1, tp=2, pp=2).",
)
def test_tc6_isolated_single_lora_step_dp1_tp2_pp2():
    """
    TC-6  dp=1, tp=2, pp=2, adapters=[a, b, c], n_steps=1.

    Exercises both Phase-1 orderings under combined tensor + pipeline parallelism.
    GPU budget: 4 (clusters run sequentially).
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = _download_model(model_id)

    os.environ.setdefault("roll_RPC_TIMEOUT", "600")
    _ray_init()

    dp, tp, pp = 1, 2, 2
    resource_manager = ResourceManager(num_nodes=1, num_gpus_per_node=torch.cuda.device_count())
    pipeline_config = _make_pipeline_config(seed=42, sequence_length=64)

    for order in ("sequential", "interleaved"):
        _run_equivalence_test(
            adapter_names=["adapter_a", "adapter_b", "adapter_c"],
            dp=dp,
            tp=tp,
            pp=pp,
            model_dir=model_dir,
            resource_manager=resource_manager,
            pipeline_config=pipeline_config,
            n_steps=3,
            phase1_order=order,
        )


# ---------------------------------------------------------------------------
# TC-7: dp=2, tp=1, pp=2, adapters=[a, b, c]  — needs 4 GPUs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="TC-7 requires >= 4 CUDA devices (dp=2, tp=1, pp=2).",
)
def test_tc7_isolated_single_lora_step_dp2_tp1_pp2():
    """
    TC-7  dp=2, tp=1, pp=2, adapters=[a, b, c], n_steps=1.

    Exercises both Phase-1 orderings under combined data + pipeline parallelism.
    GPU budget: 4 (clusters run sequentially).
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_dir = _download_model(model_id)

    os.environ.setdefault("roll_RPC_TIMEOUT", "600")
    _ray_init()

    dp, tp, pp = 2, 1, 2
    resource_manager = ResourceManager(num_nodes=1, num_gpus_per_node=torch.cuda.device_count())
    pipeline_config = _make_pipeline_config(seed=42, sequence_length=64)

    for order in ("sequential", "interleaved"):
        _run_equivalence_test(
            adapter_names=["adapter_a", "adapter_b", "adapter_c"],
            dp=dp,
            tp=tp,
            pp=pp,
            model_dir=model_dir,
            resource_manager=resource_manager,
            pipeline_config=pipeline_config,
            n_steps=3,
            phase1_order=order,
        )
