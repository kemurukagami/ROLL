import asyncio
import copy
import gc
import os
from collections import deque
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed
from vllm import RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind, BeamSearchParams
from vllm.inputs.data import TokensPrompt
from vllm.utils import random_uuid

from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto, list_of_dict_to_dict_of_list
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.third_party.vllm import create_async_llm
from roll.utils.constants import DO_TIME_SHARING
from roll.utils.functionals import concatenate_input_and_output, reduce_metrics
from roll.utils.logging import get_logger
from roll.utils.lora_routing import ensure_lora_name_in_batch, get_lora_name_array, resolve_microbatch_lora_name
from roll.utils.offload_states import OffloadStateType
from roll.platforms import current_platform


logger = get_logger()


def _normalize_lora_int_ids_loaded(value) -> list[int]:
    """Normalize LoRA adapter integer IDs returned by vLLM's list_loras RPC.

    vLLM's ``list_loras`` API has inconsistent return formats across versions and
    distributed configurations:
      - Single GPU: returns ``[id1, id2, ...]`` (flat list of ints)
      - Multi-GPU/Tensor Parallel: may return ``[[id1, id2], [id1, id2], ...]``
        where each sub-list corresponds to a different rank's view
      - Empty state: returns ``[]`` or ``[[]]``

    This helper flattens nested structures, deduplicates across ranks, and returns
    a sorted list of unique integer adapter IDs for consistent downstream handling.

    Args:
        value: The raw return value from ``await model.list_loras()``. May be
            a flat list of ints, a nested list of lists, or an empty list.

    Returns:
        A sorted list of unique integer LoRA adapter IDs. Returns an empty list
        for invalid or empty inputs.
    """
    if not isinstance(value, list) or not value:
        return []
    # Handle nested [[id,...], ...] format from multi-rank responses
    if isinstance(value[0], list):
        flat: list[int] = []
        for sub in value:
            if not isinstance(sub, list):
                continue
            for item in sub:
                if isinstance(item, int):
                    flat.append(item)
        return sorted(set(flat))
    # Handle flat [id, ...] format from single-rank responses
    return [item for item in value if isinstance(item, int)]


class VllmStrategy(InferenceStrategy):
    strategy_name = "vllm"

    def __init__(self, worker: Worker):
        super().__init__(worker)

        # Metrics snapshot infrastructure
        self._metrics_snapshots = deque(maxlen=3600)
        self._metrics_snapshot_interval = 1.0  # Snapshot every 1 second
        self._metrics_task = None

    async def initialize(self, model_provider):
        set_seed(seed=self.worker.pipeline_config.seed)
        vllm_config = copy.deepcopy(self.worker_config.strategy_args.strategy_config)
        # Must explicitly set VLLM_USE_V1 to pass this check: https://github.com/vllm-project/vllm/pull/14972
        os.environ["VLLM_USE_V1"] = str(vllm_config.pop("VLLM_USE_V1", 1))
        self.sleep_level = vllm_config.pop("sleep_level", 1)

        data_parallel_size = vllm_config.get("data_parallel_size", 1)
        if data_parallel_size > 1:
            logger.info(
                f"VllmStrategy {self.worker.cluster_name} enable data parallel {data_parallel_size=} data_parallel_rank={self.worker.rank}"
                f" data_parallel_address={os.environ['MASTER_ADDR']} data_parallel_rpc_port={os.environ['MASTER_PORT']}"
            )
            assert data_parallel_size == self.worker.world_size, f"{data_parallel_size=} != {self.worker.world_size=}"
            vllm_config.update(
                {
                    "data_parallel_rank": self.worker.rank, # set data_parallel_rank to use external load balancing
                    "data_parallel_address": os.environ["MASTER_ADDR"],
                    "data_parallel_rpc_port": os.environ["MASTER_PORT"],
                }
            )

        if self.worker_config.model_args.dtype == "fp32":
            dtype = "float32"
        elif self.worker_config.model_args.dtype == "fp16":
            dtype = "float16"
        elif self.worker_config.model_args.dtype == "bf16":
            dtype = "bfloat16"
        else:
            dtype = "auto"
        vllm_config.update(
            {
                "model": self.worker_config.model_args.model_name_or_path,
                "dtype": dtype,
                "enforce_eager": vllm_config.get("enforce_eager", False),
                "trust_remote_code": True,
                "seed": self.worker.pipeline_config.seed,
                "disable_custom_all_reduce": vllm_config.get(
                    "disable_custom_all_reduce", True
                ),  # potentially hangs in tp>1
                "enable_prefix_caching": vllm_config.get("enable_prefix_caching", True),
                "load_format": vllm_config.get("load_format", "dummy"),  # use model update passed value
                "max_num_batched_tokens": vllm_config.get("max_num_batched_tokens", 8192), # use default value of LLM class usage context
            }
        )

        # =====================================================================
        # Multi-LoRA Configuration
        # =====================================================================
        # Detection: LoRA mode is active when adapters dict is configured and non-empty.
        # This replaces the legacy lora_target field check.
        # Note: We check both `is not None` and `len() > 0` because:
        #   - adapters=None → LoRA disabled
        #   - adapters={} (empty dict) → invalid config, would crash on max_lora_rank
        adapters = self.worker_config.model_args.adapters
        self.is_lora = adapters is not None and len(adapters) > 0
        if self.is_lora:
            # -----------------------------------------------------------------
            # vLLM V1 Multi-LoRA Support:
            # -----------------------------------------------------------------
            # vLLM V1 supports multi-LoRA with prefix caching and chunked prefill:
            #   - Block hashes include LoRA adapter name via _gen_lora_extra_hash_keys()
            #   - Each request's lora_request.lora_name is part of the cache key
            #   - See: vllm/v1/core/kv_cache_utils.py:generate_block_hash_extra_keys()
            # -----------------------------------------------------------------

            # max_loras: Maximum number of LoRA adapters that can be resident in GPU
            # memory simultaneously. Set to at least configured adapters + 1 for
            # dynamic loading headroom.
            max_loras_cfg = int(vllm_config.get("max_loras", 0) or 0)
            lora_kwargs = {
                "enable_lora": True,
                "max_loras": max(max_loras_cfg, len(adapters) + 1),
                "max_lora_rank": max(a.lora_rank for a in adapters.values()),
            }
            vllm_config.update(lora_kwargs)
            # LoRA mode requires real base model weights for adapter weight initialization.
            # "dummy" load_format only works for weight broadcasting from trainer.
            vllm_config["load_format"] = "auto"

        # Guard: LoRA mode is incompatible with dummy load_format (used for weight broadcasting).
        # Users must either set load_format='auto' or disable LoRA.
        if self.is_lora and vllm_config.get("load_format") == "dummy":
            raise RuntimeError(
                "vLLM LoRA mode requires real base model weights; got load_format='dummy'. "
                "Set vllm strategy_config.load_format='auto' or disable LoRA."
            )

        # Guard: Multi-LoRA routing requires vLLM V1 engine for adapter-id RPC APIs.
        # The V0 engine does not expose the per-request adapter selection APIs needed
        # for routing different samples to different LoRA adapters in a single batch.
        if self.is_lora:
            vllm_use_v1 = int(os.environ.get("VLLM_USE_V1", "1"))
            if vllm_use_v1 != 1:
                raise RuntimeError(
                    "LoRA mode in ROLL_rlix requires VLLM_USE_V1=1. "
                    "Non-v1 engine path does not expose adapter-id APIs required by multi-LoRA routing."
                )

        logger.info(f"vllm_config: {vllm_config}")
        assert not dist.is_initialized()

        # Can not set VLLM_PORT explicitly in DP. Each call of get_engine_client_zmq_addr in
        # DPCoordinator will return the same port, which will cause port conflict.
        # https://github.com/vllm-project/vllm/blob/releases/v0.10.0/vllm/v1/engine/coordinator.py#L72
        if not data_parallel_size > 1:
            # set VLLM_PORT to avoid port conflict applied by vllm
            vllm_port = self.worker.get_free_port()
            os.environ["VLLM_PORT"] = str(vllm_port)

        self.model = await create_async_llm(resource_placement_groups=self.worker_config.resource_placement_groups, **vllm_config)

        self.tokenizer = await self.model.get_tokenizer()
        additional_special_tokens = self.tokenizer.additional_special_tokens
        special_tokens = [
            add_token
            for add_token in self.tokenizer.added_tokens_decoder.values()
            if add_token.special and add_token.content not in additional_special_tokens
        ]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}, replace_additional_special_tokens=False
        )
        logger.info(f"add {special_tokens} to additional_special_tokens: {self.tokenizer.additional_special_tokens}")

        self.worker.rank_info.dp_rank = self.worker.rank
        self.worker.rank_info.dp_size = self.worker.world_size

        self.is_model_in_gpu = True

        try:
            from vllm.v1.metrics.reader import get_metrics_snapshot
            self._metrics_task = asyncio.create_task(self._collect_metrics_snapshot())
        except Exception as e:
            logger.warning(f"Failed to create metrics collector task: {e}")

    def op_compute_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        vllm实现compute log probs在这里实现即可
        """
        pass

    async def generate(self, batch: DataProto, generation_config) -> torch.Tensor:
        # Check if beam search is requested
        if self._should_use_beam_search(generation_config):
            return await self._generate_with_beam_search(batch, generation_config)
        else:
            return await self._generate_standard(batch, generation_config)

    def _should_use_beam_search(self, generation_config) -> bool:
        """Check if beam search should be used based on generation_config."""
        return generation_config.get("num_beams", 1) > 1 or generation_config.get("use_beam_search", False)

    async def _generate_standard(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        """Standard generate method for non-beam search cases with multi-LoRA routing.

        This method handles both single-LoRA and multi-LoRA scenarios:
          - Single-LoRA: All samples use the same adapter (auto-filled if not specified)
          - Multi-LoRA: Each sample specifies its adapter via ``lora_name`` in non_tensor_batch

        The multi-LoRA routing flow:
          1. Extract per-sample adapter names from ``batch.non_tensor_batch["lora_name"]``
          2. Resolve each adapter name to its vLLM-assigned integer ID
          3. Construct a ``LoRARequest`` per sample
          4. Pass the per-sample requests to vLLM's generate API

        Args:
            batch: Input batch containing ``batch`` (tensor data) and ``non_tensor_batch``
                (metadata including optional ``lora_name`` array).
            generation_config: Generation parameters (temperature, top_p, etc.).

        Returns:
            Output tensor of shape ``(bs * num_return_sequences, input_len + max_response_len)``.
        """
        sampling_params = create_sampling_params_for_vllm(gen_kwargs=generation_config)

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        if "multi_modal_data" in batch.non_tensor_batch:
            prompts = [TokensPrompt(data) for data in batch.non_tensor_batch["multi_modal_data"]]
        else:
            prompts = [TokensPrompt(prompt_token_ids=prompt)
                for prompt in gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            ]

        # =====================================================================
        # Multi-LoRA Per-Sample Routing
        # =====================================================================
        # In multi-LoRA mode, each sample in the batch may use a different adapter.
        # The adapter assignment is determined by:
        #   1. Explicit per-sample ``lora_name`` in non_tensor_batch (producer sets this)
        #   2. Single-adapter fallback: if only one adapter is configured, all samples
        #      use it automatically (ensures backward compatibility)
        #
        # The routing validation ensures:
        #   - ``lora_name`` array length matches batch size
        #   - All referenced adapters are registered and loaded in vLLM
        # =====================================================================
        if self.is_lora:
            ensure_lora_name_in_batch(
                batch.non_tensor_batch,
                adapters=self.worker_config.model_args.adapters,
                batch_size=batch.batch["input_ids"].size(0),
            )

        lora_requests: list[LoRARequest | None] | None = None
        if self.is_lora:
            # Step 1: Extract per-sample adapter names
            lora_names = get_lora_name_array(batch.non_tensor_batch)

            # Step 2: Validate adapter count matches prompt count
            if len(lora_names) != len(prompts):
                logger.error("LoRA routing mismatch: len(lora_names)=%s len(prompts)=%s", len(lora_names), len(prompts))
                raise RuntimeError(
                    f"vLLM routing requires len(lora_name)==len(prompts), got {len(lora_names)} vs {len(prompts)}"
                )

            # Step 3: Build adapter name -> integer ID mapping
            adapters = [str(d) for d in lora_names.tolist()]
            lora_request_path = self.worker_config.model_args.model_name_or_path
            lora_int_ids_loaded = _normalize_lora_int_ids_loaded(await self.model.list_loras())
            adapter_to_int_id: dict[str, int] = {}
            for adapter in sorted(set(adapters)):
                # Validate adapter is configured
                if adapter not in self.worker_config.model_args.adapters:
                    raise RuntimeError(f"Unknown LoRA adapter requested by lora_name={adapter!r}")
                # Get vLLM-assigned integer ID
                lora_int_id = await self.get_lora_id(adapter)
                if lora_int_id is None:
                    raise RuntimeError(f"Missing LoRA adapter in vLLM engine: {adapter!r}")
                # Verify adapter is loaded (visible in list_loras)
                if lora_int_id not in lora_int_ids_loaded:
                    raise RuntimeError(
                        f"LoRA adapter id not loaded in vLLM engine: adapter={adapter!r} lora_int_id={lora_int_id}"
                    )
                adapter_to_int_id[adapter] = lora_int_id

            # Step 4: Construct per-sample LoRARequest objects
            # vLLM uses these to route each request to the correct adapter's weights.
            lora_requests = [
                LoRARequest(
                    lora_name=adapter,
                    lora_int_id=adapter_to_int_id[adapter],
                    lora_path=lora_request_path,
                )
                for adapter in adapters
            ]

        async def _generate(prompt, lora_request: LoRARequest | None):
            request_id = random_uuid()
            result_generator = self.model.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )
            output: Optional[RequestOutput] = None
            async for result in result_generator:
                output = result
            return output

        # Execute all generations in parallel, each with its LoRARequest (or None for non-LoRA mode)
        if lora_requests is None:
            vllm_outputs = await asyncio.gather(*[_generate(prompt, None) for prompt in prompts])
        else:
            vllm_outputs = await asyncio.gather(
                *[_generate(prompt, lora_request) for prompt, lora_request in zip(prompts, lora_requests, strict=True)]
            )

        # (bs * num_return_sequences, max_response_len)
        output_ids = gather_outputs_to_pad_tensor(
            request_outputs=vllm_outputs,
            pad_token_id=self.tokenizer.pad_token_id,
            device=input_ids.device,
        )

        # (bs * num_return_sequences, input_len + max_response_len)
        output = concatenate_input_and_output(
            input_ids=input_ids, output_ids=output_ids, num_return_sequences=sampling_params.n
        )

        return output

    async def _generate_with_beam_search(self, batch: DataProto, generation_config: Dict) -> torch.Tensor:
        """Generate using beam search method."""
        # Create beam search parameters
        beam_params = BeamSearchParams(
            beam_width=generation_config.get("num_beams", 1),
            max_tokens=generation_config.get("max_new_tokens", 50),
            temperature=generation_config.get("temperature", 0.0),
            ignore_eos=generation_config.get("ignore_eos", False),
            length_penalty=generation_config.get("length_penalty", 1.0),
            include_stop_str_in_output=generation_config.get("include_stop_str_in_output", False),
        )

        input_ids = batch.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = batch.batch["attention_mask"]  # left-padded attention_mask

        # Prepare prompts for beam_search
        if "multi_modal_data" in batch.non_tensor_batch:
            # For multimodal data, we need to handle it differently
            # This is a simplified approach - may need refinement based on actual multimodal format
            prompts = batch.non_tensor_batch["multi_modal_data"]
        else:
            # Convert to token lists format expected by beam_search
            token_lists = gather_unpadded_input_ids(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # Convert to TokensPrompt format expected by vLLM beam_search
            prompts = [{"prompt_token_ids": token_ids} for token_ids in token_lists]

        # Call beam_search method
        async def _beam_search(prompt):
            request_id = random_uuid()
            result_generator = self.model.beam_search(
                prompt=prompt,
                request_id=request_id,
                params=beam_params,
            )
            output: Optional[RequestOutput] = None
            async for result in result_generator:
                output = result
            return output

        beam_search_outputs = await asyncio.gather(*[_beam_search(prompt) for prompt in prompts])

        generated_token_ids = []
        for request_output in beam_search_outputs:
            for completion_output in request_output.outputs:
                generated_tokens = completion_output.token_ids
                generated_token_ids.append(torch.tensor(generated_tokens, device=input_ids.device))

        # Pad the sequences
        output_ids = pad_sequence(generated_token_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Concatenate input and output
        output = concatenate_input_and_output(
            input_ids=input_ids,
            output_ids=output_ids,
            num_return_sequences=beam_params.beam_width
        )

        return output

    async def generate_request(self, data: DataProto):
        """Generate for a single streaming request with LoRA adapter routing.

        Unlike ``_generate_standard`` which handles batch inference with per-sample
        LoRA routing, this method handles single-request streaming generation where
        each request uses exactly one LoRA adapter.

        The LoRA routing flow for single requests:
          1. Resolve the adapter name from ``non_tensor_batch`` (single value, not array)
          2. Look up the vLLM-assigned integer ID for the adapter
          3. Verify the adapter is loaded
          4. Construct and pass a single ``LoRARequest`` to vLLM

        Routing metadata is recorded in ``data.meta_info`` for observability:
          - ``routed_lora_name``: The resolved adapter name
          - ``routed_lora_int_id``: The vLLM integer ID for the adapter

        Args:
            data: Input data proto containing:
                - ``batch``: Tensor data (input_ids, attention_mask)
                - ``non_tensor_batch``: Metadata including ``lora_name``
                - ``meta_info``: Request ID, generation config, etc.

        Returns:
            DataProto with output tokens, finish reasons, and logprobs in meta_info.

        Raises:
            RuntimeError: If LoRA routing fails (adapter not found, not loaded, etc.)
        """
        # Keep meta_info writable for routing diagnostics; some callers may pass None.
        if data.meta_info is None:
            data.meta_info = {}
        collect_unfinished = data.meta_info.get("collect_unfinished", False)
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        request_id = data.meta_info["request_id"]
        generation_config = data.meta_info.get("generation_config")
        max_new_tokens = data.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
        max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
        output_kind = RequestOutputKind.CUMULATIVE if collect_unfinished else RequestOutputKind.FINAL_ONLY
        sampling_params = create_sampling_params_for_vllm(
            gen_kwargs={**generation_config, "max_new_tokens": max_new_tokens, "output_kind": output_kind}
        )
        assert sampling_params.n == 1 or not collect_unfinished, "collect_unfinished is not supported in parallel sampling"
        if "multi_modal_data" in data.non_tensor_batch:
            assert len(data.non_tensor_batch["multi_modal_data"]) == 1
            prompt_token_ids = data.non_tensor_batch["multi_modal_data"][0]["prompt_token_ids"]
            multi_modal_data = (data.non_tensor_batch["multi_modal_data"][0]["multi_modal_data"]
                                if "multi_modal_data" in data.non_tensor_batch["multi_modal_data"][0] else None)
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, multi_modal_data=multi_modal_data)
        else:
            assert input_ids.size(0) == 1, f"data['input_ids'] must have exactly one batch dimension"
            prompt_token_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            assert len(prompt_token_ids) == 1
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids[0])

        # =====================================================================
        # Single-Request LoRA Routing
        # =====================================================================
        # For streaming requests, each request uses exactly one LoRA adapter.
        # The adapter name is resolved from non_tensor_batch (single value, not array).
        # This differs from _generate_standard where we handle per-sample routing
        # within a batch.
        # =====================================================================
        if self.is_lora:
            ensure_lora_name_in_batch(
                data.non_tensor_batch,
                adapters=self.worker_config.model_args.adapters,
                batch_size=data.batch["input_ids"].size(0),
            )

        lora_request = None
        if self.is_lora:
            # Step 1: Resolve the adapter name for this single request
            routing = resolve_microbatch_lora_name(data.non_tensor_batch)

            # Step 2: Get vLLM-assigned integer ID for the adapter
            lora_name = routing.lora_name
            lora_int_id = await self.get_lora_id(lora_name)
            if lora_int_id is None:
                raise RuntimeError(f"Missing LoRA adapter in vLLM engine: {lora_name!r}")

            # Record routing decision for observability
            data.meta_info["routed_lora_name"] = lora_name
            data.meta_info["routed_lora_int_id"] = int(lora_int_id)

            # Step 3: Verify adapter is loaded (handle race condition after add_lora)
            lora_int_ids_loaded = _normalize_lora_int_ids_loaded(await self.model.list_loras())
            if lora_int_id not in lora_int_ids_loaded:
                # Fail fast if adapter not visible - add_lora should have waited
                raise RuntimeError(
                    f"LoRA adapter id not loaded: adapter={lora_name!r} lora_int_id={lora_int_id} loaded={lora_int_ids_loaded[:16]!r}"
                )

            # Step 4: Construct LoRARequest for vLLM
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=self.worker_config.model_args.model_name_or_path,
            )

        result_generator = self.model.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
        output: Optional[RequestOutput] = None
        # vLLM support partial rollout in v1 from 0.10.1, and will return finished output
        # with finish_reason setted no matter what RequestOutputKind is.
        # For compatibility, the following except block are only for v0 and older version of v1.
        try:
            async for result in result_generator:
                output = result
        except asyncio.CancelledError:
            if output is None:
                output_data = DataProto(meta_info=data.meta_info)
                output_data.meta_info["finish_reasons"] = ["abort"]
                return output_data

        output_token_ids, finish_reasons, logprobs = [], [], []
        for completion_output in output.outputs:
            output_token_ids.append(completion_output.token_ids)
            # For compatibility, older version may return unfinished result, set finish_reason of those to 'abort'.
            finish_reason = "abort" if completion_output.finish_reason is None else completion_output.finish_reason
            finish_reasons.append(finish_reason)
            if completion_output.logprobs is not None:
                logprobs.append(
                    [
                        float(lps[token_id].logprob)
                        for token_id, lps in zip(completion_output.token_ids, completion_output.logprobs)
                    ]
                )
        output_data = DataProto(meta_info=data.meta_info)
        output_data.meta_info["output_token_ids"] = output_token_ids
        output_data.meta_info["finish_reasons"] = finish_reasons
        output_data.meta_info["output_logprobs"] = logprobs
        return output_data

    async def abort_requests(self, request_ids):
        for id in request_ids:
            await self.model.abort(request_id=id)

    # offload/reload 接口
    async def load_states(self, *args, **kwargs):
        # Ensure KV/block manager exists before reset_prefix_cache. Calling reset on an
        # uninitialized engine state can block indefinitely.
        logger.info("[vllm_strategy][load_states] enter is_model_in_gpu=%s", self.is_model_in_gpu)
        if not self.is_model_in_gpu:
            logger.info("[vllm_strategy][load_states] calling model.load_states()")
            await self.model.load_states()
            self.is_model_in_gpu = True
            logger.info("[vllm_strategy][load_states] model.load_states() done")
        logger.info("[vllm_strategy][load_states] calling reset_prefix_cache()")
        await self.model.reset_prefix_cache()
        logger.info("[vllm_strategy][load_states] reset_prefix_cache() done")

    async def offload_states(self, include=None, non_blocking=False):
        await self.model.reset_prefix_cache()
        if include is None or OffloadStateType.model_params in include:
            if self.is_model_in_gpu and (self.worker.pipeline_config.is_actor_infer_colocated or DO_TIME_SHARING):
                await self.model.offload_states(self.sleep_level)
                self.is_model_in_gpu = False
        gc.collect()
        current_platform.empty_cache()
    
    def process_weights_after_loading(self,*args, **kwargs):
        # CustomAsyncLLM.process_weights_after_loading is async; return the awaitable so caller can await.
        return self.model.process_weights_after_loading()

    # =====================================================================
    # Collective Communication Group Management
    # =====================================================================
    # These methods manage process groups for distributed weight synchronization
    # between trainer (FSDP2) and inference workers. Two call styles are supported:
    #
    # 1. Dynamic comm_plan style (modern, selective model-update):
    #    Used for fine-grained control over which ranks participate in each group.
    #    setup_collective_group(comm_plan=..., backend=?, timeout_s=?)
    #
    # 2. Legacy/persistent broadcast group style:
    #    Used for traditional all-rank broadcast communication patterns.
    #    setup_collective_group(master_address=..., master_port=..., rank_offset=...,
    #                          world_size=..., group_name=..., backend=?, timeout_s=?)
    # =====================================================================

    async def setup_collective_group(self, *args, **kwargs) -> None:
        """Create a collective communication group for distributed operations.

        This method supports two calling conventions for different use cases:

        **Style 1: Dynamic comm_plan (recommended for multi-LoRA)**
            Uses a communication plan that specifies which ranks participate.
            This enables selective model updates where only relevant workers
            receive weight broadcasts for specific adapters.

            Required kwargs:
                comm_plan: Communication plan specifying participant ranks.

            Optional kwargs:
                backend: Communication backend (defaults to platform default).
                timeout_s: Timeout for group creation in seconds.

        **Style 2: Legacy broadcast group**
            Creates a persistent process group for traditional all-rank broadcasts.
            Used when all workers need to participate in weight synchronization.

            Required kwargs:
                master_address: Address of the rank 0 process.
                master_port: Port for communication.
                rank_offset: Offset to apply to local ranks.
                world_size: Total number of ranks in the group.
                group_name: Unique identifier for this process group.

            Optional kwargs:
                backend: Communication backend (defaults to platform default).
                timeout_s: Timeout for group creation in seconds.

        Raises:
            TypeError: If neither style's required arguments are provided.
        """
        # Style 1: Dynamic comm_plan based group setup
        if "comm_plan" in kwargs:
            backend = kwargs.get("backend", None)
            timeout_s = kwargs.get("timeout_s", None)
            comm_plan = kwargs["comm_plan"]
            backend = backend if backend is not None else current_platform.communication_backend
            await self.model.setup_collective_group(
                comm_plan=comm_plan, backend=backend, rank_in_cluster=self.worker.rank, timeout_s=timeout_s
            )
            return

        # Style 2: Legacy/persistent broadcast group
        required = {"master_address", "master_port", "rank_offset", "world_size", "group_name"}
        if required.issubset(kwargs.keys()):
            backend = kwargs.get("backend", None)
            timeout_s = kwargs.get("timeout_s", None)
            backend = backend if backend is not None else current_platform.communication_backend
            logger.info(f"setup_collective_group group_name={kwargs['group_name']!r}")
            await self.model.setup_collective_group(
                kwargs["master_address"],
                kwargs["master_port"],
                kwargs["rank_offset"],
                kwargs["world_size"],
                kwargs["group_name"],
                backend,
                timeout_s=timeout_s,
            )
            return

        raise TypeError(
            "VllmStrategy.setup_collective_group expects either "
            "(comm_plan=..., backend=?, timeout_s=?) "
            "or (master_address=..., master_port=..., rank_offset=..., world_size=..., group_name=..., backend=?, timeout_s=?)."
        )

    async def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False, *, broadcast_local_ranks=None):
        await self.model.broadcast_parameter(names, dtypes, shapes, group_name, is_lora, broadcast_local_ranks=broadcast_local_ranks)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False, *, ipc_local_ranks=None):
        await self.model.update_parameter_in_bucket(serialized_named_tensors, is_lora, ipc_local_ranks=ipc_local_ranks)

    async def destroy_collective_group(self, group_name: str, model_update_name: str | None = None) -> None:
        """Destroy a previously created collective communication group.

        Args:
            group_name: The name of the process group to destroy.
            model_update_name: Unused in vLLM strategy (kept for API compatibility
                with other strategies that track model_update_comm_plan state).
        """
        # vLLM has no model_update_comm_plan bookkeeping; model_update_name is unused.
        del model_update_name
        await self.model.destroy_collective_group(group_name)

    async def add_lora(
        self,
        adapter_name: str = "default",
        peft_config: dict = None,
        *,
        lora_local_ranks=None,
        wake_after_add: bool = True,
    ):
        """Register a LoRA adapter with the vLLM inference engine.

        This method handles the full lifecycle of LoRA adapter registration:
          1. Validates the adapter name against the configured adapters dict
          2. Calls vLLM's add_lora RPC with the PEFT configuration
          3. Tracks readiness via wake_after_add without follow-up visibility RPCs

        The method is designed for multi-LoRA scenarios where different samples
        in a batch may need different adapters. Each adapter must be registered
        before it can be used in inference via LoRARequest routing.

        This method always re-registers the adapter to ensure updated LoRA weights
        from the latest training step are applied. The caller must evict stale
        registrations via offload_states() before the next model_update, because
        LoRA GPU tensors are discarded for both sleep_level=1 and sleep_level=2.

        Args:
            adapter_name: Name of the adapter to register. Must match a key in
                ``worker_config.model_args.adapters``. Defaults to "default" for
                backward compatibility with single-LoRA callers.
            peft_config: PEFT configuration dict containing LoRA parameters.
                Required. The ``target_modules`` field is overwritten from the
                configured adapter spec to ensure consistency.
            wake_after_add: Whether this adapter registration should fully wake
                the vLLM engine (weights + KV cache). For multi-adapter updates,
                callers set this only on the last adapter.
        Raises:
            RuntimeError: If:
                - ``peft_config`` is None
                - ``adapter_name`` is not in the configured adapters
                - ``adapter_name="default"`` in multi-LoRA mode (FSDP2 limitation)

        Note:
            - This method intentionally avoids immediate post-registration visibility
              RPC checks (``get_lora_id``/``list_loras``) to avoid reentrancy stalls.
              Readiness is tracked via ``wake_after_add``: non-final adapters keep
              KV cache asleep, while the final adapter marks the model ready.
            - For multi-LoRA with FSDP2 trainer, use explicit adapter names instead
              of the "default" placeholder to avoid ambiguity.
        """
        # Backward-compatible: single-LoRA callers may pass only peft_config and rely on adapter_name default.
        if peft_config is None:
            raise RuntimeError("add_lora: peft_config must not be None")
        adapters = self.worker_config.model_args.adapters or {}
        if adapter_name not in adapters:
            raise RuntimeError(
                f"add_lora: unknown adapter_name={adapter_name!r}. "
                f"Valid adapters: {sorted(adapters.keys())}"
            )
        # Guard: FSDP2 model_update path does not support multi-LoRA weight broadcasting.
        # Using "default" name in multi-LoRA config would cause ambiguity.
        if adapter_name == "default" and len(adapters) > 1:
            raise RuntimeError(
                "add_lora called with adapter_name='default' in multi-LoRA mode. "
                "FSDP2 model_update path does not support multi-LoRA. "
                f"Configured adapters: {list(adapters.keys())}"
            )
        # Keep target_modules JSON-serializable and deterministic for worker-side hashing.
        peft_config["target_modules"] = sorted(adapters[adapter_name].lora_target)
        # Blocking RPC: does not return until custom_add_lora on the worker completes.
        # Inside custom_add_lora the sequence is:
        #   1. reload_model()         → wake_up(["weights"]) only (no KV cache wake-up)
        #   2. vLLM.add_lora()        → LoRA tensors loaded to GPU, adapter registered in vLLM Python cache
        #   3. register(name, id)     → _lora_names updated only after vLLM confirms success
        await self.model.add_lora(
            adapter_name,
            peft_config,
            lora_local_ranks=lora_local_ranks,
            wake_after_add=wake_after_add,
        )
        # No follow-up visibility RPCs here (get_lora_id/list_loras) to avoid
        # reentrancy hazards. Trust worker-level add_lora success and track GPU
        # readiness based on whether this call performed the final wake-up.
        self.is_model_in_gpu = wake_after_add
        logger.info(
            "[vllm_strategy][add_lora] registered adapter=%s (worker-level ok; is_model_in_gpu=%s)",
            adapter_name, self.is_model_in_gpu,
        )

    async def get_lora_id(self, adapter_name: str) -> int | None:
        """Get the integer ID assigned by vLLM for a named LoRA adapter.

        vLLM assigns unique integer IDs to each registered LoRA adapter. These IDs
        are required for constructing ``LoRARequest`` objects during inference.

        Note:
            vLLM's ``get_lora_id`` RPC may return various formats depending on the
            distributed configuration:
              - Single rank: returns ``int`` directly
              - Multi-rank via collective_rpc: returns ``[int]`` or ``[[int]]``
            This method normalizes all formats to a single ``int | None``.

        Args:
            adapter_name: The name of the LoRA adapter to query.

        Returns:
            The integer ID if the adapter is registered, or ``None`` if not found.

        Raises:
            RuntimeError: If different ranks report different IDs for the same
                adapter name, indicating a registration inconsistency.
        """
        lora_id = await self.model.get_lora_id(adapter_name)
        # Handle vLLM collective_rpc return format variations:
        # - Single rank: int
        # - Multi-rank: [int, int, ...] (one per rank) or [[int], ...]
        if isinstance(lora_id, list):
            if not lora_id:
                return None
            # Handle nested [[id], ...] format
            if len(lora_id) == 1 and isinstance(lora_id[0], list):
                inner = lora_id[0]
                return inner[0] if inner else None
            # Handle [id, id, ...] format - verify consistency across ranks
            first = lora_id[0]
            if all(x == first for x in lora_id):
                return first
            raise RuntimeError(f"Inconsistent LoRA id across ranks for adapter {adapter_name!r}: {lora_id!r}")
        return lora_id

    async def _collect_metrics_snapshot(self):
        """Collect metrics snapshots periodically in a background thread."""
        from vllm.v1.metrics.reader import get_metrics_snapshot
        while True:
            raw_metrics = get_metrics_snapshot()
            snapshot = {
                'vllm/kv_cache_usage_perc_max': [],
                'vllm/num_requests_waiting_max': [],
                'vllm/num_preemptions_max': []
            }
            for metric in raw_metrics:
                if metric.name == "vllm:kv_cache_usage_perc":
                    snapshot['vllm/kv_cache_usage_perc_max'].append(metric.value)
                elif metric.name == "vllm:num_requests_waiting":
                    snapshot['vllm/num_requests_waiting_max'].append(metric.value)
                elif metric.name == "vllm:num_preemptions":
                    snapshot['vllm/num_preemptions_max'].append(metric.value)
            self._metrics_snapshots.append(snapshot)

            await asyncio.sleep(self._metrics_snapshot_interval)

    def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get aggregated metrics for the time interval since last call.

        Args:
            metric_names: Optional list of specific metric names to filter

        Returns:
            Dictionary of metric names to aggregated values
        """
        if not self._metrics_snapshots:
            return {}
        metrics_snapshots = list_of_dict_to_dict_of_list(self._metrics_snapshots)
        self._metrics_snapshots.clear()
        return reduce_metrics(metrics_snapshots)

def gather_unpadded_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    gathered_input_ids = [ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)]
    return gathered_input_ids


def gather_outputs_to_pad_tensor(request_outputs: List["RequestOutput"], pad_token_id, device=None) -> torch.Tensor:
    if device is None:
        device = current_platform.device_type
    token_ids_list_of_lists = [
        torch.tensor(completion_output.token_ids, device=device)
        for request_output in request_outputs
        for completion_output in request_output.outputs
    ]
    output_tensor = pad_sequence(token_ids_list_of_lists, batch_first=True, padding_value=pad_token_id)
    return output_tensor


def create_sampling_params_for_vllm(gen_kwargs):
    # TODO vllm 0.10.2 support partial rollout, and do not need to set RequestOutputKind to CUMULATIVE
    output_kind = gen_kwargs.get("output_kind", RequestOutputKind.FINAL_ONLY)
    if output_kind != RequestOutputKind.FINAL_ONLY:
        assert gen_kwargs["num_return_sequences"] == 1, (
            "fetch_output only supports num_return_sequences=1 or output_kind=FINAL"
        )
    return SamplingParams(
        max_tokens=gen_kwargs["max_new_tokens"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        stop=gen_kwargs["stop_strings"],
        logprobs=gen_kwargs.get("logprobs", 0),
        output_kind=output_kind,
        include_stop_str_in_output=gen_kwargs.get("include_stop_str_in_output", True),
    )
