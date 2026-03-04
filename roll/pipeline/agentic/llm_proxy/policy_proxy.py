from typing import List, Dict, Any

import time

import ray

from roll.pipeline.agentic.llm_proxy import BaseLLMProxy, register_llm_proxy
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger


@register_llm_proxy("policy")
class PolicyProxy(BaseLLMProxy):
    """
    A proxy for policy model that invokes the policy model's engine (e.g. vllm/sglang) to perform generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger()

    def generate(self,
                 messages: List[Dict[str, str]],
                 lm_input: DataProto,
                 generation_config: Dict[str, Any]) -> DataProto:

        lm_input.meta_info["generation_config"] = generation_config
        lm_input.meta_info["pad_to_seq_len"] = False
        src_rank = lm_input.meta_info.get("src_rank")
        global_step = lm_input.meta_info.get("global_step")
        start_s = time.time()
        self.logger.info(
            f"[PolicyProxy] submit generate_one_request"
            f" src_rank={src_rank} global_step={global_step}"
        )
        lm_output: DataProto = ray.get(self.generate_scheduler.generate_one_request.remote(data=lm_input))
        elapsed_s = time.time() - start_s
        if elapsed_s >= 30.0:
            self.logger.warning(
                f"[PolicyProxy] generate_one_request slow"
                f" elapsed_s={elapsed_s:.3f}"
                f" src_rank={src_rank} global_step={global_step}"
            )
        else:
            self.logger.info(
                f"[PolicyProxy] generate_one_request done"
                f" elapsed_s={elapsed_s:.3f}"
                f" src_rank={src_rank} global_step={global_step}"
            )

        if lm_output is not None:
            lm_output.meta_info.pop("generation_config", None)

        return lm_output
