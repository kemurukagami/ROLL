from vllm.v1.engine.async_llm import AsyncLLM


class CustomAsyncLLM(AsyncLLM):
    async def custom_init_worker(self):
        await self.engine_core.collective_rpc_async(method="custom_init_worker")

    async def load_states(self):
        await self.engine_core.collective_rpc_async(method="load_states")

    async def offload_states(self, level):
        await self.reset_prefix_cache()
        await self.engine_core.collective_rpc_async(method="offload_states", args=(level,))

    async def setup_collective_group(self, *args, **kwargs):
        await self.engine_core.collective_rpc_async(method="setup_collective_group", args=args, kwargs=kwargs)

    async def broadcast_parameter(self, *args, **kwargs):
        await self.engine_core.collective_rpc_async(method="broadcast_parameter", args=args, kwargs=kwargs)

    async def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        await self.engine_core.collective_rpc_async(method="update_parameter_in_bucket", args=(serialized_named_tensors, is_lora))

    async def destroy_collective_group(self, group_name: str):
        await self.engine_core.collective_rpc_async(method="destroy_collective_group", args=(group_name,))

    async def add_lora(self, *args, **kwargs):
        await self.engine_core.collective_rpc_async(method="custom_add_lora", args=args, kwargs=kwargs)

    async def get_lora_id(self, *args, **kwargs):
        # Keep adapter-id lookup on the same collective RPC path as add_lora.
        return await self.engine_core.collective_rpc_async(method="custom_get_lora_id", args=args, kwargs=kwargs)

    async def list_loras(self) -> list[int]:
        # Expose loaded adapter ids so strategy can verify routing readiness.
        return await self.engine_core.collective_rpc_async(method="custom_list_loras")

    async def process_weights_after_loading(self):
        await self.engine_core.collective_rpc_async(method="process_weights_after_loading")
