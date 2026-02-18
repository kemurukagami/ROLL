from vllm.engine.async_llm_engine import AsyncLLMEngine

class CustomAsyncLLMEngine(AsyncLLMEngine):
    async def custom_init_worker(self):
        await self.engine.model_executor.collective_rpc(method="custom_init_worker")

    async def load_states(self):
        await self.engine.model_executor.collective_rpc(method="load_states")

    async def offload_states(self, level):
        await self.reset_prefix_cache()
        await self.engine.model_executor.collective_rpc(method="offload_states", args=(level,))

    async def setup_collective_group(self, *args, **kwargs):
        await self.engine.model_executor.collective_rpc(method="setup_collective_group", args=args, kwargs=kwargs)

    async def broadcast_parameter(self, *args, **kwargs):
        await self.engine.model_executor.collective_rpc(method="broadcast_parameter", args=args, kwargs=kwargs)

    async def update_parameter_in_bucket(self, *args, **kwargs):
        await self.engine.model_executor.collective_rpc(method="update_parameter_in_bucket", args=args, kwargs=kwargs)

    async def destroy_collective_group(self, group_name: str):
        await self.engine.model_executor.collective_rpc(method="destroy_collective_group", args=(group_name,), kwargs={})

    async def add_lora(self, *args, **kwargs):
        await self.engine.model_executor.collective_rpc(method="custom_add_lora", args=args, kwargs=kwargs)

    async def process_weights_after_loading(self):
        await self.engine.model_executor.collective_rpc(method="process_weights_after_loading")
