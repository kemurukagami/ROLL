import re
from typing import Callable

import torch.nn as nn
from megatron.core.extensions.transformer_engine import TEGroupedLinear, TELayerNormColumnParallelLinear, TELinear
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.moe.router import TopKRouter
from transformers import PreTrainedModel


def _type_tuple(*candidates):
    return tuple(candidate for candidate in candidates if isinstance(candidate, type))


_LINEAR_TYPES = _type_tuple(
    TELinear,
    TEGroupedLinear,
    TELayerNormColumnParallelLinear,
    ColumnParallelLinear,
    RowParallelLinear,
    nn.Linear,
)


def _has_materialized_weight(module) -> bool:
    weight = getattr(module, "weight", None)
    if weight is not None:
        return True
    num_gemms = int(getattr(module, "num_gemms", 0) or 0)
    for i in range(num_gemms):
        if getattr(module, f"weight{i}", None) is not None:
            return True
    return False


def set_linear_is_expert(model):
    for n, module in model.named_modules():
        if (
            ".experts." in n
            and isinstance(module, _LINEAR_TYPES)
        ):
            module.is_expert = True


def find_layers(model: "PreTrainedModel", cond: Callable):
    inner_nodes = set()
    for name, module in model.named_modules():
        name = re.sub(r"\d+\.", "{}.", name)
        if not cond(module):
            inner_nodes.add(name)
    target_module_names = set()
    for name, module in model.named_modules():
        if cond(module):
            module_name_list = name.split(".")
            module_name = module_name_list.pop()
            for inner_node in inner_nodes:
                processed_module_name = re.sub(r"\d+\.", "{}.", module_name)
                while module_name_list and inner_node.endswith(processed_module_name):
                    module_name = f"{module_name_list.pop()}.{module_name}"
            target_module_names.add(module_name)
    return list(target_module_names)


def find_all_linear_modules(model):
    return find_layers(model, lambda module: isinstance(module, _LINEAR_TYPES) and _has_materialized_weight(module))


def find_all_embedding_modules(model):
    return find_layers(model, lambda module: isinstance(module, LanguageModelEmbedding))


def find_all_router_modules(model):
    return find_layers(model, lambda module: isinstance(module, TopKRouter))
