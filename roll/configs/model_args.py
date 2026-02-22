from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch

from roll.utils.lora_routing import normalize_domain


# Inspired by: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/hparams/finetuning_args.py
@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    adapter_name: str = field(
        default="default",
        metadata={"help": "The name of the adapter to be injected."},
    )
    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    autocast_adapter_dtype: bool = field(
        default=True,
        metadata={
            "help": "Whether to autocast the adapter dtype. Defaults to `True`. Right now, "
            "this will only cast adapter weights using float16 or bfloat16 to float32, "
            "as this is typically required for stable training, and only affect select PEFT tuners."
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )


@dataclass
class ModelArguments(LoraArguments):
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapters: Optional[Dict[str, LoraArguments]] = field(
        default=None,
        metadata={"help": "List of LoRA adapter configurations."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapter weight or identifier from huggingface.co/models."},
    )
    attn_implementation: Optional[Literal["sdpa", "fa2", "auto"]] = field(
        default=None,
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    disable_gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Gradient checkpointing implementation toggle for torch.utils.checkpoint.\n"
                "- None (default): auto (use reentrant=True for MoE models; otherwise False)\n"
            )
        },
    )
    device_map: Optional[str] = field(
        default="balanced", metadata={"help": "transformer's from_pretrained device map"}
    )
    dtype: Optional[Literal["fp32", "bf16", "fp16"]] = field(
        default="bf16", metadata={"help": "Set model dtype as fp32, bf16, or fp16, otherwise use config's torch_dtype"}
    )
    model_type: Optional[
        Literal["auto_sequence_classification", "auto_token_classification", "trl", "diffusion_module"]
    ] = field(
        default=None,
        metadata={"help": "reward model type."},
    )
    num_labels: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of labels for AutoModelForTokenClassification and "
            "AutoModelForSequenceClassification."
        },
    )
    model_config_kwargs: dict = field(
        default_factory=lambda: {},
        metadata={"help": "Additional keyword arguments to pass to the model config"},
    )
    freeze_module_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "Prefix of frozen modules for partial-parameter (freeze) fine-tuning. Use commas to separate multiple modules."
        },
    )
    ulysses_size: Optional[int] = field(
        default=1,
        metadata={"help": "The group size for Ulysses attention."},
    )
    # True when adapters were auto-derived from legacy top-level lora_rank/lora_target fields.
    _derived_adapters_from_legacy_lora_fields: bool = field(default=False, repr=False)
    adapter_name_map: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        # Keep legacy top-level LoRA fields functional by canonicalizing to adapters.
        if self.adapters is None and self.lora_rank is not None and self.lora_target is not None:
            self.adapters = {
                "default": LoraArguments(
                    adapter_name="default",
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    lora_target=self.lora_target,
                )
            }
            # Mark that this config used legacy single-LoRA fields and was normalized to adapters.
            self._derived_adapters_from_legacy_lora_fields = True

        self.lora_alpha = self.lora_alpha or self.lora_rank * 2
        if self.lora_target is not None and not any(c in self.lora_target for c in ["*", "$", "|", "("]):
            # split when lora_target is not regex expression
            self.lora_target = split_arg(self.lora_target)
        self.freeze_module_prefix: Optional[List[str]] = split_arg(self.freeze_module_prefix)
        self.additional_target: Optional[List[str]] = split_arg(self.additional_target)
        if self.adapters is not None:
            normalized_adapters: dict[str, LoraArguments] = {}
            raw_to_final: dict[str, str] = {}
            seen_bases: set[str] = set()
            for raw_adapter_name, adapter_config in self.adapters.items():
                base = normalize_domain(raw_adapter_name)
                # Fail fast on normalization collisions to keep tag->adapter mapping deterministic.
                if base in seen_bases:
                    raise RuntimeError(
                        f"Adapter name collision: '{raw_adapter_name}' normalizes to '{base}' "
                        "which conflicts with an earlier adapter. Use distinct adapter names."
                    )
                seen_bases.add(base)
                adapter_config.adapter_name = base
                if adapter_config.lora_alpha is None or adapter_config.lora_alpha <= 0:
                    adapter_config.lora_alpha = adapter_config.lora_rank * 2
                if adapter_config.lora_target is not None and not any(
                    c in adapter_config.lora_target for c in ["*", "$", "|", "("]
                ):
                    adapter_config.lora_target = split_arg(adapter_config.lora_target)
                adapter_config.additional_target = split_arg(adapter_config.additional_target)
                normalized_adapters[base] = adapter_config
                raw_to_final[str(raw_adapter_name)] = base
            self.adapters = normalized_adapters
            self.adapter_name_map = raw_to_final

        dtype_mapping = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }

        self.compute_dtype = dtype_mapping[self.dtype]
        self.model_max_length = None

        if self.attn_implementation == "fa2":
            self.attn_implementation = "flash_attention_2"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
