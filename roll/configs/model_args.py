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

    #todo(tao) rename as lora_name systematically
    # Unique identifier for this adapter, used as routing key in multi-LoRA dispatch.
    # Names are normalized via normalize_domain() to lowercase slugs (e.g., "Math/v2" -> "math_v2").
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
    # Multi-LoRA support: maps normalized adapter names to their LoraArguments configs.
    # Single-LoRA configs using legacy top-level lora_rank/lora_target are auto-converted
    # to adapters={"default": LoraArguments(...)} in __post_init__.
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
    # Maps raw adapter names (as written in YAML) to their normalized slugs.
    # Used for reverse lookups when routing tags come from external sources that
    # may use the original non-normalized spelling.
    adapter_name_map: dict[str, str] = field(default_factory=dict, init=False)

    @property
    def _is_single_lora(self) -> bool:
        """True when using legacy top-level lora fields (no explicit adapters dict).

        Internal only: meaningful before __post_init__ canonicalizes single-LoRA
        into an adapters dict. After init, use is_multi_lora to distinguish.
        """
        return self.adapters is None and self.lora_rank is not None and self.lora_target is not None

    @property
    def is_multi_lora(self) -> bool:
        """True when the config carries multiple named LoRA adapters."""
        return self.adapters is not None and len(self.adapters) > 1

    @staticmethod
    def _split_arg(arg):
        """Split a comma-separated string into a list of stripped items."""
        if isinstance(arg, str):
            return [item.strip() for item in arg.split(",")]
        return arg

    def _normalize_adapters(self) -> None:
        """Normalize adapter names to lowercase slugs and apply per-adapter defaults."""
        if self.adapters is None:
            return

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
                adapter_config.lora_target = self._split_arg(adapter_config.lora_target)
            adapter_config.additional_target = self._split_arg(adapter_config.additional_target)
            normalized_adapters[base] = adapter_config
            raw_to_final[str(raw_adapter_name)] = base
        self.adapters = normalized_adapters
        self.adapter_name_map = raw_to_final

    def __post_init__(self):
        # --- LoRA mode dispatch ---
        # Multi-LoRA: adapters dict is set explicitly in config.
        # Only normalize the per-adapter configs; top-level lora_rank/lora_alpha are ignored.
        # Empty adapters dict ({}) is treated as config error — fail fast to catch typos.
        if self.adapters is not None:
            if len(self.adapters) == 0:
                raise ValueError("adapters dict is empty; remove it or add at least one adapter.")
            self._normalize_adapters()

        # Single-LoRA: top-level lora_rank + lora_target set, no adapters dict.
        # Canonicalize into a single-entry adapters dict for uniform downstream access.
        elif self._is_single_lora:
            self.lora_alpha = self.lora_alpha or self.lora_rank * 2
            self.adapters = {
                "default": LoraArguments(
                    adapter_name="default",
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    lora_target=self.lora_target,
                )
            }
            self._normalize_adapters()

        # No-LoRA: neither adapters nor lora_target set. Nothing to do.

        # --- Fields that apply regardless of LoRA mode ---
        if self.lora_target is not None and not any(c in self.lora_target for c in ["*", "$", "|", "("]):
            # split when lora_target is not regex expression
            self.lora_target = self._split_arg(self.lora_target)
        self.freeze_module_prefix: Optional[List[str]] = self._split_arg(self.freeze_module_prefix)
        self.additional_target: Optional[List[str]] = self._split_arg(self.additional_target)

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
