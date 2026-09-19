"""Load and freeze the base language model.

DEVELOPMENT.md section 5: the backbone is frozen, BF16 is preferred, generation settings are
fixed for the whole experiment, and the tokenizer/model revision is written into the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..utils.device import resolve_device, resolve_dtype


@dataclass
class Backbone:
    model: object
    tokenizer: object
    model_id: str
    revision: str
    dtype: torch.dtype
    device: torch.device
    use_chat_template: bool = True

    def describe(self) -> dict:
        return {
            "name": self.model_id,
            "revision": self.revision,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device": str(self.device),
            "use_chat_template": self.use_chat_template,
            "n_params": int(sum(p.numel() for p in self.model.parameters())),
            "n_layers": int(getattr(self.model.config, "num_hidden_layers", 0)),
            "hidden_size": int(getattr(self.model.config, "hidden_size", 0)),
        }


def load_backbone(
    model_path: str | Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
    use_chat_template: bool = True,
    revision: str = "",
) -> Backbone:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = resolve_device(device)
    torch_dtype = resolve_dtype(dtype, dev)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    )
    model.to(dev)
    model.eval()
    freeze(model)

    return Backbone(
        model=model,
        tokenizer=tokenizer,
        model_id=str(model_path),
        revision=revision or "local",
        dtype=torch_dtype,
        device=dev,
        use_chat_template=use_chat_template,
    )


def freeze(model) -> None:
    """Freeze every base parameter and disable dropout-style stochasticity."""
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def trainable_parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_prompt(tokenizer, prompt: str, use_chat_template: bool = True) -> str:
    """Apply the fixed prompt template. Recorded in the cache key, so it must not drift."""
    if not use_chat_template:
        return prompt
    if getattr(tokenizer, "chat_template", None) is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

