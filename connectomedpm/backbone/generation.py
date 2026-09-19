"""Deterministic, batched generation over a frozen backbone."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .loader import Backbone, build_prompt


@torch.no_grad()
def generate_answers(
    backbone: Backbone,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 24,
    batch_size: int = 32,
    do_sample: bool = False,
    num_beams: int = 1,
) -> list[str]:
    """Generate one answer per prompt. Decoding config is fixed across every condition."""
    model, tokenizer = backbone.model, backbone.tokenizer
    texts = [build_prompt(tokenizer, p, backbone.use_chat_template) for p in prompts]
    outputs: list[str] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(backbone.device) for k, v in enc.items()}
        generated = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=None if not do_sample else 0.7,
            top_p=None if not do_sample else 0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(text.strip() for text in decoded)
    return outputs


@torch.no_grad()
def generate_with_hooks(
    backbone: Backbone,
    prompts: Sequence[str],
    hooks,
    *,
    max_new_tokens: int = 24,
    batch_size: int = 32,
) -> list[str]:
    """Generate with a context manager (`hooks`) wrapping the whole batched generation."""
    with hooks:
        return generate_answers(
            backbone, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size
        )

