"""Train one block on one verifiable error family (DEVELOPMENT.md section 6).

The base model stays frozen; only the block's `A` and `B` receive gradients. Loss is computed
on answer tokens only, so the block is never rewarded for reproducing the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..backbone.loader import Backbone, build_prompt
from ..data.schemas import Sample
from ..utils.seed import set_seed
from .inject import BlockInjector, resolve_layer_suffix
from .parameter_block import ParameterBlock, make_block


@dataclass
class BlockTrainConfig:
    error_family: str
    layer_id: int
    module_name: str
    rank: int = 8
    scale: float = 1.0
    lr: float = 2e-3
    epochs: int = 3
    batch_size: int = 8
    max_length: int = 256
    seed: int = 0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    extra: dict = field(default_factory=dict)


def _encode_batch(backbone: Backbone, samples: Sequence[Sample], max_length: int):
    tokenizer = backbone.tokenizer
    input_ids, labels, attention = [], [], []
    for sample in samples:
        prompt_text = build_prompt(tokenizer, sample.prompt, backbone.use_chat_template)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(sample.answer, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [tokenizer.eos_token_id]
        ids = (prompt_ids + answer_ids)[:max_length]
        lab = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
        input_ids.append(ids)
        labels.append(lab)
        attention.append([1] * len(ids))

    pad_id = tokenizer.pad_token_id
    width = max(len(x) for x in input_ids)
    padded_ids = [x + [pad_id] * (width - len(x)) for x in input_ids]
    padded_lab = [x + [-100] * (width - len(x)) for x in labels]
    padded_att = [x + [0] * (width - len(x)) for x in attention]
    return (
        torch.tensor(padded_ids, dtype=torch.long),
        torch.tensor(padded_lab, dtype=torch.long),
        torch.tensor(padded_att, dtype=torch.long),
    )


def train_block(
    backbone: Backbone,
    samples: Sequence[Sample],
    config: BlockTrainConfig,
    *,
    block_id: str | None = None,
    log=None,
) -> tuple[ParameterBlock, dict]:
    """Train a single block and return (block, history)."""
    if not samples:
        raise ValueError(f"no training samples for family {config.error_family!r}")

    set_seed(config.seed)
    tokenizer = backbone.tokenizer
    path = resolve_layer_suffix(backbone.model, config.layer_id, config.module_name)
    module = dict(backbone.model.named_modules())[path]

    block = make_block(
        block_id=block_id or f"{config.error_family}_l{config.layer_id}_{config.module_name}",
        error_family=config.error_family,
        layer_id=config.layer_id,
        module_name=config.module_name,
        in_features=int(module.in_features),
        out_features=int(module.out_features),
        rank=config.rank,
        scale=config.scale,
        base_model=backbone.model_id,
        base_model_revision=backbone.revision,
        training_seed=config.seed,
    ).to(device=backbone.device, dtype=backbone.dtype)

    for param in block.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for p in block.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay,
    )

    injector = BlockInjector(backbone.model)
    injector.attach([block])
    history: list[dict] = []
    try:
        for epoch in range(config.epochs):
            order = torch.randperm(len(samples)).tolist()
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(samples), config.batch_size):
                batch_samples = [samples[i] for i in order[start:start + config.batch_size]]
                ids, labels, att = _encode_batch(backbone, batch_samples, config.max_length)
                ids = ids.to(backbone.device)
                labels = labels.to(backbone.device)
                att = att.to(backbone.device)
                out = backbone.model(input_ids=ids, attention_mask=att, labels=labels)
                loss = out.loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in block.parameters() if p.requires_grad], config.grad_clip
                    )
                optimizer.step()
                epoch_loss += float(loss)
                n_batches += 1
            row = {"epoch": epoch, "loss": epoch_loss / max(n_batches, 1)}
            history.append(row)
            if log:
                log.info("[%s] epoch %d loss %.4f", block.block_id, epoch, row["loss"])
    finally:
        injector.detach()

    block.eval()
    return block, {
        "block_id": block.block_id,
        "error_family": config.error_family,
        "n_samples": len(samples),
        "history": history,
        "module_path": path,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "rank": config.rank,
        "scale": config.scale,
        "seed": config.seed,
        "final_loss": history[-1]["loss"] if history else None,
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
    }
