"""Inject blocks into a frozen HuggingFace model with forward hooks.

The base model is never modified. `active_blocks` is a context manager that attaches exactly
the requested blocks to their target modules and removes every hook on exit, so two
conditions can never leak into each other.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

import torch
import torch.nn as nn

from .parameter_block import ParameterBlock


def list_linear_modules(model: nn.Module, module_names: Iterable[str]) -> dict[str, nn.Linear]:
    """Return `{full_name: module}` for every Linear whose leaf name is in `module_names`."""
    wanted = set(module_names)
    found: dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in wanted:
            found[name] = module
    return found


def target_module_name(layer_id: int, module_name: str) -> str:
    """Canonical Qwen/Llama-style module path: model.layers.<i>.<suffix>.<module>."""
    return f"model.layers.{layer_id}.{module_name}"


def resolve_layer_suffix(model: nn.Module, layer_id: int, module_name: str) -> str:
    """Find the actual module path for (layer, leaf-name), tolerating wrapper differences."""
    candidates = [name for name, mod in model.named_modules()
                  if isinstance(mod, nn.Linear)
                  and name.split(".")[-1] == module_name
                  and f".{layer_id}." in name]
    if not candidates:
        raise KeyError(f"no Linear named {module_name!r} found in layer {layer_id}")
    return sorted(candidates)[0]


class BlockInjector:
    """Reusable injector: build target map once, activate/deactivate blocks cheaply."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: list = []
        self._modules: dict[str, nn.Module] = {}

    def resolve(self, blocks: Iterable[ParameterBlock]) -> dict[str, str]:
        """Map each block_id to the real module path it should attach to."""
        mapping: dict[str, str] = {}
        for block in blocks:
            path = resolve_layer_suffix(self.model, block.layer_id, block.module_name)
            mapping[block.block_id] = path
            self._modules[path] = dict(self.model.named_modules())[path]
        return mapping

    def module_shapes(self, blocks: Iterable[ParameterBlock]) -> dict[str, tuple[int, int]]:
        """(in_features, out_features) for each block's target module, straight from the model."""
        named = dict(self.model.named_modules())
        out: dict[str, tuple[int, int]] = {}
        for block in blocks:
            path = resolve_layer_suffix(self.model, block.layer_id, block.module_name)
            mod = named[path]
            out[block.block_id] = (int(mod.in_features), int(mod.out_features))
        return out

    def _hook_for(self, block: ParameterBlock):
        def hook(module, inputs, output):
            if isinstance(inputs, tuple) and inputs:
                x = inputs[0]
            else:
                x = output
            # The base model usually runs in bf16 while blocks are stored in fp32. Compute the
            # correction in the block's own dtype (better conditioned) and cast the result back
            # to whatever dtype the frozen model produced.
            delta = block(x.to(block.A.dtype)).to(output.dtype)
            return output + delta

        return hook

    def attach(self, blocks: Iterable[ParameterBlock]) -> None:
        named = dict(self.model.named_modules())
        for block in blocks:
            path = resolve_layer_suffix(self.model, block.layer_id, block.module_name)
            module = named[path]
            self._handles.append(module.register_forward_hook(self._hook_for(block)))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@contextmanager
def active_blocks(model: nn.Module, blocks: Iterable[ParameterBlock]) -> Iterator[None]:
    """Temporarily attach `blocks` to `model`; always detaches, even on exception."""
    blocks = list(blocks)
    injector = BlockInjector(model)
    injector.attach(blocks)
    try:
        yield
    finally:
        injector.detach()


def assert_block_matches_module(block: ParameterBlock, module: nn.Linear) -> None:
    if int(module.in_features) != int(block.in_features):
        raise ValueError(
            f"block {block.block_id}: in_features {block.in_features} != module {module.in_features}"
        )
    if int(module.out_features) != int(block.out_features):
        raise ValueError(
            f"block {block.block_id}: out_features {block.out_features} != module {module.out_features}"
        )
