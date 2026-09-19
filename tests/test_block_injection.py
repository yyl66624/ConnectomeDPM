import numpy as np
import pytest

torch = pytest.importorskip("torch")

from connectomedpm.blocks.inject import BlockInjector, assert_block_matches_module  # noqa: E402
from connectomedpm.blocks.parameter_block import make_block  # noqa: E402


class _Tiny(torch.nn.Module):
    """Mimics the Qwen/Llama module path so injection can be tested without a real model."""

    def __init__(self, hidden=8, layers=3):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for _ in range(layers):
            layer = torch.nn.Module()
            layer.down_proj = torch.nn.Linear(hidden, hidden, bias=False)
            layer.o_proj = torch.nn.Linear(hidden, hidden, bias=False)
            self.model.layers.append(layer)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer.o_proj(layer.down_proj(x))
        return x


def test_fresh_block_is_a_no_op():
    net = _Tiny()
    x = torch.randn(2, 8)
    before = net(x).detach().clone()
    block = make_block(block_id="b", error_family="toy", layer_id=1, module_name="down_proj",
                       in_features=8, out_features=8)
    injector = BlockInjector(net)
    injector.attach([block])
    try:
        after = net(x).detach()
    finally:
        injector.detach()
    assert torch.allclose(before, after, atol=1e-6)


def test_block_changes_output_once_trained():
    net = _Tiny()
    x = torch.randn(2, 8)
    before = net(x).detach().clone()
    block = make_block(block_id="b", error_family="toy", layer_id=1, module_name="down_proj",
                       in_features=8, out_features=8)
    with torch.no_grad():
        block.B.normal_(0, 0.5)
    injector = BlockInjector(net)
    injector.attach([block])
    try:
        after = net(x).detach()
    finally:
        injector.detach()
    assert not torch.allclose(before, after, atol=1e-6)


def test_hooks_are_removed_after_detach():
    net = _Tiny()
    x = torch.randn(1, 8)
    block = make_block(block_id="b", error_family="toy", layer_id=0, module_name="o_proj",
                       in_features=8, out_features=8)
    with torch.no_grad():
        block.B.normal_(0, 1.0)
    injector = BlockInjector(net)
    injector.attach([block])
    injector.detach()
    baseline = _Tiny()
    baseline.load_state_dict(net.state_dict())
    assert torch.allclose(net(x), baseline(x), atol=1e-6)


def test_dimension_mismatch_is_caught():
    net = _Tiny()
    block = make_block(block_id="b", error_family="toy", layer_id=0, module_name="o_proj",
                       in_features=99, out_features=8)
    module = dict(net.named_modules())["model.layers.0.o_proj"]
    with pytest.raises(ValueError):
        assert_block_matches_module(block, module)
