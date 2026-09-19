"""Block bank storage + manifest (DEVELOPMENT.md section 6)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ..utils.hash import file_sha256, stable_hash
from ..utils.io import ensure_dir, write_json
from .parameter_block import BlockConfig, ParameterBlock


class BlockBank:
    """An ordered collection of blocks keyed by `block_id`."""

    def __init__(self, blocks: list[ParameterBlock] | None = None):
        self.blocks: dict[str, ParameterBlock] = {}
        for block in blocks or []:
            self.add(block)

    def add(self, block: ParameterBlock) -> None:
        if block.block_id in self.blocks:
            raise ValueError(f"duplicate block_id {block.block_id!r}")
        self.blocks[block.block_id] = block

    def get(self, block_id: str) -> ParameterBlock:
        return self.blocks[block_id]

    def ids(self) -> list[str]:
        return sorted(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks.values())

    def save(self, directory: str | Path) -> Path:
        directory = ensure_dir(directory)
        for block_id, block in self.blocks.items():
            torch.save(
                {"config": asdict(block.config), "state_dict": block.state_dict()},
                directory / f"{block_id}.pt",
            )
        return directory

    @classmethod
    def load(cls, directory: str | Path, device: str | torch.device = "cpu") -> "BlockBank":
        directory = Path(directory)
        bank = cls()
        for path in sorted(directory.glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            config = payload["config"]
            if not isinstance(config, BlockConfig):
                config = BlockConfig(**config)
            block = ParameterBlock(config)
            block.load_state_dict(payload["state_dict"])
            block = block.to(device)
            bank.add(block)
        return bank


def block_bank_manifest(directory: str | Path) -> dict:
    """Manifest required by DEVELOPMENT.md section 6: identity + hashes for every block."""
    directory = Path(directory)
    entries = []
    for path in sorted(directory.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload.get("config", {})
        if not isinstance(config, dict):
            config = asdict(config)
        shapes = {k: list(v.shape) for k, v in payload["state_dict"].items()}
        entries.append(
            {
                "block_id": config.get("block_id", path.stem),
                "file": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "config": config,
                "tensor_shapes": shapes,
            }
        )
    manifest = {
        "n_blocks": len(entries),
        "blocks": entries,
        "manifest_hash": stable_hash(
            [{k: e[k] for k in ("block_id", "sha256", "config")} for e in entries], length=32
        ),
    }
    write_json(directory / "block_bank_manifest.json", manifest)
    return manifest
