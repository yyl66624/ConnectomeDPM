"""Device / dtype resolution. Kept explicit so experiments can record exactly what ran."""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").lower()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is False")
    return torch.device(spec)


def resolve_dtype(spec: str = "auto", device: torch.device | None = None) -> torch.dtype:
    spec = (spec or "auto").lower()
    if spec == "auto":
        if device is not None and device.type == "cuda":
            # BF16 is the default for frozen-backbone work: no loss-scaling, wide range.
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }[spec]


def describe_hardware() -> dict:
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
        info["arch_list"] = torch.cuda.get_arch_list()
    return info

