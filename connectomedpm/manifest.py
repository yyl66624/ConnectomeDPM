"""Run manifests.

DEVELOPMENT.md section 21 requires every run to record enough provenance that the result can
be attributed to an exact code + data + graph + cache combination. If any of these fields is
missing, the result must not be used as main evidence.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils.device import describe_hardware
from .utils.hash import file_sha256, stable_hash
from .utils.io import ensure_dir, write_json, write_yaml


def git_commit(cwd: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd), capture_output=True, text=True, timeout=10, check=False,
        )
        commit = out.stdout.strip()
        if out.returncode == 0 and commit:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(cwd), capture_output=True, text=True, timeout=10, check=False,
            ).stdout.strip()
            return commit + ("-dirty" if dirty else "")
    except Exception:
        pass
    return "unknown"


def _safe_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return file_sha256(p)


def build_manifest(
    run_id: str,
    config: dict,
    *,
    model: dict | None = None,
    dataset: dict | None = None,
    block_bank: dict | None = None,
    graph: dict | None = None,
    router: dict | None = None,
    candidate_cache: dict | None = None,
    seed: int | None = None,
    extra: dict | None = None,
    repo_root: str | Path = ".",
) -> dict:
    """Assemble the manifest described in DEVELOPMENT.md section 21."""
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "git_commit": git_commit(repo_root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": stable_hash(config, length=32),
        "config_path": config.get("_config_path"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "hardware": describe_hardware(),
        "model": model or {},
        "dataset": dataset or {},
        "block_bank": block_bank or {},
        "graph": graph or {},
        "router": router or {},
        "candidate_cache": candidate_cache or {},
        "seed": seed,
        "training_seed": (extra or {}).get("training_seed", seed),
        "graph_seed": (extra or {}).get("graph_seed"),
        "sampling_seed": (extra or {}).get("sampling_seed", seed),
    }
    if extra:
        manifest["extra"] = {
            k: v for k, v in extra.items()
            if k not in {"training_seed", "graph_seed", "sampling_seed"}
        }
    return manifest


def write_manifest(out_dir: str | Path, manifest: dict, name: str = "manifest") -> Path:
    out_dir = ensure_dir(out_dir)
    write_json(out_dir / f"{name}.json", manifest)
    write_yaml(out_dir / f"{name}.yaml", manifest)
    return out_dir / f"{name}.yaml"


def describe_file(path: str | Path) -> dict:
    """Provenance block for a single data artefact."""
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": p.stat().st_size,
        "sha256": _safe_sha256(p),
    }

