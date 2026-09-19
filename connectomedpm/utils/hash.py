"""Hashing helpers.

Two ideas matter here:

* `file_sha256` produces the provenance hashes required by the run manifest.
* `stable_hash` produces *cache keys*. Any change to the inputs of an experiment must
  produce a different cache key, otherwise a silent stale-cache bug is possible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

_CHUNK = 1 << 20


def file_sha256(path: str | Path) -> str:
    """Streaming SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def stable_hash(obj: Any, length: int = 16) -> str:
    """Stable short hash of any JSON-serialisable object."""
    return text_sha256(canonical_json(obj))[:length]


def combine_hashes(parts: Iterable[str], length: int = 16) -> str:
    return stable_hash(sorted(str(p) for p in parts), length=length)


def cache_key(**parts: Any) -> str:
    """Build a cache key from named components.

    Callers must pass *every* component that can change the cached artefact, e.g. for the
    candidate outcome cache: base model, revision, block manifest, prompt template,
    decoding config, scorer version, dataset split hash.
    """
    missing = [k for k, v in parts.items() if v is None or v == ""]
    if missing:
        raise ValueError(f"cache_key called with empty components: {missing}")
    return stable_hash(parts, length=32)

