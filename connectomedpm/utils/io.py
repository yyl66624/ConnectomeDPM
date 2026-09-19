"""Filesystem helpers. All writes are atomic so a crashed run cannot leave half a file."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def write_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False, default=str))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_yaml(path: str | Path, obj: Any) -> Path:
    import yaml

    return atomic_write_text(path, yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))


def read_yaml(path: str | Path) -> Any:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

