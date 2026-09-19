"""Logging that always mirrors to a file so every run leaves a trace."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .io import ensure_dir


def get_logger(name: str = "connectomedpm", log_file: str | Path | None = None, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                                          datefmt="%H:%M:%S"))
        logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        already = any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file
                      for h in logger.handlers)
        if not already:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
            logger.addHandler(fh)
    return logger

