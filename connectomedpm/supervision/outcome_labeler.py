"""FIX / BREAK / UNCHANGED labels (DEVELOPMENT.md section 7).

    base wrong  -> block correct : FIX
    base correct-> block wrong   : BREAK
    otherwise                     : UNCHANGED
"""

from __future__ import annotations

FIX = "FIX"
BREAK = "BREAK"
UNCHANGED = "UNCHANGED"
LABELS: tuple[str, ...] = (FIX, BREAK, UNCHANGED)
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}


def label_outcome(base_correct: bool, block_correct: bool) -> str:
    if not base_correct and block_correct:
        return FIX
    if base_correct and not block_correct:
        return BREAK
    return UNCHANGED


def label_index(label: str) -> int:
    return LABEL_TO_INDEX[label]


def label_from_index(index: int) -> str:
    return INDEX_TO_LABEL[int(index)]

