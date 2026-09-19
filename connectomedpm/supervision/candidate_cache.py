"""The candidate outcome cache (DEVELOPMENT.md section 7).

For every sample we execute `base` plus every block exactly once, offline, and store the
result. Every router in the project shares this one cache, which is what makes the
graph-vs-null comparison fair: the only thing that varies is the router's topology.

The cache key binds the base model, block manifest, prompt template, decoding config, scorer
version and dataset split hash. Any change produces a new key and therefore a new cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

from ..backbone.generation import generate_answers
from ..backbone.loader import Backbone, build_prompt
from ..blocks.block_bank import BlockBank
from ..blocks.inject import active_blocks
from ..data.schemas import CandidateRecord, Sample
from ..utils.hash import cache_key, stable_hash
from ..utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from .outcome_labeler import label_outcome
from .scorer import SCORER_VERSION, score_answer

CACHE_VERSION = "candidate-cache-v1"


@dataclass
class DecodingConfig:
    max_new_tokens: int = 24
    batch_size: int = 48
    do_sample: bool = False
    num_beams: int = 1

    def to_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
        }


def compute_cache_key(
    backbone: Backbone,
    block_manifest_hash: str,
    decoding: DecodingConfig,
    split_hash: str,
    prompt_template: str,
) -> str:
    return cache_key(
        cache_version=CACHE_VERSION,
        base_model=backbone.model_id,
        base_model_revision=backbone.revision,
        block_manifest=block_manifest_hash,
        prompt_template=prompt_template,
        decoding=decoding.to_dict(),
        scorer=SCORER_VERSION,
        dataset_split=split_hash,
    )


def build_candidate_cache(
    backbone: Backbone,
    bank: BlockBank,
    samples: Sequence[Sample],
    *,
    block_manifest_hash: str,
    decoding: DecodingConfig | None = None,
    log=None,
) -> tuple[list[CandidateRecord], dict]:
    """Run base + every block over `samples` and return (records, cache metadata)."""
    decoding = decoding or DecodingConfig()
    block_ids = bank.ids()
    prompts = [s.prompt for s in samples]
    split_hash = stable_hash(sorted(s.sample_id for s in samples), length=32)
    key = compute_cache_key(
        backbone, block_manifest_hash, decoding, split_hash,
        prompt_template=build_prompt(backbone.tokenizer, "{prompt}", backbone.use_chat_template),
    )

    outputs: dict[str, list[str]] = {}
    outputs["base"] = generate_answers(
        backbone, prompts,
        max_new_tokens=decoding.max_new_tokens,
        batch_size=decoding.batch_size,
        do_sample=decoding.do_sample,
        num_beams=decoding.num_beams,
    )
    if log:
        log.info("base generation done (%d prompts)", len(prompts))

    for block_id in tqdm(block_ids, desc="candidate cache: blocks", disable=log is None):
        block = bank.get(block_id)
        with active_blocks(backbone.model, [block]):
            outputs[block_id] = generate_answers(
                backbone, prompts,
                max_new_tokens=decoding.max_new_tokens,
                batch_size=decoding.batch_size,
                do_sample=decoding.do_sample,
                num_beams=decoding.num_beams,
            )

    records: list[CandidateRecord] = []
    for i, sample in enumerate(samples):
        base_correct, base_score = score_answer(
            outputs["base"][i], sample.answer, sample.task_family
        )
        actions: dict[str, dict] = {
            "base": {"correct": base_correct, "score": base_score, "raw": outputs["base"][i]}
        }
        for block_id in block_ids:
            correct, score = score_answer(outputs[block_id][i], sample.answer, sample.task_family)
            actions[block_id] = {
                "correct": correct,
                "score": score,
                "outcome": label_outcome(base_correct, correct),
                "raw": outputs[block_id][i],
            }
        records.append(
            CandidateRecord(
                sample_id=sample.sample_id,
                split=sample.split,
                task_family=sample.task_family,
                template_family=sample.template_family,
                base_correct=base_correct,
                actions=actions,
                metadata={"prompt": sample.prompt, "answer": sample.answer},
            )
        )

    meta = {
        "cache_version": CACHE_VERSION,
        "cache_hash": key,
        "base_model": backbone.model_id,
        "base_model_revision": backbone.revision,
        "block_manifest_hash": block_manifest_hash,
        "block_ids": block_ids,
        "decoding": decoding.to_dict(),
        "scorer": SCORER_VERSION,
        "split_hash": split_hash,
        "n_samples": len(samples),
        "prompt_template": build_prompt(
            backbone.tokenizer, "{prompt}", backbone.use_chat_template
        ),
    }
    return records, meta


def save_cache(directory: str | Path, records: Sequence[CandidateRecord], meta: dict) -> Path:
    directory = ensure_dir(directory)
    write_jsonl(directory / "candidates.jsonl", (r.to_dict() for r in records))
    write_json(directory / "cache_meta.json", meta)
    return directory


def load_cache(directory: str | Path, expected_hash: str | None = None):
    """Load a cache, refusing to return it if the caller's expected key does not match."""
    from ..utils.io import read_json

    directory = Path(directory)
    meta = read_json(directory / "cache_meta.json")
    if expected_hash is not None and meta.get("cache_hash") != expected_hash:
        raise ValueError(
            f"stale candidate cache: stored key {meta.get('cache_hash')} != expected {expected_hash}"
        )
    records = [CandidateRecord.from_dict(row) for row in read_jsonl(directory / "candidates.jsonl")]
    return records, meta
