#!/usr/bin/env python3
"""Phase A / H1-pilot pipeline runner.

Runs the engineering closed loop of DEVELOPMENT.md (R00-R08) end to end:

    data      -> task suite, dedup audit, frozen splits        (R01)
    graph     -> MaleCNS v1.0 central-brain K=32 topology      (R02)
    nulls     -> matched null topologies                       (R03)
    blocks    -> 8-block bank + manifest                       (R04)
    cache     -> base + every block over every split           (R05/R06)
    headroom  -> oracle gate                                   (R05)
    router    -> one router per topology, then H1 comparison   (R07/R08)

Every stage writes its artefacts and a manifest, and is skipped when its output already
exists unless `--force` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from connectomedpm.audit.leakage import audit_splits  # noqa: E402
from connectomedpm.audit.graph_usage import graph_sensitivity, no_neighbor_counterfactual  # noqa: E402
from connectomedpm.audit.routing_trace import build_trace, trace_summary  # noqa: E402
from connectomedpm.backbone.feature_extractor import extract_features, feature_hash  # noqa: E402
from connectomedpm.backbone.loader import load_backbone  # noqa: E402
from connectomedpm.blocks.block_bank import BlockBank, block_bank_manifest  # noqa: E402
from connectomedpm.blocks.trainer import BlockTrainConfig, train_block  # noqa: E402
from connectomedpm.calibration.threshold import learn_threshold  # noqa: E402
from connectomedpm.data.schemas import CandidateRecord, Sample  # noqa: E402
from connectomedpm.data.splits import ALL_SPLITS, build_splits, split_summary  # noqa: E402
from connectomedpm.data.task_registry import build_task_suite  # noqa: E402
from connectomedpm.evaluation.bootstrap import clustered_bootstrap, paired_clustered_bootstrap  # noqa: E402
from connectomedpm.evaluation.deployment import evaluate_deployment  # noqa: E402
from connectomedpm.evaluation.fixed_coverage import evaluate_fixed_coverage  # noqa: E402
from connectomedpm.evaluation.metrics import compute_metrics, metrics_by_family, oracle_metrics, resolve_decisions  # noqa: E402
from connectomedpm.graph_prior.artifacts import load_graph, save_graph  # noqa: E402
from connectomedpm.graph_prior.build import build_null_topologies, build_topology, topology_hash  # noqa: E402
from connectomedpm.graph_prior.malecns_loader import read_annotations, read_edges, release_provenance  # noqa: E402
from connectomedpm.manifest import build_manifest, describe_file, write_manifest  # noqa: E402
from connectomedpm.router.dataset import build_tensors  # noqa: E402
from connectomedpm.router.inference import BASE_ACTION, route, score_blocks  # noqa: E402
from connectomedpm.router.mlp_baseline import MLPRouter  # noqa: E402
from connectomedpm.router.model import TopologyRouter  # noqa: E402
from connectomedpm.router.train import TrainConfig, fit_model  # noqa: E402
from connectomedpm.supervision.candidate_cache import (  # noqa: E402
    DecodingConfig, build_candidate_cache, compute_cache_key, load_cache, save_cache,
)
from connectomedpm.utils.config import load_config  # noqa: E402
from connectomedpm.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from connectomedpm.utils.logging import get_logger  # noqa: E402
from connectomedpm.utils.seed import set_seed  # noqa: E402


# ---------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------

class Paths:
    def __init__(self, cfg: dict):
        p = cfg["paths"]
        self.data = ensure_dir(REPO_ROOT / p["data_dir"])
        self.graphs = ensure_dir(REPO_ROOT / p["graph_dir"])
        self.blocks = ensure_dir(REPO_ROOT / p["block_dir"])
        self.caches = ensure_dir(REPO_ROOT / p["cache_dir"])
        self.results = ensure_dir(REPO_ROOT / p["results_dir"])
        self.reports = ensure_dir(REPO_ROOT / p["reports_dir"])
        self.splits = ensure_dir(self.data / "splits")
        self.manifests = ensure_dir(self.data / "manifests")


# ---------------------------------------------------------------------------------------
# Stage: data (R01)
# ---------------------------------------------------------------------------------------

def stage_data(cfg: dict, paths: Paths, log) -> dict:
    summary_path = paths.manifests / "split_summary.json"
    if summary_path.exists() and not cfg["_force"]:
        log.info("[data] reuse %s", summary_path)
        return read_json(summary_path)

    seed = int(cfg["seed"])
    set_seed(seed)
    samples = build_task_suite(
        seed=seed,
        per_template=int(cfg["data"]["per_template"]),
        family_overrides=cfg["data"].get("family_overrides") or {},
    )
    log.info("[data] generated %d samples", len(samples))

    splits = build_splits(samples, seed=seed)
    summary = split_summary(splits)

    for name, rows in splits.items():
        write_jsonl(paths.splits / f"{name}.jsonl", (r.to_dict() for r in rows))

    audit = audit_splits(splits, near_duplicate_threshold=float(cfg["data"]["dedup_near_threshold"]))
    summary["leakage_audit"] = {
        "ok": audit["ok"],
        "violations": audit["violations"],
        "template_leaks": len(audit["template_leaks"]),
        "exact_duplicate_groups": audit["exact_duplicate_groups"],
        "near_duplicate_pairs": len(audit["prompt_leaks"]),
    }
    write_json(summary_path, summary)
    log.info("[data] splits: %s", {k: v["n_samples"] for k, v in summary["splits"].items()})
    log.info("[data] leakage audit ok=%s violations=%s", audit["ok"], audit["violations"])
    return summary


# ---------------------------------------------------------------------------------------
# Stage: MaleCNS graph (R02)
# ---------------------------------------------------------------------------------------

def stage_graph(cfg: dict, paths: Paths, log) -> dict:
    release = cfg["_graph_cfg"]["release"]
    build = cfg["_graph_cfg"]["build"]
    local_dir = Path(release["local_dir"])
    weights_path = local_dir / release["files"]["weights"]
    annotations_path = local_dir / release["files"]["annotations"]

    for path in (weights_path, annotations_path):
        if not path.exists():
            raise FileNotFoundError(f"missing MaleCNS release file: {path}")

    out_dir = paths.graphs / "malecns_k32"
    if (out_dir / "graph.npz").exists() and not cfg["_force"]:
        log.info("[graph] reuse %s", out_dir)
        return {"graph_dir": str(out_dir)}

    provenance = release_provenance(
        {"weights": weights_path, "annotations": annotations_path},
        version=release["version"],
    )
    t0 = time.time()
    edges = read_edges(weights_path)
    annotations = read_annotations(annotations_path)
    log.info("[graph] loaded %d edges and %d annotations (%.1fs)",
             len(edges), len(annotations), time.time() - t0)

    real = build_topology(
        edges, annotations,
        graph_id="malecns_k32",
        k=int(build["k"]),
        region_column=build["region_column"],
        region_prefix=build["region_prefix"],
        status_column=build["status_column"],
        excluded_status=tuple(build["excluded_status"]),
        community_column=build["community_column"],
        module_column=build["module_column"],
        min_community_size=int(build["min_community_size"]),
        transform=build["edge_transform"],
        steps=int(build["steps"]),
        provenance=provenance,
    )
    ghash = topology_hash(real)
    manifest = build_manifest(
        "R02_malecns_graph", cfg,
        graph={"graph_id": real["graph_id"], "graph_type": real["graph_type"],
               "graph_hash": ghash, "n_nodes": int(real["adjacency"].shape[0]),
               "n_edges": int(real["stats"]["n_edges"])},
        extra={"graph_seed": int(cfg["matrix_seed"])},
        repo_root=REPO_ROOT,
    )
    manifest["provenance"] = provenance
    save_graph(
        out_dir,
        graph_id=real["graph_id"], graph_type=real["graph_type"],
        node_names=real["node_names"], adjacency=real["adjacency"],
        transition_in=real["transition_in"], transition_out=real["transition_out"],
        node_features=real["node_features"], feature_names=real["feature_names"],
        self_loops=real["self_loops"], node_sizes=real["node_sizes"],
        stats=real["stats"], manifest=manifest,
        mapping=real["coarse_mapping"], node_modules=real["node_modules"],
    )
    # Raw community-level matrices are kept separately: they are the input the null models
    # need, and they are too large to sit inside the JSON metadata.
    np.savez_compressed(
        out_dir / "community_level.npz",
        community_weights=real["community_weights"],
        community_sizes=real["community_sizes"],
    )
    write_json(out_dir / "community_names.json", list(real["community_names"]))

    log.info("[graph] built %s: nodes=%d edges=%d communities=%d",
             real["graph_id"], real["stats"]["n_nodes"], real["stats"]["n_edges"],
             real["stats"]["source"]["n_communities"])
    return {
        "graph_dir": str(out_dir),
        "graph_id": real["graph_id"],
        "graph_hash": ghash,
        "stats": real["stats"],
    }


# ---------------------------------------------------------------------------------------
# Stage: null graphs (R03)
# ---------------------------------------------------------------------------------------

def _load_real_with_community(paths: Paths) -> dict:
    real = load_graph(paths.graphs / "malecns_k32")
    payload = np.load(paths.graphs / "malecns_k32" / "community_level.npz")
    names = read_json(paths.graphs / "malecns_k32" / "community_names.json")
    real["community_weights"] = payload["community_weights"]
    real["community_sizes"] = payload["community_sizes"]
    real["community_names"] = names
    real["adjacency_raw"] = real["adjacency_raw"] if "adjacency_raw" in real else None
    return real


def _real_adjacency_raw(paths: Paths) -> np.ndarray:
    """Reconstruct the pre-transform community matrix at routing-node resolution.

    The stored adjacency is already log1p-transformed, so expm1 recovers the raw weights the
    null models must preserve (they permute the raw weight multiset).
    """
    real = load_graph(paths.graphs / "malecns_k32")
    return np.expm1(np.asarray(real["adjacency"], dtype=np.float64))


def stage_nulls(cfg: dict, paths: Paths, log) -> dict:
    null_cfg = cfg["_null_cfg"]
    real = load_graph(paths.graphs / "malecns_k32")
    real["adjacency_raw"] = _real_adjacency_raw(paths)
    real["config"] = {
        "edge_transform": cfg["graph"]["edge_transform"],
    }

    built: list[str] = []
    for null in build_null_topologies(
        real,
        kinds=tuple(null_cfg["kinds"]),
        seed=int(null_cfg["seed"]),
        instances=int(null_cfg["instances"]),
    ):
        out_dir = paths.graphs / null["graph_id"]
        if (out_dir / "graph.npz").exists() and not cfg["_force"]:
            built.append(null["graph_id"])
            continue
        manifest = build_manifest(
            f"R03_null_graph_{null['graph_type']}", cfg,
            graph={"graph_id": null["graph_id"], "graph_type": null["graph_type"],
                   "n_nodes": int(null["adjacency"].shape[0]),
                   "n_edges": int(null["stats"]["n_edges"])},
            extra={"graph_seed": int(null["stats"]["null_seed"])},
            repo_root=REPO_ROOT,
        )
        save_graph(
            out_dir,
            graph_id=null["graph_id"], graph_type=null["graph_type"],
            node_names=null["node_names"], adjacency=null["adjacency"],
            transition_in=null["transition_in"], transition_out=null["transition_out"],
            node_features=null["node_features"], feature_names=null["feature_names"],
            self_loops=null["self_loops"], node_sizes=null["node_sizes"],
            stats=null["stats"], manifest=manifest,
            mapping=None, node_modules=null["node_modules"],
        )
        built.append(null["graph_id"])
        log.info("[nulls] built %s edges=%d", null["graph_id"], null["stats"]["n_edges"])
    write_json(paths.manifests / "null_graphs.json", {"graphs": built})
    return {"graphs": built}


# ---------------------------------------------------------------------------------------
# Stage: blocks (R04)
# ---------------------------------------------------------------------------------------

def stage_blocks(cfg: dict, paths: Paths, log) -> dict:
    manifest_path = paths.blocks / "block_bank_manifest.json"
    if manifest_path.exists() and not cfg["_force"]:
        log.info("[blocks] reuse %s", manifest_path)
        return read_json(manifest_path)

    splits = {name: [Sample.from_dict(r) for r in read_jsonl(paths.splits / f"{name}.jsonl")]
              for name in ALL_SPLITS}
    d_block = splits["D_block"]
    if not d_block:
        raise RuntimeError("D_block is empty; run the data stage first")

    backbone = load_backbone(
        cfg["model"]["path"],
        device=cfg["model"]["device"],
        dtype=cfg["model"]["dtype"],
        use_chat_template=bool(cfg["model"]["use_chat_template"]),
    )
    log.info("[blocks] backbone: %s", backbone.describe())

    block_cfg = cfg["_block_cfg"]
    bank = BlockBank()
    history: list[dict] = []
    for i, target in enumerate(block_cfg["targets"]):
        family = target["error_family"]
        family_samples = [s for s in d_block if s.task_family == family]
        if not family_samples:
            log.warning("[blocks] no D_block samples for %s; skipping", family)
            continue
        cfg_i = BlockTrainConfig(
            error_family=family,
            layer_id=int(target["layer_id"]),
            module_name=target["module_name"],
            rank=int(block_cfg["rank"]),
            scale=float(block_cfg["scale"]),
            lr=float(block_cfg["lr"]),
            epochs=int(block_cfg["epochs"]),
            batch_size=int(block_cfg["batch_size"]),
            max_length=int(block_cfg["max_length"]),
            seed=int(block_cfg.get("seed", cfg["seed"])),
            weight_decay=float(block_cfg.get("weight_decay", 0.0)),
            grad_clip=float(block_cfg.get("grad_clip", 1.0)),
        )
        block_id = f"b{i:02d}_{family}_l{target['layer_id']}_{target['module_name']}"
        block, info = train_block(backbone, family_samples, cfg_i, block_id=block_id, log=log)
        bank.add(block)
        info["n_train_samples"] = len(family_samples)
        history.append(info)
        log.info("[blocks] %s trained on %d samples", block_id, len(family_samples))

    bank.save(paths.blocks)
    manifest = block_bank_manifest(paths.blocks)
    write_json(paths.manifests / "block_training_history.json", history)
    log.info("[blocks] bank saved: %d blocks, manifest %s",
             manifest["n_blocks"], manifest["manifest_hash"])
    return manifest


# ---------------------------------------------------------------------------------------
# Stage: candidate cache (R06) + oracle headroom (R05)
# ---------------------------------------------------------------------------------------

def _all_samples(paths: Paths) -> list[Sample]:
    rows: list[Sample] = []
    for name in ALL_SPLITS:
        path = paths.splits / f"{name}.jsonl"
        if path.exists():
            rows.extend(Sample.from_dict(r) for r in read_jsonl(path))
    return rows


def stage_cache(cfg: dict, paths: Paths, log) -> dict:
    manifest = read_json(paths.blocks / "block_bank_manifest.json")
    samples = _all_samples(paths)
    cache_dir = paths.caches / f"h1_pilot_{manifest['manifest_hash'][:12]}"
    meta_path = cache_dir / "cache_meta.json"

    backbone = load_backbone(
        cfg["model"]["path"], device=cfg["model"]["device"], dtype=cfg["model"]["dtype"],
        use_chat_template=bool(cfg["model"]["use_chat_template"]),
    )
    bank = BlockBank.load(paths.blocks, device=backbone.device)
    decoding = DecodingConfig(
        max_new_tokens=int(cfg["cache"]["max_new_tokens"]),
        batch_size=int(cfg["cache"]["batch_size"]),
        do_sample=bool(cfg["cache"]["do_sample"]),
    )
    expected = compute_cache_key(
        backbone, manifest["manifest_hash"], decoding,
        split_hash="pending", prompt_template="pending",
    )
    if meta_path.exists() and not cfg["_force"]:
        log.info("[cache] reuse %s", cache_dir)
        return read_json(meta_path)

    t0 = time.time()
    records, meta = build_candidate_cache(
        backbone, bank, samples,
        block_manifest_hash=manifest["manifest_hash"],
        decoding=decoding,
        log=log,
    )
    save_cache(cache_dir, records, meta)
    log.info("[cache] built %d records in %.1fs -> %s", len(records), time.time() - t0, cache_dir)
    return meta


def _load_candidate_records(paths: Paths) -> list[CandidateRecord]:
    manifest = read_json(paths.blocks / "block_bank_manifest.json")
    cache_dir = paths.caches / f"h1_pilot_{manifest['manifest_hash'][:12]}"
    records, meta = load_cache(cache_dir)
    return records


def stage_headroom(cfg: dict, paths: Paths, log) -> dict:
    records = _load_candidate_records(paths)
    report: dict = {"overall": oracle_metrics(records), "per_split": {}, "per_family": {}}
    for split in ALL_SPLITS:
        rows = [r for r in records if r.split == split]
        if rows:
            report["per_split"][split] = oracle_metrics(rows)
    families = sorted({r.task_family for r in records})
    for family in families:
        rows = [r for r in records if r.task_family == family]
        report["per_family"][family] = oracle_metrics(rows)

    gate = report["overall"]
    report["gate"] = {
        "oracle_net_repair_positive": bool(gate.get("oracle_net_repair", 0) > 0),
        "n_blocks_with_fixes": int(sum(1 for b in gate.get("per_block", {}).values()
                                       if b["fixes"] > 0)),
        "passed": bool(gate.get("oracle_net_repair", 0) > 0),
    }
    write_json(paths.reports / "R05_oracle_headroom.json", report)
    log.info("[headroom] base=%.4f oracle=%.4f gain=%.4f net=%d blocks_with_fixes=%d passed=%s",
             gate["base_accuracy"], gate["oracle_accuracy"], gate["oracle_gain"],
             gate["oracle_net_repair"], report["gate"]["n_blocks_with_fixes"],
             report["gate"]["passed"])
    return report


# ---------------------------------------------------------------------------------------
# Stage: router + H1 evaluation (R07/R08)
# ---------------------------------------------------------------------------------------

METHODS = ("mlp", "no_neighbor", "uniform_random", "degree_preserving", "module_preserving",
           "malecns")


def _method_graph(paths: Paths, method: str) -> dict | None:
    if method == "mlp":
        return None
    if method == "malecns":
        return load_graph(paths.graphs / "malecns_k32")
    return load_graph(paths.graphs / f"malecns_k32__{method}")


def _features_cache_path(paths: Paths, backbone_desc: dict, n: int) -> Path:
    from connectomedpm.utils.hash import stable_hash
    key = stable_hash({"model": backbone_desc, "n": n}, length=16)
    return paths.data / f"features_{key}.npz"


def _get_features(cfg: dict, paths: Paths, samples: list[Sample], log):
    from connectomedpm.utils.hash import stable_hash
    model_desc = {"path": cfg["model"]["path"],
                  "chat": bool(cfg["model"]["use_chat_template"]),
                  "pooling": "mean"}
    key = stable_hash({"model": model_desc,
                       "ids": [s.sample_id for s in samples]}, length=16)
    path = paths.data / f"features_{key}.npz"
    if path.exists() and not cfg["_force"]:
        payload = np.load(path, allow_pickle=True)
        log.info("[router] reuse features %s", path.name)
        return payload["features"], str(payload["feature_hash"])

    backbone = load_backbone(
        cfg["model"]["path"], device=cfg["model"]["device"], dtype=cfg["model"]["dtype"],
        use_chat_template=bool(cfg["model"]["use_chat_template"]),
    )
    log.info("[router] extracting features for %d samples", len(samples))
    feats = extract_features(backbone, [s.prompt for s in samples],
                             batch_size=int(cfg["cache"]["batch_size"]))
    fhash = feature_hash(feats, [s.sample_id for s in samples])
    np.savez_compressed(path, features=feats, feature_hash=fhash)
    return feats, fhash


def _select(records, indices):
    return [records[i] for i in indices]


def stage_router(cfg: dict, paths: Paths, log) -> dict:
    from connectomedpm.utils.hash import stable_hash

    records = _load_candidate_records(paths)
    samples = _all_samples(paths)
    index_of = {s.sample_id: i for i, s in enumerate(samples)}
    assert [r.sample_id for r in records] == [s.sample_id for s in samples], \
        "cache and split ordering diverged"

    features, fhash = _get_features(cfg, paths, samples, log)
    block_ids = sorted({b for r in records for b in r.actions if b != BASE_ACTION})

    split_idx = {name: [i for i, s in enumerate(samples) if s.split == name] for name in ALL_SPLITS}
    train_idx = split_idx["D_router_train"]
    dev_idx = split_idx["D_router_dev"]
    cal_idx = split_idx["D_cal"]
    test_id_idx = split_idx["D_test_ID"]
    test_ood_idx = split_idx["D_test_OOD"]

    router_cfg = cfg["router"]
    results: dict = {}
    traces: dict = {}

    for method in METHODS:
        graph = _method_graph(paths, method)
        for seed in router_cfg["seeds"]:
            run_id = f"{method}_s{seed}"
            model_dir = ensure_dir(paths.results / "R08_h1_pilot" / run_id)
            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists() and not cfg["_force"]:
                log.info("[router] reuse %s", run_id)
                results[run_id] = read_json(metrics_path)
                continue

            train_records = _select(records, train_idx)
            dev_records = _select(records, dev_idx)
            train_tensors, scaler = build_tensors(features[train_idx], train_records, block_ids)
            dev_tensors, _ = build_tensors(features[dev_idx], dev_records, block_ids, scaler)

            train_cfg = TrainConfig(
                epochs=int(router_cfg["train"]["epochs"]),
                batch_size=int(router_cfg["train"]["batch_size"]),
                lr=float(router_cfg["train"]["lr"]),
                weight_decay=float(router_cfg["train"]["weight_decay"]),
                hidden=int(router_cfg["hidden"]),
                dropout=float(router_cfg["dropout"]),
                steps=int(router_cfg["steps"]),
                temperature=float(router_cfg["temperature"]),
                class_weights=router_cfg["loss"]["class_weights"],
                early_stopping_patience=int(router_cfg["train"]["early_stopping_patience"]),
                seed=int(seed),
                use_node_features=bool(router_cfg["use_node_features"]),
                device=str(cfg["model"]["device"]),
            )

            if method == "mlp":
                model = MLPRouter(in_dim=int(features.shape[1]), n_blocks=len(block_ids),
                                  hidden=max(256, int(router_cfg["hidden"]) * 2),
                                  dropout=float(router_cfg["dropout"]))
            else:
                model = TopologyRouter(
                    in_dim=int(features.shape[1]), n_blocks=len(block_ids),
                    transition_in=graph["transition_in"],
                    transition_out=graph["transition_out"],
                    steps=int(router_cfg["steps"]),
                    hidden=int(router_cfg["hidden"]),
                    dropout=float(router_cfg["dropout"]),
                    node_features=graph.get("node_features"),
                    feature_names=graph.get("feature_names"),
                    use_node_features=bool(router_cfg["use_node_features"]),
                    temperature=float(router_cfg["temperature"]),
                )
            model, fit_summary = fit_model(
                model, train_h=train_tensors.h.numpy(), train_y=train_tensors.y.numpy(),
                dev_h=dev_tensors.h.numpy(), dev_y=dev_tensors.y.numpy(),
                config=train_cfg, log=log,
            )
            model_dir.mkdir(parents=True, exist_ok=True)

            lam = float(router_cfg["selection"]["lambda_break"])
            gamma = float(router_cfg["selection"]["gamma_cost"])
            all_h = scaler.transform(features)

            # --- fixed-coverage evaluation on ID and OOD -------------------------------
            per_split: dict = {}
            for split_name, idx in (("D_test_ID", test_id_idx), ("D_test_OOD", test_ood_idx),
                                    ("D_router_dev", dev_idx)):
                if not idx:
                    continue
                split_records = _select(records, idx)
                scores, p_fix, p_break = score_blocks(
                    model, all_h[idx], block_ids, lam=lam, gamma=gamma
                )
                split_metrics = {}
                for coverage in cfg["evaluation"]["coverages"]:
                    split_metrics[f"fixed_coverage_{float(coverage):.2f}"] = evaluate_fixed_coverage(
                        split_records, scores, block_ids, coverage=float(coverage)
                    )
                if split_name == "D_test_ID":
                    cal_records = _select(records, cal_idx)
                    cal_scores, _, _ = score_blocks(model, all_h[cal_idx], block_ids,
                                                    lam=lam, gamma=gamma)
                    calibration = learn_threshold(
                        cal_records, cal_scores, block_ids,
                        protocol=cfg["evaluation"]["calibration_protocol"],
                    )
                    split_metrics["calibration"] = {
                        "threshold": calibration["threshold"],
                        "protocol": calibration["protocol"],
                        "selection_metrics": calibration.get("selection_metrics"),
                    }
                    split_metrics["deployment_test_ID"] = evaluate_deployment(
                        split_records, scores, block_ids, threshold=calibration["threshold"]
                    )
                    per_split["calibration"] = split_metrics["calibration"]
                per_split[split_name] = split_metrics

            # --- routing trace on the primary split ------------------------------------
            primary = cfg["evaluation"]["primary_coverage"]
            scores_primary, p_fix, p_break = score_blocks(
                model, all_h[test_id_idx], block_ids, lam=lam, gamma=gamma
            )
            threshold = per_split["D_test_ID"]["calibration"]["threshold"]
            rout = route(model, all_h[test_id_idx], block_ids,
                         threshold=threshold, lam=lam, gamma=gamma)
            trace = build_trace(
                _select(records, test_id_idx),
                graph_id=method, block_ids=block_ids,
                scores=rout.scores, p_fix=rout.p_fix, p_break=rout.p_break,
                chosen=rout.chosen, intervened=rout.intervened,
                coverage_rank=rout.coverage_rank, threshold=threshold,
                request_feature_hash=fhash,
            )
            write_jsonl(model_dir / "routing_trace.jsonl", trace)
            traces[run_id] = trace

            # Per-sample decisions are stored so uncertainty can be recomputed later without
            # re-running any model (the bootstrap in the report stage relies on this).
            write_jsonl(
                model_dir / "decisions_ID.jsonl",
                (
                    {
                        "sample_id": rec.sample_id,
                        "template_family": rec.template_family,
                        "task_family": rec.task_family,
                        "score": float(scores_primary[i].max()),
                        "best_block": block_ids[int(scores_primary[i].argmax())],
                        "chosen": rout.chosen[i],
                        "intervened": bool(rout.intervened[i]),
                    }
                    for i, rec in enumerate(_select(records, test_id_idx))
                ),
            )

            # --- graph-usage audit (section 25) ----------------------------------------
            audit: dict = {}
            if method != "mlp":
                _, nn_fix, nn_break = no_neighbor_counterfactual(
                    model, all_h[test_id_idx], block_ids, lam=lam
                )
                audit["no_neighbor_counterfactual"] = graph_sensitivity(
                    scores_primary, nn_fix - lam * nn_break, block_ids, threshold=threshold
                )
                audit["graph_id"] = graph["graph_id"]
                audit["graph_hash"] = stable_hash(
                    np.round(np.asarray(graph["adjacency"], dtype=np.float64), 8).tolist(),
                    length=32,
                )

            metrics = {
                "run_id": run_id,
                "method": method,
                "seed": int(seed),
                "graph_id": None if graph is None else graph["graph_id"],
                "n_blocks": len(block_ids),
                "block_ids": block_ids,
                "feature_hash": fhash,
                "train_size": len(train_idx),
                "dev_size": len(dev_idx),
                "test_id_size": len(test_id_idx),
                "test_ood_size": len(test_ood_idx),
                "dev_action_accuracy": fit_summary.get("best_dev_action_accuracy"),
                "best_dev_loss": fit_summary.get("best_dev_loss"),
                "best_epoch": fit_summary.get("best_epoch"),
                "n_parameters": fit_summary.get("n_parameters"),
                "splits": per_split,
                "trace_summary": trace_summary(trace),
                "audit": audit,
            }
            write_json(metrics_path, metrics)
            write_json(model_dir / "fit_summary.json", fit_summary)
            write_json(model_dir / "feature_scaler.json", scaler.to_dict())
            results[run_id] = metrics
            log.info("[router] %s: dev_acc=%s ID@10%%=%s", run_id,
                     metrics["dev_action_accuracy"],
                     per_split.get("D_test_ID", {}).get("fixed_coverage_0.10", {}).get("net_repair"))

    return {"results": results, "block_ids": block_ids, "feature_hash": fhash}


# ---------------------------------------------------------------------------------------
# Stage: H1 report
# ---------------------------------------------------------------------------------------

def stage_report(cfg: dict, paths: Paths, log, router_payload: dict) -> dict:
    from connectomedpm.utils.hash import stable_hash as _sh

    results = router_payload["results"]
    if not results:
        return {}

    coverage_key = f"fixed_coverage_{float(cfg['evaluation']['primary_coverage']):.2f}"
    rows = []
    for run_id, metrics in sorted(results.items()):
        for split_name in ("D_test_ID", "D_test_OOD"):
            entry = metrics["splits"].get(split_name, {}).get(coverage_key)
            if entry:
                rows.append({
                    "method": metrics["method"], "seed": metrics["seed"], "split": split_name,
                    **{k: entry[k] for k in ("n", "coverage", "fixes", "breaks", "net_repair",
                                             "net_repair_per_1000", "selective_break_risk",
                                             "total_accuracy_delta", "base_accuracy",
                                             "final_accuracy")},
                })

    table: dict = {}
    for row in rows:
        table.setdefault(row["split"], {})[row["method"]] = row

    # Paired clustered bootstrap: MaleCNS vs every other method on D_test_ID at the primary
    # coverage, resampling template families (DEVELOPMENT.md section 19). Recomputed from the
    # saved per-sample decisions, so no model needs to be reloaded.
    boot: dict = {}
    primary_cov = float(cfg["evaluation"]["primary_coverage"])
    decisions: dict[str, dict] = {}
    for path in sorted((paths.results / "R08_h1_pilot").glob("*/decisions_ID.jsonl")):
        decisions[path.parent.name] = {row["sample_id"]: row for row in read_jsonl(path)}

    if decisions:
        ref_run = "malecns_s0" if "malecns_s0" in decisions else sorted(decisions)[0]
        ref_rows = _load_decisions_ordered(paths, ref_run)
        clusters = [row["template_family"] for row in ref_rows]
        sample_order = [row["sample_id"] for row in ref_rows]
        cache_records = _load_candidate_records(paths)
        by_id = {r.sample_id: r for r in cache_records}
        ordered_records = [by_id[sid] for sid in sample_order]
        from connectomedpm.evaluation.fixed_coverage import top_k_mask

        def _net_repair_fn(run_id: str):
            table_rows = decisions[run_id]

            def fn(idx: np.ndarray) -> float:
                subset = [ordered_records[i] for i in idx]
                scores = np.array([table_rows[r.sample_id]["score"] for r in subset])
                mask = top_k_mask(scores, primary_cov)
                chosen = [
                    table_rows[r.sample_id]["best_block"] if keep else BASE_ACTION
                    for r, keep in zip(subset, mask)
                ]
                metrics = compute_metrics(resolve_decisions(subset, chosen, intervened=mask))
                return float(metrics["net_repair"])

            return fn

        if "malecns_s0" in decisions:
            boot["malecns"] = clustered_bootstrap(
                clusters, _net_repair_fn("malecns_s0"),
                n_boot=int(cfg["evaluation"]["bootstrap"]), seed=int(cfg["seed"]),
            )
            boot["paired_vs"] = {}
            for run_id in sorted(decisions):
                if run_id == "malecns_s0":
                    continue
                method = results.get(run_id, {}).get("method", run_id)
                boot["paired_vs"][method] = paired_clustered_bootstrap(
                    clusters,
                    _net_repair_fn("malecns_s0"),
                    _net_repair_fn(run_id),
                    n_boot=int(cfg["evaluation"]["bootstrap"]), seed=int(cfg["seed"]),
                )
        boot["n_clusters"] = int(len(set(clusters)))
        boot["n_samples"] = int(len(clusters))
        boot["results_hash"] = _sh(sample_order, length=32)

    report = {
        "run": "R08_h1_pilot",
        "primary_coverage": cfg["evaluation"]["primary_coverage"],
        "n_router_seeds": len(cfg["router"]["seeds"]),
        "table": table,
        "rows": rows,
        "results_hash": _sh({k: v.get("trace_summary") for k, v in results.items()}, length=32),
    }
    write_json(paths.reports / "R08_h1_pilot.json", report)
    _write_markdown(paths.reports / "R08_h1_pilot.md", report)
    log.info("[report] wrote %s", paths.reports / "R08_h1_pilot.md")
    return report


def _load_decisions_ordered(paths: Paths, run_id: str) -> list[dict]:
    """Read one run's saved per-sample decisions, in the cache's sample order."""
    path = paths.results / "R08_h1_pilot" / run_id / "decisions_ID.jsonl"
    rows = list(read_jsonl(path))
    order_index = {row["sample_id"]: i for i, row in enumerate(rows)}
    cache_records = _load_candidate_records(paths)
    test_records = [r for r in cache_records if r.split == "D_test_ID"]
    missing = [r.sample_id for r in test_records if r.sample_id not in order_index]
    if missing:
        raise KeyError(f"{len(missing)} test samples missing from {path}")
    return [rows[order_index[r.sample_id]] for r in test_records]


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# H1 Pilot (R08)",
        "",
        f"Primary coverage: **{report['primary_coverage']:.0%}**  ",
        f"Router seeds: {report['n_router_seeds']}  ",
        f"Results hash: `{report['results_hash']}`",
        "",
        "> Pilot only: single router seed and single null instance per kind. This is a bug",
        "> hunt, a direction check and a variance estimate — not a final result.",
        "",
    ]
    for split, methods in sorted(report["table"].items()):
        lines += [
            f"## {split}", "",
            "| Method | N | Coverage | Fixes | Breaks | Net Repair | Net/1000 | Sel. Break Risk | Base Acc | Final Acc |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method, row in sorted(methods.items()):
            lines.append(
                f"| {method} | {row['n']} | {row['coverage']:.3f} | {row['fixes']} | "
                f"{row['breaks']} | {row['net_repair']} | {row['net_repair_per_1000']:.2f} | "
                f"{row['selective_break_risk']:.4f} | {row['base_accuracy']:.4f} | "
                f"{row['final_accuracy']:.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------

STAGES = ("data", "graph", "nulls", "blocks", "cache", "headroom", "router", "report")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ConnectomeDPM Phase A / H1 pipeline")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/experiment/h1_pilot.yaml"))
    parser.add_argument("--graph-config", default=str(REPO_ROOT / "configs/graph/malecns_k32.yaml"))
    parser.add_argument("--null-config", default=str(REPO_ROOT / "configs/graph/null_graphs.yaml"))
    parser.add_argument("--block-config", default=str(REPO_ROOT / "configs/block/block_bank_8.yaml"))
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["matrix_seed"] = cfg["seed"]
    cfg["_force"] = bool(args.force)
    cfg["_graph_cfg"] = load_config(args.graph_config)
    cfg["_null_cfg"] = load_config(args.null_config)
    cfg["_block_cfg"] = load_config(args.block_config)

    paths = Paths(cfg)
    log = get_logger("connectomedpm", paths.results / "pipeline.log")
    set_seed(int(cfg["seed"]))
    log.info("=== ConnectomeDPM pipeline | config=%s force=%s", args.config, args.force)

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    payload: dict = {}
    for stage in STAGES:
        if stage not in requested:
            continue
        t0 = time.time()
        log.info("--- stage: %s", stage)
        if stage == "data":
            payload["data"] = stage_data(cfg, paths, log)
        elif stage == "graph":
            payload["graph"] = stage_graph(cfg, paths, log)
        elif stage == "nulls":
            payload["nulls"] = stage_nulls(cfg, paths, log)
        elif stage == "blocks":
            payload["blocks"] = stage_blocks(cfg, paths, log)
        elif stage == "cache":
            payload["cache"] = stage_cache(cfg, paths, log)
        elif stage == "headroom":
            payload["headroom"] = stage_headroom(cfg, paths, log)
        elif stage == "router":
            payload["router"] = stage_router(cfg, paths, log)
        elif stage == "report":
            router_payload = payload.get("router") or stage_router(cfg, paths, log)
            payload["report"] = stage_report(cfg, paths, log, router_payload)
        log.info("--- stage %s done in %.1fs", stage, time.time() - t0)

    write_json(paths.results / "pipeline_summary.json",
               {k: (v if isinstance(v, dict) else str(v)) for k, v in payload.items()})
    log.info("=== pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
