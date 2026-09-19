# ConnectomeDPM

ConnectomeDPM is a research platform for **selective parameter correction driven by a real
connectome topology prior**. It freezes a base language model, builds a fixed bank of
low-rank parameter-correction blocks, maps each request to `K` graph nodes derived from the
MaleCNS connectome, propagates that request representation through a *fixed* (non-learnable)
graph operator, and predicts a per-block `fix / break / unchanged` distribution.

The scientific question is narrow and falsifiable:

> Does a real connectome topology, under a strictly matched null model, improve Net Repair,
> Break Risk, OOD generalisation and router sample efficiency?

The platform is built so that the answer can be *no*. Nothing in the code assumes MaleCNS wins.

## Status

This repository currently implements **Phase A (R00-R07)** of `DEVELOPMENT.md`: the engineering
closed loop plus the H1 pilot comparison. See `docs/PHASE_A.md` for the executed protocol and
`experiments/reports/` for generated results.

## Design principles

1. **Data correctness > block headroom > cache correctness > graph correctness > router correctness.**
2. Every graph (real and null) gets the *same* router, optimiser, search space, splits, cache and
   calibration protocol. Only the topology differs.
3. Every graph recomputes its own degrees, strengths, node features and transition operators.
4. Nothing that touches `D_test*` may influence graph construction, block selection, router
   hyper-parameters or the deployment threshold.
5. Sample-level routing traces and run manifests are always written; averages alone are never
   sufficient evidence.

## Quick start

```bash
python -m pip install -e .

# 1. Build the task suite and frozen splits
python scripts/prepare_data.py --config configs/experiment/h1_pilot.yaml

# 2. Build the MaleCNS graph and the matched null graphs
python scripts/build_malecns_graph.py --config configs/graph/malecns_k32.yaml
python scripts/build_null_graphs.py --config configs/graph/null_graphs.yaml

# 3. Train the block bank (needs a backbone; see configs/model/)
python scripts/train_blocks.py --config configs/block/block_bank_8.yaml

# 4. Oracle headroom gate, then the candidate outcome cache
python scripts/check_oracle_headroom.py --config configs/experiment/h1_pilot.yaml
python scripts/generate_candidate_cache.py --config configs/experiment/h1_pilot.yaml

# 5. Train one router per topology and evaluate H1
python scripts/train_router.py --graph malecns --config configs/experiment/h1_pilot.yaml
python scripts/evaluate_h1.py --config configs/experiment/h1_pilot.yaml
```

## Repository layout

See `DEVELOPMENT.md` section 3. The implementation follows that layout; a few directories are
intentionally absent until their phase is reached (they are listed in `DEVELOPMENT.md` as later
work: `evaluation/sample_efficiency.py`, `evaluation/ood.py`, `audit/leakage.py`, ...).

## Licence

MIT

