"""Router training (DEVELOPMENT.md sections 10 and 12).

The same function trains every topology. Hyper-parameters are selected on D_router_dev; the
caller is responsible for never looking at D_test* while doing so.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from ..utils.seed import set_seed
from .loss import WeightedActionLoss
from .model import TopologyRouter


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-4
    hidden: int = 128
    dropout: float = 0.0
    steps: int = 2
    temperature: float = 1.0
    class_weights: dict = field(default_factory=lambda: {"FIX": 1.0, "BREAK": 2.0, "UNCHANGED": 1.0})
    early_stopping_patience: int = 15
    seed: int = 0
    use_node_features: bool = False
    device: str = "auto"


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _iter_batches(n: int, batch_size: int, seed: int, epoch: int, device):
    g = torch.Generator().manual_seed(seed * 100_003 + epoch)
    order = torch.randperm(n, generator=g)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size].to(device)
        yield idx


def train_router(
    *,
    train_h: np.ndarray,
    train_y: np.ndarray,
    dev_h: np.ndarray,
    dev_y: np.ndarray,
    graph: dict,
    n_blocks: int,
    config: TrainConfig,
    log=None,
) -> tuple[TopologyRouter, dict]:
    """Train one router on one topology. Returns (model, history)."""
    set_seed(config.seed)
    device = _resolve_device(config.device)

    model = TopologyRouter(
        in_dim=int(train_h.shape[1]),
        n_blocks=int(n_blocks),
        transition_in=graph["transition_in"],
        transition_out=graph["transition_out"],
        steps=config.steps,
        hidden=config.hidden,
        dropout=config.dropout,
        node_features=graph.get("node_features"),
        feature_names=graph.get("feature_names"),
        use_node_features=config.use_node_features,
        temperature=config.temperature,
    ).to(device)

    model, summary = fit_model(model, train_h=train_h, train_y=train_y,
                               dev_h=dev_h, dev_y=dev_y, config=config, log=log)
    summary["graph_id"] = graph.get("graph_id")
    summary["graph_type"] = graph.get("graph_type")
    return model, summary


def fit_model(
    model: torch.nn.Module,
    *,
    train_h: np.ndarray,
    train_y: np.ndarray,
    dev_h: np.ndarray,
    dev_y: np.ndarray,
    config: TrainConfig,
    log=None,
) -> tuple[torch.nn.Module, dict]:
    """Generic supervised fit used by every router variant (graph and MLP alike)."""
    set_seed(config.seed)
    device = _resolve_device(config.device)
    model = model.to(device)

    criterion = WeightedActionLoss(config.class_weights)
    criterion.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay,
    )

    train_h_t = torch.as_tensor(train_h, dtype=torch.float32, device=device)
    train_y_t = torch.as_tensor(train_y, dtype=torch.long, device=device)
    dev_h_t = torch.as_tensor(dev_h, dtype=torch.float32, device=device)
    dev_y_t = torch.as_tensor(dev_y, dtype=torch.long, device=device)

    history: list[dict] = []
    best = {"dev_loss": float("inf"), "epoch": -1, "state": None}
    patience = 0

    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        seen = 0
        for idx in _iter_batches(train_h_t.shape[0], config.batch_size, config.seed, epoch, device):
            logits = model(train_h_t[idx])
            loss = criterion(logits, train_y_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            optimizer.step()
            total += float(loss.detach()) * idx.numel()
            seen += int(idx.numel())

        model.eval()
        with torch.no_grad():
            dev_logits = model(dev_h_t)
            dev_loss = float(criterion(dev_logits, dev_y_t))
            dev_pred = dev_logits.argmax(dim=-1)
            dev_acc = float((dev_pred == dev_y_t).float().mean())

        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "dev_loss": dev_loss,
            "dev_action_accuracy": dev_acc,
        }
        history.append(row)
        if log and (epoch % 10 == 0 or epoch == config.epochs - 1):
            log.info("epoch %3d | train %.4f | dev %.4f | dev acc %.4f",
                     epoch, row["train_loss"], dev_loss, dev_acc)

        if dev_loss < best["dev_loss"] - 1e-6:
            best = {
                "dev_loss": dev_loss,
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
                "dev_action_accuracy": dev_acc,
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                if log:
                    log.info("early stopping at epoch %d (best dev %.4f @ %d)",
                             epoch, best["dev_loss"], best["epoch"])
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    summary = {
        "best_epoch": best["epoch"],
        "best_dev_loss": best["dev_loss"],
        "best_dev_action_accuracy": best.get("dev_action_accuracy"),
        "n_epochs_run": len(history),
        "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "history": history,
    }
    return model, summary
