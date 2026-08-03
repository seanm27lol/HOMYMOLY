"""Gate-5 molecular transfer: graph vs cell route on OGBG-MOLHIV.

Trains the graph and cell experts independently on the official scaffold
split with BCE, early-stops on validation ROC-AUC, and reports test
ROC-AUC through the official OGB evaluator, per the plan's molecular gate.
Ring 2-cells enter only through the boundary-list representation.  Usage:

    python scripts/train_molhiv.py --seeds 0 1 2 --epochs 30 \
        --output artifacts/gate5/molhiv_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from homymoly.data.collate import collate_structured
from homymoly.data.molecular import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    MolecularHIVDataset,
)
from homymoly.models.config import ExpertConfig
from homymoly.models.experts import CellExpert, GraphExpert
from homymoly.runtime import RuntimeConfig, initialize_runtime


def _evaluate(model, loader, device, evaluator) -> float:
    model.eval()
    logits_list, label_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
            logits_list.append(output.logits.float()[:, -1].cpu())
            label_list.append(batch.labels.cpu())
    result = evaluator.eval(
        {
            "y_true": torch.cat(label_list).reshape(-1, 1).numpy(),
            "y_pred": torch.cat(logits_list).reshape(-1, 1).numpy(),
        }
    )
    return float(result["rocauc"])


def _train_route(route_name: str, model, dataset, splits, runtime, args, evaluator, seed: int):
    device = runtime.device
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, list(splits["train"])),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_structured,
        generator=generator,
    )
    eval_loaders = {
        name: DataLoader(
            Subset(dataset, list(indices)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_structured,
        )
        for name, indices in (("valid", splits["valid"]), ("test", splits["test"]))
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    best_valid = -1.0
    best_state = None
    bad_epochs = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    output.logits.float()[:, -1],
                    batch.labels.float(),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {route_name} seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        valid_auc = _evaluate(model, eval_loaders["valid"], device, evaluator)
        if valid_auc > best_valid + 1e-5:
            best_valid = valid_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_auc = _evaluate(model, eval_loaders["test"], device, evaluator)
    elapsed = time.perf_counter() - started
    # Regime-conditioned reporting: ring-bearing vs ring-free molecules.
    per_regime = {}
    for regime_name, regime_value in (("ring", 1), ("ring_free", 0)):
        indices = [
            i for i in splits["test"] if (dataset[i].num_faces > 0) == bool(regime_value)
        ]
        if indices:
            labels = [int(dataset[i].label) for i in indices]
            if len(set(labels)) < 2:
                per_regime[regime_name] = {"rocauc": None, "examples": len(indices)}
                continue
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_structured,
            )
            per_regime[regime_name] = {
                "rocauc": _evaluate(model, loader, device, evaluator),
                "examples": len(indices),
            }
    return {
        "route": route_name,
        "seed": seed,
        "valid_rocauc": best_valid,
        "test_rocauc": test_auc,
        "epochs": epoch + 1,
        "seconds": round(elapsed, 2),
        "per_regime": per_regime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap train examples")
    parser.add_argument("--output", default="artifacts/gate5/molhiv_results.json")
    args = parser.parse_args()

    runtime = initialize_runtime(RuntimeConfig(device="cuda", precision="bfloat16", deterministic=True), seed=0)
    dataset = MolecularHIVDataset(root="artifacts/molecular")
    splits = dataset.splits
    if args.limit:
        splits = {**splits, "train": splits["train"][: args.limit]}
    from ogb.graphproppred import Evaluator

    evaluator = Evaluator("ogbg-molhiv")
    config = ExpertConfig(
        node_feature_dim=NODE_FEATURE_DIM,
        edge_feature_dim=EDGE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        embedding_dim=64,
        num_layers=args.num_layers,
        dropout=0.1,
        num_classes=2,
    )
    results = []
    for seed in args.seeds:
        for route_name, cls in (("graph", GraphExpert), ("cell", CellExpert)):
            torch.manual_seed(seed)
            model = cls(config).to(runtime.device)
            params = sum(p.numel() for p in model.parameters())
            record = _train_route(route_name, model, dataset, splits, runtime, args, evaluator, seed)
            record["parameters"] = params
            results.append(record)
            print(json.dumps(record))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"results written to {output}")


if __name__ == "__main__":
    main()
