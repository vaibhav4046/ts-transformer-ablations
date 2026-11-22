"""Full experiment grid. Resumable: runs whose result file exists are skipped.

Phase 1 (lr selection): seed 0, horizon 96, lr in LR_GRID; pick the lr with the
lowest VALIDATION mse per (dataset, model). Test numbers from this phase are never used.
Phase 2 (main): SEEDS x HORIZONS at the chosen lr. Untrained baselines run once.
"""
import json
from pathlib import Path

from models import TRAINED
from train import RUNS, run

DATASETS = ["ETTh1", "ETTm1"]
HORIZONS = [96, 336]
SEEDS = [0, 1, 2, 3, 4]
LR_GRID = [1e-4, 3e-4, 1e-3]
SWEEP = Path(__file__).parent / "results" / "sweep"


def done(dataset, h, model, seed, lr) -> Path | None:
    tag = f"{dataset}_H{h}_{model}_s{seed}" + (f"_lr{lr:g}" if lr else "")
    p = RUNS / f"{tag}.json"
    return p if p.exists() else None


def ensure(dataset, h, model, seed, lr) -> dict:
    p = done(dataset, h, model, seed, lr) or run(dataset, h, model, seed, lr)
    return json.loads(p.read_text())


def main() -> None:
    chosen = {}
    for ds in DATASETS:
        for m in TRAINED:
            vals = {lr: ensure(ds, 96, m, 0, lr)["best_val_mse"] for lr in LR_GRID}
            chosen[f"{ds}/{m}"] = {"lr": min(vals, key=vals.get), "val_mse_by_lr": {str(k): v for k, v in vals.items()}}
    SWEEP.mkdir(parents=True, exist_ok=True)
    (SWEEP / "chosen_lr.json").write_text(json.dumps(chosen, indent=1))
    for ds in DATASETS:
        for h in HORIZONS:
            for m in ("naive", "seasonal_naive"):
                ensure(ds, h, m, 0, None)
            for m in TRAINED:
                for s in SEEDS:
                    ensure(ds, h, m, s, chosen[f"{ds}/{m}"]["lr"])
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
