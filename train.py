"""Train one (dataset, horizon, model, seed, lr) run; write results/runs/<tag>.json + .npz.

Usage:
  python train.py --dataset ETTh1 --pred-len 96 --model patchtst --seed 0 --lr 3e-4
  python train.py --dataset ETTh1 --pred-len 96 --model naive        # no training
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import STEPS_PER_HOUR, Windows, make_splits
from models import TRAINED, build, naive_last, seasonal_naive

SEQ_LEN = 336
BATCH = 128
RUNS = Path(__file__).parent / "results" / "runs"
MAX_EPOCHS = {"ETTh1": 30, "ETTm1": 12}
PATIENCE = 4


def windows_tensor(w: Windows, device: str) -> torch.Tensor:
    """All windows as a [n, L+H, C] view on device (no copy of the series)."""
    return w.x.to(device).unfold(0, w.seq_len + w.pred_len, 1)[: w.n].transpose(1, 2)


def predict_all(fn, win: torch.Tensor, seq_len: int, bs: int = 512) -> torch.Tensor:
    outs = [fn(win[i:i + bs, :seq_len]).float() for i in range(0, len(win), bs)]
    return torch.cat(outs)


def evaluate(fn, win: torch.Tensor, seq_len: int) -> dict:
    with torch.no_grad():
        pred = predict_all(fn, win, seq_len)
    err = pred - win[:, seq_len:]
    se = err.pow(2)
    return {
        "mse": se.mean().item(),
        "mae": err.abs().mean().item(),
        "per_window_mse": se.mean(dim=(1, 2)).cpu().numpy(),
        "per_step_mse": se.mean(dim=(0, 2)).cpu().numpy(),
        "per_channel_mse": se.mean(dim=(0, 1)).cpu().numpy(),
    }


def train(model, tr, va, lr, seed, max_epochs, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=gen).to(device)
        for i in range(0, len(tr), BATCH):
            batch = tr[perm[i:i + BATCH]]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = model(batch[:, :SEQ_LEN])
            loss = (pred.float() - batch[:, SEQ_LEN:]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        val = evaluate(model, va, SEQ_LEN)["mse"]
        hist.append(val)
        if val < best:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return best, hist


def run(dataset: str, pred_len: int, model_name: str, seed: int, lr: float | None) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"{dataset}_H{pred_len}_{model_name}_s{seed}" + (f"_lr{lr:g}" if lr else "")
    out = RUNS / f"{tag}.json"
    RUNS.mkdir(parents=True, exist_ok=True)
    tr_w, va_w, te_w, _ = make_splits(dataset, SEQ_LEN, pred_len)
    tr, va, te = (windows_tensor(w, device) for w in (tr_w, va_w, te_w))
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    info = {"dataset": dataset, "pred_len": pred_len, "model": model_name, "seed": seed, "lr": lr,
            "seq_len": SEQ_LEN, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    if model_name in TRAINED:
        model = build(model_name, tr.shape[2], SEQ_LEN, pred_len).to(device)
        info["params"] = sum(p.numel() for p in model.parameters())
        info["best_val_mse"], info["val_history"] = train(model, tr, va, lr, seed, MAX_EPOCHS[dataset], device)
        info["epochs_run"] = len(info["val_history"])
        fn = model
    elif model_name == "naive":
        fn = lambda x: naive_last(x, pred_len)  # noqa: E731
    elif model_name == "seasonal_naive":
        period = 24 * STEPS_PER_HOUR[dataset]
        fn = lambda x: seasonal_naive(x, pred_len, period)  # noqa: E731
    else:
        raise ValueError(model_name)
    if model_name not in TRAINED:
        info["best_val_mse"] = evaluate(fn, va, SEQ_LEN)["mse"]
    info["train_seconds"] = round(time.time() - t0, 2)
    res = evaluate(fn, te, SEQ_LEN)
    info["test_mse"], info["test_mae"] = res["mse"], res["mae"]
    info["per_step_mse"] = res["per_step_mse"].tolist()
    info["per_channel_mse"] = res["per_channel_mse"].tolist()
    np.savez_compressed(out.with_suffix(".npz"), per_window_mse=res["per_window_mse"].astype(np.float32))
    out.write_text(json.dumps(info, indent=1))
    print(f"{tag}: val {info['best_val_mse']:.4f} test mse {info['test_mse']:.4f} mae {info['test_mae']:.4f}"
          f" ({info['train_seconds']}s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MAX_EPOCHS))
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    run(a.dataset, a.pred_len, a.model, a.seed, a.lr)


if __name__ == "__main__":
    main()
