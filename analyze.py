"""Aggregate results/runs into tables, significance tests, error analysis and plots.

Reads only main-phase runs (chosen lr, seeds 0-4). Writes:
  results/summary.json, results/dm_tests.json, results/error_analysis.json,
  results/tables.md, results/plots/*.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

from data import make_splits
from metrics import dm_test, mean_std
from models import TRAINED
from run_all import DATASETS, HORIZONS, SEEDS, SWEEP
from train import RUNS, SEQ_LEN

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).parent / "results"
PLOTS = RES / "plots"
BASELINES = ["naive", "seasonal_naive"]
ORDER = [*BASELINES, "nlinear", "dlinear", "patchtst", "no_patch", "no_revin", "channel_mix", "no_pos", "no_attn"]


def load(ds, h, m, chosen):
    if m in BASELINES:
        tags = [f"{ds}_H{h}_{m}_s0"]
    else:
        lr = chosen[f"{ds}/{m}"]["lr"]
        tags = [f"{ds}_H{h}_{m}_s{s}_lr{lr:g}" for s in SEEDS]
    runs = [json.loads((RUNS / f"{t}.json").read_text()) for t in tags]
    pw = np.stack([np.load(RUNS / f"{t}.npz")["per_window_mse"] for t in tags])
    return runs, pw


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    chosen = json.loads((SWEEP / "chosen_lr.json").read_text())
    summary, dms, errs = {}, {}, {}
    cache = {}
    for ds in DATASETS:
        for h in HORIZONS:
            key = f"{ds}/H{h}"
            summary[key] = {}
            for m in ORDER:
                runs, pw = load(ds, h, m, chosen)
                cache[(ds, h, m)] = (runs, pw)
                mse, mse_sd = mean_std([r["test_mse"] for r in runs])
                mae, mae_sd = mean_std([r["test_mae"] for r in runs])
                summary[key][m] = {
                    "test_mse_mean": mse, "test_mse_std": mse_sd, "test_mae_mean": mae, "test_mae_std": mae_sd,
                    "n_seeds": len(runs), "lr": runs[0]["lr"], "params": runs[0].get("params", 0),
                    "train_seconds_mean": float(np.mean([r["train_seconds"] for r in runs])),
                    "epochs_mean": float(np.mean([r.get("epochs_run", 0) for r in runs])),
                    "test_mse_per_seed": [r["test_mse"] for r in runs],
                }
            # DM tests on seed-averaged per-window losses: every model vs the full transformer
            ref = cache[(ds, h, "patchtst")][1].mean(0)
            dms[key] = {}
            for m in ORDER:
                if m == "patchtst":
                    continue
                stat, p = dm_test(ref, cache[(ds, h, m)][1].mean(0), h)
                dms[key][f"patchtst_vs_{m}"] = {"dm_stat": stat, "p_value": p,
                                                "patchtst_better": stat < 0}
            errs[key] = error_analysis(ds, h, cache)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    (RES / "dm_tests.json").write_text(json.dumps(dms, indent=1))
    (RES / "error_analysis.json").write_text(json.dumps(errs, indent=1))
    write_tables(summary, dms, errs)
    plots(summary, cache)
    print((RES / "tables.md").read_text())


def error_analysis(ds, h, cache) -> dict:
    _, _, te, starts = make_splits(ds, SEQ_LEN, h)
    pt = cache[(ds, h, "patchtst")][1].mean(0)
    dl = cache[(ds, h, "dlinear")][1].mean(0)
    k = max(1, len(pt) // 10)
    top = np.argsort(pt)[::-1][:k]
    hours = starts.dt.hour.to_numpy()
    months = starts.dt.strftime("%Y-%m").to_numpy()
    runs = cache[(ds, h, "patchtst")][0]
    per_ch = np.mean([r["per_channel_mse"] for r in runs], axis=0)
    per_step = np.mean([r["per_step_mse"] for r in runs], axis=0)
    return {
        "n_test_windows": int(len(pt)),
        "top10pct_windows_share_of_total_error": float(pt[top].sum() / pt.sum()),
        "worst_window_target_start": str(starts.iloc[int(top[0])]),
        "worst_window_mse": float(pt[top[0]]),
        "frac_windows_patchtst_beats_dlinear": float((pt < dl).mean()),
        "corr_patchtst_dlinear_window_errors": float(np.corrcoef(pt, dl)[0, 1]),
        "mse_by_target_month": {mo: float(pt[months == mo].mean()) for mo in sorted(set(months))},
        "mse_by_start_hour_min_max": [float(min(pt[hours == x].mean() for x in set(hours))),
                                      float(max(pt[hours == x].mean() for x in set(hours)))],
        "per_channel_mse": dict(zip(["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"], map(float, per_ch))),
        "per_step_mse_first_last": [float(per_step[0]), float(per_step[-1])],
    }


def fmt(s):
    return f"{s['test_mse_mean']:.4f} ± {s['test_mse_std']:.4f}"


def write_tables(summary, dms, errs) -> None:
    keys = list(summary)
    lines = ["## Test MSE (mean ± std over seeds; baselines are deterministic)", "",
             "| model | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for m in ORDER:
        best = [min(summary[k], key=lambda x: summary[k][x]["test_mse_mean"]) == m for k in keys]
        cells = [("**" + fmt(summary[k][m]) + "**") if b else fmt(summary[k][m]) for k, b in zip(keys, best)]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Test MAE", "", "| model | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for m in ORDER:
        lines.append(f"| {m} | " + " | ".join(
            f"{summary[k][m]['test_mae_mean']:.4f} ± {summary[k][m]['test_mae_std']:.4f}" for k in keys) + " |")
    lines += ["", "## Ablations: % change in test MSE vs full patchtst (positive = worse); DM p-value", "",
              "| variant | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for m in ORDER:
        if m == "patchtst":
            continue
        cells = []
        for k in keys:
            d = 100 * (summary[k][m]["test_mse_mean"] / summary[k]["patchtst"]["test_mse_mean"] - 1)
            p = dms[k][f"patchtst_vs_{m}"]["p_value"]
            cells.append(f"{d:+.1f}% (p={p:.3g})")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Cost", "", "| model | params | lr | mean train s (ETTh1/H96) | mean train s (ETTm1/H336) |",
              "|---|---|---|---|---|"]
    for m in TRAINED:
        a, b = summary["ETTh1/H96"][m], summary["ETTm1/H336"][m]
        lines.append(f"| {m} | {a['params']:,} | {a['lr']:g} / {b['lr']:g} | {a['train_seconds_mean']:.0f} | "
                     f"{b['train_seconds_mean']:.0f} |")
    lines += ["", "## Error analysis (full patchtst, seed-averaged per-window MSE)", "",
              "| setting | top-10% windows share of error | patchtst beats dlinear (windows) | corr(patchtst, dlinear) |"
              " step-1 MSE | last-step MSE |", "|---|---|---|---|---|---|"]
    for k in keys:
        e = errs[k]
        lines.append(f"| {k} | {e['top10pct_windows_share_of_total_error']:.1%} | "
                     f"{e['frac_windows_patchtst_beats_dlinear']:.1%} | {e['corr_patchtst_dlinear_window_errors']:.3f} | "
                     f"{e['per_step_mse_first_last'][0]:.3f} | {e['per_step_mse_first_last'][1]:.3f} |")
    (RES / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plots(summary, cache) -> None:
    keys = list(summary)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2), sharey=False)
    for ax, k in zip(axes, keys):
        ms = [m for m in ORDER if m != "naive"]
        means = [summary[k][m]["test_mse_mean"] for m in ms]
        sds = [summary[k][m]["test_mse_std"] for m in ms]
        colors = ["#999999" if m in BASELINES else "#4C72B0" if m in ("nlinear", "dlinear")
                  else "#C44E52" if m == "patchtst" else "#DD8452" for m in ms]
        ax.bar(range(len(ms)), means, yerr=sds, color=colors, capsize=3)
        ax.set_xticks(range(len(ms)), ms, rotation=60, ha="right", fontsize=8)
        ax.set_title(k)
        lo = min(means)
        ax.set_ylim(lo * 0.9, max(means) * 1.03)
        ax.set_ylabel("test MSE (standardised)")
    fig.tight_layout()
    fig.savefig(PLOTS / "test_mse.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ds in zip(axes, DATASETS):
        for m in ["seasonal_naive", "dlinear", "nlinear", "patchtst", "no_revin"]:
            runs, _ = cache[(ds, 336, m)]
            ax.plot(np.mean([r["per_step_mse"] for r in runs], axis=0), label=m)
        ax.set_title(f"{ds}, H=336: MSE by forecast step")
        ax.set_xlabel("step ahead")
        ax.set_ylabel("MSE")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "error_by_step.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, ds in zip(axes, DATASETS):
        _, _, _, starts = make_splits(ds, SEQ_LEN, 96)
        for m in ["dlinear", "patchtst"]:
            pw = cache[(ds, 96, m)][1].mean(0)
            w = 24 * (4 if ds == "ETTm1" else 1)
            ax.plot(starts, np.convolve(pw, np.ones(w) / w, mode="same"), label=m, lw=1)
        ax.set_title(f"{ds}, H=96: per-window test MSE over time (1-day rolling mean)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "error_over_time.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
