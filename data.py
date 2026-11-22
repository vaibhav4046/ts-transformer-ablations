"""ETT loading with the standard 12/4/4-month chronological split.

Split borders follow the convention used by Informer, Autoformer, DLinear and
PatchTST so numbers are comparable to the literature: the scaler is fit on the
training months only, and val/test windows may look back into the previous
split (the input) but never predict inside it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA = Path(__file__).parent / "data"
MONTH_HOURS = 30 * 24
# rows per hour for each dataset
STEPS_PER_HOUR = {"ETTh1": 1, "ETTm1": 4}


def split_borders(dataset: str, seq_len: int) -> dict[str, tuple[int, int]]:
    """Row ranges [start, end) for each split; start includes the look-back."""
    k = STEPS_PER_HOUR[dataset]
    tr, va, te = 12 * MONTH_HOURS * k, 4 * MONTH_HOURS * k, 4 * MONTH_HOURS * k
    return {
        "train": (0, tr),
        "val": (tr - seq_len, tr + va),
        "test": (tr + va - seq_len, tr + va + te),
    }


def load_frame(dataset: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{dataset}.csv", parse_dates=["date"])


def standardize(values: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[:train_end].mean(axis=0)
    std = values[:train_end].std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (values - mean) / std, mean, std


class Windows(Dataset):
    """Sliding (input, target) windows over a [T, C] array, stride 1."""

    def __init__(self, series: np.ndarray, seq_len: int, pred_len: int):
        self.x = torch.as_tensor(series, dtype=torch.float32)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.n = len(series) - seq_len - pred_len + 1
        if self.n <= 0:
            raise ValueError("series shorter than seq_len + pred_len")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        s = i + self.seq_len
        return self.x[i:s], self.x[s:s + self.pred_len]


def make_splits(dataset: str, seq_len: int, pred_len: int):
    """Return (train, val, test) Windows plus the test target start timestamps."""
    df = load_frame(dataset)
    values = df.drop(columns=["date"]).to_numpy(dtype=np.float64)
    borders = split_borders(dataset, seq_len)
    scaled, _, _ = standardize(values, borders["train"][1])
    out = {}
    for name, (a, b) in borders.items():
        out[name] = Windows(scaled[a:b], seq_len, pred_len)
    a = borders["test"][0]
    starts = df["date"].iloc[a + seq_len: a + seq_len + len(out["test"])].reset_index(drop=True)
    return out["train"], out["val"], out["test"], starts
