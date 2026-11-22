import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import Windows, split_borders, standardize  # noqa: E402
from metrics import dm_test, mean_std  # noqa: E402
from models import (TRAINED, VARIANTS, DLinear, PatchConfig, PatchTransformer, RevIN,  # noqa: E402
                    build, n_patches, naive_last, seasonal_naive)


def test_split_borders_etth1_standard():
    b = split_borders("ETTh1", 336)
    assert b["train"] == (0, 8640)
    assert b["val"] == (8640 - 336, 8640 + 2880)
    assert b["test"] == (8640 + 2880 - 336, 8640 + 5760)


def test_split_borders_ettm1_is_4x():
    b = split_borders("ETTm1", 96)
    assert b["train"][1] == 4 * 8640 and b["test"][1] == 4 * 14400


def test_standardize_uses_train_only():
    x = np.concatenate([np.zeros((10, 1)) + [[1.0]], np.zeros((10, 1)) + [[100.0]]])
    x[:10:2] = 3.0  # train part mean 2, std 1
    z, mean, std = standardize(x, 10)
    assert mean[0] == pytest.approx(2.0) and std[0] == pytest.approx(1.0)
    assert z[15, 0] == pytest.approx(98.0)


def test_windows_alignment():
    series = np.arange(20, dtype=float)[:, None]
    w = Windows(series, seq_len=5, pred_len=3)
    assert len(w) == 20 - 5 - 3 + 1
    x, y = w[4]
    assert x[:, 0].tolist() == [4, 5, 6, 7, 8] and y[:, 0].tolist() == [9, 10, 11]


def test_windows_too_short():
    with pytest.raises(ValueError):
        Windows(np.zeros((5, 1)), 4, 3)


def test_revin_roundtrip():
    r = RevIN(3)
    x = torch.randn(4, 50, 3) * 5 + 2
    assert torch.allclose(r.denorm(r.norm(x)), x, atol=1e-4)


def test_patch_count():
    assert n_patches(336, 16, 8) == 42
    assert n_patches(336, 1, 1) == 337


@pytest.mark.parametrize("name", TRAINED)
def test_all_models_output_shape(name):
    m = build(name, c=7, seq_len=64, pred_len=24)
    assert m(torch.randn(2, 64, 7)).shape == (2, 24, 7)


def test_channel_independence_holds_and_mixing_leaks():
    torch.manual_seed(0)
    x = torch.randn(2, 64, 4)
    x2 = x.clone()
    x2[:, :, 3] += torch.randn(2, 64)
    ci = PatchTransformer(4, 64, 8, PatchConfig()).eval()
    mix = PatchTransformer(4, 64, 8, PatchConfig(channel_independent=False)).eval()
    with torch.no_grad():
        assert torch.allclose(ci(x)[..., 0], ci(x2)[..., 0], atol=1e-5)
        assert not torch.allclose(mix(x)[..., 0], mix(x2)[..., 0], atol=1e-5)


def test_variants_differ_from_full_in_one_switch():
    full = VARIANTS["patchtst"]
    for name, cfg in VARIANTS.items():
        diff = [k for k in full.__dataclass_fields__ if getattr(cfg, k) != getattr(full, k)]
        if name == "patchtst":
            assert diff == []
        elif name == "no_patch":
            assert set(diff) == {"patch_len", "stride"}
        else:
            assert len(diff) == 1, (name, diff)


def test_naive_baselines():
    x = torch.arange(10.0).view(1, 10, 1)
    assert naive_last(x, 3).flatten().tolist() == [9, 9, 9]
    assert seasonal_naive(x, 5, 3).flatten().tolist() == [7, 8, 9, 7, 8]


def test_dlinear_constant_input_learns_nothing_wrong():
    m = DLinear(16, 4)
    out = m(torch.ones(1, 16, 2))
    assert out.shape == (1, 4, 2)


def test_patch_transformer_can_fit_a_sine():
    torch.manual_seed(0)
    t = torch.arange(0, 400, dtype=torch.float32)
    series = torch.sin(t * 2 * torch.pi / 24).unsqueeze(1)
    xs = torch.stack([series[i:i + 48] for i in range(0, 300, 3)])
    ys = torch.stack([series[i + 48:i + 60] for i in range(0, 300, 3)])
    m = PatchTransformer(1, 48, 12, PatchConfig(patch_len=8, stride=4, d_model=16, layers=1, dropout=0.0))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        loss = (m(xs) - ys).pow(2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 * first


def test_dm_identical_losses():
    a = np.random.default_rng(0).random(200)
    stat, p = dm_test(a, a, h=5)
    assert stat == 0.0 and p == 1.0


def test_dm_detects_better_model_and_is_antisymmetric():
    rng = np.random.default_rng(1)
    base = rng.random(500)
    better = base * 0.5
    s1, p1 = dm_test(better, base, h=1)
    s2, p2 = dm_test(base, better, h=1)
    assert s1 < 0 and p1 < 1e-6
    assert s1 == pytest.approx(-s2) and p1 == pytest.approx(p2)


def test_dm_noise_not_significant():
    rng = np.random.default_rng(2)
    a, b = rng.random(300), rng.random(300)
    assert dm_test(a, b, h=1)[1] > 0.01


def test_mean_std():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert m == 2.0 and s == pytest.approx(1.0)
