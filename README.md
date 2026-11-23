# ts-transformer-ablations

[![tests](https://github.com/vaibhav4046/ts-transformer-ablations/actions/workflows/tests.yml/badge.svg)](https://github.com/vaibhav4046/ts-transformer-ablations/actions/workflows/tests.yml)

A controlled PyTorch study: **does a small patch transformer beat linear models for time-series forecasting, and which of its parts matter?**
Datasets: ETTh1 and ETTm1, horizons 96 and 336, look-back 336. Baselines: naive, seasonal naive, NLinear, DLinear. Five ablations of the transformer. Five seeds each, held-out test months, Diebold-Mariano tests, error analysis.

**Short answer (details in [REPORT.md](REPORT.md)):** the transformer is within about 1% of DLinear everywhere (not significantly different). Channel independence is the only design choice with a large, consistent effect (mixing channels is +13% to +19% worse). Removing the attention layers did not hurt, and was the best model on ETTh1.

| test MSE (mean ± std, 5 seeds) | ETTh1 H96 | ETTh1 H336 | ETTm1 H96 | ETTm1 H336 |
|---|---|---|---|---|
| seasonal_naive | 0.5122 | 0.6499 | 0.4227 | 0.4961 |
| nlinear | 0.3824 ± 0.0036 | 0.4408 ± 0.0027 | 0.3119 ± 0.0053 | 0.3818 ± 0.0060 |
| dlinear | 0.3792 ± 0.0018 | 0.4490 ± 0.0047 | 0.3006 ± 0.0004 | 0.3717 ± 0.0003 |
| patchtst (329k params) | 0.3800 ± 0.0015 | 0.4475 ± 0.0042 | 0.2976 ± 0.0063 | 0.3675 ± 0.0042 |
| channel_mix | 0.4533 ± 0.0020 | 0.5207 ± 0.0048 | 0.3356 ± 0.0041 | 0.4227 ± 0.0098 |
| no_attn | 0.3733 ± 0.0031 | 0.4319 ± 0.0041 | 0.3078 ± 0.0034 | 0.3756 ± 0.0023 |

## Reproduce

```
pip install -r requirements.txt     # plus torch for your CUDA
python prepare_data.py              # downloads ETT (CC BY-ND 4.0, not redistributed)
python run_all.py                   # ~5 h on an RTX 3050 6 GB, resumable
python analyze.py                   # tables, DM tests, plots
pytest -q
```

## Layout

- `models.py` patch transformer with switchable parts, NLinear, DLinear, naive baselines
- `train.py`, `run_all.py` one run / the whole grid (lr picked on validation only)
- `metrics.py` Diebold-Mariano test; `analyze.py` aggregation and error analysis
- `results/` every raw run (`runs/`), lr sweep, `summary.json`, `dm_tests.json`, `error_analysis.json`, `tables.md`, `plots/`

Code: MIT. The ETT data is by Zhou et al. and is not included.
