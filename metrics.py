"""Error metrics and the Diebold-Mariano test for comparing two forecasters."""
import numpy as np
from scipy import stats


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """Diebold-Mariano (1995) test with the Harvey et al. (1997) small-sample fix.

    loss_a, loss_b: per-window losses in time order. h: forecast horizon; the
    long-run variance uses a Bartlett/Newey-West window of h-1 lags because
    overlapping h-step windows are autocorrelated up to lag h-1.
    Returns (statistic, two-sided p-value). Negative statistic = A has lower loss.
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = len(d)
    if n < 3:
        raise ValueError("need at least 3 paired losses")
    mean = d.mean()
    dc = d - mean
    lags = min(h - 1, n - 1)
    var = dc @ dc / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    if var <= 0:
        return 0.0, 1.0
    stat = mean / np.sqrt(var / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) if n + 1 - 2 * h + h * (h - 1) / n > 0 else 1.0
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def mean_std(xs) -> tuple[float, float]:
    a = np.asarray(xs, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
