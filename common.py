"""Shared helpers: data loading, splits, portfolio accounting, metrics."""
import numpy as np
import pandas as pd

import config as C


# ----------------------------------------------------------------------- directories
def ensure_dirs():
    for d in (C.DATA_DIR, C.RESULTS_DIR, C.MODELS_DIR,
              C.RESULTS_DIR / "returns" / "val", C.RESULTS_DIR / "returns" / "test"):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------- loading
def load_processed():
    """Return (features, simple_returns, bench_returns) aligned on the same dates."""
    feats = pd.read_csv(C.DATA_DIR / "features.csv", index_col=0, parse_dates=True)
    feats.columns = pd.MultiIndex.from_tuples([tuple(c.split("|")) for c in feats.columns])
    rets = pd.read_csv(C.DATA_DIR / "simple_returns.csv", index_col=0, parse_dates=True)
    bench = pd.read_csv(C.DATA_DIR / "bench_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return feats, rets, bench


def split_bounds(index, window=C.WINDOW):
    """Row-index bounds (s, e) for each split.

    The agent observes row t, acts, and earns the return of row t+1.  So a split
    (s, e) produces returns for rows s+1 .. e.  Validation starts at the last
    training row so its first earned return is the first validation day.
    """
    n = len(index)
    tr_last = index.searchsorted(pd.Timestamp(C.TRAIN_END), side="right") - 1
    va_last = index.searchsorted(pd.Timestamp(C.VAL_END), side="right") - 1
    return {
        "train": (window - 1, tr_last),
        "val": (tr_last, va_last),
        "test": (va_last, n - 1),
    }


def rf_daily():
    return (1.0 + C.RF_ANNUAL) ** (1.0 / 252.0) - 1.0


# ----------------------------------------------------------------------- portfolio accounting
def portfolio_step(w_drift, w_target, r_next_assets, rf_d, cost):
    """One day of trading.

    w_drift  : current holdings (n assets + cash) after yesterday's price moves
    w_target : desired holdings (n assets + cash), sums to 1
    r_next   : next-day simple returns of the n risky assets
    Cost is charged on traded risky-asset value (cash moves are free).
    Returns (net_return, turnover, new_drifted_weights).
    """
    r_full = np.append(r_next_assets, rf_d)
    turnover = float(np.abs(w_target[:-1] - w_drift[:-1]).sum())
    gross = float(w_target @ r_full)
    net = gross - cost * turnover
    w_new = w_target * (1.0 + r_full) / (1.0 + gross)
    return net, turnover, w_new


# ----------------------------------------------------------------------- metrics
def metrics(net_ret, rf_annual=C.RF_ANNUAL, periods=252):
    r = np.asarray(net_ret, dtype=float)
    rf_d = (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    ex = r - rf_d
    n = len(r)
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min())
    ann_ret = float(equity[-1] ** (periods / n) - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods))
    sd = ex.std(ddof=1)
    sharpe = float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0
    down = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    sortino = float(ex.mean() / down * np.sqrt(periods)) if down > 0 else 0.0
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar,
    }


def save_returns(path, index, net, turnover):
    df = pd.DataFrame({"net_ret": net, "turnover": turnover}, index=index)
    df.index.name = "date"
    df.to_csv(path)
