"""Per-compound shift minimising expected soft-threshold error."""
from __future__ import annotations

import numpy as np

GRID = np.arange(-2.0, 2.0001, 0.02)
N_BINS = 5


def residual_bins(centre, n_bins: int = N_BINS):
    """Assign compounds to predicted-activity bins, for conditioning residuals."""
    edges = np.quantile(centre, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(centre, edges)


def optimal_shift(centre, resid_pools, bins, lo_half, hi_half, grid=GRID):
    """Assign compounds to predicted-activity bins for conditioning residuals."""
    shifts = np.zeros(len(centre))
    for b, pool in resid_pools.items():
        sel = np.flatnonzero(bins == b)
        if sel.size == 0 or pool.size == 0:
            continue
        r = pool[None, None, :]
        s = grid[None, :, None]
        lo_h = lo_half[sel][:, None, None]
        hi_h = hi_half[sel][:, None, None]
        loss = (np.clip(s - r - hi_h, 0, None) + np.clip(r - lo_h - s, 0, None)).mean(axis=2)
        shifts[sel] = grid[loss.argmin(axis=1)]
    return shifts


def pools_from(centre, y_true, bins, exclude=None):
    """Residual pools per bin, optionally excluding one fold's own compounds."""
    resid = y_true - centre
    keep = np.ones(len(resid), bool) if exclude is None else ~exclude
    return {b: resid[(bins == b) & keep] for b in np.unique(bins)}


def global_affine(centre, y_true, lo, hi, grid=GRID):
    """Baseline comparator: one scale and one shift per isoform, fitted on the metric."""
    from metric import st_rae

    best, best_s = (1.0, 0.0), np.inf
    mu = float(centre.mean())
    for k in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
        scaled = mu + (centre - mu) * k
        for b in grid:
            s = st_rae(y_true, scaled + b, lo, hi)
            if s < best_s:
                best_s, best = s, (k, b)
    return best, best_s


def optimal_shift_uncertain(centre, pools, bins, lo_half, hi_half, grid=GRID, chunk=128):
    """Assign compounds to predicted-activity bins for conditioning residuals."""
    shifts = np.zeros(len(centre))
    for b, pool in pools.items():
        sel = np.flatnonzero(bins == b)
        if sel.size == 0 or len(pool[0]) == 0:
            continue
        r, e_lo, e_hi = pool
        for s0 in range(0, sel.size, chunk):
            idx = sel[s0:s0 + chunk]
            lo_d = np.clip(lo_half[idx][:, None] + e_lo[None, :], 0.01, None)
            hi_d = np.clip(hi_half[idx][:, None] + e_hi[None, :], 0.01, None)
            rr = r[None, None, :]
            ss = grid[None, :, None]
            loss = (
                np.clip(ss - rr - hi_d[:, None, :], 0, None)
                + np.clip(rr - lo_d[:, None, :] - ss, 0, None)
            ).mean(axis=2)
            shifts[idx] = grid[loss.argmin(axis=1)]
    return shifts


def joint_pools(centre, y_true, lo_half_pred, hi_half_pred, lo_true, hi_true, bins, exclude=None):
    """Residual and band-half error drawn as one triple per compound, per bin."""
    keep = np.ones(len(y_true), bool) if exclude is None else ~exclude
    resid = y_true - centre
    e_lo = (y_true - lo_true) - lo_half_pred
    e_hi = (hi_true - y_true) - hi_half_pred
    out = {}
    for b in np.unique(bins):
        m = (bins == b) & keep
        out[b] = (resid[m], e_lo[m], e_hi[m])
    return out
