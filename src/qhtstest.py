"""Do the stacked qHTS probabilities improve the blend? Dev picks, holdout scored once."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from holdout import holdout_mask  # noqa: E402
from metric import band_objective, st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, N_FOLDS, _fit  # noqa: E402

P = dict(BASE_PARAMS, learning_rate=0.02, num_leaves=31, feature_fraction=0.25)
ROUNDS, SEEDS = 900, (0, 1)
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
DS = np.arange(-0.6, 0.81, 0.05)


def cal(yd, zd, lod, hid, yh, zh, loh, hih):
    k = float(np.clip(pearsonr(yd, zd)[0], 0.3, 0.95))
    u = (zd - zd.mean()) / zd.std()
    d = float(DS[int(np.argmin([st_rae(yd, yd.mean() + q + u * k * yd.std(), lod, hid)
                                for q in DS]))])
    v = (zh - zh.mean()) / zh.std()
    return st_rae(yh, yh.mean() + d + v * k * yh.std(), loh, hih)


def main() -> None:
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    Xb = np.hstack([np.load(CACHE / "X_train.npy"),
                    np.load(CACHE / "aux_train.npy")]).astype(np.float32)
    Q = np.load(CACHE / "qhts_tr.npy")
    Xq = np.hstack([Xb, Q]).astype(np.float32)
    G = np.load(CACHE / "scaffold_groups.npy", allow_pickle=True)
    hold = holdout_mask()
    A, B = [], []
    for iso in ISO:
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        dev, ho = np.flatnonzero(obs & ~hold), np.flatnonzero(obs & hold)
        yd, lod, hid = y[dev], lo[dev], hi[dev]
        yh, loh, hih = y[ho], lo[ho], hi[ho]
        base = float(yd.mean())
        row = []
        for tag, X in (("plain", Xb), ("+qhts", Xq)):
            oof = np.full(len(dev), np.nan)
            for tr_j, va_j in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
                tr = dev[tr_j]; b = float(y[tr].mean()); a = 0.0
                for s in SEEDS:
                    m = _fit(X[tr], y[tr], P, ROUNDS, s,
                             obj=band_objective(lo[tr], hi[tr]),
                             init_score=np.full(len(tr), b))
                    a = a + (m.predict(X[dev[va_j]]) + b) / len(SEEDS)
                oof[va_j] = a
            a = 0.0
            for s in SEEDS:
                m = _fit(X[dev], y[dev], P, ROUNDS, s,
                         obj=band_objective(lod, hid),
                         init_score=np.full(len(dev), base))
                a = a + (m.predict(X[ho]) + base) / len(SEEDS)
            row.append((tag, st_rae(yd, oof, lod, hid), oof, a))
        for tag, dsc, oof, hp in row:
            print(f"  {iso} {tag:6s}: dev {dsc:.4f} | holdout raw "
                  f"{st_rae(yh, hp, loh, hih):.4f} calibrated "
                  f"{cal(yd, oof, lod, hid, yh, hp, loh, hih):.4f} "
                  f"rho={spearmanr(yh, hp)[0]:.3f}", flush=True)
        A.append(cal(yd, row[0][2], lod, hid, yh, row[0][3], loh, hih))
        B.append(cal(yd, row[1][2], lod, hid, yh, row[1][3], loh, hih))
    print(f"\nMACRO calibrated plain  {np.mean(A):.4f}")
    print(f"MACRO calibrated +qhts  {np.mean(B):.4f}  delta {np.mean(B)-np.mean(A):+.4f}")


if __name__ == "__main__":
    main()
