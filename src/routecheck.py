"""Does the Emax route still pay once the dev-fitted calibration is applied after it?

The route reshapes the prediction distribution and so does the calibration, so a gain
before calibration can vanish after it. Both arms get identical treatment: weights and
calibration from dev, holdout scored once.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from holdout import holdout_mask  # noqa: E402
from metric import st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, N_FOLDS, _fit  # noqa: E402

P = dict(BASE_PARAMS, learning_rate=0.02, num_leaves=31, feature_fraction=0.25)
ROUNDS, SEEDS = 900, (0, 1)
W = np.round(np.arange(0, 1.01, 0.05), 2)
DS = np.arange(-0.6, 0.81, 0.05)
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def cal(yd, zd, lod, hid, yh, zh, loh, hih):
    """Fit scale at rho and shift on dev; return the holdout score."""
    k = float(np.clip(pearsonr(yd, zd)[0], 0.3, 0.95))
    u = (zd - zd.mean()) / zd.std()
    d = float(DS[int(np.argmin([st_rae(yd, yd.mean() + q + u * k * yd.std(), lod, hid)
                                for q in DS]))])
    v = (zh - zh.mean()) / zh.std()
    return st_rae(yh, yh.mean() + d + v * k * yh.std(), loh, hih), k, d


def main() -> None:
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    X = np.hstack([np.load(CACHE / "X_train.npy"),
                   np.load(CACHE / "aux_train.npy")]).astype(np.float32)
    G = np.load(CACHE / "scaffold_groups.npy", allow_pickle=True)
    hold = holdout_mask()
    emx = pd.read_csv("data/cyp-challenge-TRAIN_Emax.csv")
    A, B = [], []
    for iso in ISO:
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        dev, ho = np.flatnonzero(obs & ~hold), np.flatnonzero(obs & hold)
        tz = np.load(CACHE / f"trees_parts_{iso}.npz")
        zt = np.load(CACHE / f"tabpfn_oof_{iso}.npz")
        zh = np.load(CACHE / f"tabpfn_medium_{iso}.npz")
        yd, lod, hid = y[dev], lo[dev], hi[dev]
        yh, loh, hih = y[ho], lo[ho], hi[ho]
        w = float(min(W, key=lambda q: st_rae(
            yd, q * zt["oof"] + (1 - q) * tz["oof"], lod, hid)))
        bd = w * zt["oof"] + (1 - w) * tz["oof"]
        bh = w * zh["hold"] + (1 - w) * tz["hold"]

        col = f"{iso}_EmaxVsPosCtrl_direct_inhibition"
        mp = dict(zip(emx["SMILES"], pd.to_numeric(emx[col], errors="coerce")))
        Ev = np.array([mp.get(s, np.nan) for s in meta["SMILES"]], float)
        eok = np.isfinite(Ev)
        inc = bool(pearsonr(Ev[obs & eok], y[obs & eok])[0] > 0)

        def _e(rf, xp):
            r = rf[np.isfinite(Ev[rf])]
            a = 0.0
            for s in SEEDS:
                a = a + _fit(X[r], Ev[r], dict(P, objective="regression"),
                             ROUNDS, s).predict(xp) / len(SEEDS)
            return a

        oe = np.full(len(dev), np.nan)
        rt = np.full(len(dev), np.nan)
        for tr_j, va_j in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
            oe[va_j] = _e(dev[tr_j], X[dev[va_j]])
        for tr_j, va_j in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
            m = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(
                oe[tr_j], y[dev[tr_j]])
            rt[va_j] = m.predict(oe[va_j])
        hr = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(
            oe, yd).predict(_e(dev, X[ho]))
        w2 = float(min(W, key=lambda q: st_rae(yd, q * rt + (1 - q) * bd, lod, hid)))

        a, ka, da = cal(yd, bd, lod, hid, yh, bh, loh, hih)
        b, kb, db = cal(yd, w2 * rt + (1 - w2) * bd, lod, hid,
                        yh, w2 * hr + (1 - w2) * bh, loh, hih)
        A.append(a); B.append(b)
        print(f"{iso}: w2={w2:.2f} | calibrated holdout  no-route {a:.4f}  "
              f"with-route {b:.4f}  delta {b-a:+.4f}", flush=True)
    print(f"\nMACRO calibrated, no route   {np.mean(A):.4f}")
    print(f"MACRO calibrated, with route {np.mean(B):.4f}   delta {np.mean(B)-np.mean(A):+.4f}")


if __name__ == "__main__":
    main()
