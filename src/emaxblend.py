"""Does the CYP2D6 Emax route survive on top of the TabPFN blend?

Trees alone are the weak model, so a gain there may not carry. Here the Emax-route
prediction enters as a third blend component, with weights chosen on dev
out-of-fold rows and the holdout scored once.
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


def main() -> None:
    iso = sys.argv[1]
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    X = np.hstack([np.load(CACHE / "X_train.npy"),
                   np.load(CACHE / "aux_train.npy")]).astype(np.float32)
    G = np.load(CACHE / "scaffold_groups.npy", allow_pickle=True)
    hold = holdout_mask()
    c = f"{iso}_pIC50_direct_inhibition"
    y = meta[c].to_numpy(float)
    lo = meta[f"{c}_conf_low"].to_numpy(float)
    hi = meta[f"{c}_conf_high"].to_numpy(float)
    obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    dev, ho = np.flatnonzero(obs & ~hold), np.flatnonzero(obs & hold)

    em = pd.read_csv("data/cyp-challenge-TRAIN_Emax.csv")
    col = f"{iso}_EmaxVsPosCtrl_direct_inhibition"
    mp = dict(zip(em["SMILES"], pd.to_numeric(em[col], errors="coerce")))
    E = np.array([mp.get(s, np.nan) for s in meta["SMILES"]], float)
    ok = np.isfinite(E)
    inc = bool(pearsonr(E[obs & ok], y[obs & ok])[0] > 0)

    # Emax route, out-of-fold on dev and once for the holdout.
    def fit_emax(rows_fit, rows_pred):
        r = rows_fit[np.isfinite(E[rows_fit])]
        acc = 0.0
        for s in SEEDS:
            m = _fit(X[r], E[r], dict(P, objective="regression"), ROUNDS, s)
            acc = acc + m.predict(X[rows_pred]) / len(SEEDS)
        return acc

    oof_e = np.full(len(dev), np.nan)
    for tr_j, va_j in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
        oof_e[va_j] = fit_emax(dev[tr_j], dev[va_j])
    ho_e = fit_emax(dev, ho)
    imap = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(oof_e, y[dev])
    # dev-side Emax route must itself be out-of-fold through the map
    dev_route = np.full(len(dev), np.nan)
    for tr_j, va_j in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
        m2 = IsotonicRegression(increasing=inc, out_of_bounds="clip").fit(
            oof_e[tr_j], y[dev[tr_j]])
        dev_route[va_j] = m2.predict(oof_e[va_j])
    ho_route = imap.predict(ho_e)

    tz = np.load(CACHE / f"trees_parts_{iso}.npz")
    zt = np.load(CACHE / f"tabpfn_oof_{iso}.npz")
    zh = np.load(CACHE / f"tabpfn_medium_{iso}.npz")
    assert (zt["dev"] == dev).all() and (zh["ho"] == ho).all()
    dev_tre, ho_tre = tz["oof"], tz["hold"]
    dev_tab, ho_tab = zt["oof"], zh["hold"]

    yd, lod, hid = y[dev], lo[dev], hi[dev]
    yh, loh, hih = y[ho], lo[ho], hi[ho]
    # two-stage weights, both chosen on dev only
    w1 = float(min(W, key=lambda w: st_rae(yd, w * dev_tab + (1 - w) * dev_tre, lod, hid)))
    base_d = w1 * dev_tab + (1 - w1) * dev_tre
    base_h = w1 * ho_tab + (1 - w1) * ho_tre
    w2 = float(min(W, key=lambda w: st_rae(yd, w * dev_route + (1 - w) * base_d, lod, hid)))
    final_h = w2 * ho_route + (1 - w2) * base_h
    print(f"{iso}: w_tabpfn={w1:.2f} w_emaxroute={w2:.2f} (both from dev)")
    print(f"  dev   base {st_rae(yd, base_d, lod, hid):.4f} "
          f"+route {st_rae(yd, w2*dev_route+(1-w2)*base_d, lod, hid):.4f}")
    a = st_rae(yh, base_h, loh, hih); b = st_rae(yh, final_h, loh, hih)
    print(f"  HOLDOUT base {a:.4f} r={pearsonr(yh, base_h)[0]:.3f}")
    print(f"  HOLDOUT +route {b:.4f} r={pearsonr(yh, final_h)[0]:.3f}  delta {b-a:+.4f}")


if __name__ == "__main__":
    main()
