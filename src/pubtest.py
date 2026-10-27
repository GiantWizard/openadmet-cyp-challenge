"""Do Veith qHTS rows improve the challenge regression? Dev picks the weight,
the locked holdout is scored once.

Public actives become a band of +/-0.6 around their bias-corrected pIC50, which is the
cross-assay residual spread implied by the agreement measured on shared compounds.
Censored rows become an interval open downward below the top dose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from holdout import holdout_mask  # noqa: E402
from metric import st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, N_FOLDS, _fit  # noqa: E402

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
# Veith reads higher than Octant; measured on shared training compounds.
BIAS = {"CYP2C9": 0.40, "CYP3A4": 0.34, "CYP2D6": 0.45}
HALF = 0.6          # cross-assay residual spread
TOP = 4.40
P = dict(BASE_PARAMS, learning_rate=0.02, num_leaves=31, feature_fraction=0.25)
ROUNDS, SEEDS = 900, (0, 1)
WP = [0.0, 0.1, 0.3, 0.6, 1.0]


def wband(lo, hi, w, delta=0.5):
    def obj(preds, dataset):
        del dataset
        excess = np.clip(preds - hi, 0, None) - np.clip(lo - preds, 0, None)
        absx = np.abs(excess)
        grad = np.where(absx <= delta, excess, delta * np.sign(excess)) * w
        hess = np.where(absx <= delta, 1.0, 1e-3) * w
        return grad, hess
    return obj


def main() -> None:
    iso = sys.argv[1]
    actives_only = "--actives" in sys.argv
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    X = np.load(CACHE / "X_train.npy").astype(np.float32)
    G = np.load(CACHE / "scaffold_groups.npy", allow_pickle=True)
    hold = holdout_mask()
    c = f"{iso}_pIC50_direct_inhibition"
    y = meta[c].to_numpy(float)
    lo = meta[f"{c}_conf_low"].to_numpy(float)
    hi = meta[f"{c}_conf_high"].to_numpy(float)
    obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    dev, ho = np.flatnonzero(obs & ~hold), np.flatnonzero(obs & hold)

    pm = pd.read_parquet(CACHE / "pub_meta.parquet")
    PX = np.load(CACHE / "pub_X.npy")
    if iso not in pm.columns:
        print(f"{iso}: no public labels"); return
    v = pm[iso].to_numpy(float)
    sel = np.flatnonzero(pd.notna(pm[iso]))
    if actives_only:
        sel = sel[np.isfinite(v[sel])]
    b = BIAS.get(iso, 0.40)
    plo, phi = np.empty(len(sel)), np.empty(len(sel))
    for i, j in enumerate(sel):
        if np.isfinite(v[j]):
            plo[i], phi[i] = v[j] - b - HALF, v[j] - b + HALF
        else:
            plo[i], phi[i] = -5.0, TOP - b
    PXs = PX[sel]
    print(f"{iso}: dev {len(dev)} holdout {len(ho)} public {len(sel)} "
          f"({int(np.isfinite(v[sel]).sum())} with AC50)", flush=True)

    base = float(y[dev].mean())
    results = {}
    for wp in WP:
        oof = np.full(len(dev), np.nan)
        for tr_j, va_j in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
            tr = dev[tr_j]
            if wp > 0:
                xx = np.vstack([X[tr], PXs])
                ll = np.concatenate([lo[tr], plo]); hh = np.concatenate([hi[tr], phi])
                ww = np.concatenate([np.ones(len(tr)), np.full(len(sel), wp)])
            else:
                xx, ll, hh, ww = X[tr], lo[tr], hi[tr], np.ones(len(tr))
            acc = 0.0
            for s in SEEDS:
                m = _fit(xx, np.zeros(len(xx)), P, ROUNDS, s, obj=wband(ll, hh, ww),
                         init_score=np.full(len(xx), base))
                acc = acc + (m.predict(X[dev[va_j]]) + base) / len(SEEDS)
            oof[va_j] = acc
        sc = st_rae(y[dev], oof, lo[dev], hi[dev])
        results[wp] = sc
        print(f"  wp={wp:<5} dev OOF ST-RAE {sc:.4f}  r={pearsonr(y[dev], oof)[0]:.3f}",
              flush=True)

    bestw = min(results, key=results.get)
    print(f"  dev picks wp={bestw}", flush=True)
    out = {}
    for tag, wp in (("baseline", 0.0), ("public", bestw)):
        if wp > 0:
            xx = np.vstack([X[dev], PXs])
            ll = np.concatenate([lo[dev], plo]); hh = np.concatenate([hi[dev], phi])
            ww = np.concatenate([np.ones(len(dev)), np.full(len(sel), wp)])
        else:
            xx, ll, hh, ww = X[dev], lo[dev], hi[dev], np.ones(len(dev))
        acc = 0.0
        for s in SEEDS:
            m = _fit(xx, np.zeros(len(xx)), P, ROUNDS, s, obj=wband(ll, hh, ww),
                     init_score=np.full(len(xx), base))
            acc = acc + (m.predict(X[ho]) + base) / len(SEEDS)
        out[tag] = (st_rae(y[ho], acc, lo[ho], hi[ho]), pearsonr(y[ho], acc)[0])
    print(f"  HOLDOUT baseline {out['baseline'][0]:.4f} r={out['baseline'][1]:.3f}")
    print(f"  HOLDOUT +public  {out['public'][0]:.4f} r={out['public'][1]:.3f}  "
          f"(wp={bestw}) delta {out['public'][0]-out['baseline'][0]:+.4f}")


if __name__ == "__main__":
    main()
