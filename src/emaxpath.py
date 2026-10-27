"""Is Emax a better route to pIC50 than predicting pIC50 directly?

CYP2D6 is read out by mass spec rather than fluorescence, which inverts its
Emax-to-pIC50 sign and makes that relationship much tighter (Spearman 0.770) than on
the other three. If Emax is also easier to learn from structure, the composition
structure -> Emax -> pIC50 can beat structure -> pIC50. Scored on the locked holdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from holdout import holdout_mask  # noqa: E402
from metric import band_objective, st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, N_FOLDS, _fit  # noqa: E402

P = dict(BASE_PARAMS, learning_rate=0.02, num_leaves=31, feature_fraction=0.25)
ROUNDS, SEEDS = 900, (0, 1)


def main() -> None:
    iso = sys.argv[1]
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    X = np.hstack([np.load(CACHE / "X_train.npy"),
                   np.load(CACHE / "aux_train.npy")]).astype(np.float32)
    G = np.load(CACHE / "scaffold_groups.npy", allow_pickle=True)
    hold = holdout_mask()

    em = pd.read_csv("data/cyp-challenge-TRAIN_Emax.csv")
    ecol = f"{iso}_EmaxVsPosCtrl_direct_inhibition"
    emap = dict(zip(em["SMILES"], pd.to_numeric(em[ecol], errors="coerce")))
    E = np.array([emap.get(s, np.nan) for s in meta["SMILES"]], float)

    c = f"{iso}_pIC50_direct_inhibition"
    y = meta[c].to_numpy(float)
    lo = meta[f"{c}_conf_low"].to_numpy(float)
    hi = meta[f"{c}_conf_high"].to_numpy(float)
    obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(E)
    dev, ho = np.flatnonzero(obs & ~hold), np.flatnonzero(obs & hold)
    print(f"{iso}: dev {len(dev)} holdout {len(ho)} | "
          f"Emax-pIC50 rho {spearmanr(y[obs], E[obs])[0]:+.3f}", flush=True)

    # Route 1, direct: structure -> pIC50, using the band objective.
    base = float(y[dev].mean())
    dp = np.zeros(len(ho))
    for s in SEEDS:
        m = _fit(X[dev], y[dev], P, ROUNDS, s, obj=band_objective(lo[dev], hi[dev]),
                 init_score=np.full(len(dev), base))
        dp += (m.predict(X[ho]) + base) / len(SEEDS)

    # Route 2, via Emax: structure -> Emax on L2, then a monotone Emax -> pIC50 map
    # fitted out-of-fold on dev so the mapping never sees its own training rows.
    eb = float(E[dev].mean())
    oof_e = np.full(len(dev), np.nan)
    for tr_j, va_j in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
        tr = dev[tr_j]
        acc = 0.0
        for s in SEEDS:
            m = _fit(X[tr], E[tr], dict(P, objective="regression"), ROUNDS, s)
            acc = acc + m.predict(X[dev[va_j]]) / len(SEEDS)
        oof_e[va_j] = acc
    ep = np.zeros(len(ho))
    for s in SEEDS:
        m = _fit(X[dev], E[dev], dict(P, objective="regression"), ROUNDS, s)
        ep += m.predict(X[ho]) / len(SEEDS)
    inc = spearmanr(E[dev], y[dev])[0] > 0
    iso_map = IsotonicRegression(increasing=bool(inc), out_of_bounds="clip")
    iso_map.fit(oof_e, y[dev])
    vp = iso_map.predict(ep)

    # CONTROL: the direct prediction through the same isotonic map. If this matches
    # the via-Emax number, the gain is the output distribution, not Emax.
    oof_d = np.full(len(dev), np.nan)
    for tr_j, va_j in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
        tr = dev[tr_j]
        b2 = float(y[tr].mean())
        acc = 0.0
        for s in SEEDS:
            m = _fit(X[tr], y[tr], P, ROUNDS, s, obj=band_objective(lo[tr], hi[tr]),
                     init_score=np.full(len(tr), b2))
            acc = acc + (m.predict(X[dev[va_j]]) + b2) / len(SEEDS)
        oof_d[va_j] = acc
    ctrl_map = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(oof_d, y[dev])
    cp = ctrl_map.predict(dp)
    print(f"  Emax learnability: holdout r(Emax_true, Emax_pred) "
          f"{pearsonr(E[ho], ep)[0]:.3f}", flush=True)
    for tag, p in (("direct       ", dp), ("direct+isotonic", cp), ("via Emax     ", vp),
                   ("mean 50/50   ", 0.5 * dp + 0.5 * vp)):
        print(f"  {tag}: holdout ST-RAE {st_rae(y[ho], p, lo[ho], hi[ho]):.4f} "
              f"r={pearsonr(y[ho], p)[0]:.3f} rho={spearmanr(y[ho], p)[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
