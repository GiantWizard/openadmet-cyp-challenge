"""qHTS-augmented trees inside the TabPFN blend. Weights and calibration from dev,
holdout scored once. Caches the tree parts so the blind submission can reuse them.
"""
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
ROUNDS, SEEDS = 900, (0, 1, 2)
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
W = np.round(np.arange(0, 1.01, 0.05), 2)
DS = np.arange(-0.6, 0.81, 0.05)


def cal(yd, zd, lod, hid, yh, zh, loh, hih):
    k = float(np.clip(pearsonr(yd, zd)[0], 0.3, 0.95))
    u = (zd - zd.mean()) / zd.std()
    d = float(DS[int(np.argmin([st_rae(yd, yd.mean() + q + u * k * yd.std(), lod, hid)
                                for q in DS]))])
    v = (zh - zh.mean()) / zh.std()
    return st_rae(yh, yh.mean() + d + v * k * yh.std(), loh, hih), k, d


def main() -> None:
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    X = np.hstack([np.load(CACHE / "X_train.npy"), np.load(CACHE / "aux_train.npy"),
                   np.load(CACHE / "qhts_tr.npy")]).astype(np.float32)
    XT = np.hstack([np.load(CACHE / "X_test.npy"), np.load(CACHE / "aux_test.npy"),
                    np.load(CACHE / "qhts_te.npy")]).astype(np.float32)
    G = np.load(CACHE / "scaffold_groups.npy", allow_pickle=True)
    hold = holdout_mask()
    old, new = [], []
    for iso in ISO:
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        dev, ho = np.flatnonzero(obs & ~hold), np.flatnonzero(obs & hold)
        yd, lod, hid = y[dev], lo[dev], hi[dev]
        yh, loh, hih = y[ho], lo[ho], hi[ho]

        f = CACHE / f"trees_qhts_{iso}.npz"
        if f.exists():
            z = np.load(f); toof, tho, tte = z["oof"], z["hold"], z["test"]
        else:
            toof = np.full(len(dev), np.nan); tho = np.zeros(len(ho))
            for tr_j, va_j in GroupKFold(N_FOLDS).split(dev, dev, G[dev]):
                tr = dev[tr_j]; b = float(y[tr].mean()); av = 0.0; ah = 0.0
                for s in SEEDS:
                    m = _fit(X[tr], y[tr], P, ROUNDS, s,
                             obj=band_objective(lo[tr], hi[tr]),
                             init_score=np.full(len(tr), b))
                    av = av + (m.predict(X[dev[va_j]]) + b) / len(SEEDS)
                    ah = ah + (m.predict(X[ho]) + b) / len(SEEDS)
                toof[va_j] = av; tho += ah / N_FOLDS
            allr = np.flatnonzero(obs); b = float(y[allr].mean()); tte = np.zeros(len(XT))
            for s in SEEDS:
                m = _fit(X[allr], y[allr], P, ROUNDS, s,
                         obj=band_objective(lo[allr], hi[allr]),
                         init_score=np.full(len(allr), b))
                tte += (m.predict(XT) + b) / len(SEEDS)
            np.savez_compressed(f, oof=toof, hold=tho, test=tte)

        oz = np.load(CACHE / f"trees_parts_{iso}.npz")
        zt = np.load(CACHE / f"tabpfn_oof_{iso}.npz")
        zh = np.load(CACHE / f"tabpfn_medium_{iso}.npz")
        for tag, (do, hp) in {"old trees": (oz["oof"], oz["hold"]),
                              "qhts trees": (toof, tho)}.items():
            w = float(min(W, key=lambda q: st_rae(yd, q * zt["oof"] + (1 - q) * do,
                                                  lod, hid)))
            bd = w * zt["oof"] + (1 - w) * do
            bh = w * zh["hold"] + (1 - w) * hp
            sc, k, d = cal(yd, bd, lod, hid, yh, bh, loh, hih)
            (old if tag == "old trees" else new).append(sc)
            print(f"  {iso} {tag:11s}: w_tabpfn={w:.2f} trees-alone-ho "
                  f"{st_rae(yh, hp, loh, hih):.4f} | blend cal {sc:.4f} "
                  f"rho={spearmanr(yh, bh)[0]:.3f}", flush=True)
    print(f"\nMACRO blend, old trees  {np.mean(old):.4f}")
    print(f"MACRO blend, qhts trees {np.mean(new):.4f}  delta {np.mean(new)-np.mean(old):+.4f}")


if __name__ == "__main__":
    main()
