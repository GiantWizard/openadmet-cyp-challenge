"""TabPFN on frozen embeddings plus the stacked qHTS probabilities.

The trees only carry 30 to 50 percent of the blend, so giving them the qHTS features
moved little. TabPFN is the dominant component. Holdout arm only, to see whether this
is worth the full out-of-fold run.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path("clean/cache")
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def soft(p, lo, hi):
    return np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)


def st_rae(y, p, lo, hi):
    return float(soft(p, lo, hi).sum() / soft(np.full_like(y, y.mean()), lo, hi).sum())


def main() -> None:
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    from tabpfn import TabPFNRegressor
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    hold = np.load(CACHE / "holdout_mask.npy")
    E = np.load(CACHE / "emb_medium_inhib.npy")
    Q = np.load(CACHE / "qhts_tr.npy")
    res = {}
    for iso in ISOFORMS:
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        dev, ho = np.flatnonzero(obs & ~hold), np.flatnonzero(obs & hold)
        for tag, src in (("emb", E), ("emb+qhts", np.hstack([E, Q]))):
            n = min(200, len(dev), src.shape[1])
            pipe = Pipeline([("i", SimpleImputer(strategy="median")),
                             ("s", StandardScaler()),
                             ("p", PCA(n_components=n, random_state=42))])
            Xd = pipe.fit_transform(src[dev]).astype(np.float32)
            Xh = pipe.transform(src[ho]).astype(np.float32)
            t0 = time.time()
            m = TabPFNRegressor(device="cpu", random_state=42,
                                ignore_pretraining_limits=True,
                                fit_mode="low_memory", memory_saving_mode=True, n_jobs=2)
            m.fit(Xd, y[dev])
            p = m.predict(Xh)
            s = st_rae(y[ho], p, lo[ho], hi[ho])
            res.setdefault(tag, []).append(s)
            print(f"  {iso} {tag:9s}: holdout ST-RAE {s:.4f} "
                  f"r={pearsonr(y[ho], p)[0]:.3f} rho={spearmanr(y[ho], p)[0]:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    for k, v in res.items():
        print(f"MACRO {k:9s} {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
