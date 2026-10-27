"""TabPFN on embeddings plus qHTS probabilities: dev out-of-fold, holdout, and blind.

Mirrors tabpfn_oof.py and tabpfn_reg.py so the blend can swap this in for the
embedding-only parts. Resumable per isoform.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path("clean/cache")
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
N_FOLDS = 5


def soft(p, lo, hi):
    return np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)


def st_rae(y, p, lo, hi):
    return float(soft(p, lo, hi).sum() / soft(np.full_like(y, y.mean()), lo, hi).sum())


def prep(a, b, n=200):
    n = min(n, a.shape[0], a.shape[1])
    pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("p", PCA(n_components=n, random_state=42))])
    return pipe.fit_transform(a).astype(np.float32), pipe.transform(b).astype(np.float32)


def main() -> None:
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    from tabpfn import TabPFNRegressor
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    hold = np.load(CACHE / "holdout_mask.npy")
    G = np.load(CACHE / "scaffold_groups.npy", allow_pickle=True)
    S = np.hstack([np.load(CACHE / "emb_medium_inhib.npy"),
                   np.load(CACHE / "qhts_tr.npy")])
    ST = np.hstack([np.load(CACHE / "emb_medium_test.npy"),
                    np.load(CACHE / "qhts_te.npy")])
    print(f"features {S.shape} test {ST.shape}", flush=True)

    def mk():
        return TabPFNRegressor(device="cpu", random_state=42,
                               ignore_pretraining_limits=True, fit_mode="low_memory",
                               memory_saving_mode=True, n_jobs=2)

    want = sys.argv[1:] or ISOFORMS
    for iso in ISOFORMS:
        if iso not in want:
            continue
        out = CACHE / f"tabqhts_{iso}.npz"
        if out.exists():
            print(f"  {iso}: cached", flush=True)
            continue
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        dev, ho = np.flatnonzero(obs & ~hold), np.flatnonzero(obs & hold)
        t0 = time.time()

        oof = np.full(len(dev), np.nan)
        for k, (tr, va) in enumerate(GroupKFold(N_FOLDS).split(dev, dev, G[dev])):
            Xtr, Xva = prep(S[dev[tr]], S[dev[va]])
            m = mk(); m.fit(Xtr, y[dev[tr]])
            oof[va] = m.predict(Xva)
            print(f"    {iso} fold {k} ({time.time()-t0:.0f}s)", flush=True)

        Xd, Xh = prep(S[dev], S[ho])
        m = mk(); m.fit(Xd, y[dev])
        hp = m.predict(Xh)
        allr = np.flatnonzero(obs)
        Xa, Xt = prep(S[allr], ST)
        m2 = mk(); m2.fit(Xa, y[allr])
        tp = m2.predict(Xt)
        np.savez_compressed(out, oof=oof, dev=dev, hold=hp, ho=ho, test=tp)
        print(f"  {iso}: dev OOF {st_rae(y[dev], oof, lo[dev], hi[dev]):.4f} | "
              f"holdout {st_rae(y[ho], hp, lo[ho], hi[ho]):.4f} "
              f"r={pearsonr(y[ho], hp)[0]:.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
