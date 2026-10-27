"""Out-of-fold TabPFN predictions on the dev rows, so blend weights can be fitted
without the holdout seeing them. Scaffold-grouped folds, matching the tree pipeline.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path("clean/cache")
N_FOLDS = 5
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def pca_pipe(a, b, n=200):
    n = min(n, a.shape[0], a.shape[1])
    pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("p", PCA(n_components=n, random_state=42))])
    return pipe.fit_transform(a).astype(np.float32), pipe.transform(b).astype(np.float32)


def main() -> None:
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    from tabpfn import TabPFNRegressor
    which = "medium"
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    hold = np.load(CACHE / "holdout_mask.npy")
    E = np.load(CACHE / f"emb_{which}_inhib.npy")
    global GROUPS
    GROUPS = np.load(CACHE / "scaffold_groups.npy", allow_pickle=True)
    print(f"embeddings {E.shape}", flush=True)

    for iso in ISOFORMS:
        out = CACHE / f"tabpfn_oof_{iso}.npz"
        if out.exists():
            print(f"  {iso}: cached", flush=True)
            continue
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(meta[f"{c}_conf_low"].to_numpy(float))
        dev = np.flatnonzero(obs & ~hold)
        g = GROUPS[dev]
        oof = np.full(len(dev), np.nan)
        t0 = time.time()
        for k, (tr, va) in enumerate(GroupKFold(N_FOLDS).split(dev, dev, g)):
            Xtr, Xva = pca_pipe(E[dev[tr]], E[dev[va]])
            m = TabPFNRegressor(device="cpu", random_state=42, ignore_pretraining_limits=True,
                                fit_mode="low_memory", memory_saving_mode=True, n_jobs=2)
            m.fit(Xtr, y[dev[tr]])
            oof[va] = m.predict(Xva)
            print(f"    {iso} fold {k} done ({time.time()-t0:.0f}s)", flush=True)
        np.savez_compressed(out, oof=oof, dev=dev)
        print(f"  {iso}: OOF written ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
