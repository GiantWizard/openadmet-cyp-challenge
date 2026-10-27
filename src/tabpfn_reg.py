"""TabPFN on frozen CYP-pretrained embeddings, scored on the locked holdout.

Resumable: an isoform whose .npz already exists is reloaded rather than refitted.
Inference is chunked because the first pass thrashed swap on this machine; the
ensemble size is unchanged so results stay comparable across isoforms.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CACHE = Path("clean/cache")
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def soft(p, lo, hi):
    return np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)


def st_rae(y, p, lo, hi):
    return float(soft(p, lo, hi).sum() / soft(np.full_like(y, y.mean()), lo, hi).sum())


def pca_pipe(a, b, n=200):
    n = min(n, a.shape[0], a.shape[1])
    pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("p", PCA(n_components=n, random_state=42))])
    return pipe.fit_transform(a).astype(np.float32), pipe.transform(b).astype(np.float32)


def main() -> None:
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    from tabpfn import TabPFNRegressor
    dev = "cpu"
    which = sys.argv[1] if len(sys.argv) > 1 else "medium"
    want = sys.argv[2:] or ISOFORMS
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    hold = np.load(CACHE / "holdout_mask.npy")
    E = np.load(CACHE / f"emb_{which}_inhib.npy")
    ET = np.load(CACHE / f"emb_{which}_test.npy")
    print(f"device {dev} | embeddings {which} {E.shape}", flush=True)

    def mk():
        return TabPFNRegressor(device=dev, random_state=42, ignore_pretraining_limits=True,
                               fit_mode="low_memory", memory_saving_mode=True, n_jobs=2)

    res = []
    for iso in ISOFORMS:
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        out = CACHE / f"tabpfn_{which}_{iso}.npz"
        if out.exists() and iso not in sys.argv[2:]:
            d = np.load(out)
            ho, p = d["ho"], d["hold"]
            s, r = st_rae(y[ho], p, lo[ho], hi[ho]), pearsonr(y[ho], p)[0]
            res.append((s, r))
            print(f"  {iso}: cached  ST-RAE {s:.4f}  Pearson {r:.3f}", flush=True)
            continue
        if iso not in want:
            continue
        dev_rows = np.flatnonzero(obs & ~hold)
        ho = np.flatnonzero(obs & hold)
        Xd, Xh = pca_pipe(E[dev_rows], E[ho])
        t0 = time.time()
        m = mk()
        m.fit(Xd, y[dev_rows])
        p = m.predict(Xh)
        s, r = st_rae(y[ho], p, lo[ho], hi[ho]), pearsonr(y[ho], p)[0]
        res.append((s, r))
        print(f"  {iso}: holdout ST-RAE {s:.4f}  Pearson {r:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
        # Refit on every labelled compound for the blind set.
        allr = np.flatnonzero(obs)
        Xa, Xt = pca_pipe(E[allr], ET)
        m2 = mk()
        m2.fit(Xa, y[allr])
        tp = m2.predict(Xt)
        np.savez_compressed(out, hold=p, ho=ho, test=tp)
        print(f"  {iso}: blind written ({time.time()-t0:.0f}s total)", flush=True)
    if len(res) == 4:
        print(f"\nMACRO ST-RAE {np.mean([a for a, _ in res]):.4f}  "
              f"Pearson {np.mean([b for _, b in res]):.3f}", flush=True)


if __name__ == "__main__":
    main()
