"""Similarity-weighted read-across: nearest neighbours by Tanimoto, for blend diversity."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout import holdout_mask  # noqa: E402
from metric import st_rae  # noqa: E402
from pipeline import CACHE, ISOFORMS, N_FOLDS, scaffold_groups  # noqa: E402

K = 10
POWER = 3.0      # sharpens the weighting toward the closest analogues
MIN_SIM = 0.25


def tanimoto(a_bits, b_bits, a_pop, b_pop):
    inter = a_bits @ b_bits.T
    return inter / (a_pop[:, None] + b_pop[None, :] - inter + 1e-9)


def predict(sim, y_ref, fallback):
    """Weighted mean of the K most similar reference compounds."""
    out = np.full(sim.shape[0], fallback)
    k = min(K, sim.shape[1])
    if k == 0:
        return out
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    s = sim[rows, idx]
    w = np.where(s >= MIN_SIM, s, 0.0) ** POWER
    tot = w.sum(1)
    hit = tot > 0
    out[hit] = (w[hit] * y_ref[idx][hit]).sum(1) / tot[hit]
    return out


def main() -> None:
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    bits = (np.load(CACHE / "X_train.npy")[:, :2048] > 0).astype(np.float32)
    pop = bits.sum(1)
    hold = holdout_mask()

    for iso in ISOFORMS:
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        dev = np.flatnonzero(obs & ~hold)
        ho = np.flatnonzero(obs & hold)
        groups = scaffold_groups(meta["SMILES"].iloc[dev])

        oof = np.zeros(len(dev))
        for tr_j, va_j in GroupKFold(n_splits=N_FOLDS).split(dev, dev, groups):
            tr, va = dev[tr_j], dev[va_j]
            sim = tanimoto(bits[va], bits[tr], pop[va], pop[tr])
            oof[va_j] = predict(sim, y[tr], float(y[tr].mean()))
        sim_ho = tanimoto(bits[ho], bits[dev], pop[ho], pop[dev])
        hp = predict(sim_ho, y[dev], float(y[dev].mean()))

        np.savez_compressed(CACHE / f"readacross_{iso}.npz", oof=oof, hold=hp,
                            dev=dev, ho=ho)
        print(f"  {iso}: dev OOF {st_rae(y[dev], oof, lo[dev], hi[dev]):.4f}", flush=True)


if __name__ == "__main__":
    main()
