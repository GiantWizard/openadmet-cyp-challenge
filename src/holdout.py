"""Locked holdout: a fixed slice of scaffold groups, scored once at the very end."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import CACHE, scaffold_groups  # noqa: E402

FRACTION = 0.15
SEED = 20261010


def holdout_mask() -> np.ndarray:
    """Boolean over the 4,905 training rows; True means locked away."""
    f = CACHE / "holdout_mask.npy"
    if f.exists():
        return np.load(f)
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    groups = scaffold_groups(meta["SMILES"])
    uniq = np.unique(groups)
    rng = np.random.default_rng(SEED)
    locked = set(rng.choice(uniq, int(round(len(uniq) * FRACTION)), replace=False))
    mask = np.array([g in locked for g in groups])
    np.save(f, mask)
    return mask


if __name__ == "__main__":
    m = holdout_mask()
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    print(f"locked {m.sum()} of {len(m)} compounds ({m.mean():.1%})")
    for iso in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
        lab = meta[f"{iso}_pIC50_direct_inhibition"].notna().to_numpy()
        print(f"  {iso}: {int((lab & m).sum())} labelled in holdout, "
              f"{int((lab & ~m).sum())} available for development")
