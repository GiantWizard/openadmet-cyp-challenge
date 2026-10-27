"""Is the stacked-arm AUC gain real, or an artifact of mismatched fold splits?

Everything is rebuilt inside a locked holdout: the arm models see only development
compounds, the classifier sees only development compounds, and the holdout is scored
once. If the gain survives this it is real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from pipeline import BASE_PARAMS, CACHE, _fit, scaffold_groups  # noqa: E402

CP = dict(BASE_PARAMS, objective="binary", learning_rate=0.03, num_leaves=31,
          min_data_in_leaf=20, feature_fraction=0.3)
ROUNDS, SEEDS = 600, (0, 1)
ISO = ["CYP2D6", "CYP3A4"]
ARMS = ["pIC50_direct_inhibition", "pIC50_TDI_condition"]
HOLD_FRAC = 0.20


def main() -> None:
    tdi = pd.read_csv("data/cyp-challenge-TRAIN_TDI.csv")
    X = np.load(CACHE / "X_tdi_train.npy").astype(np.float32)
    Q = np.load(CACHE / "qhts_tdi.npy")
    g = np.asarray(scaffold_groups(tdi["SMILES"]))
    rng = np.random.default_rng(11)
    ug = np.unique(g)
    hg = set(rng.choice(ug, int(len(ug) * HOLD_FRAC), replace=False))
    hold = np.array([x in hg for x in g])
    print(f"locked holdout: {int(hold.sum())} of {len(g)} rows\n", flush=True)

    # Arm features: fitted ONLY on development rows, applied to both halves.
    A = np.full((len(tdi), len(ISO) * len(ARMS)), np.nan)
    col = 0
    for iso in ISO:
        for arm in ARMS:
            y = pd.to_numeric(tdi[f"{iso}_{arm}"], errors="coerce").to_numpy(float)
            ok = np.isfinite(y)
            dv = np.flatnonzero(ok & ~hold)
            oof = np.full(len(dv), np.nan)
            for tr_i, va_i in GroupKFold(5).split(dv, dv, g[dv]):
                oof[va_i] = _fit(X[dv[tr_i]], y[dv[tr_i]], BASE_PARAMS, 400, 0
                                 ).predict(X[dv[va_i]])
            A[dv, col] = oof
            full = _fit(X[dv], y[dv], BASE_PARAMS, 400, 0)
            A[np.flatnonzero(hold), col] = full.predict(X[hold])
            col += 1
    for j in range(len(ISO)):
        d, t = 2 * j, 2 * j + 1
        A = np.column_stack([A, A[:, t] - A[:, d]])
    A = np.nan_to_num(A, nan=-999)

    sets = {"base": X,
            "base+arms": np.hstack([X, A]).astype(np.float32),
            "base+arms+qhts": np.hstack([X, A, Q]).astype(np.float32)}
    for iso in ISO:
        y = pd.to_numeric(tdi[f"{iso}_is_TDI"], errors="coerce").to_numpy(float)
        ok = np.isfinite(y)
        dv, ho = np.flatnonzero(ok & ~hold), np.flatnonzero(ok & hold)
        print(f"{iso}: dev {len(dv)} holdout {len(ho)}", flush=True)
        for tag, XX in sets.items():
            hp = 0.0
            for s in SEEDS:
                hp = hp + _fit(XX[dv], y[dv], CP, ROUNDS, s).predict(XX[ho]) / len(SEEDS)
            print(f"   {tag:16s} HOLDOUT AUC {roc_auc_score(y[ho], hp):.4f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
