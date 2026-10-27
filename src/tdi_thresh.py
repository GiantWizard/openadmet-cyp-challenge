"""Optimal TDI calling rate under the BLIND prevalence, not the training one.

The TDI training and holdout sets are about 0.22 positive on CYP2D6 while the blind set
is 0.0707, so an MCC measured at a given calling rate on our own data does not transfer.
Here the ROC is estimated from held-out predictions and MCC is computed analytically at
the blind prevalence, which is what the leaderboard actually scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from pipeline import BASE_PARAMS, CACHE, _fit, scaffold_groups  # noqa: E402

CP = dict(BASE_PARAMS, objective="binary", learning_rate=0.03, num_leaves=31,
          min_data_in_leaf=20, feature_fraction=0.3)
ROUNDS, SEEDS = 600, (0, 1)
BLIND = {"CYP2D6": 0.0707, "CYP3A4": 0.2915}
SHIPPED = {"CYP2D6": 0.196, "CYP3A4": 0.320}


def mcc_from_rates(tpr, fpr, p):
    """MCC implied by an operating point at prevalence p."""
    tp, fn = tpr * p, (1 - tpr) * p
    fp, tn = fpr * (1 - p), (1 - fpr) * (1 - p)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)


def main() -> None:
    tdi = pd.read_csv("data/cyp-challenge-TRAIN_TDI.csv")
    X = np.load(CACHE / "X_tdi_train.npy").astype(np.float32)
    g = np.asarray(scaffold_groups(tdi["SMILES"]))
    for iso, p in BLIND.items():
        y = pd.to_numeric(tdi[f"{iso}_is_TDI"], errors="coerce").to_numpy(float)
        obs = np.flatnonzero(np.isfinite(y))
        oof = np.full(len(obs), np.nan)
        for tr_j, va_j in GroupKFold(5).split(obs, obs, g[obs]):
            tr = obs[tr_j]
            a = 0.0
            for s in SEEDS:
                a = a + _fit(X[tr], y[tr], CP, ROUNDS, s).predict(X[obs[va_j]]) / len(SEEDS)
            oof[va_j] = a
        fpr, tpr, thr = roc_curve(y[obs], oof)
        m = mcc_from_rates(tpr, fpr, p)
        k = int(np.nanargmax(m))
        call = tpr[k] * p + fpr[k] * (1 - p)
        # what our shipped rate implies
        q = np.quantile(oof, 1 - SHIPPED[iso])
        j = int(np.argmin(np.abs(thr - q)))
        call_ship = tpr[j] * p + fpr[j] * (1 - p)
        print(f"{iso}: blind prevalence {p:.4f}  our OOF AUC "
              f"{np.trapezoid(tpr, fpr):.4f}  (n={len(obs)})")
        print(f"   best achievable MCC at this prevalence  {m[k]:.4f} "
              f"at calling rate {call:.3f} (tpr {tpr[k]:.3f} fpr {fpr[k]:.3f})")
        print(f"   MCC at our shipped rate {SHIPPED[iso]:.3f}        {m[j]:.4f} "
              f"(effective calls {call_ship:.3f})")
        print(f"   gain available from retuning            {m[k]-m[j]:+.4f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
