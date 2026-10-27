"""MCC at blind prevalence across calling rates, with our shipped rates marked."""
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
MARK = {"CYP2D6": {"v2/v3": 0.196}, "CYP3A4": {"v3": 0.320, "v2": 0.4147}}


def mcc_at(tpr, fpr, p):
    tp, fn = tpr * p, (1 - tpr) * p
    fp, tn = fpr * (1 - p), (1 - fpr) * (1 - p)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)


def main() -> None:
    tdi = pd.read_csv("data/cyp-challenge-TRAIN_TDI.csv")
    X = np.load(CACHE / "X_tdi_train.npy").astype(np.float32)
    g = np.asarray(scaffold_groups(tdi["SMILES"]))
    out = {}
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
        m = mcc_at(tpr, fpr, p)
        calls = tpr * p + fpr * (1 - p)
        print(f"\n{iso} (blind prevalence {p:.4f}, our OOF AUC "
              f"{np.trapezoid(tpr, fpr):.4f})")
        for r in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            j = int(np.argmin(np.abs(calls - r)))
            print(f"   calling {r:.2f} -> MCC {m[j]:.4f}")
        k = int(np.nanargmax(m))
        out[iso] = {}
        for lab, r in MARK[iso].items():
            q = np.quantile(oof, 1 - r)
            j = int(np.argmin(np.abs(thr - q)))
            out[iso][lab] = m[j]
            print(f"   OUR {lab:6s} at {r:.3f} -> MCC {m[j]:.4f}")
        out[iso]["best"] = m[k]
        print(f"   OPTIMUM at {calls[k]:.3f} -> MCC {m[k]:.4f}")
    print()
    for lab in ("v2", "v3"):
        a = out["CYP2D6"]["v2/v3"]
        b = out["CYP3A4"][lab]
        print(f"predicted macro MCC for {lab}: {(a+b)/2:.4f}")
    print(f"predicted macro MCC at optimum : "
          f"{(out['CYP2D6']['best']+out['CYP3A4']['best'])/2:.4f}")
    print("actual scored: v2 0.2806 (rank 77), v3 0.2685 (rank 91)")


if __name__ == "__main__":
    main()
