"""TDI with the stacked assay arms restored, plus qHTS, thresholded at blind prevalence.

v4 applied correct thresholds to a ranker rebuilt without the stacked arm features, and
lost about 0.012 MCC to v2 as a result. Here the arms are back: each isoform's two assay
conditions predicted out of fold, plus their difference, which is what the TDI label
actually is. Feature set chosen on out-of-fold AUC, calling rate chosen by maximising
MCC analytically at the blind prevalence from this model's own ROC.
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

DATA = Path("data")
OUT = Path("clean/submissions")
CP = dict(BASE_PARAMS, objective="binary", learning_rate=0.03, num_leaves=31,
          min_data_in_leaf=20, feature_fraction=0.3)
ROUNDS, SEEDS = 600, (0, 1, 2)
ISO = ["CYP2D6", "CYP3A4"]
ARMS = ["pIC50_direct_inhibition", "pIC50_TDI_condition"]
BLIND = {"CYP2D6": 0.0707, "CYP3A4": 0.2915}


def mcc_at(tpr, fpr, p):
    tp, fn = tpr * p, (1 - tpr) * p
    fp, tn = fpr * (1 - p), (1 - fpr) * (1 - p)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, (tp * tn - fp * fn) / np.maximum(den, 1e-12), 0.0)


def arms(X, tdi, XT, g):
    f = CACHE / "tdi_arms_v5.npz"
    if f.exists():
        z = np.load(f)
        return z["tr"], z["te"]
    atr = np.full((len(tdi), len(ISO) * len(ARMS)), np.nan)
    ate = np.zeros((len(XT), len(ISO) * len(ARMS)))
    col = 0
    for iso in ISO:
        for arm in ARMS:
            y = pd.to_numeric(tdi[f"{iso}_{arm}"], errors="coerce").to_numpy(float)
            obs = np.flatnonzero(np.isfinite(y))
            oof = np.full(len(obs), np.nan)
            for tr_i, va_i in GroupKFold(5).split(obs, obs, g[obs]):
                m = _fit(X[obs[tr_i]], y[obs[tr_i]], BASE_PARAMS, 400, 0)
                oof[va_i] = m.predict(X[obs[va_i]])
            atr[obs, col] = oof
            ate[:, col] = _fit(X[obs], y[obs], BASE_PARAMS, 400, 0).predict(XT)
            print(f"  arm {iso} {arm}: n={len(obs)} r={np.corrcoef(oof, y[obs])[0,1]:.3f}",
                  flush=True)
            col += 1
    for j in range(len(ISO)):
        d, t = 2 * j, 2 * j + 1
        atr = np.column_stack([atr, atr[:, t] - atr[:, d]])
        ate = np.column_stack([ate, ate[:, t] - ate[:, d]])
    np.savez_compressed(f, tr=atr, te=ate)
    return atr, ate


def main() -> None:
    tdi = pd.read_csv(DATA / "cyp-challenge-TRAIN_TDI.csv")
    te = pd.read_csv(DATA / "cyp-challenge-TEST-BLINDED.csv")
    X = np.load(CACHE / "X_tdi_train.npy").astype(np.float32)
    XT = np.load(CACHE / "X_test.npy").astype(np.float32)
    Q = np.load(CACHE / "qhts_tdi.npy")
    QT = np.load(CACHE / "qhts_te.npy")
    g = np.asarray(scaffold_groups(tdi["SMILES"]))
    atr, ate = arms(X, tdi, XT, g)

    sets = {
        "base": (X, XT),
        "base+arms": (np.hstack([X, atr]), np.hstack([XT, ate])),
        "base+arms+qhts": (np.hstack([X, atr, Q]), np.hstack([XT, ate, QT])),
    }
    sub = te[["SMILES", "Molecule_Name"]].copy()
    for iso in ISO:
        p = BLIND[iso]
        y = pd.to_numeric(tdi[f"{iso}_is_TDI"], errors="coerce").to_numpy(float)
        obs = np.flatnonzero(np.isfinite(y))
        best = None
        for tag, (XX, XXT) in sets.items():
            XX = np.nan_to_num(XX, nan=-999).astype(np.float32)
            oof = np.full(len(obs), np.nan)
            for tr_j, va_j in GroupKFold(5).split(obs, obs, g[obs]):
                tr = obs[tr_j]
                a = 0.0
                for s in SEEDS:
                    a = a + _fit(XX[tr], y[tr], CP, ROUNDS, s).predict(XX[obs[va_j]]) / len(SEEDS)
                oof[va_j] = a
            fpr, tpr, _ = roc_curve(y[obs], oof)
            auc = float(np.trapezoid(tpr, fpr))
            m = mcc_at(tpr, fpr, p)
            k = int(np.nanargmax(m))
            rate = float(tpr[k] * p + fpr[k] * (1 - p))
            print(f"  {iso} {tag:16s} OOF AUC {auc:.4f} | best MCC {m[k]:.4f} "
                  f"at rate {rate:.3f}", flush=True)
            if best is None or auc > best[0]:
                best = (auc, tag, rate, m[k], XX, np.nan_to_num(XXT, nan=-999).astype(np.float32))
        auc, tag, rate, mcc, XX, XXT = best
        tp = 0.0
        for s in SEEDS:
            tp = tp + _fit(XX[obs], y[obs], CP, ROUNDS, s).predict(XXT) / len(SEEDS)
        cut = float(np.quantile(tp, 1 - rate))
        sub[f"{iso}_is_TDI"] = (tp >= cut)
        print(f"  {iso} CHOSEN {tag} (AUC {auc:.4f}) rate {rate:.3f} "
              f"predicted MCC {mcc:.4f} | blind calls {float((tp>=cut).mean()):.4f}\n",
              flush=True)
    OUT.mkdir(exist_ok=True)
    o = OUT / "submission_classification_v5.csv"
    sub.to_csv(o, index=False)
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
