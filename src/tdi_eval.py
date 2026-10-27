"""Honest TDI evaluation: locked scaffold holdout, ranker measured threshold-free.

Earlier work tuned the calling rate on the same out-of-fold predictions it reported,
and the leaderboard showed the gap is ranking rather than threshold. Here AUC carries
the ranker comparison and MCC is reported at a threshold set from the blind positive
rate, which differs sharply from the training rate on CYP2D6.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from pipeline import BASE_PARAMS, CACHE, _fit, scaffold_groups  # noqa: E402

CP = dict(BASE_PARAMS, objective="binary", learning_rate=0.03, num_leaves=31,
          min_data_in_leaf=20, feature_fraction=0.3)
ROUNDS, SEEDS = 600, (0, 1)
ISO = ["CYP2D6", "CYP3A4"]
BLIND_RATE = {"CYP2D6": 0.0707, "CYP3A4": 0.2915}
HOLD_FRAC = 0.18


def main() -> None:
    tdi = pd.read_csv("data/cyp-challenge-TRAIN_TDI.csv")
    X = np.load(CACHE / "X_tdi_train.npy").astype(np.float32)
    Q = np.load(CACHE / "qhts_tdi.npy")
    g = np.asarray(scaffold_groups(tdi["SMILES"]))
    rng = np.random.default_rng(7)
    ug = np.unique(g)
    hold_g = set(rng.choice(ug, int(len(ug) * HOLD_FRAC), replace=False))
    hold = np.array([x in hold_g for x in g])
    print(f"holdout groups {len(hold_g)} covering {int(hold.sum())} of {len(g)} rows\n")

    for iso in ISO:
        y = pd.to_numeric(tdi[f"{iso}_is_TDI"], errors="coerce").to_numpy(float)
        obs = np.isfinite(y)
        dev, ho = np.flatnonzero(obs & ~hold), np.flatnonzero(obs & hold)
        print(f"{iso}: dev {len(dev)} (pos {y[dev].mean():.3f}) "
              f"holdout {len(ho)} (pos {y[ho].mean():.3f}) "
              f"| blind rate {BLIND_RATE[iso]:.3f}", flush=True)
        for tag, XX in (("base", X), ("+qhts", np.hstack([X, Q]).astype(np.float32))):
            oof = np.full(len(dev), np.nan)
            for tr_j, va_j in GroupKFold(5).split(dev, dev, g[dev]):
                tr = dev[tr_j]
                a = 0.0
                for s in SEEDS:
                    m = _fit(XX[tr], y[tr], CP, ROUNDS, s)
                    a = a + m.predict(XX[dev[va_j]]) / len(SEEDS)
                oof[va_j] = a
            hp = 0.0
            for s in SEEDS:
                m = _fit(XX[dev], y[dev], CP, ROUNDS, s)
                hp = hp + m.predict(XX[ho]) / len(SEEDS)
            # threshold from the blind positive rate, fixed on dev, applied to holdout
            thr = float(np.quantile(oof, 1 - BLIND_RATE[iso]))
            mcc_blind = matthews_corrcoef(y[ho], (hp >= thr).astype(int))
            # and the dev-tuned MCC threshold, for comparison
            grid = np.quantile(oof, np.linspace(0.5, 0.995, 60))
            best = max(grid, key=lambda t: matthews_corrcoef(y[dev], (oof >= t).astype(int)))
            mcc_tuned = matthews_corrcoef(y[ho], (hp >= best).astype(int))
            print(f"   {tag:6s} dev AUC {roc_auc_score(y[dev], oof):.4f} | "
                  f"HOLDOUT AUC {roc_auc_score(y[ho], hp):.4f} "
                  f"MCC@blindrate {mcc_blind:.4f} (calls {float((hp>=thr).mean()):.3f}) "
                  f"MCC@devtuned {mcc_tuned:.4f} (calls {float((hp>=best).mean()):.3f})",
                  flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
