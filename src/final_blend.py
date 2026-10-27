"""Three-way blend, weights fitted per fold, then one scoring run on the locked holdout."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision import global_affine  # noqa: E402
from holdout import holdout_mask  # noqa: E402
from metric import st_rae  # noqa: E402
from pipeline import CACHE, ISOFORMS, N_FOLDS, scaffold_groups  # noqa: E402

STEP = 0.1
WEIGHTS = [w for w in itertools.product(np.round(np.arange(0, 1.01, STEP), 2), repeat=3)
           if abs(sum(w) - 1.0) < 1e-9]


def best_weights(parts, y, lo, hi):
    return min(WEIGHTS, key=lambda w: st_rae(y, sum(wi * p for wi, p in zip(w, parts)), lo, hi))


def main() -> None:
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    hold = holdout_mask()
    hold_rows = np.flatnonzero(hold)
    dm_oof = np.load(CACHE / "dmpnn_ho_oof_interval.npy")
    dm_hold = np.load(CACHE / "dmpnn_ho_hold_interval.npy")

    dev_scores, hold_scores, singles = {}, {}, {k: [] for k in ("ecfp", "alt", "dmpnn")}
    for j, iso in enumerate(ISOFORMS):
        tz = np.load(CACHE / f"trees_ho_{iso}.npz")
        az = np.load(CACHE / f"altview_{iso}.npz")
        dev, ho = tz["dev"], tz["ho"]
        assert (az["dev"] == dev).all() and (az["ho"] == ho).all()

        c = f"{iso}_pIC50_direct_inhibition"
        y_all = meta[c].to_numpy(float)
        lo_all = meta[f"{c}_conf_low"].to_numpy(float)
        hi_all = meta[f"{c}_conf_high"].to_numpy(float)
        y, lo, hi = y_all[dev], lo_all[dev], hi_all[dev]

        pos = {r: i for i, r in enumerate(hold_rows)}
        hsel = np.array([pos[r] for r in ho])
        dev_parts = [tz["oof"], az["oof"], dm_oof[dev, j]]
        ho_parts = [tz["hold"], az["hold"], dm_hold[hsel, j]]
        for k, p in zip(("ecfp", "alt", "dmpnn"), dev_parts):
            singles[k].append(st_rae(y, p, lo, hi))

        groups = scaffold_groups(meta["SMILES"].iloc[dev])
        folds = np.empty(len(y), int)
        for k, (_, va) in enumerate(GroupKFold(N_FOLDS).split(dev, dev, groups)):
            folds[va] = k

        # Honest dev estimate: weights and affine fitted on the other folds.
        blended = np.empty(len(y))
        for k in range(N_FOLDS):
            h = folds == k
            w = best_weights([p[~h] for p in dev_parts], y[~h], lo[~h], hi[~h])
            m = sum(wi * p for wi, p in zip(w, dev_parts))
            kk, bb = global_affine(m[~h], y[~h], lo[~h], hi[~h])[0]
            mu = m[~h].mean()
            blended[h] = mu + (m[h] - mu) * kk + bb
        dev_scores[iso] = st_rae(y, blended, lo, hi)

        # Final model: weights and affine fitted on all dev rows, applied to the holdout.
        w = best_weights(dev_parts, y, lo, hi)
        m_dev = sum(wi * p for wi, p in zip(w, dev_parts))
        kk, bb = global_affine(m_dev, y, lo, hi)[0]
        mu = m_dev.mean()
        m_ho = sum(wi * p for wi, p in zip(w, ho_parts))
        pred_ho = mu + (m_ho - mu) * kk + bb
        hold_scores[iso] = st_rae(y_all[ho], pred_ho, lo_all[ho], hi_all[ho])
        print(f"{iso}: dev blend {dev_scores[iso]:.4f} | HOLDOUT {hold_scores[iso]:.4f} "
              f"| w={w} k={kk:.2f} b={bb:+.2f} n_hold={len(ho)}", flush=True)

    print()
    for k, v in singles.items():
        print(f"  single {k:6s} dev {np.mean(v):.4f}")
    print(f"\nMACRO dev blend  {np.mean(list(dev_scores.values())):.4f}")
    print(f"MACRO HOLDOUT    {np.mean(list(hold_scores.values())):.4f}   "
          f"<- scored once, on compounds no model or choice has seen")


if __name__ == "__main__":
    main()
