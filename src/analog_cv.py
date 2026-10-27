"""Validation matched to the blind set's similarity profile, not cluster-disjoint."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout import holdout_mask  # noqa: E402
from metric import band_objective, st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, ISOFORMS, N_FOLDS, _fit  # noqa: E402
from readacross import predict as read_predict, tanimoto  # noqa: E402

SIM_FLOOR = 0.5      # the blind set has a training analogue this close for 93.5% of it
ROUNDS = 900
SEED = 0
PARAMS_ECFP = dict(BASE_PARAMS, learning_rate=0.02, num_leaves=31, feature_fraction=0.3)
PARAMS_ALT = dict(BASE_PARAMS, learning_rate=0.02, num_leaves=31, feature_fraction=0.25)


def main() -> None:
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    ecfp = np.hstack([np.load(CACHE / "X_train.npy"), np.load(CACHE / "aux_train.npy"),
                      np.load(CACHE / "pub_train.npy")]).astype(np.float32)
    alt = np.hstack([np.load(CACHE / "altview_X.npy"), np.load(CACHE / "aux_train.npy"),
                     np.load(CACHE / "pub_train.npy")]).astype(np.float32)
    bits = (np.load(CACHE / "X_train.npy")[:, :2048] > 0).astype(np.float32)
    pop = bits.sum(1)
    hold = holdout_mask()

    agg = {k: [] for k in ("ecfp", "alt", "read", "blend")}
    for iso in ISOFORMS:
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        dev = np.flatnonzero(obs & ~hold)

        preds = {k: np.full(len(dev), np.nan) for k in ("ecfp", "alt", "read")}
        keep = np.zeros(len(dev), bool)
        for tr_j, va_j in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(dev):
            tr, va = dev[tr_j], dev[va_j]
            sim = tanimoto(bits[va], bits[tr], pop[va], pop[tr])
            # Score only where the blind set's regime is reproduced.
            keep[va_j] = sim.max(1) >= SIM_FLOOR
            base = float(y[tr].mean())
            for key, x, p in (("ecfp", ecfp, PARAMS_ECFP), ("alt", alt, PARAMS_ALT)):
                m = _fit(x[tr], y[tr], p, ROUNDS, SEED,
                         obj=band_objective(lo[tr], hi[tr]),
                         init_score=np.full(len(tr), base))
                preds[key][va_j] = m.predict(x[va]) + base
            preds["read"][va_j] = read_predict(sim, y[tr], base)

        s = {}
        yk, lok, hik = y[dev][keep], lo[dev][keep], hi[dev][keep]
        for k in ("ecfp", "alt", "read"):
            s[k] = st_rae(yk, preds[k][keep], lok, hik)
        b = (preds["ecfp"] + preds["alt"] + preds["read"]) / 3
        s["blend"] = st_rae(yk, b[keep], lok, hik)
        for k in agg:
            agg[k].append(s[k])
        print(f"{iso}: n={int(keep.sum())}/{len(dev)} analogue-rich | "
              + "  ".join(f"{k}={s[k]:.4f}" for k in ("ecfp", "alt", "read", "blend")),
              flush=True)

    print("\nMACRO, validation matched to the blind set's similarity profile:")
    for k in ("ecfp", "alt", "read", "blend"):
        print(f"  {k:6s} {np.mean(agg[k]):.4f}")
    print("\n(scaffold-disjoint CV for reference: ecfp 0.7100, alt 0.6977, read 1.0867)")


if __name__ == "__main__":
    main()
