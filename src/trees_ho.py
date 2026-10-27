"""Tree models on development rows only: small grid search, then holdout predictions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holdout import holdout_mask  # noqa: E402
from metric import band_objective, st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, ISOFORMS, N_FOLDS, _fit, scaffold_groups  # noqa: E402

GRID = {
    "base":      dict(num_leaves=31, min_data_in_leaf=20, feature_fraction=0.3, rounds=700),
    "small":     dict(num_leaves=15, min_data_in_leaf=30, feature_fraction=0.2, rounds=900),
    "large":     dict(num_leaves=63, min_data_in_leaf=15, feature_fraction=0.3, rounds=500),
    "sparse":    dict(num_leaves=31, min_data_in_leaf=20, feature_fraction=0.1, rounds=900),
    "reg":       dict(num_leaves=31, min_data_in_leaf=40, feature_fraction=0.3, rounds=900,
                      lambda_l2=10.0),
    "slow":      dict(num_leaves=31, min_data_in_leaf=20, feature_fraction=0.3, rounds=1400,
                      learning_rate=0.015),
}
FINAL_SEEDS = (0, 1, 2)


def views():
    x = np.hstack([np.load(CACHE / "X_train.npy"), np.load(CACHE / "aux_train.npy"),
                   np.load(CACHE / "pub_train.npy")]).astype(np.float32)
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    hold = holdout_mask()
    out = {}
    for iso in ISOFORMS:
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        dev, ho = obs & ~hold, obs & hold
        out[iso] = dict(x=x, y=y, lo=lo, hi=hi, dev=np.flatnonzero(dev),
                        ho=np.flatnonzero(ho),
                        groups=scaffold_groups(meta["SMILES"][dev]))
    return out


def run(v, cfg, seeds):
    """Dev OOF and holdout predictions (holdout = mean of the fold models)."""
    cfg = dict(cfg)
    rounds = cfg.pop("rounds")
    params = dict(BASE_PARAMS, **cfg)
    dev, ho = v["dev"], v["ho"]
    x, y, lo, hi = v["x"], v["y"], v["lo"], v["hi"]
    oof = np.zeros(len(dev))
    hp = np.zeros(len(ho))
    for tr_j, va_j in GroupKFold(n_splits=N_FOLDS).split(dev, dev, v["groups"]):
        tr, va = dev[tr_j], dev[va_j]
        base = float(y[tr].mean())
        for seed in seeds:
            m = _fit(x[tr], y[tr], params, rounds, seed,
                     obj=band_objective(lo[tr], hi[tr]), init_score=np.full(len(tr), base))
            oof[va_j] += (m.predict(x[va]) + base) / len(seeds)
            hp += (m.predict(x[ho]) + base) / len(seeds) / N_FOLDS
    return oof, hp


def main():
    vs = views()
    scores = {}
    for name, cfg in GRID.items():
        s = []
        for iso, v in vs.items():
            oof, _ = run(v, cfg, (0,))
            d = v["dev"]
            s.append(st_rae(v["y"][d], oof, v["lo"][d], v["hi"][d]))
        scores[name] = float(np.mean(s))
        print(f"  grid {name:7s} dev macro {scores[name]:.4f}  "
              + " ".join(f"{x:.4f}" for x in s), flush=True)
    best = min(scores, key=scores.get)
    print(f"chosen: {best}", flush=True)

    for iso, v in vs.items():
        oof, hp = run(v, GRID[best], FINAL_SEEDS)
        np.savez_compressed(CACHE / f"trees_ho_{iso}.npz", oof=oof, hold=hp,
                            dev=v["dev"], ho=v["ho"])
        d = v["dev"]
        print(f"  {iso}: dev OOF {st_rae(v['y'][d], oof, v['lo'][d], v['hi'][d]):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
