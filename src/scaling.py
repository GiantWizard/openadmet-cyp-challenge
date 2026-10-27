"""Interval-vs-MSE gap as a function of training set size."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import band_objective, st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, ISOFORMS, N_FOLDS, _fit, scaffold_groups  # noqa: E402

FRACTIONS = (0.15, 0.3, 0.6, 1.0)
ROUNDS = 500
SEED = 0


def subsample(groups, frac, rng):
    """Keep a fraction of scaffold groups, so less data means less chemistry."""
    uniq = np.unique(groups)
    keep = set(rng.choice(uniq, max(1, int(round(len(uniq) * frac))), replace=False))
    return np.array([g in keep for g in groups])


def main() -> None:
    # One isoform per invocation: the full feature matrix plus LightGBM's own copies
    # was enough to get the all-in-one version killed.
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    isoforms = only if only else list(ISOFORMS)
    meta = pd.read_parquet(CACHE / "meta_train.parquet")

    out = []
    for iso in isoforms:
        d = np.load(CACHE / f"cv_pub_{iso}.npz")
        y, lo, hi = d["y"], d["lo"], d["hi"]
        xtr = np.hstack([
            np.load(CACHE / "X_train.npy"),
            np.load(CACHE / "aux_train.npy"),
            np.load(CACHE / "pub_train.npy"),
        ]).astype(np.float32)
        x = np.ascontiguousarray(xtr[d["obs"]])
        del xtr
        groups = scaffold_groups(meta["SMILES"][d["obs"]])

        folds = np.empty(len(y), int)
        for k, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(x, y, groups)):
            folds[va] = k

        for frac in FRACTIONS:
            rng = np.random.default_rng(SEED)
            p_l2 = np.zeros(len(y))
            p_bd = np.zeros(len(y))
            for k in range(N_FOLDS):
                va = folds == k
                tr = ~va
                if frac < 1.0:
                    tr = tr & subsample(groups, frac, rng)
                if tr.sum() < 50:
                    p_l2[va] = y[tr].mean() if tr.sum() else y.mean()
                    p_bd[va] = p_l2[va]
                    continue
                base = float(y[tr].mean())
                p_l2[va] = _fit(x[tr], y[tr], BASE_PARAMS, ROUNDS, SEED).predict(x[va])
                m = _fit(x[tr], y[tr], BASE_PARAMS, ROUNDS, SEED,
                         obj=band_objective(lo[tr], hi[tr]),
                         init_score=np.full(int(tr.sum()), base))
                p_bd[va] = m.predict(x[va]) + base
            s_l2 = st_rae(y, p_l2, lo, hi)
            s_bd = st_rae(y, p_bd, lo, hi)
            out.append(dict(iso=iso, frac=frac, l2=s_l2, band=s_bd,
                            gap=100 * (s_bd - s_l2) / s_l2))
            print(f"{iso} frac={frac:<5.2f} L2={s_l2:.4f} band={s_bd:.4f} "
                  f"gap={out[-1]['gap']:+.2f}%", flush=True)

    df = pd.DataFrame(out)
    f = CACHE / "scaling.csv"
    if f.exists() and only:
        df = pd.concat([pd.read_csv(f), df], ignore_index=True).drop_duplicates(
            ["iso", "frac"], keep="last")
    df.to_csv(f, index=False)
    if set(df.iso) != set(ISOFORMS):
        print(f"\n(partial: {sorted(set(df.iso))})")
        return
    print("\nmacro across isoforms, by training fraction:")
    print(f"{'frac':>6} {'L2':>9} {'band':>9} {'gap%':>8}")
    for frac in FRACTIONS:
        s = df[df.frac == frac]
        l2, bd = s.l2.mean(), s.band.mean()
        print(f"{frac:6.2f} {l2:9.4f} {bd:9.4f} {100*(bd-l2)/l2:+8.2f}")
    print("\nIf the gap narrows as the model strengthens, the objective is a crutch.")
    print("If it is flat or widens, it is a property of the scoring rule.")


if __name__ == "__main__":
    main()
