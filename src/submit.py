"""Fit on all labelled data, recalibrate, write the regression submission."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision import global_affine, joint_pools, optimal_shift_uncertain  # noqa: E402
from metric import band_objective, st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, ISOFORMS, N_FOLDS, N_ROUNDS, SEEDS, _fit  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "submissions"
OUT.mkdir(exist_ok=True)

# --public uses the checkpoints and features that include the PubChem qHTS panel.
USE_PUBLIC = "--public" in sys.argv
TAG = "_pub" if USE_PUBLIC else ""

# Predicted-activity bins the residual pools are conditioned on. Chosen on CYP1A2 and
# then held fixed, so only that isoform's decision-layer figure is a tuned result.
N_BINS = 10


def bin_edges(values, n=N_BINS):
    return np.quantile(values, np.linspace(0, 1, n + 1)[1:-1])


def load(iso, xtr_full):
    """Rebuild one isoform's training view from its cross-validation checkpoint."""
    d = np.load(CACHE / f"cv{TAG}_{iso}.npz")
    return dict(
        x=xtr_full[d["obs"]], y=d["y"], lo=d["lo"], hi=d["hi"], groups=d["groups"],
        oof_l2=d["oof_l2"], oof_band=d["oof_band"],
        oof_lo=d["oof_lo"], oof_hi=d["oof_hi"],
    )


def validate(res) -> dict:
    """Score every stage out of fold, never reusing a held-out fold's own residuals."""
    y, lo, hi = res["y"], res["lo"], res["hi"]
    lo_h, hi_h = res["oof_lo"], res["oof_hi"]
    c = res["oof_band"]

    folds = np.empty(len(y), int)
    for k, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(res["x"], y, res["groups"])):
        folds[va] = k

    bins = np.digitize(c, bin_edges(c))
    aff = np.empty_like(c)
    dec = np.empty_like(c)
    orc = np.empty_like(c)
    for k in np.unique(folds):
        h = folds == k
        (kk, bb), _ = global_affine(c[~h], y[~h], lo[~h], hi[~h])
        mu = float(c[~h].mean())
        aff[h] = mu + (c[h] - mu) * kk + bb
        dec[h] = c[h] + optimal_shift_uncertain(
            c[h], joint_pools(c, y, lo_h, hi_h, lo, hi, bins, exclude=h),
            bins[h], lo_h[h], hi_h[h],
        )
        # Ceiling: the same layer handed the true band halves, which separates the
        # idea's worth from the band-half models' accuracy.
        orc[h] = c[h] + optimal_shift_uncertain(
            c[h], joint_pools(c, y, y - lo, hi - y, lo, hi, bins, exclude=h),
            bins[h], (y - lo)[h], (hi - y)[h],
        )

    return {
        "l2": st_rae(y, res["oof_l2"], lo, hi),
        "band": st_rae(y, c, lo, hi),
        "band+affine": st_rae(y, aff, lo, hi),
        "band+decision": st_rae(y, dec, lo, hi),
        "oracle": st_rae(y, orc, lo, hi),
    }


def main() -> None:
    tr_parts = [np.load(CACHE / "X_train.npy"), np.load(CACHE / "aux_train.npy")]
    te_parts = [np.load(CACHE / "X_test.npy"), np.load(CACHE / "aux_test.npy")]
    if USE_PUBLIC:
        tr_parts.append(np.load(CACHE / "pub_train.npy"))
        te_parts.append(np.load(CACHE / "pub_test.npy"))
    xtr = np.hstack(tr_parts).astype(np.float32)
    xt = np.hstack(te_parts).astype(np.float32)
    meta_te = pd.read_parquet(CACHE / "meta_test.parquet")

    sub = meta_te[["SMILES", "Molecule_Name"]].copy()
    alt = meta_te[["SMILES", "Molecule_Name"]].copy()
    report = {}

    for iso in ISOFORMS:
        res = load(iso, xtr)
        report[iso] = validate(res)
        print(f"{iso}: validated", flush=True)

        x, y, lo, hi = res["x"], res["y"], res["lo"], res["hi"]
        base = float(np.mean(y))
        centre = np.zeros(len(xt))
        for seed in SEEDS:
            m = _fit(x, y, BASE_PARAMS, N_ROUNDS, seed,
                     obj=band_objective(lo, hi), init_score=np.full(len(y), base))
            centre += (m.predict(xt) + base) / len(SEEDS)

        lo_te = np.clip(_fit(x, y - lo, BASE_PARAMS, 400, 0).predict(xt), 0.01, None)
        hi_te = np.clip(_fit(x, hi - y, BASE_PARAMS, 400, 0).predict(xt), 0.01, None)

        c_oof = res["oof_band"]
        edges = bin_edges(c_oof)
        pools = joint_pools(c_oof, y, res["oof_lo"], res["oof_hi"], lo, hi,
                            np.digitize(c_oof, edges))
        shift = optimal_shift_uncertain(centre, pools, np.digitize(centre, edges),
                                        lo_te, hi_te)

        (k_aff, b_aff), _ = global_affine(c_oof, y, lo, hi)
        mu = float(c_oof.mean())
        pred = mu + (centre - mu) * k_aff + b_aff

        sub[f"{iso}_pIC50_direct_inhibition"] = pred
        alt[f"{iso}_pIC50_direct_inhibition"] = centre + shift
        report[iso].update(affine=(k_aff, b_aff), mean=float(pred.mean()),
                           sd=float(pred.std()))
        print(f"  {iso}: k={k_aff:.2f} b={b_aff:+.2f} "
              f"test mean={pred.mean():.2f} sd={pred.std():.2f}", flush=True)

    path = OUT / f"submission_regression{TAG}.csv"
    sub.to_csv(path, index=False)
    alt.to_csv(OUT / f"submission_regression{TAG}_decision_layer.csv", index=False)
    print(f"\nwrote {path}  rows={len(sub)}\n")

    cols = ("l2", "band", "band+affine", "band+decision", "oracle")
    hdr = f"{'isoform':8s}" + "".join(f"{c:>15s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for iso in ISOFORMS:
        print(f"{iso:8s}" + "".join(f"{report[iso][c]:15.4f}" for c in cols))
    print("-" * len(hdr))
    print(f"{'MACRO':8s}" + "".join(
        f"{np.mean([report[i][c] for i in ISOFORMS]):15.4f}" for c in cols))


if __name__ == "__main__":
    main()
