"""Merge nb25 fold files into per-isoform arm files in the layout the blend scripts read
(dev, oof, ho, hold, test), using whichever folds exist.

Usage: python nb25_arm.py <dir with nb25_f*.npz> <out dir>

A fold file holds predictions for all four scored heads over every compound in its fold (te), the
challenge holdout (ho) and the blind set (tp). An isoform's arm keeps only rows with a label for
that isoform. Holdout and blind predictions are the mean over the folds present."""
import sys, numpy as np, pandas as pd
from pathlib import Path
C = Path("clean/cache"); ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
files = sorted(src.glob("nb25_f[0-9].npz")); assert files, f"no nb25_f*.npz in {src}"
F = [np.load(f) for f in files]
print("folds:", [f.stem[-1] for f in files], "| best epochs", [int(z["best_epoch"]) for z in F],
      "| epochs", [int(z["epochs"]) for z in F])
meta = pd.read_csv(C / "meta_train.csv.gz")
te = np.concatenate([z["te"] for z in F]); oof = np.concatenate([z["oof"] for z in F])
assert len(np.unique(te)) == len(te), "a compound appears in two folds"
ho = F[0]["ho"]
for z in F[1:]: assert np.array_equal(z["ho"], ho), "holdout rows differ between folds"
hp = np.mean([z["hp"] for z in F], 0); tp = np.nanmean([z["tp"] for z in F], 0)
for j, iso in enumerate(ISO):
    y = meta[f"{iso}_pIC50_direct_inhibition"].to_numpy(float)
    k = np.isfinite(y[te]); h = np.isfinite(y[ho])
    np.savez(out / f"nb25_{iso}.npz", dev=te[k], oof=oof[k, j], ho=ho[h], hold=hp[h, j], test=tp[:, j])
    print(f"{iso}: dev {k.sum()} | holdout {h.sum()} | blind {np.isfinite(tp[:, j]).sum()}")
