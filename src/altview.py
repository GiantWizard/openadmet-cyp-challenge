"""A third model on different fingerprints, for blend diversity rather than solo score."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import standardise  # noqa: E402
from holdout import holdout_mask  # noqa: E402
from metric import band_objective, st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, ISOFORMS, N_FOLDS, _fit, scaffold_groups  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ROUNDS = 900
SEEDS = (0, 1, 2)
PARAMS = dict(BASE_PARAMS, learning_rate=0.02, num_leaves=31, feature_fraction=0.25)


def build_features(smiles) -> np.ndarray:
    """Atom-pair, topological-torsion, RDKit path and MACCS keys."""
    ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    tt = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
    rd = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024, maxPath=6)
    rows = []
    for smi in smiles:
        mol = standardise(smi)
        if mol is None:
            rows.append(np.zeros(2048 + 1024 + 1024 + 167, np.float32))
            continue
        rows.append(np.concatenate([
            ap.GetCountFingerprintAsNumPy(mol).astype(np.float32),
            tt.GetCountFingerprintAsNumPy(mol).astype(np.float32),
            rd.GetFingerprintAsNumPy(mol).astype(np.float32),
            np.array(GetMACCSKeysFingerprint(mol), np.float32),
        ]))
    return np.vstack(rows)


def main() -> None:
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    f = CACHE / "altview_X.npy"
    if f.exists():
        x = np.load(f)
    else:
        x = build_features(meta["SMILES"].tolist())
        np.save(f, x)
    # Keep the stacked public and screen features: they are cheap and already built.
    x = np.hstack([x, np.load(CACHE / "aux_train.npy"),
                   np.load(CACHE / "pub_train.npy")]).astype(np.float32)
    print(f"alt features {x.shape}", flush=True)

    hold = holdout_mask()
    for iso in ISOFORMS:
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        dev = np.flatnonzero(obs & ~hold)
        ho = np.flatnonzero(obs & hold)
        groups = scaffold_groups(meta["SMILES"].iloc[dev])

        oof = np.zeros(len(dev))
        hp = np.zeros(len(ho))
        for tr_j, va_j in GroupKFold(n_splits=N_FOLDS).split(dev, dev, groups):
            tr, va = dev[tr_j], dev[va_j]
            base = float(y[tr].mean())
            for seed in SEEDS:
                m = _fit(x[tr], y[tr], PARAMS, ROUNDS, seed,
                         obj=band_objective(lo[tr], hi[tr]),
                         init_score=np.full(len(tr), base))
                oof[va_j] += (m.predict(x[va]) + base) / len(SEEDS)
                hp += (m.predict(x[ho]) + base) / len(SEEDS) / N_FOLDS
        np.savez_compressed(CACHE / f"altview_{iso}.npz", oof=oof, hold=hp, dev=dev, ho=ho)
        print(f"  {iso}: dev OOF {st_rae(y[dev], oof, lo[dev], hi[dev]):.4f}", flush=True)


if __name__ == "__main__":
    main()
