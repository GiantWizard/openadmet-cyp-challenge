"""qHTS probabilities for the TDI training compounds, so the classifier can use them.

qhts_tr.npy is aligned to the 4,905 inhibition rows; the TDI track has 6,145. Same five
classifiers, predicted onto the TDI feature matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import KFold

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from pipeline import BASE_PARAMS, CACHE, _fit  # noqa: E402

RDLogger.DisableLog("rdApp.*")
RAW = Path("public_raw")
GENES = {1544: "CYP1A2", 1557: "CYP2C19", 1559: "CYP2C9", 1565: "CYP2D6", 1576: "CYP3A4"}
CP = dict(BASE_PARAMS, objective="binary", learning_rate=0.05, num_leaves=63,
          feature_fraction=0.3)
ROUNDS = 400


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def main() -> None:
    PX = np.load(CACHE / "qhts_X.npy")
    smi = pd.read_csv(RAW / "all_smiles.csv").drop_duplicates("CID")
    col = next(c for c in ["CanonicalSMILES", "ConnectivitySMILES", "SMILES"]
               if c in smi.columns)
    cid2k = {c: canon(s) for c, s in zip(smi["CID"], smi[col])}
    blind = {canon(s) for s in pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv").SMILES}
    blind.discard(None)
    d = pd.read_csv(RAW / "aid1851_concise.csv", low_memory=False)
    d = d[d["Activity Outcome"].isin(["Active", "Inactive"])].copy()
    d["iso"] = d["Target GeneID"].map(GENES)
    d = d.dropna(subset=["iso"])
    d["k"] = d["CID"].map(cid2k)
    d = d[d["k"].notna() & ~d["k"].isin(blind)]
    d["act"] = (d["Activity Outcome"] == "Active").astype(int)
    wide = d.pivot_table(index="k", columns="iso", values="act", aggfunc="max")
    ok = ~np.isnan(PX).all(axis=1)
    PX, wide = PX[ok], wide[ok]

    XT = np.load(CACHE / "X_tdi_train.npy").astype(np.float32)
    out = np.zeros((len(XT), len(GENES)), np.float32)
    for j, iso in enumerate(sorted(set(GENES.values()))):
        lab = wide[iso].to_numpy(float)
        sel = np.flatnonzero(np.isfinite(lab))
        acc = np.zeros(len(XT)); n = 0
        for tr_i, _ in KFold(4, shuffle=True, random_state=0).split(PX[sel]):
            m = _fit(PX[sel][tr_i], lab[sel][tr_i], CP, ROUNDS, 0)
            acc += m.predict(XT); n += 1
        out[:, j] = acc / n
        print(f"  {iso}: {len(sel)} calls", flush=True)
    np.save(CACHE / "qhts_tdi.npy", out)
    print(f"wrote clean/cache/qhts_tdi.npy {out.shape}")


if __name__ == "__main__":
    main()
