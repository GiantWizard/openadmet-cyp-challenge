"""AID 1851 binary qHTS calls as stacked probability features.

Five isoform classifiers trained on the public panel, their out-of-fold probabilities
attached to every challenge compound. Binary calls are weaker than AC50 but there are
roughly 13,000 of them per isoform against our 1,300 to 2,300 labels, and CYP1A2 is
the one endpoint with no public potency data at all.
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
from features import featurise  # noqa: E402
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
    smi = pd.read_csv(RAW / "all_smiles.csv")
    col = next(c for c in ["CanonicalSMILES", "ConnectivitySMILES", "SMILES"]
               if c in smi.columns)
    smi = smi.drop_duplicates("CID")
    cid2raw = dict(zip(smi["CID"], smi[col]))
    cid2k = {c: canon(s) for c, s in cid2raw.items()}

    te = pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv")
    blind = {canon(s) for s in te.SMILES}; blind.discard(None)

    d = pd.read_csv(RAW / "aid1851_concise.csv", low_memory=False)
    d = d[d["Activity Outcome"].isin(["Active", "Inactive"])].copy()
    d["iso"] = d["Target GeneID"].map(GENES)
    d = d.dropna(subset=["iso"])
    d["k"] = d["CID"].map(cid2k)
    d = d[d["k"].notna() & ~d["k"].isin(blind)]
    d["act"] = (d["Activity Outcome"] == "Active").astype(int)

    wide = d.pivot_table(index="k", columns="iso", values="act", aggfunc="max")
    keys = wide.index.to_numpy()
    raws = [cid2raw[d.loc[d["k"] == k, "CID"].iloc[0]] for k in keys] \
        if False else None
    k2cid = dict(zip(d["k"], d["CID"]))
    raws = [cid2raw[k2cid[k]] for k in keys]
    print(f"public panel: {len(keys)} compounds, blind rows excluded", flush=True)

    cache = CACHE / "qhts_X.npy"
    if cache.exists():
        PX = np.load(cache)
    else:
        PX, _ = featurise(raws)
        np.save(cache, PX.astype(np.float32))
        PX = np.load(cache)
    ok = ~np.isnan(PX).all(axis=1)
    PX, wide = PX[ok], wide[ok]
    print(f"featurised {PX.shape}", flush=True)

    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    XTR = np.load(CACHE / "X_train.npy").astype(np.float32)
    XTE = np.load(CACHE / "X_test.npy").astype(np.float32)
    ftr = np.zeros((len(XTR), len(GENES)), np.float32)
    fte = np.zeros((len(XTE), len(GENES)), np.float32)
    for j, iso in enumerate(sorted(set(GENES.values()))):
        if iso not in wide.columns:
            continue
        lab = wide[iso].to_numpy(float)
        sel = np.flatnonzero(np.isfinite(lab))
        x, yy = PX[sel], lab[sel]
        acc_tr = np.zeros(len(XTR)); acc_te = np.zeros(len(XTE)); n = 0
        for tr_i, _ in KFold(4, shuffle=True, random_state=0).split(x):
            m = _fit(x[tr_i], yy[tr_i], CP, ROUNDS, 0)
            acc_tr += m.predict(XTR); acc_te += m.predict(XTE); n += 1
        ftr[:, j] = acc_tr / n
        fte[:, j] = acc_te / n
        print(f"  {iso}: {len(sel)} calls, active rate {yy.mean():.3f}", flush=True)
    np.save(CACHE / "qhts_tr.npy", ftr)
    np.save(CACHE / "qhts_te.npy", fte)
    print("wrote clean/cache/qhts_tr.npy and qhts_te.npy")


if __name__ == "__main__":
    main()
