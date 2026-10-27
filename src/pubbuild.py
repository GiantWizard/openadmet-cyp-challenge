"""Veith qHTS rows as interval labels on the challenge's own feature space.

Actives carry an AC50, so they become a point with a narrow band. Inactives were
censored above the 39.8 uM top dose, which is an interval open downward, exactly the
shape band_objective already consumes. Blind-set compounds are dropped outright.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from features import featurise  # noqa: E402

RDLogger.DisableLog("rdApp.*")
RAW = Path("public_raw")
CACHE = Path("clean/cache")
FILES = {"CYP2C9": "veith_883", "CYP3A4": "veith_884",
         "CYP2D6": "veith_891", "CYP2C19": "veith_899"}
TOP_DOSE_PIC50 = 4.40   # 39.8 uM, the uniform maximum across all four assays


def canon(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m else None


def main() -> None:
    smi = pd.read_csv(RAW / "all_smiles.csv")
    col = next(c for c in ["CanonicalSMILES", "ConnectivitySMILES", "SMILES"]
               if c in smi.columns)
    smi["k"] = [canon(s) for s in smi[col]]
    smi = smi.dropna(subset=["k"]).drop_duplicates("CID")
    cid2 = dict(zip(smi["CID"], zip(smi["k"], smi[col])))

    blind = {canon(s) for s in pd.read_csv("data/cyp-challenge-TEST-BLINDED.csv").SMILES}
    blind.discard(None)

    recs = {}
    for iso, f in FILES.items():
        d = pd.read_csv(RAW / f"{f}.csv", low_memory=False)
        d = d[d["Activity Outcome"].isin(["Active", "Inactive"])].copy()
        d["ac"] = pd.to_numeric(d["Activity Value [uM]"], errors="coerce")
        for cid, out, ac in zip(d["CID"], d["Activity Outcome"], d["ac"]):
            if cid not in cid2:
                continue
            k, raw = cid2[cid]
            if k in blind:
                continue
            r = recs.setdefault(k, {"smiles": raw})
            if out == "Active" and np.isfinite(ac):
                r[iso] = 6 - np.log10(ac)
            elif out == "Inactive":
                r.setdefault(iso, -np.inf)   # censored at the top dose

    tab = pd.DataFrame.from_dict(recs, orient="index").reset_index(drop=True)
    print(f"public compounds kept: {len(tab)}  (blind overlaps dropped)", flush=True)
    for iso in FILES:
        if iso in tab:
            v = tab[iso]
            print(f"  {iso}: {int(v.notna().sum())} labelled "
                  f"({int(np.isfinite(v).sum())} with AC50, "
                  f"{int((v == -np.inf).sum())} censored)", flush=True)

    x, _ = featurise(tab["smiles"].tolist())
    ok = ~np.isnan(x).all(axis=1)
    x, tab = x[ok], tab[ok].reset_index(drop=True)
    print(f"featurised: {x.shape}", flush=True)
    np.save(CACHE / "pub_X.npy", x.astype(np.float32))
    tab.to_parquet(CACHE / "pub_meta.parquet")
    print("wrote clean/cache/pub_X.npy and pub_meta.parquet")


if __name__ == "__main__":
    main()
