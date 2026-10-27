"""PubChem AID 1851 qHTS panel, stacked into one activity feature per isoform."""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import featurise  # noqa: E402
from pipeline import BASE_PARAMS, CACHE  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "public_raw"

# PubChem target GeneIDs for the panel's five isoforms.
GENES = {1544: "CYP1A2", 1557: "CYP2C19", 1559: "CYP2C9", 1565: "CYP2D6", 1576: "CYP3A4"}

CLF_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
    feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbosity=-1, num_threads=8,
)
ROUNDS = 500


def build_table() -> pd.DataFrame:
    """One row per compound, one active/inactive column per isoform."""
    conc = pd.read_csv(RAW / "aid1851_concise.csv", low_memory=False)
    smi = pd.read_csv(RAW / "aid1851_smiles.csv")
    # PubChem has renamed this property over time; take whichever form came back.
    candidates = ["CanonicalSMILES", "ConnectivitySMILES", "SMILES", "IsomericSMILES"]
    smi_col = next((c for c in candidates if c in smi.columns), None)
    if smi_col is None:
        raise ValueError(f"no SMILES column in {list(smi.columns)}")

    # Inconclusive calls carry no information about activity; drop rather than guess.
    conc = conc[conc["Activity Outcome"].isin(["Active", "Inactive"])].copy()
    conc["iso"] = conc["Target GeneID"].map(GENES)
    conc = conc.dropna(subset=["iso"])
    conc["active"] = (conc["Activity Outcome"] == "Active").astype(int)

    wide = conc.pivot_table(index="CID", columns="iso", values="active", aggfunc="max")
    wide.columns = [f"{c}_qhts_active" for c in wide.columns]
    return wide.join(smi.set_index("CID")[smi_col].rename("SMILES"), how="inner").reset_index()


def main() -> None:
    table = build_table()
    print(f"qHTS table: {len(table)} compounds", flush=True)

    cache = CACHE / "qhts_features.npy"
    if cache.exists():
        x_pub = np.load(cache)
    else:
        x_pub, _ = featurise(table["SMILES"].tolist())
        np.save(cache, x_pub)
    ok = ~np.isnan(x_pub).all(axis=1)
    x_pub, table = x_pub[ok], table[ok].reset_index(drop=True)
    print(f"featurised: {x_pub.shape}", flush=True)

    # The challenge compounds these predictions will be attached to.
    x_tr = np.load(CACHE / "X_train.npy")
    x_te = np.load(CACHE / "X_test.npy")

    cols = [c for c in table.columns if c.endswith("_qhts_active")]
    pub_tr = np.zeros((len(x_tr), len(cols)))
    pub_te = np.zeros((len(x_te), len(cols)))
    for j, col in enumerate(cols):
        y = table[col].to_numpy(float)
        obs = np.isfinite(y)
        m = lgb.train(dict(CLF_PARAMS, seed=0),
                      lgb.Dataset(x_pub[obs], y[obs]), ROUNDS)
        pub_tr[:, j] = m.predict(x_tr)
        pub_te[:, j] = m.predict(x_te)
        print(f"  {col}: n={int(obs.sum())} active_rate={y[obs].mean():.3f} "
              f"| mean p(train)={pub_tr[:, j].mean():.3f} p(test)={pub_te[:, j].mean():.3f}",
              flush=True)

    np.save(CACHE / "pub_train.npy", pub_tr)
    np.save(CACHE / "pub_test.npy", pub_te)
    print(f"wrote stacked public features: {pub_tr.shape} / {pub_te.shape}")


if __name__ == "__main__":
    main()
