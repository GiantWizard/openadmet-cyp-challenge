"""Assemble the modelling table: labels, bands, and auxiliary assay arms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import featurise_frame  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parents[1] / "cache"
OUT.mkdir(exist_ok=True)

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def main() -> None:
    train = pd.read_csv(DATA / "cyp-challenge-TRAIN_inhibition.csv")
    test = pd.read_csv(DATA / "cyp-challenge-TEST-BLINDED.csv")
    tdi = pd.read_csv(DATA / "cyp-challenge-TRAIN_TDI.csv")
    single = pd.read_csv(DATA / "cyp-challenge-single-concentration-TRAIN.csv")

    # Single-concentration screen pivots to one log2fc column per enzyme. This is the
    # only arm that records a compound which did nothing, so it carries the inactives.
    sc = single.pivot_table(
        index="Molecule_Name", columns="enzyme", values="log2fc_estimate", aggfunc="mean"
    )
    sc.columns = [f"{c}_log2fc" for c in sc.columns]
    sc_d = single.pivot_table(
        index="Molecule_Name", columns="enzyme", values="cohens_d", aggfunc="mean"
    )
    sc_d.columns = [f"{c}_cohensd" for c in sc_d.columns]
    screen = sc.join(sc_d).reset_index()

    tdi_cols = ["Molecule_Name"] + [
        c for c in tdi.columns if c.endswith("_pIC50_TDI_condition") or c.endswith("_is_TDI")
    ]

    frames = {}
    for name, df in (("train", train), ("test", test)):
        df = df.copy()
        x, keys = featurise_frame(df)
        df["conn_key"] = keys
        df = df.merge(screen, on="Molecule_Name", how="left")
        df = df.merge(tdi[tdi_cols], on="Molecule_Name", how="left")
        np.save(OUT / f"X_{name}.npy", x)
        df.to_parquet(OUT / f"meta_{name}.parquet")
        frames[name] = (x, df)
        print(name, "X", x.shape, "meta", df.shape, "missing mols", int(np.isnan(x).all(1).sum()))

    xtr, mtr = frames["train"]
    for iso in ISOFORMS:
        c = f"{iso}_pIC50_direct_inhibition"
        print(iso, "labelled", int(mtr[c].notna().sum()), "screen cov",
              int(mtr[f"{iso}_log2fc"].notna().sum()))


if __name__ == "__main__":
    main()
