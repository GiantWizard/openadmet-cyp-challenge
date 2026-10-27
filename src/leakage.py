"""Blind-set connectivity keys, for excluding them from any public training source."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import connectivity_key, standardise  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
CACHE = Path(__file__).resolve().parents[1] / "cache"


def blind_keys() -> set[str]:
    f = CACHE / "blind_keys.csv"
    if f.exists():
        return set(pd.read_csv(f)["key"])
    test = pd.read_csv(DATA / "cyp-challenge-TEST-BLINDED.csv")
    keys = {connectivity_key(standardise(s)) for s in test["SMILES"]}
    keys.discard(None)
    pd.DataFrame({"key": sorted(keys)}).to_csv(f, index=False)
    return keys


def drop_blind(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Remove any row whose structure matches a blind-set compound."""
    keys = blind_keys()
    k = [connectivity_key(standardise(s)) for s in df[smiles_col]]
    keep = [key not in keys for key in k]
    dropped = len(df) - sum(keep)
    if dropped:
        print(f"  dropped {dropped} row(s) overlapping the blind set", flush=True)
    return df[keep].reset_index(drop=True)
