"""Featurisation: ECFP4 counts, RDKit descriptors, and a standardised InChIKey block."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ECFP_BITS = 2048
ECFP_RADIUS = 2

_uncharger = rdMolStandardize.Uncharger()
_largest = rdMolStandardize.LargestFragmentChooser()
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)

_DESC = [(n, f) for n, f in Descriptors.descList if n != "Ipc"]


def standardise(smiles: str):
    """Desalt and neutralise, so salt/stereo variants of one compound align."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _largest.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def connectivity_key(mol) -> str | None:
    """InChIKey connectivity block: the first 14 chars, ignoring stereo and charge."""
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)[:14]
    except Exception:
        return None


def _descriptors(mol) -> np.ndarray:
    out = np.empty(len(_DESC), dtype=np.float64)
    for i, (_, fn) in enumerate(_DESC):
        try:
            out[i] = fn(mol)
        except Exception:
            out[i] = np.nan
    return out


DESC_NAMES = [n for n, _ in _DESC]
FEATURE_NAMES = [f"ecfp_{i}" for i in range(ECFP_BITS)] + DESC_NAMES


def featurise(smiles_list) -> tuple[np.ndarray, list]:
    """Return (X, mols) with ECFP4 counts concatenated to RDKit descriptors."""
    rows, mols = [], []
    for smi in smiles_list:
        mol = standardise(smi)
        mols.append(mol)
        if mol is None:
            rows.append(np.full(ECFP_BITS + len(_DESC), np.nan))
            continue
        fp = _morgan.GetCountFingerprintAsNumPy(mol).astype(np.float64)
        rows.append(np.concatenate([fp, _descriptors(mol)]))
    x = np.vstack(rows)
    # Descriptors occasionally emit infinities; LightGBM treats NaN as missing.
    x[~np.isfinite(x)] = np.nan
    return x, mols


def featurise_frame(df: pd.DataFrame, smiles_col: str = "SMILES"):
    x, mols = featurise(df[smiles_col].tolist())
    keys = [connectivity_key(m) for m in mols]
    return x, keys
