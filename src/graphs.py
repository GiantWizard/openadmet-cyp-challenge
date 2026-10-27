"""Molecular graphs for the D-MPNN: atom/bond features and batched index arrays."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 34, 1]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRID = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

ATOM_DIM = len(ELEMENTS) + 1 + len(DEGREES) + 1 + len(CHARGES) + 1 + len(NUM_HS) + 1 \
    + len(HYBRID) + 1 + 3
BOND_DIM = len(BOND_TYPES) + 1 + 3


def _onehot(value, choices):
    v = [0.0] * (len(choices) + 1)
    v[choices.index(value) if value in choices else len(choices)] = 1.0
    return v


def atom_features(a) -> list[float]:
    return (
        _onehot(a.GetAtomicNum(), ELEMENTS)
        + _onehot(a.GetTotalDegree(), DEGREES)
        + _onehot(a.GetFormalCharge(), CHARGES)
        + _onehot(a.GetTotalNumHs(), NUM_HS)
        + _onehot(a.GetHybridization(), HYBRID)
        + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() * 0.01]
    )


def bond_features(b) -> list[float]:
    return _onehot(b.GetBondType(), BOND_TYPES) + [
        float(b.GetIsConjugated()), float(b.IsInRing()), float(b.GetStereo() != 0)
    ]


def mol_to_graph(mol):
    """Directed-edge graph: atom features, per-edge features, and reverse-edge index."""
    n = mol.GetNumAtoms()
    af = np.array([atom_features(a) for a in mol.GetAtoms()], np.float32)
    src, dst, bf = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        bf += [f, f]
    if not src:
        return af, np.zeros((0,), np.int64), np.zeros((0,), np.int64), \
            np.zeros((0, BOND_DIM), np.float32), np.zeros((0,), np.int64)
    rev = np.arange(len(src), dtype=np.int64) ^ 1  # pairs were appended adjacently
    return (af, np.array(src, np.int64), np.array(dst, np.int64),
            np.array(bf, np.float32), rev)


def graphs_from_smiles(smiles_list):
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles("C")
        out.append(mol_to_graph(mol))
    return out


def collate(graphs, idx):
    """Batch selected graphs into flat arrays with offsets into the atom/edge pools."""
    afs, srcs, dsts, bfs, revs, mol_of_atom = [], [], [], [], [], []
    a_off = e_off = 0
    for k, g in enumerate(idx):
        af, src, dst, bf, rev = graphs[g]
        afs.append(af)
        srcs.append(src + a_off)
        dsts.append(dst + a_off)
        bfs.append(bf)
        revs.append(rev + e_off)
        mol_of_atom.append(np.full(len(af), k, np.int64))
        a_off += len(af)
        e_off += len(src)
    return (
        np.concatenate(afs),
        np.concatenate(srcs) if e_off else np.zeros((0,), np.int64),
        np.concatenate(dsts) if e_off else np.zeros((0,), np.int64),
        np.concatenate(bfs) if e_off else np.zeros((0, BOND_DIM), np.float32),
        np.concatenate(revs) if e_off else np.zeros((0,), np.int64),
        np.concatenate(mol_of_atom),
        len(idx),
    )
