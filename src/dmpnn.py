"""D-MPNN with multi-task heads, trained with either the interval or MSE loss."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphs import ATOM_DIM, BOND_DIM, collate, graphs_from_smiles  # noqa: E402
from metric import st_rae  # noqa: E402
from pipeline import CACHE, ISOFORMS, N_FOLDS, scaffold_groups  # noqa: E402
from holdout import holdout_mask  # noqa: E402
from leakage import blind_keys  # noqa: E402
from features import connectivity_key, standardise  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "public_raw"
GENES = {1544: "CYP1A2", 1557: "CYP2C19", 1559: "CYP2C9", 1565: "CYP2D6", 1576: "CYP3A4"}
ISO5 = sorted(set(GENES.values()))

HIDDEN = 300
DEPTH = 4
FFN_HIDDEN = 300
DROPOUT = 0.1
EPOCHS = 40
BATCH = 64
LR = 8e-4
AUX_WEIGHT = 0.3
DELTA = 0.5
WARMUP = 2

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DMPNN(nn.Module):
    def __init__(self, n_scored, n_aux_reg, n_aux_clf, extra_dim):
        super().__init__()
        self.W_i = nn.Linear(ATOM_DIM + BOND_DIM, HIDDEN, bias=False)
        self.W_h = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.W_o = nn.Linear(ATOM_DIM + HIDDEN, HIDDEN)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT)
        self.ffn = nn.Sequential(
            nn.Linear(HIDDEN + extra_dim, FFN_HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(FFN_HIDDEN, FFN_HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
        )
        self.scored = nn.Linear(FFN_HIDDEN, n_scored)
        self.aux_reg = nn.Linear(FFN_HIDDEN, n_aux_reg)
        self.aux_clf = nn.Linear(FFN_HIDDEN, n_aux_clf)

    def forward(self, af, src, dst, bf, rev, mol_of_atom, n_mols, extra):
        n_atoms = af.shape[0]
        if src.numel() == 0:
            h_atom = self.act(self.W_o(torch.cat([af, af.new_zeros(n_atoms, HIDDEN)], 1)))
        else:
            h0 = self.act(self.W_i(torch.cat([af[src], bf], 1)))
            h = h0
            for _ in range(DEPTH - 1):
                # Sum incoming messages at each atom, then remove the reverse edge.
                agg = af.new_zeros(n_atoms, HIDDEN).index_add_(0, dst, h)
                h = self.drop(self.act(h0 + self.W_h(agg[src] - h[rev])))
            agg = af.new_zeros(n_atoms, HIDDEN).index_add_(0, dst, h)
            h_atom = self.drop(self.act(self.W_o(torch.cat([af, agg], 1))))

        pooled = af.new_zeros(n_mols, HIDDEN).index_add_(0, mol_of_atom, h_atom)
        counts = af.new_zeros(n_mols).index_add_(
            0, mol_of_atom, torch.ones_like(mol_of_atom, dtype=af.dtype))
        pooled = pooled / counts.clamp(min=1).unsqueeze(1)

        z = self.ffn(torch.cat([pooled, extra], 1))
        return self.scored(z), self.aux_reg(z), self.aux_clf(z)


def interval_loss(pred, lo, hi, mask):
    excess = torch.clamp(pred - hi, min=0) - torch.clamp(lo - pred, min=0)
    a = excess.abs()
    per = torch.where(a <= DELTA, 0.5 * excess ** 2, DELTA * (a - 0.5 * DELTA))
    return (per * mask).sum() / mask.sum().clamp(min=1)


def mse_loss(pred, y, mask):
    return ((0.5 * (pred - y) ** 2) * mask).sum() / mask.sum().clamp(min=1)


def load_data():
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    conc = pd.read_csv(RAW / "aid1851_concise.csv", low_memory=False)
    smi = pd.read_csv(RAW / "aid1851_smiles.csv")
    col = next(c for c in ("CanonicalSMILES", "ConnectivitySMILES", "SMILES")
               if c in smi.columns)
    conc = conc[conc["Activity Outcome"].isin(["Active", "Inactive"])].copy()
    conc["iso"] = conc["Target GeneID"].map(GENES)
    conc = conc.dropna(subset=["iso"])
    conc["active"] = (conc["Activity Outcome"] == "Active").astype(int)
    wide = conc.pivot_table(index="CID", columns="iso", values="active", aggfunc="max")
    pub = wide.join(smi.set_index("CID")[col].rename("SMILES"), how="inner").reset_index()

    n_ch, n_pub = len(meta), len(pub)
    smiles = meta["SMILES"].tolist() + pub["SMILES"].tolist()
    n = n_ch + n_pub

    y = np.zeros((n, 4), np.float32); lo = np.zeros_like(y)
    hi = np.zeros_like(y); m = np.zeros_like(y)
    for j, iso in enumerate(ISOFORMS):
        c = f"{iso}_pIC50_direct_inhibition"
        v = meta[c].to_numpy(float)
        y[:n_ch, j] = np.nan_to_num(v)
        lo[:n_ch, j] = np.nan_to_num(meta[f"{c}_conf_low"].to_numpy(float))
        hi[:n_ch, j] = np.nan_to_num(meta[f"{c}_conf_high"].to_numpy(float))
        m[:n_ch, j] = np.isfinite(v)

    ar = np.zeros((n, 4), np.float32); arm = np.zeros_like(ar)
    for j, iso in enumerate(ISOFORMS):
        v = meta[f"{iso}_log2fc"].to_numpy(float)
        ar[:n_ch, j] = np.nan_to_num(v); arm[:n_ch, j] = np.isfinite(v)

    ac = np.zeros((n, 5), np.float32); acm = np.zeros_like(ac)
    for j, iso in enumerate(ISO5):
        if iso in pub.columns:
            v = pub[iso].to_numpy(float)
            ac[n_ch:, j] = np.nan_to_num(v); acm[n_ch:, j] = np.isfinite(v)

    # Global descriptors ride alongside the learned representation.
    desc_ch = np.load(CACHE / "X_train.npy")[:, 2048:]
    desc_pub = np.load(CACHE / "qhts_features.npy")[:, 2048:]
    ok = ~np.isnan(np.load(CACHE / "qhts_features.npy")).all(axis=1)
    desc = np.vstack([desc_ch, desc_pub])
    assert len(desc) == n or ok.sum() == n_pub, "descriptor/row mismatch"
    return dict(smiles=smiles, y=y, lo=lo, hi=hi, m=m, ar=ar, arm=arm, ac=ac, acm=acm,
                desc=desc.astype(np.float32), n_ch=n_ch, meta=meta)


def prep_desc(desc, train_rows):
    med = np.nanmedian(desc[train_rows], axis=0)
    d = np.where(np.isfinite(desc), desc, med)
    mu, sd = d[train_rows].mean(0), d[train_rows].std(0) + 1e-6
    return np.clip((d - mu) / sd, -10, 10).astype(np.float32)


def run_fold(graphs, data, desc, tr_rows, va_rows, use_interval, seed, log):
    torch.manual_seed(seed); np.random.seed(seed)
    t = lambda a: torch.as_tensor(a, device=DEVICE)
    Y, LO, HI, M = t(data["y"]), t(data["lo"]), t(data["hi"]), t(data["m"])
    AR, ARM, AC, ACM = t(data["ar"]), t(data["arm"]), t(data["ac"]), t(data["acm"])
    D = t(desc)

    model = DMPNN(4, 4, 5, desc.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    steps = EPOCHS * max(1, len(tr_rows) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=WARMUP / EPOCHS)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    done = 0

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr_rows)
        for s in range(0, len(perm) - 1, BATCH):
            idx = perm[s:s + BATCH]
            if len(idx) < 2:
                continue
            af, src, dst, bf, rev, moa, nm = collate(graphs, idx)
            ps, pr, pc = model(t(af), t(src), t(dst), t(bf), t(rev), t(moa), nm, D[idx])
            main = (interval_loss(ps, LO[idx], HI[idx], M[idx]) if use_interval
                    else mse_loss(ps, Y[idx], M[idx]))
            l_reg = ((0.5 * (pr - AR[idx]) ** 2) * ARM[idx]).sum() / ARM[idx].sum().clamp(min=1)
            l_clf = (bce(pc, AC[idx]) * ACM[idx]).sum() / ACM[idx].sum().clamp(min=1)
            (main + AUX_WEIGHT * (l_reg + l_clf)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); opt.zero_grad()
            if done < steps - 1:
                sched.step(); done += 1
        if ep % 10 == 0:
            log(f"      epoch {ep}/{EPOCHS}")

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(va_rows), 256):
            idx = va_rows[s:s + 256]
            af, src, dst, bf, rev, moa, nm = collate(graphs, idx)
            ps, _, _ = model(t(af), t(src), t(dst), t(bf), t(rev), t(moa), nm, D[idx])
            preds.append(ps.cpu().numpy())
    return np.vstack(preds)


def public_leak_rows(smiles, n_ch):
    """Public rows whose structure matches a blind-set compound."""
    f = CACHE / "pub_keys.npy"
    if f.exists():
        keys = np.load(f, allow_pickle=True)
    else:
        keys = np.array([connectivity_key(standardise(s)) for s in smiles[n_ch:]], dtype=object)
        np.save(f, keys, allow_pickle=True)
    bk = blind_keys()
    return np.array([n_ch + i for i, k in enumerate(keys) if k in bk], np.int64)


def main():
    args = sys.argv[1:]
    seeds = [int(a) for a in args if a.isdigit()] or [0]
    use_holdout = "--holdout" in args
    losses = [True] if "--interval-only" in args else [False, True]
    tag = "_ho" if use_holdout else ""
    data = load_data()
    n_ch = data["n_ch"]
    log = lambda s: print(s, flush=True)
    log(f"device {DEVICE} | seeds {seeds} | holdout {use_holdout}")

    gfile = CACHE / "graphs.npy"
    if gfile.exists():
        graphs = list(np.load(gfile, allow_pickle=True))
    else:
        graphs = graphs_from_smiles(data["smiles"])
        np.save(gfile, np.array(graphs, dtype=object), allow_pickle=True)

    leak = public_leak_rows(data["smiles"], n_ch)
    pub_rows = np.setdiff1d(np.arange(n_ch, len(graphs)), leak)
    log(f"public rows {len(pub_rows)} (dropped {len(leak)} blind-set overlaps)")

    hold = holdout_mask() if use_holdout else np.zeros(n_ch, bool)
    dev, hold_rows = np.flatnonzero(~hold), np.flatnonzero(hold)
    groups = scaffold_groups(data["meta"]["SMILES"])

    for use_interval in losses:
        name = "interval" if use_interval else "mse"
        oof = np.full((n_ch, 4), np.nan, np.float32)
        hpred = np.zeros((len(hold_rows), 4), np.float32)
        splits = GroupKFold(n_splits=N_FOLDS).split(
            np.zeros((len(dev), 1)), np.zeros(len(dev)), groups[dev])
        for k, (tr_j, va_j) in enumerate(splits):
            tr_i, va_i = dev[tr_j], dev[va_j]
            train_rows = np.concatenate([tr_i, pub_rows])
            desc = prep_desc(data["desc"], train_rows)
            eval_rows = np.concatenate([va_i, hold_rows])
            acc = np.zeros((len(eval_rows), 4), np.float32)
            for seed in seeds:
                t0 = time.time()
                acc += run_fold(graphs, data, desc, train_rows, eval_rows,
                                use_interval, seed, log) / len(seeds)
                log(f"    {name} fold {k+1} seed {seed} in {time.time() - t0:.0f}s")
            oof[va_i] = acc[:len(va_i)]
            hpred += acc[len(va_i):] / N_FOLDS
        np.save(CACHE / f"dmpnn{tag}_oof_{name}.npy", oof)
        np.save(CACHE / f"dmpnn{tag}_hold_{name}.npy", hpred)
        scores = {}
        for j, iso in enumerate(ISOFORMS):
            mk = (data["m"][:n_ch, j] > 0) & ~hold
            scores[iso] = st_rae(data["y"][:n_ch, j][mk], oof[mk, j],
                                 data["lo"][:n_ch, j][mk], data["hi"][:n_ch, j][mk])
        log(f"{name} (dev OOF): " + "  ".join(f"{i}={scores[i]:.4f}" for i in ISOFORMS)
            + f"  MACRO={np.mean(list(scores.values())):.4f}")


if __name__ == "__main__":
    main()
