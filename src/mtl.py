"""Multi-task net: shared encoder, masked heads, interval or MSE on the scored outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import st_rae  # noqa: E402
from pipeline import CACHE, ISOFORMS, N_FOLDS, scaffold_groups  # noqa: E402

RAW = Path(__file__).resolve().parents[2] / "public_raw"
GENES = {1544: "CYP1A2", 1557: "CYP2C19", 1559: "CYP2C9", 1565: "CYP2D6", 1576: "CYP3A4"}

HIDDEN = (1024, 512)
DROPOUT = 0.25
EPOCHS = 60
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
AUX_WEIGHT = 0.3        # auxiliary heads inform the encoder without dominating it
DELTA = 0.5             # same Huber width as the tree objective, for comparability
torch.set_num_threads(8)


class MultiTask(nn.Module):
    def __init__(self, n_in: int, n_scored: int, n_aux_reg: int, n_aux_clf: int):
        super().__init__()
        layers, d = [], n_in
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(DROPOUT)]
            d = h
        self.encoder = nn.Sequential(*layers)
        self.scored = nn.Linear(d, n_scored)
        self.aux_reg = nn.Linear(d, n_aux_reg)
        self.aux_clf = nn.Linear(d, n_aux_clf)

    def forward(self, x):
        z = self.encoder(x)
        return self.scored(z), self.aux_reg(z), self.aux_clf(z)


def interval_loss(pred, lo, hi, mask):
    """Zero inside the credible band, Huber-shaped on the excess outside it."""
    excess = torch.clamp(pred - hi, min=0) - torch.clamp(lo - pred, min=0)
    a = excess.abs()
    per = torch.where(a <= DELTA, 0.5 * excess ** 2, DELTA * (a - 0.5 * DELTA))
    return (per * mask).sum() / mask.sum().clamp(min=1)


def mse_loss(pred, y, mask):
    per = 0.5 * (pred - y) ** 2
    return (per * mask).sum() / mask.sum().clamp(min=1)


def build_matrix():
    """Challenge compounds stacked on top of the public panel, with per-head masks."""
    x_ch = np.load(CACHE / "X_train.npy")
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    x_pub = np.load(CACHE / "qhts_features.npy")

    conc = pd.read_csv(RAW / "aid1851_concise.csv", low_memory=False)
    smi = pd.read_csv(RAW / "aid1851_smiles.csv")
    col = next(c for c in ("CanonicalSMILES", "ConnectivitySMILES", "SMILES")
               if c in smi.columns)
    conc = conc[conc["Activity Outcome"].isin(["Active", "Inactive"])].copy()
    conc["iso"] = conc["Target GeneID"].map(GENES)
    conc = conc.dropna(subset=["iso"])
    conc["active"] = (conc["Activity Outcome"] == "Active").astype(int)
    wide = conc.pivot_table(index="CID", columns="iso", values="active", aggfunc="max")
    pub = wide.join(smi.set_index("CID")[col], how="inner")
    ok = ~np.isnan(x_pub).all(axis=1)
    x_pub, pub = x_pub[ok], pub[ok]

    n_ch, n_pub = len(x_ch), len(x_pub)
    x = np.vstack([x_ch, x_pub]).astype(np.float32)

    # Scored heads: pIC50 per isoform, challenge compounds only.
    y = np.zeros((n_ch + n_pub, 4), np.float32)
    lo = np.zeros_like(y); hi = np.zeros_like(y); m = np.zeros_like(y)
    for j, iso in enumerate(ISOFORMS):
        c = f"{iso}_pIC50_direct_inhibition"
        v = meta[c].to_numpy(float)
        ok_j = np.isfinite(v)
        y[:n_ch, j] = np.nan_to_num(v)
        lo[:n_ch, j] = np.nan_to_num(meta[f"{c}_conf_low"].to_numpy(float))
        hi[:n_ch, j] = np.nan_to_num(meta[f"{c}_conf_high"].to_numpy(float))
        m[:n_ch, j] = ok_j

    # Auxiliary regression: the single-concentration screen arms.
    ar = np.zeros((n_ch + n_pub, 4), np.float32); arm = np.zeros_like(ar)
    for j, iso in enumerate(ISOFORMS):
        v = meta[f"{iso}_log2fc"].to_numpy(float)
        ar[:n_ch, j] = np.nan_to_num(v); arm[:n_ch, j] = np.isfinite(v)

    # Auxiliary classification: the public panel's five isoforms.
    iso5 = sorted(GENES.values())
    ac = np.zeros((n_ch + n_pub, 5), np.float32); acm = np.zeros_like(ac)
    for j, iso in enumerate(iso5):
        if iso in pub.columns:
            v = pub[iso].to_numpy(float)
            ac[n_ch:, j] = np.nan_to_num(v); acm[n_ch:, j] = np.isfinite(v)

    groups = scaffold_groups(meta["SMILES"])
    return dict(x=x, y=y, lo=lo, hi=hi, m=m, ar=ar, arm=arm, ac=ac, acm=acm,
                n_ch=n_ch, groups=groups, meta=meta)


def normalise(x, train_rows):
    """Log-compress the counts, then standardise on training rows only."""
    x = x.copy()
    x[:, :2048] = np.log1p(np.clip(x[:, :2048], 0, None))
    d = x[:, 2048:]
    med = np.nanmedian(d[train_rows], axis=0)
    d = np.where(np.isfinite(d), d, med)
    mu, sd = d[train_rows].mean(0), d[train_rows].std(0) + 1e-6
    x[:, 2048:] = np.clip((d - mu) / sd, -10, 10)
    return x


def run_fold(data, tr_rows, va_rows, use_interval, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    x = normalise(data["x"], tr_rows)
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32))
    X = t(x)
    Y, LO, HI, M = t(data["y"]), t(data["lo"]), t(data["hi"]), t(data["m"])
    AR, ARM, AC, ACM = t(data["ar"]), t(data["arm"]), t(data["ac"]), t(data["acm"])

    model = MultiTask(X.shape[1], 4, 4, 5)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * max(1, len(tr_rows) // BATCH + 1))
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for _ in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr_rows)
        for s in range(0, len(perm), BATCH):
            idx = perm[s:s + BATCH]
            if len(idx) < 2:
                continue
            ps, pr, pc = model(X[idx])
            if use_interval:
                main = interval_loss(ps, LO[idx], HI[idx], M[idx])
            else:
                main = mse_loss(ps, Y[idx], M[idx])
            l_reg = ((0.5 * (pr - AR[idx]) ** 2) * ARM[idx]).sum() / ARM[idx].sum().clamp(min=1)
            l_clf = (bce(pc, AC[idx]) * ACM[idx]).sum() / ACM[idx].sum().clamp(min=1)
            loss = main + AUX_WEIGHT * (l_reg + l_clf)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()

    model.eval()
    with torch.no_grad():
        pred, _, _ = model(X[va_rows])
    return pred.numpy()


def main():
    data = build_matrix()
    n_ch = data["n_ch"]
    pub_rows = np.arange(n_ch, len(data["x"]))
    print(f"matrix {data['x'].shape}: {n_ch} challenge + {len(pub_rows)} public", flush=True)

    results = {}
    for use_interval in (False, True):
        name = "interval" if use_interval else "mse"
        oof = np.zeros((n_ch, 4), np.float32)
        gkf = GroupKFold(n_splits=N_FOLDS)
        dummy = np.zeros((n_ch, 1))
        for k, (tr_i, va_i) in enumerate(gkf.split(dummy, np.zeros(n_ch), data["groups"])):
            # Public compounds are always in training; folds split the challenge deck.
            tr_rows = np.concatenate([tr_i, pub_rows])
            oof[va_i] = run_fold(data, tr_rows, va_i, use_interval)
            print(f"  {name} fold {k+1}/{N_FOLDS}", flush=True)

        scores = {}
        for j, iso in enumerate(ISOFORMS):
            mask = data["m"][:n_ch, j] > 0
            scores[iso] = st_rae(data["y"][:n_ch, j][mask], oof[mask, j],
                                 data["lo"][:n_ch, j][mask], data["hi"][:n_ch, j][mask])
        results[name] = scores
        np.save(CACHE / f"mtl_oof_{name}.npy", oof)
        print(f"{name}: " + "  ".join(f"{i}={scores[i]:.4f}" for i in ISOFORMS)
              + f"  MACRO={np.mean(list(scores.values())):.4f}", flush=True)

    ml = np.mean(list(results["mse"].values()))
    mi = np.mean(list(results["interval"].values()))
    print(f"\nmulti-task MSE      {ml:.4f}")
    print(f"multi-task interval {mi:.4f}")
    print(f"gap {100*(mi-ml)/ml:+.2f}%   (trees: -4.25% challenge-only, -4.83% with public)")


if __name__ == "__main__":
    main()
