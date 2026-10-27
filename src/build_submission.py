"""Refit all three components on every labelled compound and write the blend submission."""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dmpnn as D  # noqa: E402
from altview import PARAMS as ALT_PARAMS, ROUNDS as ALT_ROUNDS, build_features  # noqa: E402
from decision import global_affine  # noqa: E402
from features import connectivity_key, standardise  # noqa: E402
from graphs import collate, graphs_from_smiles  # noqa: E402
from leakage import blind_keys  # noqa: E402
from metric import band_objective, st_rae  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, ISOFORMS, _fit  # noqa: E402
from trees_ho import GRID  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parents[1] / "submissions"
OUT.mkdir(exist_ok=True)
SEEDS = (0, 1, 2)
TREE_CFG = GRID["slow"]
STEP = 0.1
WEIGHTS = [w for w in itertools.product(np.round(np.arange(0, 1.01, STEP), 2), repeat=3)
           if abs(sum(w) - 1.0) < 1e-9]
log = lambda s: print(s, flush=True)


def clean_public_features():
    """Stacked qHTS features with blind-set overlaps dropped from the source panel."""
    f_tr, f_te = CACHE / "pub_train_clean.npy", CACHE / "pub_test_clean.npy"
    if f_tr.exists() and f_te.exists():
        return np.load(f_tr), np.load(f_te)
    import public_data as P
    table = P.build_table()
    x_pub = np.load(CACHE / "qhts_features.npy")
    ok = ~np.isnan(x_pub).all(axis=1)
    x_pub, table = x_pub[ok], table[ok].reset_index(drop=True)
    bk = blind_keys()
    keys = np.load(CACHE / "pub_keys.npy", allow_pickle=True)
    keep = np.array([k not in bk for k in keys])
    log(f"  public panel: dropping {int((~keep).sum())} blind-set overlap(s)")
    x_pub, table = x_pub[keep], table[keep].reset_index(drop=True)

    x_tr = np.load(CACHE / "X_train.npy")
    x_te = np.load(CACHE / "X_test.npy")
    cols = [c for c in table.columns if c.endswith("_qhts_active")]
    pt = np.zeros((len(x_tr), len(cols)))
    pe = np.zeros((len(x_te), len(cols)))
    for j, col in enumerate(cols):
        y = table[col].to_numpy(float)
        obs = np.isfinite(y)
        m = lgb.train(dict(P.CLF_PARAMS, seed=0), lgb.Dataset(x_pub[obs], y[obs]), P.ROUNDS)
        pt[:, j] = m.predict(x_tr)
        pe[:, j] = m.predict(x_te)
    np.save(f_tr, pt); np.save(f_te, pe)
    return pt, pe


def alt_test_features():
    f = CACHE / "altview_X_test.npy"
    if f.exists():
        return np.load(f)
    te = pd.read_csv(DATA / "cyp-challenge-TEST-BLINDED.csv")
    x = build_features(te["SMILES"].tolist())
    np.save(f, x)
    return x


def tree_preds(x_tr, x_te, y, lo, hi, obs, params, rounds):
    base = float(y[obs].mean())
    out = np.zeros(len(x_te))
    for seed in SEEDS:
        m = _fit(x_tr[obs], y[obs], params, rounds, seed,
                 obj=band_objective(lo[obs], hi[obs]),
                 init_score=np.full(int(obs.sum()), base))
        out += (m.predict(x_te) + base) / len(SEEDS)
    return out


def dmpnn_preds(meta):
    """Train on every labelled challenge compound plus the public panel, predict the test set."""
    f = CACHE / "dmpnn_test_pred.npy"
    if f.exists():
        return np.load(f)
    data = D.load_data()
    n_ch = data["n_ch"]
    te = pd.read_csv(DATA / "cyp-challenge-TEST-BLINDED.csv")
    n_all = len(data["smiles"])

    graphs = list(np.load(CACHE / "graphs.npy", allow_pickle=True))
    tg = CACHE / "graphs_test.npy"
    if tg.exists():
        test_graphs = list(np.load(tg, allow_pickle=True))
    else:
        test_graphs = graphs_from_smiles(te["SMILES"].tolist())
        np.save(tg, np.array(test_graphs, dtype=object), allow_pickle=True)
    graphs = graphs + test_graphs

    # Extend every target array over the appended test rows, all masked out.
    pad = lambda a, w: np.vstack([a, np.zeros((len(te), w), np.float32)])
    for k, w in (("y", 4), ("lo", 4), ("hi", 4), ("m", 4),
                 ("ar", 4), ("arm", 4), ("ac", 5), ("acm", 5)):
        data[k] = pad(data[k], w)
    desc_te = np.load(CACHE / "X_test.npy")[:, 2048:]
    data["desc"] = np.vstack([data["desc"], desc_te.astype(np.float32)])

    leak = D.public_leak_rows(data["smiles"], n_ch)
    pub_rows = np.setdiff1d(np.arange(n_ch, n_all), leak)
    train_rows = np.concatenate([np.arange(n_ch), pub_rows])
    test_rows = np.arange(n_all, n_all + len(te))
    desc = D.prep_desc(data["desc"], train_rows)

    acc = np.zeros((len(te), 4), np.float32)
    for seed in SEEDS:
        t0 = time.time()
        acc += D.run_fold(graphs, data, desc, train_rows, test_rows, True, seed, log) / len(SEEDS)
        log(f"  dmpnn seed {seed} in {time.time() - t0:.0f}s")
    np.save(f, acc)
    return acc


def main() -> None:
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    te = pd.read_csv(DATA / "cyp-challenge-TEST-BLINDED.csv")
    log("public features")
    pub_tr, pub_te = clean_public_features()
    log("alternate fingerprints for the test set")
    alt_te_raw = alt_test_features()

    ecfp_tr = np.hstack([np.load(CACHE / "X_train.npy"), np.load(CACHE / "aux_train.npy"),
                         pub_tr]).astype(np.float32)
    ecfp_te = np.hstack([np.load(CACHE / "X_test.npy"), np.load(CACHE / "aux_test.npy"),
                         pub_te]).astype(np.float32)
    alt_tr = np.hstack([np.load(CACHE / "altview_X.npy"), np.load(CACHE / "aux_train.npy"),
                        pub_tr]).astype(np.float32)
    alt_te = np.hstack([alt_te_raw, np.load(CACHE / "aux_test.npy"), pub_te]).astype(np.float32)

    log("D-MPNN on all labelled compounds")
    dm_te = dmpnn_preds(meta)

    hold_rows = np.flatnonzero(np.load(CACHE / "holdout_mask.npy"))
    dm_oof = np.load(CACHE / "dmpnn_ho_oof_interval.npy")

    sub = te[["SMILES", "Molecule_Name"]].copy()
    report = {}
    cfg = dict(TREE_CFG); rounds = cfg.pop("rounds")
    tree_params = dict(BASE_PARAMS, **cfg)

    for j, iso in enumerate(ISOFORMS):
        c = f"{iso}_pIC50_direct_inhibition"
        y = meta[c].to_numpy(float)
        lo = meta[f"{c}_conf_low"].to_numpy(float)
        hi = meta[f"{c}_conf_high"].to_numpy(float)
        obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)

        p_ecfp = tree_preds(ecfp_tr, ecfp_te, y, lo, hi, obs, tree_params, rounds)
        p_alt = tree_preds(alt_tr, alt_te, y, lo, hi, obs, ALT_PARAMS, ALT_ROUNDS)
        parts_te = [p_ecfp, p_alt, dm_te[:, j]]

        # Weights and recalibration come from the development out-of-fold predictions.
        tz = np.load(CACHE / f"trees_ho_{iso}.npz")
        az = np.load(CACHE / f"altview_{iso}.npz")
        dev = tz["dev"]
        parts_dev = [tz["oof"], az["oof"], dm_oof[dev, j]]
        yd, lod, hid = y[dev], lo[dev], hi[dev]
        w = min(WEIGHTS, key=lambda w: st_rae(
            yd, sum(wi * p for wi, p in zip(w, parts_dev)), lod, hid))
        m_dev = sum(wi * p for wi, p in zip(w, parts_dev))
        (kk, bb), _ = global_affine(m_dev, yd, lod, hid)
        mu = float(m_dev.mean())

        m_te = sum(wi * p for wi, p in zip(w, parts_te))
        pred = mu + (m_te - mu) * kk + bb
        sub[c] = pred
        report[iso] = (w, kk, bb, float(pred.mean()), float(pred.std()))
        log(f"{iso}: w={w} k={kk:.2f} b={bb:+.2f} test mean={pred.mean():.2f} "
            f"sd={pred.std():.2f}")

    path = OUT / "submission_regression_blend.csv"
    sub.to_csv(path, index=False)
    log(f"\nwrote {path} rows={len(sub)}")


if __name__ == "__main__":
    main()
