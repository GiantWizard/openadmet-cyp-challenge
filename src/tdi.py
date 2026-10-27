"""TDI classification track."""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import featurise_frame  # noqa: E402
from pipeline import BASE_PARAMS, CACHE, N_FOLDS, SEEDS, scaffold_groups  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
OUT = Path(__file__).resolve().parents[1] / "submissions"
OUT.mkdir(exist_ok=True)

TDI_ISOFORMS = ["CYP2D6", "CYP3A4"]
ARMS = ["pIC50_direct_inhibition", "pIC50_TDI_condition"]

CLF_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=20,
    feature_fraction=0.3,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbosity=-1,
    num_threads=8,
)
N_ROUNDS_CLF = 600


def _features():
    """Featurise the TDI training compounds and the blind set, with caching."""
    f_tr, f_te = CACHE / "X_tdi_train.npy", CACHE / "X_test.npy"
    tdi = pd.read_csv(DATA / "cyp-challenge-TRAIN_TDI.csv")
    if f_tr.exists():
        return np.load(f_tr), tdi, np.load(f_te)
    x, _ = featurise_frame(tdi)
    np.save(f_tr, x)
    return x, tdi, np.load(f_te)


def _stacked_arms(x, tdi, xte, groups):
    """Out-of-fold predictions of both assay arms, as features for the classifier."""
    cache = CACHE / "tdi_arms.npz"
    if cache.exists():
        z = np.load(cache)
        print("  arms: cached", flush=True)
        return z["tr"], z["te"]
    aux_tr = np.full((len(tdi), len(TDI_ISOFORMS) * len(ARMS)), np.nan)
    aux_te = np.zeros((len(xte), len(TDI_ISOFORMS) * len(ARMS)))
    col = 0
    for iso in TDI_ISOFORMS:
        for arm in ARMS:
            y = tdi[f"{iso}_{arm}"].to_numpy(float)
            obs = np.isfinite(y)
            xo, yo, go = x[obs], y[obs], groups[obs]
            oof = np.full(obs.sum(), np.nan)
            for tr_i, va_i in GroupKFold(n_splits=N_FOLDS).split(xo, yo, go):
                m = lgb.train(dict(BASE_PARAMS, seed=0), lgb.Dataset(xo[tr_i], yo[tr_i]), 400)
                oof[va_i] = m.predict(xo[va_i])
            aux_tr[obs, col] = oof
            full = lgb.train(dict(BASE_PARAMS, seed=0), lgb.Dataset(xo, yo), 400)
            aux_te[:, col] = full.predict(xte)
            print(f"  arm {iso} {arm}: n={obs.sum()} r={np.corrcoef(oof, yo)[0, 1]:.3f}", flush=True)
            col += 1
    # The label is a shift, so hand the classifier the shift itself.
    for j, _ in enumerate(TDI_ISOFORMS):
        d, t = 2 * j, 2 * j + 1
        aux_tr = np.column_stack([aux_tr, aux_tr[:, t] - aux_tr[:, d]])
        aux_te = np.column_stack([aux_te, aux_te[:, t] - aux_te[:, d]])
    np.savez_compressed(CACHE / "tdi_arms.npz", tr=aux_tr, te=aux_te)
    return aux_tr, aux_te


def best_threshold(y, prob):
    """Threshold maximising MCC, searched over the observed probability grid."""
    grid = np.quantile(prob, np.linspace(0.01, 0.99, 197))
    scores = [matthews_corrcoef(y, prob >= t) for t in grid]
    k = int(np.argmax(scores))
    return float(grid[k]), float(scores[k])


def main() -> None:
    x, tdi, xte = _features()
    groups = scaffold_groups(tdi["SMILES"])
    print("stacking both assay arms", flush=True)
    aux_tr, aux_te = _stacked_arms(x, tdi, xte, groups)
    xf = np.hstack([x, aux_tr]).astype(np.float32)
    xtf = np.hstack([xte, aux_te]).astype(np.float32)

    sub = pd.read_csv(DATA / "cyp-challenge-TEST-BLINDED.csv")[["SMILES", "Molecule_Name"]]
    alt = sub.copy()
    report = {}

    for iso in TDI_ISOFORMS:
        lab = tdi[f"{iso}_is_TDI"]
        obs = lab.notna().to_numpy()
        y = lab[obs].astype(int).to_numpy()
        xo, go = xf[obs], groups[obs]

        oof = np.zeros(len(y))
        for tr_i, va_i in GroupKFold(n_splits=N_FOLDS).split(xo, y, go):
            p = np.zeros(len(va_i))
            for seed in SEEDS:
                m = lgb.train(
                    dict(CLF_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed),
                    lgb.Dataset(xo[tr_i], y[tr_i]), N_ROUNDS_CLF,
                )
                p += m.predict(xo[va_i]) / len(SEEDS)
            oof[va_i] = p

        thr, mcc = best_threshold(y, oof)
        mcc_half = matthews_corrcoef(y, oof >= 0.5)
        oof_rate = float((oof >= thr).mean())
        report[iso] = dict(n=len(y), pos=float(y.mean()), thr=thr, mcc=mcc,
                           mcc_half=mcc_half, oof_rate=oof_rate)

        p_te = np.zeros(len(xtf))
        for seed in SEEDS:
            m = lgb.train(
                dict(CLF_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed),
                lgb.Dataset(xo, y), N_ROUNDS_CLF,
            )
            p_te += m.predict(xtf) / len(SEEDS)
        # Carry the calling rate across, not the probability cut: the blind set's
        # probability distribution is shifted.
        thr_rate = float(np.quantile(p_te, 1.0 - oof_rate))
        report[iso]["thr_rate"] = thr_rate
        report[iso]["test_pos_abs"] = float((p_te >= thr).mean())
        report[iso]["test_pos_rate"] = float((p_te >= thr_rate).mean())
        sub[f"{iso}_is_TDI"] = p_te >= thr_rate
        alt[f"{iso}_is_TDI"] = p_te >= thr
        np.savez_compressed(CACHE / f"tdi_prob_{iso}.npz", oof=oof, y=y, te=p_te)

    path = OUT / "submission_classification.csv"
    sub.to_csv(path, index=False)
    alt.to_csv(OUT / "submission_classification_abs_threshold.csv", index=False)
    print(f"\nwrote {path} rows={len(sub)}")
    for iso, r in report.items():
        print(f"{iso}: n={r['n']} train_pos={r['pos']:.3f} | MCC@0.5={r['mcc_half']:.4f} "
              f"MCC@tuned={r['mcc']:.4f} (thr={r['thr']:.3f}, oof calls {r['oof_rate']:.3f})")
        print(f"         test calls: absolute cut {r['test_pos_abs']:.3f} | "
              f"rate-matched {r['test_pos_rate']:.3f} (thr={r['thr_rate']:.3f})")
    print(f"MACRO MCC tuned = {np.mean([r['mcc'] for r in report.values()]):.4f}")


if __name__ == "__main__":
    main()
