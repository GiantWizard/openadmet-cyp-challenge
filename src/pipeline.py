"""Train the CYP direct-inhibition models and score them under cluster CV."""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import band_eval, band_objective, soft_abs_error, st_rae  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CACHE = Path(__file__).resolve().parents[1] / "cache"
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
N_FOLDS = 5
SEEDS = (0, 1, 2)

BASE_PARAMS = dict(
    objective="regression",
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
N_ROUNDS = 700


def scaffold_groups(smiles) -> np.ndarray:
    """Group by Bemis-Murcko scaffold, so analog clusters cannot straddle folds."""
    groups, index = [], {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ""
        except Exception:
            scaf = ""
        groups.append(index.setdefault(scaf or smi, len(index)))
    return np.asarray(groups)


def _fit(x, y, params, rounds, seed, obj=None, feval=None, init_score=None):
    p = dict(params, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    if obj is not None:
        p = dict(p, objective=obj)
    ds = lgb.Dataset(x, label=y, init_score=init_score, free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=rounds, feval=feval)


def _predict(model, x, init=0.0):
    return model.predict(x) + init


def build_aux_features(xtr, mtr, xte) -> tuple[np.ndarray, np.ndarray]:
    """Group by Bemis-Murcko scaffold so analog clusters cannot straddle folds."""
    groups = scaffold_groups(mtr["SMILES"])
    aux_tr = np.full((len(mtr), len(ISOFORMS)), np.nan)
    aux_te = np.zeros((xte.shape[0], len(ISOFORMS)))
    for j, iso in enumerate(ISOFORMS):
        y = mtr[f"{iso}_log2fc"].to_numpy(float)
        obs = np.isfinite(y)
        xo, yo, go = xtr[obs], y[obs], groups[obs]
        oof = np.full(obs.sum(), np.nan)
        for tr_i, va_i in GroupKFold(n_splits=N_FOLDS).split(xo, yo, go):
            m = _fit(xo[tr_i], yo[tr_i], BASE_PARAMS, 400, 0)
            oof[va_i] = m.predict(xo[va_i])
        aux_tr[obs, j] = oof
        full = _fit(xo, yo, BASE_PARAMS, 400, 0)
        aux_te[:, j] = full.predict(xte)
        print(f"  aux {iso}: n={obs.sum()} oof_r={np.corrcoef(oof, yo)[0, 1]:.3f}")
    return aux_tr, aux_te


def run_isoform(iso, xtr, mtr, xte, aux_tr, aux_te, tag=""):
    """Group by Bemis-Murcko scaffold so analog clusters cannot straddle folds."""
    col = f"{iso}_pIC50_direct_inhibition"
    y = mtr[col].to_numpy(float)
    lo = mtr[f"{col}_conf_low"].to_numpy(float)
    hi = mtr[f"{col}_conf_high"].to_numpy(float)
    obs = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)

    x = np.hstack([xtr, aux_tr]).astype(np.float32)[obs]
    xt = np.hstack([xte, aux_te]).astype(np.float32)
    y, lo, hi = y[obs], lo[obs], hi[obs]
    groups = scaffold_groups(mtr["SMILES"][obs])

    oof = {k: np.zeros(len(y)) for k in ("l2", "band")}
    oof_lo = np.zeros(len(y))
    oof_hi = np.zeros(len(y))

    for tr_i, va_i in GroupKFold(n_splits=N_FOLDS).split(x, y, groups):
        base = float(np.mean(y[tr_i]))
        p_l2 = np.zeros(len(va_i))
        p_band = np.zeros(len(va_i))
        for seed in SEEDS:
            m = _fit(x[tr_i], y[tr_i], BASE_PARAMS, N_ROUNDS, seed)
            p_l2 += m.predict(x[va_i]) / len(SEEDS)
            init = np.full(len(tr_i), base)
            mb = _fit(
                x[tr_i], y[tr_i], BASE_PARAMS, N_ROUNDS, seed,
                obj=band_objective(lo[tr_i], hi[tr_i]),
                feval=band_eval(lo[tr_i], hi[tr_i]),
                init_score=init,
            )
            p_band += (mb.predict(x[va_i]) + base) / len(SEEDS)
        oof["l2"][va_i] = p_l2
        oof["band"][va_i] = p_band
        # Band-half models: how much slack the metric allows around a compound.
        mlo = _fit(x[tr_i], (y - lo)[tr_i], BASE_PARAMS, 400, 0)
        mhi = _fit(x[tr_i], (hi - y)[tr_i], BASE_PARAMS, 400, 0)
        oof_lo[va_i] = np.clip(mlo.predict(x[va_i]), 0.01, None)
        oof_hi[va_i] = np.clip(mhi.predict(x[va_i]), 0.01, None)

    scores = {k: st_rae(y, v, lo, hi) for k, v in oof.items()}
    np.savez_compressed(
        CACHE / f"cv{tag}_{iso}.npz",
        y=y, lo=lo, hi=hi, groups=groups, obs=obs,
        oof_l2=oof["l2"], oof_band=oof["band"], oof_lo=oof_lo, oof_hi=oof_hi,
    )
    return dict(iso=iso, scores=scores)


def main():
    """Run one isoform per invocation, so a long fit cannot take the others with it."""
    which = sys.argv[1] if len(sys.argv) > 1 else "aux"
    # --public appends the stacked qHTS predictions, so the same cross-validation
    # can be run with and without the public data and the two compared directly.
    use_public = "--public" in sys.argv
    xtr = np.load(CACHE / "X_train.npy")
    xte = np.load(CACHE / "X_test.npy")
    mtr = pd.read_parquet(CACHE / "meta_train.parquet")

    if which == "aux":
        print("building stacked screen features", flush=True)
        aux_tr, aux_te = build_aux_features(xtr, mtr, xte)
        np.save(CACHE / "aux_train.npy", aux_tr)
        np.save(CACHE / "aux_test.npy", aux_te)
        return

    aux_tr = np.load(CACHE / "aux_train.npy")
    aux_te = np.load(CACHE / "aux_test.npy")
    tag = ""
    if use_public:
        aux_tr = np.hstack([aux_tr, np.load(CACHE / "pub_train.npy")])
        aux_te = np.hstack([aux_te, np.load(CACHE / "pub_test.npy")])
        tag = "_pub"
        print(f"  with public features: aux now {aux_tr.shape[1]} columns", flush=True)
    r = run_isoform(which, xtr, mtr, xte, aux_tr, aux_te, tag=tag)
    print(f"{which}{tag}: ST-RAE  L2={r['scores']['l2']:.4f}  band={r['scores']['band']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
