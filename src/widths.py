"""Band-half models. L1 on log half-width so the fit lands on the median, not the tail."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import BASE_PARAMS, CACHE, ISOFORMS, N_FOLDS, _fit  # noqa: E402

# Absolute-error fits, so each model lands on the conditional median half-width.
WIDTH_PARAMS = dict(BASE_PARAMS, objective="regression_l1")
WIDTH_ROUNDS = 400
# A credible interval can be reported as essentially zero-width; floor it before the
# log so one degenerate curve cannot dominate the fit.
FLOOR = 0.02


def fit_widths(x, y, lo, hi, groups, xt):
    """Out-of-fold and full-fit band halves for one isoform."""
    targets = {"lo": np.clip(y - lo, FLOOR, None), "hi": np.clip(hi - y, FLOOR, None)}
    oof, test = {}, {}
    for name, target in targets.items():
        t = np.log(target)
        o = np.empty(len(y))
        for tr_i, va_i in GroupKFold(n_splits=N_FOLDS).split(x, t, groups):
            o[va_i] = _fit(x[tr_i], t[tr_i], WIDTH_PARAMS, WIDTH_ROUNDS, 0).predict(x[va_i])
        oof[name] = np.exp(o)
        test[name] = np.exp(_fit(x, t, WIDTH_PARAMS, WIDTH_ROUNDS, 0).predict(xt))
    return oof, test, targets


def main() -> None:
    xtr = np.hstack([
        np.load(CACHE / "X_train.npy"), np.load(CACHE / "aux_train.npy")
    ]).astype(np.float32)
    xt = np.hstack([
        np.load(CACHE / "X_test.npy"), np.load(CACHE / "aux_test.npy")
    ]).astype(np.float32)

    for iso in ISOFORMS:
        d = np.load(CACHE / f"cv_{iso}.npz")
        x = xtr[d["obs"]]
        oof, test, targets = fit_widths(x, d["y"], d["lo"], d["hi"], d["groups"], xt)
        np.savez_compressed(
            CACHE / f"widths_{iso}.npz",
            oof_lo=oof["lo"], oof_hi=oof["hi"], te_lo=test["lo"], te_hi=test["hi"],
        )
        # Report correlation on the log scale, where the fit was made, alongside the
        # medians: a model can correlate well and still sit at the wrong level.
        msg = [f"{iso}:"]
        for name in ("lo", "hi"):
            t, p = targets[name], oof[name]
            r = np.corrcoef(np.log(t), np.log(p))[0, 1]
            old = d[f"oof_{name}"]
            msg.append(
                f" {name}: r_log={r:.3f} median true={np.median(t):.3f} "
                f"new={np.median(p):.3f} old={np.median(old):.3f}"
            )
        print("".join(msg), flush=True)


if __name__ == "__main__":
    main()
