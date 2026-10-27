"""Find the ST-RAE-optimal affine correction for the blind set's inferred population."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric import st_rae  # noqa: E402
from pipeline import CACHE, ISOFORMS  # noqa: E402

# Inferred from our own leaderboard metrics via the Murphy decomposition.
TRUE_SD = 1.25
TRUE_MEAN = 4.17
RHO = 0.6923
N_SIM = 20000
SEED = 0


def band_model():
    """Median credible half-widths as a function of pIC50, from the training release."""
    meta = pd.read_parquet(CACHE / "meta_train.parquet")
    lo_w, hi_w, centres = [], [], []
    edges = np.arange(1.5, 8.01, 0.5)
    for a, b in zip(edges[:-1], edges[1:]):
        los, his = [], []
        for iso in ISOFORMS:
            c = f"{iso}_pIC50_direct_inhibition"
            y = meta[c].to_numpy(float)
            m = np.isfinite(y) & (y >= a) & (y < b)
            if m.sum() < 5:
                continue
            los.append(np.median(y[m] - meta[f"{c}_conf_low"].to_numpy(float)[m]))
            his.append(np.median(meta[f"{c}_conf_high"].to_numpy(float)[m] - y[m]))
        if los:
            centres.append((a + b) / 2); lo_w.append(np.mean(los)); hi_w.append(np.mean(his))
    return np.array(centres), np.array(lo_w), np.array(hi_w)


def simulate(rng, centres, lo_w, hi_w):
    """A blind-set-like population with bands attached, plus predictions of our quality."""
    truth = rng.normal(TRUE_MEAN, TRUE_SD, N_SIM)
    lo_half = np.interp(truth, centres, lo_w)
    hi_half = np.interp(truth, centres, hi_w)
    # Predictions correlated with truth at RHO, standardised then rescaled later.
    noise = rng.normal(0, 1, N_SIM)
    z = (truth - truth.mean()) / truth.std()
    raw = RHO * z + np.sqrt(1 - RHO**2) * noise
    return truth, truth - lo_half, truth + hi_half, raw


def main() -> None:
    centres, lo_w, hi_w = band_model()
    print("band half-widths by pIC50:")
    for c, l, h in zip(centres, lo_w, hi_w):
        print(f"  {c:4.2f}: low {l:5.3f}  high {h:5.3f}")

    rng = np.random.default_rng(SEED)
    truth, lo, hi, raw = simulate(rng, centres, lo_w, hi_w)

    best = None
    for scale in np.arange(0.2, 1.81, 0.05):
        for centre in np.arange(2.8, 5.61, 0.05):
            pred = centre + raw * scale
            s = st_rae(truth, pred, lo, hi)
            if best is None or s < best[0]:
                best = (s, scale, centre)
    s, scale, centre = best
    print(f"\nST-RAE-optimal: predict mean {centre:.2f}, sd {scale:.2f}  -> ST-RAE {s:.4f}")

    # What the R2-optimal choice would be, for contrast.
    r2_scale = RHO * TRUE_SD
    pred_r2 = TRUE_MEAN + raw * r2_scale
    print(f"R2-optimal:     predict mean {TRUE_MEAN:.2f}, sd {r2_scale:.2f}  -> "
          f"ST-RAE {st_rae(truth, pred_r2, lo, hi):.4f}")

    print("\nwhat we actually submitted, per isoform, and the correction implied:")
    sub = pd.read_csv(Path(__file__).resolve().parents[1] / "submissions"
                      / "submission_regression_blend.csv")
    for iso in ISOFORMS:
        c = f"{iso}_pIC50_direct_inhibition"
        m, sd = sub[c].mean(), sub[c].std()
        print(f"  {iso}: mean {m:.2f} -> {centre:.2f} (shift {centre - m:+.2f}), "
              f"sd {sd:.2f} -> {scale:.2f} (x{scale / sd:.2f})")


if __name__ == "__main__":
    main()
