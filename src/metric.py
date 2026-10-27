"""The challenge's soft-threshold RAE, and a band-aware LightGBM objective."""
from __future__ import annotations

import numpy as np


def soft_abs_error(pred, lo, hi):
    """Distance outside the credible band; zero inside it."""
    return np.clip(pred - hi, 0, None) + np.clip(lo - pred, 0, None)


def st_rae(y_true, y_pred, lo, hi) -> float:
    """Soft-threshold RAE, matching evaluation/custom_scoring_functions.py."""
    num = soft_abs_error(np.asarray(y_pred), lo, hi).sum()
    den = soft_abs_error(np.full_like(np.asarray(y_true, float), np.mean(y_true)), lo, hi).sum()
    return float(num / den)


def band_objective(lo, hi, delta: float = 0.5):
    """LightGBM objective matching the metric: zero inside the band, Huber outside."""

    def obj(preds, dataset):
        y = dataset.get_label()  # unused: the band carries the target
        del y
        excess = np.clip(preds - hi, 0, None) - np.clip(lo - preds, 0, None)
        absx = np.abs(excess)
        grad = np.where(absx <= delta, excess, delta * np.sign(excess))
        hess = np.where(absx <= delta, 1.0, 1e-3)
        return grad, hess

    return obj


def band_eval(lo, hi):
    """Matching eval hook, so early stopping tracks the scored quantity."""

    def feval(preds, dataset):
        y = dataset.get_label()
        return "st_rae", st_rae(y, preds, lo, hi), False

    return feval
