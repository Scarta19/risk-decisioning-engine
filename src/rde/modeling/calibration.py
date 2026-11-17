"""Probability calibration.

Cost-based decisions multiply probabilities by money, so a score of 0.10 must mean "about
1 in 10 of these is fraud". Gradient boosting scores are usually well ranked but not
well calibrated, so we fit a calibrator on a dedicated out-of-time window and pick between
Platt scaling and isotonic regression on a held-out slice of that window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    method: str  # "platt" | "isotonic" | "identity"
    model: object | None = None

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if self.method == "platt":
            return self.model.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return np.clip(self.model.predict(raw), EPS, 1 - EPS)
        return raw


def _fit(method: str, raw: np.ndarray, y: np.ndarray) -> Calibrator:
    if method == "platt":
        lr = LogisticRegression(C=1e6)
        lr.fit(_logit(raw).reshape(-1, 1), y)
        return Calibrator("platt", lr)
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw, y)
        return Calibrator("isotonic", iso)
    return Calibrator("identity")


def fit_calibrator(
    raw: np.ndarray, y: np.ndarray, holdout_frac: float = 0.33
) -> tuple[Calibrator, dict[str, float]]:
    """Select a calibrator on a time-ordered holdout, then refit it on all of ``raw``.

    ``raw`` and ``y`` must be in time order so the holdout is the most recent slice.
    Returns the fitted calibrator and the holdout log loss of every candidate.
    """
    raw, y = np.asarray(raw, float), np.asarray(y, int)
    cut = int(len(raw) * (1 - holdout_frac))
    scores = {}
    for method in ("identity", "platt", "isotonic"):
        cal = _fit(method, raw[:cut], y[:cut])
        scores[method] = float(log_loss(y[cut:], cal.transform(raw[cut:]), labels=[0, 1]))
    best = min(scores, key=scores.get)
    return _fit(best, raw, y), scores
