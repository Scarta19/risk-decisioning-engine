"""Daily monitors that only use information available on the day they run.

Four families, from cheapest and fastest to slowest and most trustworthy:

* ``feature_psi`` and ``score_psi``: population stability of inputs and calibrated scores,
  last 7 days against the reference window. Label-free, available immediately, but blind to
  fraud that looks normal and noisy for shifts that cost nothing.
* ``decision_mix``: decline and review rates against the reference window (binomial z-score).
  Label-free; the only family that can see a rise in false declines, because declined events
  are never labelled.
* ``early_labels``: the age-adjusted early-label test. Every approved event in the last
  ``cohort_days`` is expected to produce ``p * F(age)`` confirmed frauds by today, where ``p``
  is its calibrated score and ``F`` is the historical label-delay curve. More confirmed frauds
  than expected means the model is approving fraud it does not recognise, weeks before the
  labels mature.
* ``matured_labels``: the same test on the newest fully matured cohort. Exact, but it can only
  fire ``maturity_days`` after the problem starts. This is what most teams run today.

An alarm is raised when a monitor's rule holds on ``persistence`` consecutive days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rde.evaluation.metrics import population_stability_index
from rde.features.state import FEATURES

DAY = pd.Timedelta(days=1)
MONITORS = ("feature_psi", "score_psi", "decision_mix", "early_labels", "matured_labels")


@dataclass(frozen=True)
class MonitorConfig:
    window_days: int = 7  # label-free monitors compare the last week with the reference
    cohort_days: int = 60  # approved events whose early labels are tested (= maturity)
    psi_threshold: float = 0.25  # common rule of thumb for a major shift
    z_threshold: float = 4.0
    poisson_alpha: float = 1e-3
    min_excess_ratio: float = 1.5  # ignore statistically significant but tiny excesses
    persistence: int = 2


def poisson_cdf(observed: int, expected: float) -> float:
    """P(X <= observed) for X ~ Poisson(expected), computed in log space."""
    if observed < 0:
        return 0.0
    if expected <= 0:
        return 1.0
    log_terms = [
        k * math.log(expected) - expected - math.lgamma(k + 1) for k in range(observed + 1)
    ]
    m = max(log_terms)
    return float(min(math.exp(m) * sum(math.exp(t - m) for t in log_terms), 1.0))


def poisson_sf(observed: int, expected: float) -> float:
    """P(X >= observed) for X ~ Poisson(expected)."""
    return 1.0 if observed <= 0 else max(1.0 - poisson_cdf(observed - 1, expected), 0.0)


@dataclass
class DelayCurve:
    """Empirical CDF of label delays for approved fraud, from fully matured history."""

    delays: np.ndarray

    @classmethod
    def fit(cls, delays) -> DelayCurve:
        d = np.sort(np.asarray(delays, float))
        if len(d) == 0:
            raise ValueError("need at least one matured fraud label to fit the delay curve")
        return cls(d)

    def __call__(self, age_days) -> np.ndarray:
        age = np.asarray(age_days, float)
        return np.searchsorted(self.delays, age, side="right") / len(self.delays)


@dataclass
class Reference:
    """What the monitors know at deployment time."""

    features: pd.DataFrame
    scores: np.ndarray
    decline_rate: float
    review_rate: float
    approved_calibration: float  # observed / expected fraud among approved reference events
    delay_curve: DelayCurve
    # Fraud arrives in campaigns, so daily decision rates vary far more than a binomial
    # model predicts. The z-scores are divided by sqrt(dispersion), estimated from the
    # day-to-day variation before deployment (quasi-binomial).
    decline_dispersion: float = 1.0
    review_dispersion: float = 1.0


def dispersion(daily_rates, daily_counts) -> float:
    """Quasi-binomial dispersion: observed variance of daily rates / binomial variance."""
    r, n = np.asarray(daily_rates, float), np.asarray(daily_counts, float)
    keep = n > 0
    r, n = r[keep], n[keep]
    if len(r) < 3:
        return 1.0
    mean = float(np.average(r, weights=n))
    binomial = float(np.mean(max(mean * (1 - mean), 1e-12) / n))
    return max(float(np.var(r, ddof=1)) / binomial, 1.0)


def _z(rate: float, ref: float, n: int, phi: float = 1.0) -> float:
    if n == 0:
        return 0.0
    se = math.sqrt(phi * max(ref * (1 - ref), 1e-12) / n)
    return (rate - ref) / se


def _label_test(observed: int, expected: float, cfg: MonitorConfig) -> tuple[float, str]:
    """Two-sided test of confirmed frauds against the model's own prediction.

    ``excess``: more fraud than predicted, the model is approving fraud it does not see.
    ``deficit``: far less than predicted, the model over-scores some traffic; since its
    declines are never labelled, this is the label-side hint of false declines.
    """
    if expected <= 0:
        return (math.nan, "")
    ratio = observed / expected
    if poisson_sf(observed, expected) < cfg.poisson_alpha and ratio >= cfg.min_excess_ratio:
        return ratio, "excess"
    if poisson_cdf(observed, expected) < cfg.poisson_alpha and ratio <= 1 / cfg.min_excess_ratio:
        return ratio, "deficit"
    return ratio, ""


def run_monitors(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    ref: Reference,
    days: pd.DatetimeIndex,
    maturity_days: int,
    cfg: MonitorConfig | None = None,
) -> pd.DataFrame:
    """One row per day with every monitor's statistic and whether its rule held.

    ``frame`` holds the production events with ``p``, ``decision``, ``is_fraud`` and
    ``label_at``. Each day ``T`` is the end of that day: only events before ``T`` and labels
    with ``label_at <= T`` are used.
    """
    cfg = cfg or MonitorConfig()
    ts = frame["timestamp"]
    p = frame["p"].to_numpy(float)
    dec = frame["decision"].to_numpy()
    approved = dec == "approve"
    fraud_known_at = frame["label_at"].where(frame["is_fraud"].astype(bool) & approved)
    rows = []
    for day in days:
        T = day + DAY
        week = ((ts >= T - cfg.window_days * DAY) & (ts < T)).to_numpy()
        n = int(week.sum())
        row = {"day": day, "events": n}

        if n:
            psis = {
                f: population_stability_index(ref.features[f], features.loc[week, f])
                for f in FEATURES
            }
            top = max(psis, key=psis.get)
            row["feature_psi"], row["feature_psi_top"] = psis[top], top
            row["score_psi"] = population_stability_index(pd.Series(ref.scores), pd.Series(p[week]))
            row["decline_rate"] = float((dec[week] == "decline").mean())
            row["review_rate"] = float((dec[week] == "review").mean())
        else:
            row.update(
                feature_psi=0.0,
                feature_psi_top="",
                score_psi=0.0,
                decline_rate=0.0,
                review_rate=0.0,
            )
        # Dispersion measured on single days carries over to a multi-day mean, since
        # var(mean of k days) = phi * r(1 - r) / (k * n_day) = phi * r(1 - r) / n.
        row["decline_z"] = _z(row["decline_rate"], ref.decline_rate, n, ref.decline_dispersion)
        row["review_z"] = _z(row["review_rate"], ref.review_rate, n, ref.review_dispersion)

        # Early labels: approved events of the last cohort_days, aged at T.
        cohort = ((ts >= T - cfg.cohort_days * DAY) & (ts < T)).to_numpy() & approved
        age = ((T - ts[cohort]).dt.total_seconds() / 86_400).to_numpy()
        exp_early = float(np.sum(p[cohort] * ref.delay_curve(age)) * ref.approved_calibration)
        obs_early = int((fraud_known_at[cohort] <= T).sum())
        r, kind = _label_test(obs_early, exp_early, cfg)
        row.update(
            early_observed=obs_early,
            early_expected=exp_early,
            early_ratio=r,
            early_signal=kind,
            early_hit=bool(kind),
        )

        # Matured labels: the newest window_days of events that are now fully matured.
        hi = T - maturity_days * DAY
        mat = ((ts >= hi - cfg.window_days * DAY) & (ts < hi)).to_numpy() & approved
        exp_mat = float(np.sum(p[mat]) * ref.approved_calibration)
        obs_mat = int(frame["is_fraud"].to_numpy()[mat].sum())
        r, kind = _label_test(obs_mat, exp_mat, cfg)
        row.update(
            matured_observed=obs_mat,
            matured_expected=exp_mat,
            matured_ratio=r,
            matured_signal=kind,
            matured_hit=bool(kind),
        )

        row["feature_psi_hit"] = row["feature_psi"] > cfg.psi_threshold
        row["score_psi_hit"] = row["score_psi"] > cfg.psi_threshold
        row["decision_mix_hit"] = max(abs(row["decline_z"]), abs(row["review_z"])) > cfg.z_threshold
        rows.append(row)

    out = pd.DataFrame(rows).set_index("day")
    hits = {
        "feature_psi": "feature_psi_hit",
        "score_psi": "score_psi_hit",
        "decision_mix": "decision_mix_hit",
        "early_labels": "early_hit",
        "matured_labels": "matured_hit",
    }
    for name, col in hits.items():
        streak = out[col].astype(int).groupby((~out[col]).cumsum()).cumsum()
        out[f"alarm_{name}"] = streak >= cfg.persistence
    return out
