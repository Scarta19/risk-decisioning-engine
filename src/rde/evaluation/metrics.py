"""Evaluation helpers: ranking, calibration, drift and decision economics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from rde.decision.policy import DecisionPolicy, realized_cost


def ranking_metrics(y, p) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "base_rate": float(np.mean(y)),
    }


def calibration_metrics(y, p, n_bins: int = 15) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece": expected_calibration_error(y, p, n_bins),
    }


def reliability_table(y, p, n_bins: int = 15) -> pd.DataFrame:
    """Equal-frequency bins: mean predicted vs observed fraud rate."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    t = reliability_table(y, p, n_bins)
    return float(np.sum(t["n"] * (t["predicted"] - t["observed"]).abs()) / t["n"].sum())


def population_stability_index(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """PSI with quantile bins from ``expected``; NaN is treated as its own bin."""
    e, a = expected.to_numpy(float), actual.to_numpy(float)
    edges = np.unique(np.nanquantile(e, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(x):
        counts = np.histogram(x[~np.isnan(x)], bins=edges)[0].astype(float)
        counts = np.append(counts, np.isnan(x).sum())
        return np.clip(counts / max(len(x), 1), 1e-4, None)

    pe, pa = dist(e), dist(a)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def decision_summary(
    name: str,
    policy: DecisionPolicy,
    decisions: np.ndarray,
    y: np.ndarray,
    amount: np.ndarray,
    event_type: np.ndarray,
) -> dict[str, float | str]:
    y = np.asarray(y, int)
    cost = realized_cost(policy, decisions, y, amount, event_type)
    fraud, legit = y == 1, y == 0
    stopped = np.isin(decisions, ["review", "decline"])
    reviewed = decisions == "review"
    return {
        "policy": name,
        "total_cost": float(cost.sum()),
        "cost_per_1k_events": float(cost.mean() * 1000),
        "approve_rate": float(np.mean(decisions == "approve")),
        "review_rate": float(reviewed.mean()),
        "decline_rate": float(np.mean(decisions == "decline")),
        "fraud_intercepted_rate": float(stopped[fraud].mean()) if fraud.any() else float("nan"),
        "fraud_amount_intercepted_rate": float(
            amount[fraud & stopped].sum() / max(amount[fraud].sum(), 1e-9)
        ),
        "legit_declined_rate": float(np.mean(decisions[legit] == "decline")),
        "review_precision": float(y[reviewed].mean()) if reviewed.any() else float("nan"),
    }
