"""End-to-end offline pipeline: simulate -> features -> train -> calibrate -> decide -> evaluate."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rde import __version__
from rde.config import Config
from rde.data.synthetic import generate_events
from rde.decision.policy import DecisionPolicy, realized_cost
from rde.engine import RiskEngine
from rde.evaluation import metrics as M
from rde.features.state import FEATURES, FeatureState, build_feature_matrix
from rde.modeling.calibration import fit_calibrator
from rde.modeling.train import predict_raw, train_model

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    engine: RiskEngine
    metrics: dict[str, Any]
    test_frame: pd.DataFrame  # test events with scores and decisions, for reporting
    calib_frame: pd.DataFrame
    events: pd.DataFrame | None = None  # all simulated events, in time order
    features: pd.DataFrame | None = None  # point-in-time features, aligned with ``events``
    scores: pd.Series | None = None  # calibrated probability for every event


def _split(events: pd.DataFrame, cfg: Config) -> pd.Series:
    day = (events["timestamp"] - events["timestamp"].min().normalize()).dt.days
    s = cfg.split
    return pd.Series(
        np.select(
            [day < s.train_end_day, day < s.valid_end_day, day < s.calib_end_day],
            ["train", "valid", "calib"],
            default="test",
        ),
        index=events.index,
    )


def _best_single_threshold(policy, p, y, amount, etype) -> float:
    """Approve/decline-only baseline: one global score cut-off that minimises cost."""
    grid = np.unique(np.quantile(p, np.linspace(0.5, 1.0, 501)))
    costs = [
        realized_cost(policy, np.where(p >= t, "decline", "approve"), y, amount, etype).sum()
        for t in grid
    ]
    return float(grid[int(np.argmin(costs))])


def run_pipeline(cfg: Config, events: pd.DataFrame | None = None) -> PipelineResult:
    if events is None:
        log.info("Simulating events")
        events = generate_events(cfg.simulation)
    log.info("Building point-in-time features for %d events", len(events))
    state = FeatureState()
    X = build_feature_matrix(events, state)
    y = events["is_fraud"].astype(int)
    split = _split(events, cfg)
    part = {k: split == k for k in ("train", "valid", "calib", "test")}

    log.info("Training LightGBM")
    booster = train_model(
        X[part["train"]], y[part["train"]], X[part["valid"]], y[part["valid"]], cfg.model
    )

    raw = pd.Series(predict_raw(booster, X), index=X.index)
    calibrator, cal_scores = fit_calibrator(raw[part["calib"]], y[part["calib"]])
    p = pd.Series(calibrator.transform(raw), index=X.index)

    policy = DecisionPolicy(cfg.policy)
    amount, etype = events["amount"].to_numpy(float), events["event_type"].to_numpy()
    c, t = part["calib"].to_numpy(), part["test"].to_numpy()
    review_price = policy.fit_review_price(p[c], amount[c], etype[c])
    threshold = _best_single_threshold(
        policy, p[c].to_numpy(), y[c].to_numpy(), amount[c], etype[c]
    )

    # Test-window decisions for each policy.
    pt, yt, at, et = p[t].to_numpy(), y[t].to_numpy(), amount[t], etype[t]
    no_review = np.where(
        policy.expected_costs(pt, at, et)[:, 2] < policy.expected_costs(pt, at, et)[:, 0],
        "decline",
        "approve",
    )
    full = policy.decide(pt, at, et)
    decisions = {
        "approve_all": np.full(len(pt), "approve"),
        "single_threshold": np.where(pt >= threshold, "decline", "approve"),
        "cost_policy_no_review": no_review,
        "cost_policy": full,
    }
    policy_table = [M.decision_summary(k, policy, d, yt, at, et) for k, d in decisions.items()]

    test_frame = events[t].assign(raw_score=raw[t].to_numpy(), p=pt, decision=full)
    pattern_recall = (
        test_frame[test_frame["is_fraud"] == 1]
        .assign(intercepted=lambda d: d["decision"].isin(["review", "decline"]))
        .groupby("fraud_pattern")
        .agg(events=("event_id", "size"), intercepted_rate=("intercepted", "mean"))
        .reset_index()
        .to_dict("records")
    )
    psi = {f: M.population_stability_index(X.loc[part["train"], f], X.loc[t, f]) for f in FEATURES}
    importance = dict(
        zip(FEATURES, booster.feature_importance("gain").round(1).tolist(), strict=True)
    )

    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(events.drop(columns=["timestamp", "account_created_at"]))
        .to_numpy()
        .tobytes()
    ).hexdigest()[:12]
    metadata = {
        "model_version": f"{__version__}+{data_hash}",
        "package_version": __version__,
        "data_hash": data_hash,
        "best_iteration": int(booster.best_iteration),
        "calibration_method": calibrator.method,
        "review_price": review_price,
        "single_threshold": threshold,
        "trained_on": {
            k: str(events.loc[part[k], "timestamp"].min().date())
            + " to "
            + str(events.loc[part[k], "timestamp"].max().date())
            for k in part
        },
    }
    metrics = {
        "metadata": metadata,
        "data": {
            "events": int(len(events)),
            "fraud_rate": float(y.mean()),
            "by_split": {
                k: {"events": int(v.sum()), "fraud_rate": float(y[v].mean())}
                for k, v in part.items()
            },
            "by_pattern": events["fraud_pattern"].value_counts().to_dict(),
        },
        "ranking_test": M.ranking_metrics(yt, pt),
        "calibration_selection_logloss": cal_scores,
        "calibration_test": {
            "raw": M.calibration_metrics(yt, raw[t].to_numpy()),
            "calibrated": M.calibration_metrics(yt, pt),
        },
        "policies_test": policy_table,
        "pattern_recall_test": pattern_recall,
        "psi_train_vs_test": psi,
        "feature_importance_gain": importance,
        "config": cfg.to_dict(),
    }
    # The serving state starts from a snapshot of entity history at the end of the data,
    # so the API does not treat every known user, device or IP as brand new.
    metadata["state_snapshot_at"] = str(events["timestamp"].max())
    engine = RiskEngine(booster, calibrator, policy, metadata, state)
    calib_frame = events[c].assign(raw_score=raw[c].to_numpy(), p=p[c].to_numpy())
    return PipelineResult(engine, metrics, test_frame, calib_frame, events, X, p)


def save_outputs(result: PipelineResult, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result.engine.save(artifacts_dir / "model_bundle.joblib")
    (artifacts_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2, default=str))
