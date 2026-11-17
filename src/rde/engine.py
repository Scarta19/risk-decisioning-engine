"""The deployable unit: model + calibrator + decision policy + feature state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from rde.decision.policy import ACTIONS, DecisionPolicy
from rde.features.state import FEATURES, FeatureState
from rde.modeling.calibration import Calibrator
from rde.modeling.train import predict_raw, reason_codes

BUNDLE_VERSION = 2


@dataclass
class RiskEngine:
    booster: lgb.Booster
    calibrator: Calibrator
    policy: DecisionPolicy
    metadata: dict[str, Any] = field(default_factory=dict)
    state: FeatureState = field(default_factory=FeatureState)

    # -- batch ----------------------------------------------------------------------
    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(predict_raw(self.booster, X))

    # -- online ---------------------------------------------------------------------
    def decide_event(self, event: Mapping[str, Any], record: bool = True) -> dict[str, Any]:
        features = self.state.compute(event)
        if record:
            self.state.update(event)
        X = pd.DataFrame([features], columns=FEATURES)
        raw = float(predict_raw(self.booster, X)[0])
        p = float(self.calibrator.transform(np.array([raw]))[0])
        etype = event["event_type"]
        amount = float(event.get("amount") or 0.0)
        costs = self.policy.expected_costs([p], [amount], [etype])[0]
        decision = ACTIONS[int(np.argmin(costs))]
        return {
            "event_id": event["event_id"],
            "decision": decision,
            "fraud_probability": p,
            "raw_score": raw,
            "expected_cost": {a: round(float(c), 4) for a, c in zip(ACTIONS, costs, strict=True)},
            "reasons": reason_codes(self.booster, X)[0],
            "features": {k: (None if np.isnan(v) else v) for k, v in features.items()},
            "model_version": self.metadata.get("model_version", "unknown"),
        }

    # -- persistence ----------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "bundle_version": BUNDLE_VERSION,
                "booster": self.booster.model_to_string(),
                "best_iteration": self.booster.best_iteration,
                "calibrator": self.calibrator,
                "policy_config": self.policy.config,
                "review_price": self.policy.review_price,
                "features": FEATURES,
                "metadata": self.metadata,
                "state": self.state,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> RiskEngine:
        raw = joblib.load(path)
        if raw.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError(f"Unsupported bundle version {raw.get('bundle_version')}")
        if raw["features"] != FEATURES:
            raise ValueError("Bundle feature list does not match this code version")
        booster = lgb.Booster(model_str=raw["booster"])
        booster.best_iteration = raw["best_iteration"]
        policy = DecisionPolicy(raw["policy_config"], raw["review_price"])
        return cls(booster, raw["calibrator"], policy, raw["metadata"], raw["state"])

    def reset_state(self) -> None:
        """Forget all entity history (cold start)."""
        self.state = FeatureState()
