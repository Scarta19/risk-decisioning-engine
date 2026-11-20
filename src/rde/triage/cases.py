"""Case store: what an analyst (or the triage agent) can look up about a review case.

Only information available *at decision time* is exposed:

* the case itself, its model score, expected costs and reason codes;
* earlier events from the same user, device or IP prefix, without their labels;
* resolved reference cases from an earlier time window, with their confirmed outcome.

Labels of the cases being triaged are never reachable through the store, so the agent
cannot read the answer. The evaluation reads them separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rde.decision.policy import ACTIONS
from rde.engine import RiskEngine
from rde.features.state import FEATURES
from rde.modeling.train import reason_codes

ENTITY_COLUMNS = {"user": "user_id", "device": "device_id", "ip": "ip_prefix"}
TYPOLOGY = {
    "card_testing": "card_testing",
    "account_takeover": "account_takeover",
    "bot_farm": "bot_or_install_farm",
    "install_farm": "bot_or_install_farm",
    "friendly_fraud": "friendly_fraud",
}


def _num(value: Any, digits: int = 3) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


@dataclass
class CaseStore:
    engine: RiskEngine
    events: pd.DataFrame  # every event, labels included (never returned for open cases)
    features: pd.DataFrame  # aligned with events
    scores: pd.Series  # calibrated probability per event
    decisions: pd.Series  # engine decision per event (NaN outside the scored windows)
    open_cases: pd.Index  # event index of the cases to triage
    reference: pd.Index  # event index of resolved cases whose outcome is known

    # -- construction -------------------------------------------------------------------
    @classmethod
    def from_pipeline(cls, result) -> CaseStore:
        """Open cases = test-window events routed to review; reference = calibration window."""
        if result.events is None or result.features is None or result.scores is None:
            raise ValueError("Pipeline result has no events/features; rerun with this version")
        events, X, p = result.events, result.features, result.scores
        engine = result.engine
        decisions = pd.Series(
            engine.policy.decide(
                p.to_numpy(), events["amount"].to_numpy(float), events["event_type"]
            ),
            index=events.index,
        )
        test_ids = set(result.test_frame["event_id"])
        calib_ids = set(result.calib_frame["event_id"])
        in_test = events["event_id"].isin(test_ids)
        open_cases = events.index[in_test & (decisions == "review")]
        reference = events.index[events["event_id"].isin(calib_ids)]
        return cls(engine, events, X, p, decisions, open_cases, reference)

    def __post_init__(self) -> None:
        self._by_id = pd.Series(self.events.index, index=self.events["event_id"])
        ref = self.features.loc[self.reference, FEATURES].astype(float)
        self._mu = ref.mean()
        self._sd = ref.std().replace(0, 1.0)
        self._ref_matrix = ((ref - self._mu) / self._sd).fillna(0.0).to_numpy()

    # -- helpers --------------------------------------------------------------------------
    def case_ids(self) -> list[str]:
        return self.events.loc[self.open_cases, "event_id"].tolist()

    def _row(self, case_id: str) -> int:
        try:
            idx = int(self._by_id[case_id])
        except KeyError as exc:
            raise KeyError(f"Unknown case_id {case_id!r}") from exc
        if idx not in self.open_cases:
            raise KeyError(f"{case_id!r} is not an open review case")
        return idx

    def label(self, case_id: str) -> tuple[int, str]:
        """Ground truth, for evaluation only. Not exposed as a tool."""
        e = self.events.loc[self._row(case_id)]
        return int(e["is_fraud"]), str(e["fraud_pattern"])

    def expected_costs(self, case_id: str) -> dict[str, float]:
        idx = self._row(case_id)
        e = self.events.loc[idx]
        costs = self.engine.policy.expected_costs(
            [self.scores.loc[idx]], [float(e["amount"] or 0.0)], [e["event_type"]]
        )[0]
        return {a: round(float(c), 2) for a, c in zip(ACTIONS, costs, strict=True)}

    # -- tools ----------------------------------------------------------------------------
    def get_case_summary(self, case_id: str) -> dict[str, Any]:
        idx = self._row(case_id)
        e = self.events.loc[idx]
        x = self.features.loc[[idx], FEATURES]
        age_days = (e["timestamp"] - e["account_created_at"]).total_seconds() / 86_400
        return {
            "case_id": case_id,
            "timestamp": str(e["timestamp"]),
            "event_type": e["event_type"],
            "amount_usd": _num(e["amount"], 2),
            "payment_method": e["payment_method"],
            "user_country": e["user_country"],
            "ip_country": e["ip_country"],
            "account_age_days": _num(age_days, 1),
            "click_to_conversion_sec": _num(e["click_to_conversion_sec"], 1),
            "acquisition_source": e["source_id"],
            "model": {
                "fraud_probability": _num(self.scores.loc[idx], 4),
                "engine_decision": "review",
                "expected_cost_usd": self.expected_costs(case_id),
                "top_risk_factors": reason_codes(self.engine.booster, x, top_k=5)[0],
            },
            "features": {f: _num(v) for f, v in x.iloc[0].items()},
        }

    def get_entity_history(
        self, case_id: str, entity: str, lookback_hours: float = 72.0, limit: int = 10
    ) -> dict[str, Any]:
        if entity not in ENTITY_COLUMNS:
            raise ValueError(f"entity must be one of {sorted(ENTITY_COLUMNS)}")
        lookback_hours = float(min(max(lookback_hours, 1.0), 720.0))
        idx = self._row(case_id)
        e = self.events.loc[idx]
        col = ENTITY_COLUMNS[entity]
        start = e["timestamp"] - pd.Timedelta(hours=lookback_hours)
        prior = self.events[
            (self.events[col] == e[col])
            & (self.events["timestamp"] < e["timestamp"])
            & (self.events["timestamp"] >= start)
        ]
        recent = prior.sort_values("timestamp", ascending=False).head(int(min(max(limit, 1), 25)))
        return {
            "case_id": case_id,
            "entity": entity,
            "lookback_hours": lookback_hours,
            "prior_events": int(len(prior)),
            "distinct_users": int(prior["user_id"].nunique()),
            "distinct_devices": int(prior["device_id"].nunique()),
            "distinct_ip_countries": int(prior["ip_country"].nunique()),
            "purchase_amount_usd": _num(prior["amount"].fillna(0).sum(), 2),
            "most_recent": [
                {
                    "hours_before_case": _num(
                        (e["timestamp"] - r["timestamp"]).total_seconds() / 3600, 2
                    ),
                    "event_type": r["event_type"],
                    "amount_usd": _num(r["amount"], 2),
                    "ip_country": r["ip_country"],
                    "same_user": bool(r["user_id"] == e["user_id"]),
                    "same_device": bool(r["device_id"] == e["device_id"]),
                    "model_fraud_probability": _num(self.scores.get(i), 4),
                }
                for i, r in recent.iterrows()
            ],
            "note": "Outcomes of these earlier events are not known yet.",
        }

    def find_similar_resolved_cases(self, case_id: str, k: int = 8) -> dict[str, Any]:
        k = int(min(max(k, 1), 20))
        idx = self._row(case_id)
        x = (self.features.loc[idx, FEATURES].astype(float) - self._mu) / self._sd
        dist = np.sqrt(((self._ref_matrix - x.fillna(0.0).to_numpy()) ** 2).sum(axis=1))
        order = np.argsort(dist)[:k]
        rows = self.reference[order]
        ref = self.events.loc[rows]
        out = []
        for i, d in zip(rows, dist[order], strict=True):
            r = self.events.loc[i]
            out.append(
                {
                    "distance": _num(d, 3),
                    "event_type": r["event_type"],
                    "amount_usd": _num(r["amount"], 2),
                    "model_fraud_probability": _num(self.scores.loc[i], 4),
                    "outcome": "fraud" if r["is_fraud"] else "legitimate",
                    "confirmed_typology": TYPOLOGY.get(str(r["fraud_pattern"]))
                    if r["is_fraud"]
                    else None,
                }
            )
        return {
            "case_id": case_id,
            "k": k,
            "fraud_share_among_neighbours": _num(ref["is_fraud"].mean(), 3),
            "neighbours": out,
            "note": (
                "Resolved cases from an earlier window. Distance is on standardised model features."
            ),
        }
