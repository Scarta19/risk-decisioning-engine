"""Cost-based approve / review / decline policy.

For each event with calibrated fraud probability ``p`` and amount ``a`` the expected cost
of each action is::

    approve = p * (approve_fraud_fixed + approve_fraud_amount_rate * a)
    decline = (1 - p) * (decline_legit_fixed + decline_legit_amount_rate * a)
    review  = review_fixed + review_price
              + p * (1 - review_fraud_catch_rate) * fraud_loss(a)
              + (1 - p) * review_legit_fixed

and the policy picks the cheapest action. Thresholds are therefore not hand-picked: they
fall out of the unit economics and depend on the amount at stake.

Manual review is a scarce resource. ``review_price`` is a shadow price (a Lagrange
multiplier) on analyst capacity: it is tuned on the calibration window so that the share
of events sent to review stays within ``review_capacity``. Because it is a single number,
the exact same rule runs in batch evaluation and in the online API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rde.config import EventCosts, PolicyConfig

ACTIONS = ("approve", "review", "decline")
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


@dataclass
class DecisionPolicy:
    config: PolicyConfig
    review_price: float = 0.0

    def _costs(self, event_type: str) -> EventCosts:
        try:
            return self.config.costs[event_type]
        except KeyError as exc:
            raise ValueError(f"No cost configuration for event_type={event_type!r}") from exc

    def expected_costs(
        self, p: np.ndarray, amount: np.ndarray, event_type: np.ndarray, review_price=None
    ) -> np.ndarray:
        """Return an (n, 3) array of expected costs in ACTIONS order."""
        p = np.asarray(p, float)
        amount = np.asarray(amount, float)
        event_type = np.asarray(event_type)
        price = self.review_price if review_price is None else review_price
        out = np.empty((len(p), 3))
        for etype in np.unique(event_type):
            m = event_type == etype
            c = self._costs(str(etype))
            fraud_loss = c.approve_fraud_fixed + c.approve_fraud_amount_rate * amount[m]
            legit_loss = c.decline_legit_fixed + c.decline_legit_amount_rate * amount[m]
            out[m, 0] = p[m] * fraud_loss
            out[m, 1] = (
                c.review_fixed
                + price
                + p[m] * (1 - c.review_fraud_catch_rate) * fraud_loss
                + (1 - p[m]) * c.review_legit_fixed
            )
            out[m, 2] = (1 - p[m]) * legit_loss
        return out

    def decide(self, p, amount, event_type) -> np.ndarray:
        costs = self.expected_costs(p, amount, event_type)
        return np.array(ACTIONS)[np.argmin(costs, axis=1)]

    def fit_review_price(self, p, amount, event_type) -> float:
        """Smallest shadow price that keeps the review rate within capacity."""
        base = self.expected_costs(p, amount, event_type, review_price=0.0)
        savings = np.minimum(base[:, 0], base[:, 2]) - base[:, 1]  # >0 means review wins
        cap = int(np.floor(self.config.review_capacity * len(savings)))
        positive = np.sort(savings[savings > 0])[::-1]
        # Just above the (cap+1)-th largest saving, so ties and float rounding in the cost
        # comparison cannot push the review rate over capacity.
        price = 0.0 if len(positive) <= cap else float(positive[cap] * (1 + 1e-9) + 1e-9)
        self.review_price = price
        return price


def realized_cost(
    policy: DecisionPolicy, decisions: np.ndarray, y: np.ndarray, amount, event_type
) -> np.ndarray:
    """Cost actually incurred per event given the true label (review uses expected catch).

    The shadow price is excluded: it is a planning device, not money spent.
    """
    y = np.asarray(y, int)
    p_true = y.astype(float)
    costs = policy.expected_costs(p_true, amount, event_type, review_price=0.0)
    idx = np.array([ACTION_INDEX[d] for d in decisions], dtype=int)
    return costs[np.arange(len(y)), idx]
