import numpy as np
import pytest

from rde.config import PolicyConfig
from rde.decision.policy import ACTIONS, DecisionPolicy, realized_cost


@pytest.fixture
def policy():
    return DecisionPolicy(PolicyConfig())


def test_expected_cost_formula(policy):
    c = policy.config.costs["purchase"]
    p, a = 0.2, 100.0
    costs = policy.expected_costs([p], [a], ["purchase"])[0]
    fraud_loss = c.approve_fraud_fixed + c.approve_fraud_amount_rate * a
    assert costs[0] == pytest.approx(p * fraud_loss)
    assert costs[2] == pytest.approx(
        (1 - p) * (c.decline_legit_fixed + c.decline_legit_amount_rate * a)
    )
    assert costs[1] == pytest.approx(
        c.review_fixed
        + p * (1 - c.review_fraud_catch_rate) * fraud_loss
        + (1 - p) * c.review_legit_fixed
    )


def test_decisions_are_monotone_in_risk(policy):
    p = np.linspace(0, 1, 2001)
    for amount in (5.0, 50.0, 500.0):
        d = policy.decide(p, np.full_like(p, amount), np.full(len(p), "purchase"))
        rank = np.array([ACTIONS.index(x) for x in d])
        assert np.all(np.diff(rank) >= 0), f"non-monotone decisions at amount={amount}"
    assert d[0] == "approve" and d[-1] == "decline"


def test_larger_amounts_get_stricter(policy):
    p = np.full(2, 0.03)
    d = policy.decide(p, np.array([10.0, 2000.0]), np.array(["purchase", "purchase"]))
    assert d[0] == "approve" and d[1] != "approve"


def test_review_price_enforces_capacity(policy):
    rng = np.random.default_rng(0)
    n = 20_000
    p = rng.beta(0.5, 8, n)
    amount = rng.lognormal(4, 1, n)
    etype = np.full(n, "purchase")
    unconstrained = np.mean(policy.decide(p, amount, etype) == "review")
    assert unconstrained > policy.config.review_capacity  # the test is meaningful
    price = policy.fit_review_price(p, amount, etype)
    assert price > 0
    assert np.mean(policy.decide(p, amount, etype) == "review") <= policy.config.review_capacity


def test_realized_cost_uses_truth(policy):
    d = np.array(["approve", "decline", "review", "approve"])
    y = np.array([1, 0, 1, 0])
    a = np.full(4, 100.0)
    cost = realized_cost(policy, d, y, a, np.full(4, "purchase"))
    c = policy.config.costs["purchase"]
    assert cost[0] == pytest.approx(c.approve_fraud_fixed + 100.0)
    assert cost[1] == pytest.approx(c.decline_legit_fixed + 30.0)
    assert cost[3] == 0.0


def test_unknown_event_type(policy):
    with pytest.raises(ValueError, match="No cost configuration"):
        policy.expected_costs([0.1], [1.0], ["refund"])
