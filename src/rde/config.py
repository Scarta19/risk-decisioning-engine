"""Typed configuration loaded from YAML.

Every number that shapes the simulation, the time split or the business costs lives in
``configs/*.yaml`` so that experiments are reproducible and assumptions are explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 7
    start: str = "2026-01-01"
    n_days: int = 120
    n_users: int = 15_000
    n_sources: int = 40
    n_bad_sources: int = 3
    new_user_share: float = 0.3
    # Fraud intensities are per 10,000 users per simulated day, so that smaller test
    # configurations keep roughly the same fraud mix as the default one.
    card_testing_campaigns_per_10k_users_day: float = 0.25
    ato_incidents_per_10k_users_day: float = 1.6
    bot_campaigns_per_10k_users_day: float = 0.16
    friendly_fraud_rate: float = 0.002


@dataclass(frozen=True)
class SplitConfig:
    """Out-of-time split, in days from the simulation start (end-exclusive)."""

    train_end_day: int = 75
    valid_end_day: int = 90
    calib_end_day: int = 105


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 50
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    seed: int = 7


@dataclass(frozen=True)
class EventCosts:
    """Illustrative unit economics for one event type. All values are assumptions."""

    approve_fraud_fixed: float
    approve_fraud_amount_rate: float
    decline_legit_fixed: float
    decline_legit_amount_rate: float
    review_fixed: float
    review_legit_fixed: float
    review_fraud_catch_rate: float


@dataclass(frozen=True)
class PolicyConfig:
    review_capacity: float = 0.02
    costs: dict[str, EventCosts] = field(
        default_factory=lambda: {
            "purchase": EventCosts(15.0, 1.0, 5.0, 0.3, 3.0, 1.0, 0.9),
            "signup": EventCosts(2.5, 0.0, 6.0, 0.0, 1.5, 0.5, 0.9),
        }
    )


@dataclass(frozen=True)
class Config:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, raw: dict[str, Any] | None):
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def config_from_dict(raw: dict[str, Any]) -> Config:
    policy_raw = dict(raw.get("policy") or {})
    costs_raw = policy_raw.pop("costs", None)
    policy = _build(PolicyConfig, policy_raw)
    if costs_raw is not None:
        policy = PolicyConfig(
            review_capacity=policy.review_capacity,
            costs={k: EventCosts(**v) for k, v in costs_raw.items()},
        )
    cfg = Config(
        simulation=_build(SimulationConfig, raw.get("simulation")),
        split=_build(SplitConfig, raw.get("split")),
        model=_build(ModelConfig, raw.get("model")),
        policy=policy,
    )
    s = cfg.split
    if not 0 < s.train_end_day < s.valid_end_day < s.calib_end_day < cfg.simulation.n_days:
        raise ValueError("split days must satisfy 0 < train < valid < calib < n_days")
    return cfg


def load_config(path: str | Path) -> Config:
    with open(path) as fh:
        return config_from_dict(yaml.safe_load(fh) or {})
