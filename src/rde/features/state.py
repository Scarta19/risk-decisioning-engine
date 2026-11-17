"""Point-in-time feature computation shared by training and serving.

Training/serving skew is one of the most common ways production risk models fail. Here a
single class, :class:`FeatureState`, owns every feature definition. Offline, the training
set is built by replaying historical events through it in time order; online, the API
calls the very same ``process`` method per request. Each event only sees events that were
processed *before* it, so the offline features contain no future information.

The state is in-memory and single-process, which is enough for a reference
implementation. A production deployment would back the same interface with a shared,
TTL-based store (for example Redis) keyed by entity.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

HOUR = 3600.0
DAY = 86_400.0
MONTH = 30 * DAY

PAYMENT_CODES = {"card": 0, "wallet": 1, "bank": 2}

FEATURES: list[str] = [
    "is_purchase",
    "amount",
    "log_amount",
    "payment_method_code",
    "log_account_age_days",
    "log_click_to_conversion_sec",
    "geo_mismatch",
    "hour_of_day",
    "user_events_1h",
    "user_events_24h",
    "user_amount_24h",
    "user_purchases_30d",
    "amount_to_user_mean_30d",
    "user_new_device",
    "device_events_1h",
    "device_users_24h",
    "ip_events_1h",
    "ip_users_24h",
    "source_signups_1h",
]


def _epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.timestamp()
    if isinstance(value, datetime):
        return value.timestamp()
    return pd.Timestamp(value).timestamp()


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


@dataclass
class _Window:
    """Events for one entity within a fixed look-back horizon (24h by default)."""

    items: deque
    horizon: float = DAY

    def evict(self, now: float) -> None:
        while self.items and self.items[0][0] <= now - self.horizon:
            self.items.popleft()

    def count_since(self, now: float, horizon: float) -> int:
        n = 0
        for ts, *_ in reversed(self.items):
            if ts <= now - horizon:
                break
            n += 1
        return n


def _day_window() -> _Window:
    return _Window(deque())


def _month_window() -> _Window:
    return _Window(deque(), MONTH)


class FeatureState:
    """Holds rolling entity state and turns a raw event into a model feature vector."""

    def __init__(self) -> None:
        # Module-level factories (not lambdas) keep the state picklable, so a snapshot can
        # ship inside the model bundle.
        self._user = defaultdict(_day_window)  # (ts, amount)
        self._device = defaultdict(_day_window)  # (ts, user)
        self._ip = defaultdict(_day_window)  # (ts, user)
        self._source = defaultdict(_day_window)  # (ts,)
        # Purchases over 30 days rather than all-time totals: cumulative counters grow with
        # the age of the system and drift between training and serving.
        self._user_30d = defaultdict(_month_window)  # (ts, amount)
        self._user_devices: dict[str, set[str]] = defaultdict(set)

    def compute(self, event: Mapping[str, Any]) -> dict[str, float]:
        """Features for ``event`` using only previously processed events (does not record it)."""
        now = _epoch(event["timestamp"])
        user, device, ip = event["user_id"], event["device_id"], event["ip_prefix"]
        is_purchase = event["event_type"] == "purchase"
        amount = float(event.get("amount") or 0.0)

        uw, dw, iw = self._user.get(user), self._device.get(device), self._ip.get(ip)
        for w in (uw, dw, iw):
            if w is not None:
                w.evict(now)

        u30 = self._user_30d.get(user)
        if u30 is not None:
            u30.evict(now)
        n_30d = len(u30.items) if u30 else 0
        mean_30d = sum(a for _, a in u30.items) / n_30d if n_30d else math.nan
        known_user = user in self._user_devices

        age_days = max(now - _epoch(event["account_created_at"]), 0.0) / DAY
        ctc = event.get("click_to_conversion_sec")
        source = event.get("source_id")
        sw = self._source.get(source) if source and source != "organic" else None
        if sw is not None:
            sw.evict(now)

        return {
            "is_purchase": float(is_purchase),
            "amount": amount,
            "log_amount": math.log1p(amount),
            "payment_method_code": float(PAYMENT_CODES.get(event.get("payment_method"), -1)),
            "log_account_age_days": math.log1p(age_days),
            "log_click_to_conversion_sec": math.nan if _missing(ctc) else math.log1p(ctc),
            "geo_mismatch": float(event["ip_country"] != event["user_country"]),
            "hour_of_day": float(pd.Timestamp(now, unit="s").hour),
            "user_events_1h": float(uw.count_since(now, HOUR) if uw else 0),
            "user_events_24h": float(len(uw.items) if uw else 0),
            "user_amount_24h": float(sum(a for _, a in uw.items) if uw else 0.0),
            "user_purchases_30d": float(n_30d),
            "amount_to_user_mean_30d": (
                amount / mean_30d if is_purchase and mean_30d > 0 else math.nan
            ),
            "user_new_device": float(known_user and device not in self._user_devices[user]),
            "device_events_1h": float(dw.count_since(now, HOUR) if dw else 0),
            "device_users_24h": float(len({u for _, u in dw.items}) if dw else 0),
            "ip_events_1h": float(iw.count_since(now, HOUR) if iw else 0),
            "ip_users_24h": float(len({u for _, u in iw.items}) if iw else 0),
            "source_signups_1h": float(sw.count_since(now, HOUR) if sw else 0),
        }

    def update(self, event: Mapping[str, Any]) -> None:
        """Record ``event`` so that later events can see it."""
        now = _epoch(event["timestamp"])
        user = event["user_id"]
        amount = float(event.get("amount") or 0.0)
        self._user[user].items.append((now, amount))
        self._device[event["device_id"]].items.append((now, user))
        self._ip[event["ip_prefix"]].items.append((now, user))
        source = event.get("source_id")
        if event["event_type"] == "signup" and source and source != "organic":
            self._source[source].items.append((now,))
        if event["event_type"] == "purchase":
            self._user_30d[user].items.append((now, amount))
        self._user_devices[user].add(event["device_id"])

    def process(self, event: Mapping[str, Any]) -> dict[str, float]:
        features = self.compute(event)
        self.update(event)
        return features


def build_feature_matrix(
    events: pd.DataFrame | Iterable[Mapping[str, Any]], state: FeatureState | None = None
) -> pd.DataFrame:
    """Replay events in time order and return one feature row per event (same index).

    Pass ``state`` to keep the resulting entity state (e.g. to snapshot it for serving).
    """
    df = events if isinstance(events, pd.DataFrame) else pd.DataFrame(list(events))
    order = np.argsort(df["timestamp"].to_numpy(), kind="mergesort")
    records = df.iloc[order].to_dict("records")
    state = FeatureState() if state is None else state
    rows = [state.process(r) for r in records]
    feats = pd.DataFrame(rows, columns=FEATURES, index=df.index[order])
    return feats.loc[df.index]
