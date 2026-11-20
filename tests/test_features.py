import math

import pandas as pd

from rde.features.state import FEATURES, FeatureState, build_feature_matrix


def _event(i, minutes, user="u1", device="d1", ip="ip1", amount=10.0, etype="purchase", **kw):
    base = {
        "event_id": f"e{i}",
        "timestamp": pd.Timestamp("2026-01-10", tz="UTC") + pd.Timedelta(minutes=minutes),
        "event_type": etype,
        "user_id": user,
        "device_id": device,
        "ip_prefix": ip,
        "ip_country": "US",
        "user_country": "US",
        "source_id": "organic",
        "payment_method": "card" if etype == "purchase" else None,
        "amount": amount if etype == "purchase" else 0.0,
        "account_created_at": pd.Timestamp("2026-01-01", tz="UTC"),
        "click_to_conversion_sec": None,
    }
    base.update(kw)
    return base


def test_windows_count_only_prior_events():
    s = FeatureState()
    f0 = s.process(_event(0, 0))
    assert f0["user_events_1h"] == 0 and f0["user_events_24h"] == 0
    s.process(_event(1, 45))
    f2 = s.process(_event(2, 90))  # first event is now outside 1h but inside 24h
    assert f2["user_events_1h"] == 1
    assert f2["user_events_24h"] == 2
    assert f2["user_amount_24h"] == 20.0
    f3 = s.process(_event(3, 60 * 25))  # >24h after the first two
    assert f3["user_events_24h"] == 1


def test_shared_device_and_new_device():
    s = FeatureState()
    s.process(_event(0, 0, user="a", device="shared"))
    s.process(_event(1, 1, user="b", device="shared"))
    f = s.process(_event(2, 2, user="c", device="shared"))
    assert f["device_users_24h"] == 2
    assert f["user_new_device"] == 0  # unknown user: not a "new device" signal
    f = s.process(_event(3, 3, user="a", device="other"))
    assert f["user_new_device"] == 1


def test_amount_ratio_and_missing_values():
    s = FeatureState()
    first = s.process(_event(0, 0, amount=20.0))
    assert math.isnan(first["amount_to_user_mean_30d"])
    second = s.process(_event(1, 10, amount=60.0))
    assert second["amount_to_user_mean_30d"] == 3.0
    signup = s.process(_event(2, 20, user="new", etype="signup", click_to_conversion_sec=5.0))
    assert signup["log_click_to_conversion_sec"] == math.log1p(5.0)
    assert signup["payment_method_code"] == -1


def test_no_future_leakage(small_events):
    """Features of the first N events must not change when later events are added."""
    n = len(small_events) // 2
    full = build_feature_matrix(small_events)
    prefix = build_feature_matrix(small_events.iloc[:n])
    pd.testing.assert_frame_equal(full.iloc[:n], prefix, check_exact=True)


def test_matrix_has_expected_columns(small_events):
    X = build_feature_matrix(small_events.iloc[:500])
    assert list(X.columns) == FEATURES
    assert X.index.equals(small_events.index[:500])
