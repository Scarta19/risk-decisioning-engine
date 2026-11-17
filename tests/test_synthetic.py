from rde.data.synthetic import EVENT_COLUMNS, generate_events


def test_schema_and_ordering(small_events):
    assert list(small_events.columns) == EVENT_COLUMNS
    assert small_events["timestamp"].is_monotonic_increasing
    assert small_events["event_id"].is_unique


def test_deterministic_for_seed(small_cfg, small_events):
    again = generate_events(small_cfg.simulation)
    assert again.equals(small_events)


def test_fraud_mix_is_realistic(small_events):
    rate = small_events["is_fraud"].mean()
    assert 0.005 < rate < 0.08
    patterns = set(small_events.loc[small_events["is_fraud"] == 1, "fraud_pattern"])
    assert {"card_testing", "account_takeover", "bot_farm"} <= patterns
    assert (small_events.loc[small_events["is_fraud"] == 0, "fraud_pattern"] == "legit").all()


def test_field_consistency(small_events):
    signups = small_events[small_events["event_type"] == "signup"]
    purchases = small_events[small_events["event_type"] == "purchase"]
    assert (signups["amount"] == 0).all()
    assert purchases["payment_method"].notna().all()
    assert purchases["click_to_conversion_sec"].isna().all()
    assert (small_events["account_created_at"] <= small_events["timestamp"]).all()
