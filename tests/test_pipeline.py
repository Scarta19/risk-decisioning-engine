import numpy as np
import pandas as pd

from rde.engine import RiskEngine
from rde.evaluation.report import render_markdown
from rde.features.state import build_feature_matrix


def test_pipeline_produces_useful_model(pipeline_result):
    m = pipeline_result.metrics
    assert m["ranking_test"]["roc_auc"] > 0.75
    costs = {p["policy"]: p["cost_per_1k_events"] for p in m["policies_test"]}
    assert costs["cost_policy"] < costs["approve_all"]


def test_bundle_round_trip_and_training_serving_parity(pipeline_result, bundle_path, small_events):
    engine = RiskEngine.load(bundle_path)
    test = pipeline_result.test_frame.head(200)
    X = build_feature_matrix(small_events).loc[test.index]
    np.testing.assert_allclose(engine.score(X), test["p"].to_numpy(), rtol=1e-9)

    # Replaying history through the online path reproduces the offline features exactly.
    online = RiskEngine.load(bundle_path)
    online.reset_state()
    records = small_events.loc[: test.index[-1]].to_dict("records")
    served = {}
    for r in records:
        out = online.decide_event(r)
        served[r["event_id"]] = out
    for idx, row in test.iterrows():
        out = served[row["event_id"]]
        assert out["decision"] == row["decision"]
        assert abs(out["fraud_probability"] - row["p"]) < 1e-9
        offline = X.loc[idx].to_dict()
        for k, v in out["features"].items():
            assert (v is None and np.isnan(offline[k])) or v == offline[k], k


def test_report_renders(pipeline_result):
    md = render_markdown(pipeline_result.metrics)
    assert "synthetic data" in md and "| Policy |" in md


def test_bundle_ships_state_snapshot(bundle_path, small_events):
    engine = RiskEngine.load(bundle_path)
    last = small_events.iloc[-1].to_dict()
    later = last | {"event_id": "later", "timestamp": last["timestamp"] + pd.Timedelta(minutes=1)}
    feats = engine.decide_event(later, record=False)["features"]
    assert feats["user_events_1h"] >= 1  # the snapshot knows this user's recent activity
