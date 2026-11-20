import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ROOT


@pytest.fixture(scope="module")
def client(bundle_path, monkeypatch_module):
    monkeypatch_module.setenv("RDE_MODEL_PATH", str(bundle_path))
    from rde.serving.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _example():
    return json.loads((ROOT / "examples" / "purchase.json").read_text())


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/v1/model").json()
    assert info["model_version"] and "review_price" in info


def test_decision_contract(client):
    r = client.post("/v1/decisions", json=_example(), params={"record": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in {"approve", "review", "decline"}
    assert 0 < body["fraud_probability"] < 1
    assert set(body["expected_cost"]) == {"approve", "review", "decline"}
    assert min(body["expected_cost"], key=body["expected_cost"].get) == body["decision"]


def test_burst_raises_risk(client):
    base = _example() | {
        "user_id": "burst_user",
        "device_id": "burst_dev",
        "ip_prefix": "burst_ip",
        "amount": 3.0,
        "account_created_at": "2026-05-01T03:00:00Z",
    }
    first = client.post("/v1/decisions", json=base | {"event_id": "b0"}).json()
    last = None
    for i in range(1, 15):
        ts = f"2026-05-01T03:{12 + i:02d}:00Z"
        last = client.post(
            "/v1/decisions", json=base | {"event_id": f"b{i}", "timestamp": ts}
        ).json()
    assert last["features"]["user_events_1h"] == 14
    assert last["fraud_probability"] > first["fraud_probability"]


@pytest.mark.parametrize(
    "patch",
    [
        {"event_type": "refund"},
        {"amount": -5},
        {"timestamp": "2026-05-01T03:12:00"},  # naive timestamp
        {"payment_method": None},  # purchase without payment method
        {"unexpected": 1},
    ],
)
def test_validation_errors(client, patch):
    assert client.post("/v1/decisions", json=_example() | patch).status_code == 422
