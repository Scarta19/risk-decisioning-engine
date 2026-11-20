import json

import numpy as np
import pytest

from rde.triage import evaluate as E
from rde.triage.agent import TOOLS, TriageAgent
from rde.triage.cases import CaseStore
from rde.triage.llm import ScriptedLLM, heuristic_script, response, text, tool_call


@pytest.fixture(scope="module")
def store(pipeline_result):
    return CaseStore.from_pipeline(pipeline_result)


@pytest.fixture(scope="module")
def case_id(store):
    return store.case_ids()[0]


def _submit(**overrides):
    args = dict(
        recommendation="escalate",
        confidence=0.5,
        suspected_pattern="unclear",
        evidence=[{"fact": "p is 0.3", "source_tool": "get_case_summary"}],
        rationale="Mixed signals.",
    )
    args.update(overrides)
    return tool_call("submit_triage", "sub", **args)


def test_open_cases_are_the_review_lane(store, pipeline_result):
    tf = pipeline_result.test_frame
    assert len(store.case_ids()) == int((tf["decision"] == "review").sum()) > 0


def test_tools_never_leak_labels_of_open_cases(store):
    for cid in store.case_ids()[:10]:
        payload = json.dumps(
            [
                store.get_case_summary(cid),
                store.get_entity_history(cid, "user", 720),
                store.get_entity_history(cid, "device", 720),
                store.get_entity_history(cid, "ip", 720),
            ]
        )
        for word in ("is_fraud", "fraud_pattern", "outcome", "account_takeover", "legit"):
            assert word not in payload


def test_reference_cases_are_resolved_before_any_open_case(store):
    ref_end = store.events.loc[store.reference, "timestamp"].max()
    open_start = store.events.loc[store.open_cases, "timestamp"].min()
    assert ref_end < open_start


def test_entity_history_only_looks_backwards(store, case_id):
    h = store.get_entity_history(case_id, "device", 720)
    assert all(e["hours_before_case"] > 0 for e in h["most_recent"])


def test_similar_cases_come_from_the_reference_window(store, case_id):
    out = store.find_similar_resolved_cases(case_id, 5)
    assert len(out["neighbours"]) == 5
    assert {n["outcome"] for n in out["neighbours"]} <= {"fraud", "legitimate"}


def test_unknown_or_closed_case_is_rejected(store):
    with pytest.raises(KeyError):
        store.get_case_summary("does-not-exist")
    closed = store.events.loc[store.reference[0], "event_id"]
    with pytest.raises(KeyError):
        store.get_case_summary(closed)


def test_tool_schemas_are_strict():
    for t in TOOLS:
        assert t["strict"] is True
        assert t["input_schema"]["additionalProperties"] is False


def test_agent_records_a_valid_submission(store, case_id):
    llm = ScriptedLLM(
        [
            response(tool_call("get_case_summary", "t1", case_id=case_id)),
            response(_submit(recommendation="decline", suspected_pattern="account_takeover")),
        ]
    )
    r = TriageAgent(store, llm).triage(case_id)
    assert r.error is None and r.final == "decline"
    assert r.tool_calls == ["get_case_summary", "submit_triage"]
    assert r.input_tokens == 200


def test_invalid_submission_is_returned_as_tool_error_then_fixed(store, case_id):
    llm = ScriptedLLM(
        [
            response(_submit(confidence=3.0)),
            response(_submit(recommendation="approve", confidence=0.9)),
        ]
    )
    r = TriageAgent(store, llm).triage(case_id)
    assert r.final == "approve" and r.turns == 2


def test_tool_errors_do_not_crash_the_loop(store, case_id):
    llm = ScriptedLLM(
        [response(tool_call("get_case_summary", "t1", case_id="nope")), response(_submit())]
    )
    r = TriageAgent(store, llm).triage(case_id)
    assert r.error is None and r.final == "escalate"


@pytest.mark.parametrize(
    "script,error",
    [
        ([response(text("hmm")), response(text("still thinking"))], "no_submission"),
        ([response(text("no"), stop_reason="refusal")], "refusal"),
        ([response(tool_call("get_case_summary", "t", case_id="x"))] * 20, "max_turns"),
    ],
)
def test_anything_but_a_submission_escalates(store, case_id, script, error):
    r = TriageAgent(store, ScriptedLLM(script), max_turns=4).triage(case_id)
    assert r.error == error and r.final == "escalate"


def test_evaluation_baselines_are_consistent(store):
    ids = E.sample_cases(store, None)
    agent = TriageAgent(store, ScriptedLLM(heuristic_script(store)))
    results = E.run(store, agent, ids, workers=1)
    s = E.summarise(store, results, "scripted")
    rows = {t["strategy"].split(" ")[0]: t for t in s["strategies"]}
    assert rows["oracle"]["fraud_cost_usd"] == 0
    assert rows["escalate_all"]["escalation_rate"] == 1.0
    # escalating everything costs exactly the realised review cost
    review = E._costs(store, ids, ["escalate"] * len(ids))
    assert rows["escalate_all"]["fraud_cost_usd"] == pytest.approx(float(np.sum(review)), abs=0.01)
    assert s["agent"]["errors"] == 0


def test_stratified_sample_keeps_both_classes(store):
    ids = E.sample_cases(store, 20, seed=1)
    labels = {store.label(c)[0] for c in ids}
    assert len(ids) == 20 and labels == {0, 1}
