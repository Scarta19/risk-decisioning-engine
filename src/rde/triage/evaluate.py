"""Evaluate the triage agent against labelled review cases.

Every number is computed against the true label of each case, with the same illustrative
unit economics the decision policy uses. Escalated cases are charged the realised cost of a
human review, so the agent can only save money by approving or declining correctly.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from rde.decision.policy import realized_cost
from rde.triage.agent import TriageAgent, TriageResult
from rde.triage.cases import TYPOLOGY, CaseStore
from rde.triage.llm import cost_usd


def sample_cases(store: CaseStore, n: int | None, seed: int = 0) -> list[str]:
    """All open cases, or a sample stratified by label so both classes are represented."""
    ids = store.case_ids()
    if n is None or n >= len(ids):
        return ids
    rng = random.Random(seed)
    fraud = [c for c in ids if store.label(c)[0] == 1]
    legit = [c for c in ids if store.label(c)[0] == 0]
    n_fraud = max(1, round(n * len(fraud) / len(ids))) if fraud else 0
    return sorted(rng.sample(fraud, min(n_fraud, len(fraud))) + rng.sample(legit, n - n_fraud))


def _costs(store: CaseStore, case_ids: list[str], actions: list[str]) -> np.ndarray:
    ev = store.events.set_index("event_id").loc[case_ids]
    review_to_policy = ["review" if a == "escalate" else a for a in actions]
    return realized_cost(
        store.engine.policy,
        np.array(review_to_policy),
        ev["is_fraud"].to_numpy(int),
        ev["amount"].fillna(0).to_numpy(float),
        ev["event_type"].to_numpy(),
    )


def run(store: CaseStore, agent: TriageAgent, case_ids: list[str], workers: int = 4):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(agent.triage, case_ids))


def summarise(store: CaseStore, results: list[TriageResult], model: str) -> dict[str, Any]:
    ids = [r.case_id for r in results]
    labels = [store.label(c) for c in ids]
    y = np.array([lab[0] for lab in labels])
    probs = np.array([store.get_case_summary(c)["model"]["fraud_probability"] for c in ids])
    agent_actions = [r.final for r in results]
    strategies = {
        "escalate_all (today: every case to an analyst)": ["escalate"] * len(ids),
        "model_only (decline if p >= 0.5, else approve)": [
            "decline" if q >= 0.5 else "approve" for q in probs
        ],
        "agent": agent_actions,
        "oracle (perfect information, lower bound)": [
            "decline" if v == 1 else "approve" for v in y
        ],
    }
    llm_costs = [
        cost_usd(model, r.input_tokens, r.output_tokens, r.cache_read_tokens, r.cache_write_tokens)
        for r in results
    ]
    llm_total = None if any(c is None for c in llm_costs) else float(sum(llm_costs))
    table = []
    for name, actions in strategies.items():
        c = _costs(store, ids, actions)
        a = np.array(actions)
        auto = a != "escalate"
        correct = ((a == "decline") & (y == 1)) | ((a == "approve") & (y == 0))
        table.append(
            {
                "strategy": name,
                "fraud_cost_usd": round(float(c.sum()), 2),
                "per_case_usd": round(float(c.mean()), 3),
                "escalation_rate": round(float((~auto).mean()), 3),
                "automated_accuracy": round(float(correct[auto].mean()), 3) if auto.any() else None,
                "fraud_approved": int(((a == "approve") & (y == 1)).sum()),
                "legit_declined": int(((a == "decline") & (y == 0)).sum()),
            }
        )
    fraud_idx = [i for i, v in enumerate(y) if v == 1]
    pattern_hits = [results[i].suspected_pattern == TYPOLOGY.get(labels[i][1]) for i in fraud_idx]
    grounded = [
        all(e.get("source_tool") in r.tool_calls for e in r.evidence) for r in results if r.evidence
    ]
    lat = np.array([r.latency_s for r in results])
    base = table[0]["fraud_cost_usd"]
    agent_row = table[2]
    return {
        "model": model,
        "cases": len(ids),
        "fraud_share": round(float(y.mean()), 3),
        "strategies": table,
        "agent": {
            "errors": int(sum(r.error is not None for r in results)),
            "error_kinds": sorted({r.error.split(":")[0] for r in results if r.error}),
            "pattern_accuracy_on_fraud": round(float(np.mean(pattern_hits)), 3)
            if pattern_hits
            else None,
            "evidence_from_tools_actually_called": round(float(np.mean(grounded)), 3)
            if grounded
            else None,
            "mean_tool_calls": round(float(np.mean([len(r.tool_calls) for r in results])), 2),
            "latency_p50_s": round(float(np.percentile(lat, 50)), 2),
            "latency_p90_s": round(float(np.percentile(lat, 90)), 2),
            "llm_cost_total_usd": None if llm_total is None else round(llm_total, 4),
            "llm_cost_per_case_usd": None if llm_total is None else round(llm_total / len(ids), 4),
            "net_saving_vs_escalate_all_usd": None
            if llm_total is None
            else round(base - agent_row["fraud_cost_usd"] - llm_total, 2),
        },
    }


def write_report(summary: dict[str, Any], results: list[TriageResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "triage_summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "triage_cases.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")
    a = summary["agent"]
    lines = [
        "# Triage agent evaluation",
        "",
        f"Model: `{summary['model']}` · cases: {summary['cases']} · "
        f"fraud share among them: {summary['fraud_share']:.1%}. "
        "Synthetic data, illustrative costs.",
        "",
        "| Strategy | Fraud + friction cost | Per case | Sent to analysts "
        "| Accuracy when automated | Fraud approved | Good customers declined |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in summary["strategies"]:
        acc = "n/a" if t["automated_accuracy"] is None else f"{t['automated_accuracy']:.1%}"
        lines.append(
            f"| {t['strategy']} | ${t['fraud_cost_usd']:,.2f} | ${t['per_case_usd']:.2f} | "
            f"{t['escalation_rate']:.0%} | {acc} | {t['fraud_approved']} | {t['legit_declined']} |"
        )
    lines += [
        "",
        "| Agent metric | Value |",
        "|---|---|",
        *[f"| {k.replace('_', ' ')} | {v} |" for k, v in a.items()],
        "",
        "Costs exclude analyst salaries beyond the per-review cost in the config. "
        "LLM cost uses list prices in `rde/triage/llm.py`.",
    ]
    path = out_dir / "triage_evaluation.md"
    path.write_text("\n".join(lines) + "\n")
    return path
