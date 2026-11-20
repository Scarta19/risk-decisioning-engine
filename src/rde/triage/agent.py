"""Triage agent: a tool-using LLM loop that prepares a recommendation for a review case.

The agent never decides on its own authority. It reads the case through the same tools an
analyst would use and ends by calling ``submit_triage`` with a recommendation
(approve, decline or escalate to a human), a confidence, a suspected fraud pattern and the
evidence it relied on. ``escalate`` keeps the case in the human queue, so the safe default
under uncertainty costs exactly what the review lane costs today.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from rde.triage.cases import CaseStore

RECOMMENDATIONS = ("approve", "decline", "escalate")
PATTERNS = (
    "card_testing",
    "account_takeover",
    "bot_or_install_farm",
    "friendly_fraud",
    "none",
    "unclear",
)
SOURCES = ("get_case_summary", "get_entity_history", "find_similar_resolved_cases")


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_case_summary",
        "description": (
            "Return the review case: event details, the model's calibrated fraud probability, "
            "expected cost of each action in USD, the top risk factors (TreeSHAP) and all "
            "model features. Call this first."
        ),
        "strict": True,
        "input_schema": _schema({"case_id": {"type": "string"}}, ["case_id"]),
    },
    {
        "name": "get_entity_history",
        "description": (
            "Return earlier events from the same user, device or IP prefix within a look-back "
            "window (1 to 720 hours): counts, distinct users and devices, and the most recent "
            "events with their model scores. Outcomes of these events are not known yet."
        ),
        "strict": True,
        "input_schema": _schema(
            {
                "case_id": {"type": "string"},
                "entity": {"type": "string", "enum": ["user", "device", "ip"]},
                "lookback_hours": {"type": "number"},
            },
            ["case_id", "entity", "lookback_hours"],
        ),
    },
    {
        "name": "find_similar_resolved_cases",
        "description": (
            "Return the k (1 to 20) most similar resolved cases from an earlier window, with "
            "their confirmed outcome and fraud typology, and the fraud share among them."
        ),
        "strict": True,
        "input_schema": _schema(
            {"case_id": {"type": "string"}, "k": {"type": "integer"}}, ["case_id", "k"]
        ),
    },
    {
        "name": "submit_triage",
        "description": (
            "Submit the final triage for the case. Call exactly once, after gathering "
            "evidence. Every evidence item must quote a fact returned by one of the tools."
        ),
        "strict": True,
        "input_schema": _schema(
            {
                "recommendation": {"type": "string", "enum": list(RECOMMENDATIONS)},
                "confidence": {"type": "number"},
                "suspected_pattern": {"type": "string", "enum": list(PATTERNS)},
                "evidence": {
                    "type": "array",
                    "items": _schema(
                        {
                            "fact": {"type": "string"},
                            "source_tool": {"type": "string", "enum": list(SOURCES)},
                        },
                        ["fact", "source_tool"],
                    ),
                },
                "rationale": {"type": "string"},
            },
            ["recommendation", "confidence", "suspected_pattern", "evidence", "rationale"],
        ),
    },
]


def system_prompt(store: CaseStore) -> str:
    costs = store.engine.policy.config.costs
    lines = []
    for etype, c in costs.items():
        lines.append(
            f"- {etype}: approving fraud costs ${c.approve_fraud_fixed:g} + "
            f"{c.approve_fraud_amount_rate:g} x amount; declining a legitimate customer costs "
            f"${c.decline_legit_fixed:g} + {c.decline_legit_amount_rate:g} x amount; "
            f"a human review costs about ${c.review_fixed:g} of analyst time."
        )
    return (
        "You are a fraud operations analyst assistant. A risk model has routed a case to "
        "manual review because its expected cost is lowest there. Your job is to prepare "
        "the review: investigate the case with the tools and recommend approve, decline or "
        "escalate (leave it for a human analyst).\n\n"
        "Unit economics (illustrative, used to judge your recommendations):\n"
        + "\n".join(lines)
        + "\n\nHow to work:\n"
        "1. Start with get_case_summary.\n"
        "2. Look at the history of the user, the device and the IP prefix where it could "
        "change your view, and compare with similar resolved cases.\n"
        "3. Recommend approve or decline only when the evidence clearly points one way and "
        "the cost of being wrong is small relative to a review. Otherwise escalate. "
        "Escalating is never penalised beyond the cost of a review.\n"
        "4. Finish by calling submit_triage once. Quote concrete numbers from tool results "
        "as evidence; do not state facts the tools did not return. Confidence is your "
        "probability that the recommendation is the right call (0 to 1).\n"
        "All data here is synthetic."
    )


# -- model interface ----------------------------------------------------------------------
class LLM(Protocol):
    model: str

    def create(self, system: str, tools: list[dict], messages: list[dict]) -> Any: ...


@dataclass
class TriageResult:
    case_id: str
    recommendation: str | None = None
    confidence: float | None = None
    suspected_pattern: str | None = None
    evidence: list[dict[str, str]] = field(default_factory=list)
    rationale: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None

    @property
    def final(self) -> str:
        """The action actually taken: anything other than a valid submission escalates."""
        return self.recommendation if self.error is None and self.recommendation else "escalate"

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["final"] = self.final
        return d


def _usage(result: TriageResult, usage: Any) -> None:
    if usage is None:
        return
    result.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
    result.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
    result.cache_read_tokens += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    result.cache_write_tokens += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)


def _validate_submission(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("recommendation") not in RECOMMENDATIONS:
        raise ValueError(f"recommendation must be one of {RECOMMENDATIONS}")
    if args.get("suspected_pattern") not in PATTERNS:
        raise ValueError(f"suspected_pattern must be one of {PATTERNS}")
    conf = float(args.get("confidence", -1))
    if not 0.0 <= conf <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    evidence = args.get("evidence") or []
    if not evidence:
        raise ValueError("at least one evidence item is required")
    return {**args, "confidence": conf}


class TriageAgent:
    def __init__(self, store: CaseStore, llm: LLM, max_turns: int = 8) -> None:
        self.store = store
        self.llm = llm
        self.max_turns = max_turns
        self.system = system_prompt(store)

    def _run_tool(self, name: str, args: dict[str, Any]) -> Any:
        s = self.store
        if name == "get_case_summary":
            return s.get_case_summary(args["case_id"])
        if name == "get_entity_history":
            return s.get_entity_history(
                args["case_id"], args["entity"], float(args.get("lookback_hours", 72))
            )
        if name == "find_similar_resolved_cases":
            return s.find_similar_resolved_cases(args["case_id"], int(args.get("k", 8)))
        raise ValueError(f"Unknown tool {name!r}")

    def triage(self, case_id: str) -> TriageResult:
        result = TriageResult(case_id=case_id)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Triage review case {case_id}."}
        ]
        nudged = False
        start = time.perf_counter()
        try:
            for _ in range(self.max_turns):
                response = self.llm.create(self.system, TOOLS, messages)
                result.turns += 1
                _usage(result, getattr(response, "usage", None))
                if response.stop_reason == "refusal":
                    result.error = "refusal"
                    break
                if response.stop_reason == "max_tokens":
                    result.error = "max_tokens"
                    break
                messages.append({"role": "assistant", "content": response.content})
                calls = [b for b in response.content if b.type == "tool_use"]
                if not calls:
                    if nudged:
                        result.error = "no_submission"
                        break
                    nudged = True
                    messages.append(
                        {"role": "user", "content": "Call submit_triage now with your answer."}
                    )
                    continue
                tool_results = []
                submitted = False
                for call in calls:
                    result.tool_calls.append(call.name)
                    args = dict(call.input or {})
                    try:
                        if call.name == "submit_triage":
                            sub = _validate_submission(args)
                            result.recommendation = sub["recommendation"]
                            result.confidence = sub["confidence"]
                            result.suspected_pattern = sub["suspected_pattern"]
                            result.evidence = list(sub["evidence"])
                            result.rationale = sub["rationale"]
                            submitted = True
                            content = "Recorded."
                        else:
                            content = json.dumps(self._run_tool(call.name, args), default=str)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": call.id, "content": content}
                        )
                    except (KeyError, ValueError, TypeError) as exc:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": f"Error: {exc}",
                                "is_error": True,
                            }
                        )
                if submitted:
                    break
                messages.append({"role": "user", "content": tool_results})
            else:
                result.error = "max_turns"
            if result.recommendation is None and result.error is None:
                result.error = "no_submission"
        except Exception as exc:  # network, auth, rate limit after SDK retries
            result.error = f"{type(exc).__name__}: {exc}"
        result.latency_s = round(time.perf_counter() - start, 3)
        return result
