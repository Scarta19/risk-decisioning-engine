"""Model backends for the triage agent."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "claude-opus-5-5"

# USD per million tokens (Anthropic list prices, November 2026). Verify before relying on
# them: https://www.anthropic.com/pricing
PRICES = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.10, "cache_write": 2.5},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int):
    p = PRICES.get(model)
    if p is None:
        return None
    return (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_read * p["cache_read"]
        + cache_write * p["cache_write"]
    ) / 1e6


class AnthropicLLM:
    """Claude via the Messages API.

    * adaptive thinking with an explicit effort level;
    * the static system prompt and tool list are cached across turns and cases;
    * server-side fallback: if the model declines on policy grounds, the API retries the
      same request on a fallback model inside the same call.
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", client=None) -> None:
        import anthropic

        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def create(self, system: str, tools: list[dict], messages: list[dict]) -> Any:
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )


# -- offline backend for tests and CI -----------------------------------------------------
@dataclass
class _Block:
    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict | None = None


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _Response:
    content: list[_Block]
    stop_reason: str
    usage: _Usage = field(default_factory=_Usage)


def tool_call(name: str, call_id: str = "toolu_1", **args) -> _Block:
    return _Block(type="tool_use", id=call_id, name=name, input=args)


def text(value: str) -> _Block:
    return _Block(type="text", text=value)


def response(*blocks: _Block, stop_reason: str | None = None) -> _Response:
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return _Response(list(blocks), stop_reason, _Usage(input_tokens=100, output_tokens=20))


class ScriptedLLM:
    """Returns pre-built responses. ``script`` is a list, or a function of the case_id
    (read from the first user message) that returns a list."""

    model = "scripted"

    def __init__(self, script: Iterable[_Response] | Callable[[str], Iterable[_Response]]):
        self._script = script
        self._queue: list[_Response] = []

    def create(self, system: str, tools: list[dict], messages: list[dict]) -> Any:
        if len(messages) == 1:  # new conversation
            first = messages[0]["content"]
            case_id = first.rsplit(" ", 1)[-1].rstrip(".")
            src = self._script(case_id) if callable(self._script) else self._script
            self._queue = list(src)
        if not self._queue:
            return response(text("I have nothing more to add."))
        return self._queue.pop(0)


def heuristic_script(store) -> Callable[[str], list[_Response]]:
    """A deterministic stand-in analyst for smoke runs: reads the summary and neighbours and
    escalates unless the neighbours are unanimous. Not a benchmark, only a harness check."""

    def script(case_id: str) -> list[_Response]:
        nb = store.find_similar_resolved_cases(case_id, 8)
        share = nb["fraud_share_among_neighbours"] or 0.0
        rec = "decline" if share >= 0.99 else "approve" if share <= 0.0 else "escalate"
        return [
            response(tool_call("get_case_summary", "t1", case_id=case_id)),
            response(tool_call("find_similar_resolved_cases", "t2", case_id=case_id, k=8)),
            response(
                tool_call(
                    "submit_triage",
                    "t3",
                    recommendation=rec,
                    confidence=0.6,
                    suspected_pattern="unclear",
                    evidence=[
                        {
                            "fact": f"fraud share among 8 neighbours is {share}",
                            "source_tool": "find_similar_resolved_cases",
                        }
                    ],
                    rationale="Heuristic stand-in.",
                )
            ),
        ]

    return script
