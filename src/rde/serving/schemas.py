"""Request and response contracts for the decision API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    event_type: Literal["signup", "purchase"]
    user_id: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=64)
    ip_prefix: str = Field(min_length=1, max_length=64)
    ip_country: str = Field(min_length=2, max_length=2)
    user_country: str = Field(min_length=2, max_length=2)
    source_id: str = Field(default="organic", max_length=64)
    payment_method: Literal["card", "wallet", "bank"] | None = None
    amount: float = Field(default=0.0, ge=0, le=1_000_000)
    account_created_at: datetime
    click_to_conversion_sec: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> Event:
        if self.timestamp.tzinfo is None or self.account_created_at.tzinfo is None:
            raise ValueError("timestamps must include a timezone (e.g. 2026-04-20T13:05:00Z)")
        if self.event_type == "purchase" and self.payment_method is None:
            raise ValueError("purchase events require payment_method")
        return self


class Reason(BaseModel):
    feature: str
    value: float | None
    contribution: float


class Decision(BaseModel):
    event_id: str
    decision: Literal["approve", "review", "decline"]
    fraud_probability: float
    raw_score: float
    expected_cost: dict[str, float]
    reasons: list[Reason]
    features: dict[str, float | None]
    model_version: str
