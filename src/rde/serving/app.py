"""FastAPI decision service.

The model bundle is loaded once at start-up from ``RDE_MODEL_PATH``. Every request is
featurised by the same :class:`~rde.features.state.FeatureState` used in training, then
scored, calibrated and decided by the cost policy. Online state is in-process; run a
single worker per state store (see README, "Production notes").
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request

from rde import __version__
from rde.engine import RiskEngine
from rde.serving.schemas import Decision, Event

log = logging.getLogger("rde.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("RDE_MODEL_PATH", "artifacts/model_bundle.joblib")
    app.state.engine = RiskEngine.load(path)
    app.state.lock = threading.Lock()
    log.info("Loaded model %s from %s", app.state.engine.metadata.get("model_version"), path)
    yield


app = FastAPI(
    title="Risk Decisioning Engine",
    version=__version__,
    description="Calibrated fraud risk scores and cost-based approve/review/decline decisions.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/model")
def model_info(request: Request) -> dict:
    engine: RiskEngine = request.app.state.engine
    return {**engine.metadata, "review_capacity": engine.policy.config.review_capacity}


@app.post("/v1/decisions", response_model=Decision)
def decide(
    event: Event,
    request: Request,
    record: bool = Query(True, description="Add the event to online state after scoring"),
) -> Decision:
    engine: RiskEngine = request.app.state.engine
    start = time.perf_counter()
    try:
        # Feature state is shared and mutable: serialise compute+update.
        with request.app.state.lock:
            result = engine.decide_event(event.model_dump(), record=record)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    log.info(
        "decision event_id=%s decision=%s p=%.4f latency_ms=%.1f",
        result["event_id"],
        result["decision"],
        result["fraud_probability"],
        (time.perf_counter() - start) * 1000,
    )
    return Decision(**result)
