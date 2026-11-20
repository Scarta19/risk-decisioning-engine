# syntax=docker/dockerfile:1
# Stage 1: install the package and train a model bundle from the synthetic simulator.
# Training is deterministic for a given config, so the image is reproducible and no
# binary model files need to live in Git.
FROM python:3.12-slim AS train
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .
ARG RDE_CONFIG=configs/default.yaml
RUN rde run --config "$RDE_CONFIG" --artifacts /app/artifacts --no-report

# Stage 2: slim runtime with only the package and the trained bundle.
FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=train /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=train /usr/local/bin/uvicorn /usr/local/bin/rde /usr/local/bin/
COPY --from=train /app/artifacts /app/artifacts
ENV RDE_MODEL_PATH=/app/artifacts/model_bundle.joblib \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
# One worker on purpose: online feature state is in-process (see README).
CMD ["uvicorn", "rde.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
