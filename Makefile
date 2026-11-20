.PHONY: install test lint run serve docker

install:
	pip install -e ".[dev]"

lint:
	ruff check . && ruff format --check .

test:
	pytest

run:
	rde run --config configs/default.yaml

serve:
	rde serve --model artifacts/model_bundle.joblib

docker:
	docker build -t risk-decisioning-engine . && docker run --rm -p 8000:8000 risk-decisioning-engine
