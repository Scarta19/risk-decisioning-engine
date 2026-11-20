from pathlib import Path

import pytest

from rde.config import load_config
from rde.data.synthetic import generate_events
from rde.pipeline import run_pipeline, save_outputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def small_cfg():
    return load_config(ROOT / "configs" / "small.yaml")


@pytest.fixture(scope="session")
def small_events(small_cfg):
    return generate_events(small_cfg.simulation)


@pytest.fixture(scope="session")
def pipeline_result(small_cfg, small_events):
    return run_pipeline(small_cfg, small_events)


@pytest.fixture(scope="session")
def bundle_path(pipeline_result, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    save_outputs(pipeline_result, out)
    return out / "model_bundle.joblib"
