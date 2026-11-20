import pytest

from rde.config import Config, config_from_dict, load_config
from tests.conftest import ROOT


def test_default_yaml_matches_code_defaults():
    assert load_config(ROOT / "configs" / "default.yaml") == Config()


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown keys"):
        config_from_dict({"simulation": {"n_userz": 10}})


def test_split_order_is_validated():
    with pytest.raises(ValueError, match="split days"):
        config_from_dict({"split": {"train_end_day": 100, "valid_end_day": 90}})
