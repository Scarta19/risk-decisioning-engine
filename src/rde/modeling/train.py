"""LightGBM risk model training with early stopping on an out-of-time window."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from rde.config import ModelConfig
from rde.features.state import FEATURES


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cfg: ModelConfig,
) -> lgb.Booster:
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l2": cfg.lambda_l2,
        "seed": cfg.seed,
        "deterministic": True,
        "num_threads": 1,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(X_train[FEATURES], label=y_train, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid[FEATURES], label=y_valid, reference=dtrain)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )


def predict_raw(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(X[FEATURES], num_iteration=booster.best_iteration)


def reason_codes(booster: lgb.Booster, X: pd.DataFrame, top_k: int = 3) -> list[list[dict]]:
    """Top features pushing each score up, from LightGBM's exact TreeSHAP contributions."""
    contrib = booster.predict(X[FEATURES], num_iteration=booster.best_iteration, pred_contrib=True)[
        :, :-1
    ]
    out = []
    for row, values in zip(contrib, X[FEATURES].to_numpy(), strict=True):
        idx = np.argsort(-row)[:top_k]
        out.append(
            [
                {"feature": FEATURES[i], "value": _clean(values[i]), "contribution": float(row[i])}
                for i in idx
                if row[i] > 0
            ]
        )
    return out


def _clean(v: float) -> float | None:
    return None if v is None or np.isnan(v) else float(v)
