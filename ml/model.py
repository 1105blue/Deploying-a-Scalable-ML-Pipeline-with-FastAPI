# ml/model.py
"""
Model utilities for the Census Income classification project.
"""

from __future__ import annotations

import pickle
from typing import Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1."""
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train and return a fast, reliable classifier."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def inference(model: Any, X: np.ndarray) -> np.ndarray:
    """Run model inference and return predictions."""
    return model.predict(X)


def save_model(model_or_obj: Any, path: str) -> None:
    """Serialize a model/encoder/lb to disk."""
    with open(path, "wb") as f:
        pickle.dump(model_or_obj, f)


def load_model(path: str) -> Any:
    """Load a pickled object from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data: pd.DataFrame,
    column_name: str,
    slice_value,
    categorical_features: List[str],
    label: str,
    encoder,
    lb,
    model: Any,
) -> Tuple[float, float, float]:
    """Compute metrics for the subset where column_name == slice_value."""
    df_slice = data[data[column_name] == slice_value].copy()
    if df_slice.empty:
        return 1.0, 1.0, 1.0

    Xs, ys, _, _ = process_data(
        df_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, Xs)
    return compute_model_metrics(ys, preds)
