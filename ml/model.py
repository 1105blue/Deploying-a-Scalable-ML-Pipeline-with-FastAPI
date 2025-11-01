# ml/model.py
"""
Model utilities for the Census Income classification project.

Implements:
- train_model(X_train, y_train)
- compute_model_metrics(y, preds)
- inference(model, X)
- save_model(obj, path)
- load_model(path)
- performance_on_categorical_slice(...)

Designed to align with the project rubric and work with ml/data.process_data.
"""

from __future__ import annotations

import pickle
from typing import Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


# --------------------------- Metrics ---------------------------

def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 (fbeta with beta=1).

    Parameters
    ----------
    y : np.ndarray
        Ground-truth binary labels (0/1).
    preds : np.ndarray
        Predicted binary labels (0/1).

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


# --------------------------- Train ---------------------------

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Train and return a classifier.

    Notes
    -----
    RandomForest is used for fast, reliable training on one-hot features.
    This keeps CI runs quick and avoids solver convergence issues.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# --------------------------- Inference ---------------------------

def inference(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Run model inference and return predicted labels (0/1).
    """
    return model.predict(X)


# --------------------------- Persistence ---------------------------

def save_model(model_or_obj: Any, path: str) -> None:
    """
    Serialize a model/encoder/label-binarizer to a pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump(model_or_obj, f)


def load_model(path: str) -> Any:
    """
    Load and return a pickled object from `path`.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------- Slice Performance ---------------------------

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
    """
    Compute precision/recall/F1 for the subset where `column_name == slice_value`.

    Parameters
    ----------
    data : pd.DataFrame
        The full evaluation dataframe (includes features + label column).
    column_name : str
        Categorical column to slice on (e.g., "education").
    slice_value : Any
        Exact value within `column_name` to filter (e.g., "Bachelors").
    categorical_features : list[str]
        List of categorical feature names used in process_data.
    label : str
        Name of the target column (e.g., "salary").
    encoder : OneHotEncoder
        Fitted encoder from training.
    lb : LabelBinarizer
        Fitted label binarizer from training.
    model : Any
        Trained classifier.

    Returns
    -------
    precision, recall, fbeta : tuple[float, float, float]
        Metrics for this specific slice. If slice is empty, returns (1.0, 1.0, 1.0).
    """
    # Filter to the slice
    df_slice = data[data[column_name] == slice_value].copy()
    if df_slice.empty:
        # Avoid divide-by-zero; caller can ignore these
        return 1.0, 1.0, 1.0

    # Process slice with existing encoder/lb (training=False!)
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
