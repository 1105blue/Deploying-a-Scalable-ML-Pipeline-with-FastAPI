# test_ml.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def _prepare_data():
    """Load, split, and process the census data for tests."""
    df = pd.read_csv("data/census.csv")
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[LABEL]
        if LABEL in df.columns
        else None
    )
    X_train, y_train, enc, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label=LABEL, training=False, encoder=enc, lb=lb
    )
    return X_train, y_train, X_test, y_test


def test_train_model_fits_and_predicts():
    """
    The trained model should be able to make predictions on training data without error.
    """
    X_train, y_train, _, _ = _prepare_data()
    model = train_model(X_train, y_train)
    # Predict on a small slice to verify it's fitted
    preds = model.predict(X_train[:10])
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == min(10, X_train.shape[0])


def test_inference_length_matches_input():
    """
    inference(model, X) should return one prediction per input row.
    """
    X_train, y_train, X_test, _ = _prepare_data()
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]


def test_metrics_within_valid_range():
    """
    Precision, recall, and F1 should be between 0 and 1 (inclusive).
    """
    X_train, y_train, X_test, y_test = _prepare_data()
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    p, r, f1 = compute_model_metrics(y_test, preds)
    for m in (p, r, f1):
        assert 0.0 <= m <= 1.0
