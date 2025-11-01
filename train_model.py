# train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# ----- Paths -----
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")
MODEL_DIR = os.path.join(PROJECT_PATH, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ----- Load data -----
print(DATA_PATH)
data = pd.read_csv(DATA_PATH)

# ----- Train/test split -----
train, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["salary"]
)

# ----- Categorical features -----
cat_features = [
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

# ----- Process data -----
X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=cat_features,
    label=LABEL,
    training=True,
)

X_test, y_test, _, _ = process_data(
    X=test,
    categorical_features=cat_features,
    label=LABEL,
    training=False,
    encoder=encoder,
    lb=lb,
)

# ----- Train model -----
model = train_model(X_train, y_train)

# ----- Save model objects -----
model_path = os.path.join(MODEL_DIR, "model.pkl")
encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
lb_path = os.path.join(MODEL_DIR, "lb.pkl")
save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

# (Optional) Reload to prove saving works
model = load_model(model_path)

# ----- Inference & metrics -----
preds = inference(model, X_test)
p, r, f1 = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

# ----- Slice metrics -----
# Start fresh each run
with open("slice_output.txt", "w") as f:
    f.write("")

for col in cat_features:
    for slice_value in sorted(test[col].dropna().unique()):
        sp, sr, sf1 = performance_on_categorical_slice(
            data=test,
            column_name=col,                # <-- correct kw
            slice_value=slice_value,       # <-- correct kw
            categorical_features=cat_features,
            label=LABEL,
            encoder=encoder,
            lb=lb,
            model=model,
        )
        count = int((test[col] == slice_value).sum())
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slice_value}, Count: {count:,}", file=f)
            print(f"Precision: {sp:.4f} | Recall: {sr:.4f} | F1: {sf1:.4f}", file=f)
