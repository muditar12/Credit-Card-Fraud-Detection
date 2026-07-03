"""
predict.py
----------
Loads the trained model + scaler and scores new transactions for fraud
probability. Demonstrates how the pipeline would be used in production /
in a portfolio demo.

Run:
    python src/predict.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "creditcard.csv"


def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, scaler


def predict_transactions(df_features: pd.DataFrame):
    """df_features must have the same columns as the training features
    (Time, V1..V28, Amount), in that order, without the Class column."""
    model, scaler = load_artifacts()
    X_scaled = scaler.transform(df_features.values)
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = model.predict(X_scaled)
    return pred, proba


if __name__ == "__main__":
    # Demo: score a handful of random transactions from the dataset
    df = pd.read_csv(DATA_PATH)
    sample = df.sample(10, random_state=1)
    features = sample.drop(columns=["Class"])

    pred, proba = predict_transactions(features)

    out = sample[["Class"]].copy()
    out["Predicted"] = pred
    out["Fraud_Probability"] = np.round(proba, 4)
    print(out.to_string(index=False))
