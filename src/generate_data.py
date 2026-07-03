"""
generate_data.py
-----------------
Generates a synthetic transaction dataset that mirrors the structure and
statistical properties of the well-known anonymized credit-card-fraud
dataset (Time, V1-V28 PCA-style features, Amount, Class), including a
realistic ~0.17% fraud rate. Useful for offline development, testing,
and demos when the original proprietary dataset isn't available.

Run:
    python src/generate_data.py
Output:
    data/creditcard.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N_SAMPLES = 50000
FRAUD_RATE = 0.0017  # ~0.17%, matches real-world class imbalance
N_PCA_FEATURES = 28

def generate_dataset(n_samples=N_SAMPLES, fraud_rate=FRAUD_RATE, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    n_fraud = max(int(n_samples * fraud_rate), 10)
    n_legit = n_samples - n_fraud

    # --- Legitimate transactions ---
    # V1-V28: roughly standard-normal, low correlation (mimics PCA components)
    legit_pca = rng.normal(loc=0.0, scale=1.0, size=(n_legit, N_PCA_FEATURES))
    legit_time = rng.uniform(0, 172800, size=n_legit)  # 2 days, in seconds
    legit_amount = np.round(np.abs(rng.gamma(shape=2.0, scale=40.0, size=n_legit)), 2)
    legit_class = np.zeros(n_legit, dtype=int)

    # --- Fraudulent transactions ---
    # Shifted / higher-variance PCA components + different amount distribution,
    # simulating the statistical drift fraud typically introduces.
    fraud_pca = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, N_PCA_FEATURES))
    shift_idx = rng.choice(N_PCA_FEATURES, size=8, replace=False)
    fraud_pca[:, shift_idx] += rng.normal(loc=3.0, scale=1.5, size=(n_fraud, 8))
    fraud_time = rng.uniform(0, 172800, size=n_fraud)
    fraud_amount = np.round(np.abs(rng.gamma(shape=1.2, scale=150.0, size=n_fraud)), 2)
    fraud_class = np.ones(n_fraud, dtype=int)

    # --- Combine ---
    pca = np.vstack([legit_pca, fraud_pca])
    time = np.concatenate([legit_time, fraud_time])
    amount = np.concatenate([legit_amount, fraud_amount])
    cls = np.concatenate([legit_class, fraud_class])

    cols = [f"V{i}" for i in range(1, N_PCA_FEATURES + 1)]
    df = pd.DataFrame(pca, columns=cols)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = cls

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "creditcard.csv"
    df.to_csv(out_path, index=False)

    fraud_count = int(df["Class"].sum())
    print(f"Generated {len(df):,} transactions -> {out_path}")
    print(f"Fraudulent: {fraud_count:,} ({fraud_count / len(df) * 100:.3f}%)")
    print(f"Legitimate: {len(df) - fraud_count:,} ({(1 - fraud_count / len(df)) * 100:.3f}%)")
