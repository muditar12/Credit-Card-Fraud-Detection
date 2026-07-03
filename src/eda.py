"""
eda.py
------
Quick exploratory data analysis for the credit card transaction dataset.
Generates summary stats and a handful of plots (class balance, amount
distribution by class, correlation heatmap) into results/plots/.

Run:
    python src/eda.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
PLOTS_DIR = ROOT / "results" / "plots"


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum().sum(), "total missing cells")
    print("\nClass balance:\n", df["Class"].value_counts())
    print("\nFraud rate: {:.4f}%".format(df["Class"].mean() * 100))
    print("\nAmount summary by class:\n", df.groupby("Class")["Amount"].describe())

    # Class balance bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Class"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#4a5568", "#e53e3e"]
    )
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"], rotation=0)
    ax.set_title("Class Distribution (raw data)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_class_balance.png", dpi=150)
    plt.close(fig)

    # Amount distribution by class
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[df.Class == 0]["Amount"], bins=50, alpha=0.6, label="Legit", density=True)
    ax.hist(df[df.Class == 1]["Amount"], bins=50, alpha=0.6, label="Fraud", density=True)
    ax.set_xlim(0, 500)
    ax.set_xlabel("Transaction Amount")
    ax.set_ylabel("Density")
    ax.set_title("Transaction Amount Distribution by Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_amount_distribution.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap (subset of features for readability)
    corr = df.corr(numeric_only=True)["Class"].drop("Class").sort_values()
    fig, ax = plt.subplots(figsize=(6, 8))
    corr.plot(kind="barh", ax=ax, color="#2b6cb0")
    ax.set_title("Feature Correlation with Class (Fraud)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_feature_correlation.png", dpi=150)
    plt.close(fig)

    print(f"\nEDA plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
