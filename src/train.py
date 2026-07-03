"""
train.py
--------
End-to-end pipeline for the Credit Card Fraud Detection project:

 1. Load transaction data
 2. Preprocess (scale Time/Amount, train/test split with stratification)
 3. Handle severe class imbalance with SMOTE (on the training set only)
 4. Train multiple classifiers (Logistic Regression, Random Forest,
    Gradient Boosting)
 5. Evaluate with metrics suited to imbalanced classification:
    Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix
 6. Save plots (ROC curve, Precision-Recall curve, confusion matrices,
    feature importance) and the best model to disk

Run:
    python src/train.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score
)

from smote import smote_oversample

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = ROOT / "models"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/generate_data.py` first, "
            f"or drop in the real creditcard.csv (Time, V1-V28, Amount, Class)."
        )
    return pd.read_csv(DATA_PATH)


def preprocess(df):
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], zero_division=0))
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    return y_pred, y_proba, cm


def plot_confusion_matrix(cm, name):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(curves):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_pr_curves(curves):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (precision, recall, ap) in curves.items():
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "precision_recall_curves.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="#2b6cb0")
    ax.set_title(f"Top 15 Feature Importances - {name}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"feature_importance_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close(fig)


def plot_class_distribution(y_before, y_after):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, y, title in zip(axes, [y_before, y_after], ["Before SMOTE", "After SMOTE"]):
        counts = pd.Series(y).value_counts().sort_index()
        ax.bar(["Legit", "Fraud"], counts.values, color=["#4a5568", "#e53e3e"])
        ax.set_title(title)
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution_before_after_smote.png", dpi=150)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    feature_names = [c for c in df.columns if c != "Class"]
    print(f"Dataset shape: {df.shape} | Fraud rate: {df['Class'].mean() * 100:.3f}%")

    print("Preprocessing (scaling + stratified train/test split)...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("Applying SMOTE to balance the training set...")
    t0 = time.time()
    X_train_res, y_train_res = smote_oversample(X_train, y_train, random_state=RANDOM_STATE)
    print(f"SMOTE done in {time.time() - t0:.1f}s. "
          f"Train size: {len(y_train)} -> {len(y_train_res)}")
    plot_class_distribution(y_train, y_train_res)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

    results = {}
    roc_curves_data = {}
    pr_curves_data = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train_res, y_train_res)
        print(f"  trained in {time.time() - t0:.1f}s")
        trained_models[name] = model

        y_pred, y_proba, cm = evaluate_model(name, model, X_test, y_test, results)
        plot_confusion_matrix(cm, name)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves_data[name] = (fpr, tpr, results[name]["roc_auc"])

        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_curves_data[name] = (prec, rec, results[name]["pr_auc"])

        plot_feature_importance(model, feature_names, name)

    plot_roc_curves(roc_curves_data)
    plot_pr_curves(pr_curves_data)

    # Pick best model by F1 (a balanced metric appropriate for imbalanced fraud data)
    best_name = max(results, key=lambda n: results[n]["f1_score"])
    best_model = trained_models[best_name]
    print(f"\nBest model by F1-score: {best_name} ({results[best_name]['f1_score']})")

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    summary = {
        "dataset_shape": list(df.shape),
        "fraud_rate_percent": round(df["Class"].mean() * 100, 4),
        "train_size_before_smote": int(len(y_train)),
        "train_size_after_smote": int(len(y_train_res)),
        "test_size": int(len(y_test)),
        "best_model": best_name,
        "results": results,
    }
    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}")
    print(f"Best model saved to {MODELS_DIR / 'best_model.pkl'}")


if __name__ == "__main__":
    main()
