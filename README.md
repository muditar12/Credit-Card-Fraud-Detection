# Credit Card Fraud Detection

## Overview

This project builds and evaluates machine learning models for detecting fraudulent credit card transactions. It includes data preprocessing, exploratory data analysis (EDA), class imbalance handling using SMOTE, model training, evaluation, and prediction.

The project uses the **Credit Card Fraud Detection** dataset from **Kaggle**.

---

## Project Structure

```text
credit-fraud-detection/
│── data/
│── models/
│── results/
│   ├── metrics_summary.json
│   └── plots/
│── src/
│   ├── eda.py
│   ├── generate_data.py
│   ├── predict.py
│   ├── smote.py
│   └── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature scaling
- Class imbalance handling using SMOTE
- Multiple machine learning models for fraud detection
- Model evaluation using standard classification metrics
- Confusion matrices, ROC curves, Precision-Recall curves, and feature importance visualizations
- Saved trained model and scaler for inference

---

## Dataset

This project uses the **Credit Card Fraud Detection** dataset available on Kaggle.

The dataset is **not included** in this repository due to GitHub's file size limitations.

Download it from Kaggle and place:

```text
creditcard.csv
```

inside the `data/` directory.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/credit-fraud-detection.git
cd credit-fraud-detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run exploratory analysis:

```bash
python src/eda.py
```

Apply SMOTE preprocessing:

```bash
python src/smote.py
```

Train the model:

```bash
python src/train.py
```

Generate predictions:

```bash
python src/predict.py
```

---

## Results

The project generates:

- Model evaluation metrics
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Saved trained model (`best_model.pkl`)
- Saved feature scaler (`scaler.pkl`)

All outputs are stored in the `results/` and `models/` directories.

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib
- XGBoost

---

