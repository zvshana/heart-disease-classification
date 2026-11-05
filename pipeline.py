"""
Heart disease risk classification - reproducible pipeline.

A script version of `classification_project.ipynb`: it runs the same
preprocessing steps, trains the three final models with the hyperparameters
selected in the notebook, and reports the metrics computed on the test set.

Usage:
    python pipeline.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach"]
NOMINAL_COLS = ["cp", "restecg"]

# Hyperparameters selected by the tuning curves in the notebook.
BEST_PARAMS = {
    "Logistic Regression": {"C": 0.01},
    "Decision Tree": {"max_depth": 4},
    "Random Forest": {"n_estimators": 100},
}


def load_data(path=DATA_PATH):
    """Load the raw dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_raw_data(df):
    """Drop duplicate rows and rows whose label is not a valid binary class."""
    df = df.drop_duplicates().reset_index(drop=True)
    return df[df[TARGET_COL].isin([0, 1])].reset_index(drop=True)


def split_data(df):
    """80/20 train-test split with a fixed seed."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


class Preprocessor:
    """
    Learns every cleaning parameter from the training set only and applies the
    same transformation to unseen data, to prevent data leakage:
      1. Negative (physically impossible) values -> NaN
      2. IQR outliers in continuous columns -> NaN
      3. Imputation: median for continuous columns, mode for the rest
      4. One-hot encoding of nominal columns (drop_first=True)
      5. Standard scaling
    """

    def fit(self, X):
        X = X.copy()
        X[X < 0] = np.nan

        self.bounds_ = {}
        for col in CONTINUOUS_COLS:
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        X = self._mask_outliers(X)

        self.fill_values_ = {
            col: X[col].median() if col in CONTINUOUS_COLS else X[col].mode()[0]
            for col in X.columns
        }
        X = X.fillna(self.fill_values_)

        encoded = pd.get_dummies(X, columns=NOMINAL_COLS, drop_first=True, dtype=int)
        self.feature_names_ = list(encoded.columns)
        self.scaler_ = StandardScaler().fit(encoded)
        return self

    def transform(self, X):
        X = X.copy()
        X[X < 0] = np.nan
        X = self._mask_outliers(X).fillna(self.fill_values_)
        encoded = pd.get_dummies(X, columns=NOMINAL_COLS, drop_first=True, dtype=int)
        encoded = encoded.reindex(columns=self.feature_names_, fill_value=0)
        return pd.DataFrame(self.scaler_.transform(encoded), columns=self.feature_names_, index=X.index)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _mask_outliers(self, X):
        for col, (low, high) in self.bounds_.items():
            X.loc[(X[col] < low) | (X[col] > high), col] = np.nan
        return X


def build_models():
    """The three final classifiers with their tuned hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, **BEST_PARAMS["Logistic Regression"]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, **BEST_PARAMS["Decision Tree"]),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, **BEST_PARAMS["Random Forest"]),
    }


def evaluate(model, X_train, y_train, X_test, y_test):
    """Train/test accuracy, high-risk class metrics and 5-fold CV accuracy on the training set."""
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    return {
        "train_acc": accuracy_score(y_train, model.predict(X_train)),
        "test_acc": accuracy_score(y_test, y_pred),
        "precision_high_risk": precision[1],
        "recall_high_risk": recall[1],
        "f1_high_risk": f1[1],
        "cv_acc_mean": cv_scores.mean(),
        "cv_acc_std": cv_scores.std(),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }


def run_pipeline(path=DATA_PATH):
    """Run the full pipeline and return (results, fitted models)."""
    df = clean_raw_data(load_data(path))
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    models, results = build_models(), {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        results[name] = evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
    return results, models


def print_report(results):
    table = pd.DataFrame(
        {name: {k: v for k, v in m.items() if k != "confusion_matrix"} for name, m in results.items()}
    ).T
    table.columns = ["Train Acc", "Test Acc", "Precision (1)", "Recall (1)", "F1 (1)", "CV Acc (mean)", "CV Acc (std)"]
    print("\n=== Model comparison (test set: 61 patients) ===")
    print(table.astype(float).round(4).to_string())

    for name, m in results.items():
        cm = pd.DataFrame(m["confusion_matrix"],
                          index=["Actual: Low Risk (0)", "Actual: High Risk (1)"],
                          columns=["Pred: Low Risk (0)", "Pred: High Risk (1)"])
        print(f"\n--- {name}: confusion matrix ---\n{cm.to_string()}")

    best = max(results, key=lambda n: results[n]["test_acc"])
    print(f"\nBest test accuracy: {best} ({results[best]['test_acc']:.2%})")


if __name__ == "__main__":
    results, _ = run_pipeline()
    print_report(results)
