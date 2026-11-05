"""
Tests for the heart disease classification pipeline.

Run with:
    pytest -v
"""
import numpy as np
import pandas as pd
import pytest

import pipeline as pl


@pytest.fixture(scope="module")
def raw_df():
    return pl.load_data()


@pytest.fixture(scope="module")
def splits(raw_df):
    return pl.split_data(pl.clean_raw_data(raw_df))


@pytest.fixture(scope="module")
def fitted(splits):
    X_train, X_test, _, _ = splits
    prep = pl.Preprocessor().fit(X_train)
    return prep, prep.transform(X_train), prep.transform(X_test)


@pytest.fixture(scope="module")
def pipeline_output():
    return pl.run_pipeline()


# ---------- Data ----------

def test_dataset_schema(raw_df):
    expected = {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"}
    assert set(raw_df.columns) == expected
    assert len(raw_df) == 303


def test_cleaning_removes_duplicates_and_invalid_labels():
    df = pd.DataFrame({"age": [50, 50, 60, 70], "target": [1, 1, 0, 5]})
    cleaned = pl.clean_raw_data(df)
    assert len(cleaned) == 2
    assert set(cleaned["target"]) <= {0, 1}


def test_split_sizes_and_no_overlap(splits):
    X_train, X_test, y_train, y_test = splits
    assert len(X_train) == len(y_train) == 241
    assert len(X_test) == len(y_test) == 61
    assert set(X_train.index).isdisjoint(X_test.index)


# ---------- Preprocessing ----------

def test_no_missing_values_after_preprocessing(fitted):
    _, X_train_scaled, X_test_scaled = fitted
    assert not X_train_scaled.isnull().values.any()
    assert not X_test_scaled.isnull().values.any()


def test_train_and_test_share_feature_space(fitted):
    prep, X_train_scaled, X_test_scaled = fitted
    assert list(X_train_scaled.columns) == list(X_test_scaled.columns) == prep.feature_names_
    for col in ["cp_1", "cp_2", "cp_3", "restecg_1", "restecg_2"]:
        assert col in prep.feature_names_
    assert "cp" not in prep.feature_names_ and "restecg" not in prep.feature_names_


def test_training_features_are_standardized(fitted):
    _, X_train_scaled, _ = fitted
    assert np.allclose(X_train_scaled.mean(), 0, atol=1e-8)
    assert np.allclose(X_train_scaled.std(ddof=0), 1, atol=1e-8)


def test_preprocessing_statistics_come_from_train_only(splits):
    """Fitting must not change when the test set is altered (no data leakage)."""
    X_train, X_test, _, _ = splits
    prep_a = pl.Preprocessor().fit(X_train)
    prep_b = pl.Preprocessor().fit(X_train)
    X_test_modified = X_test.copy()
    X_test_modified["chol"] = 10_000
    prep_b.transform(X_test_modified)
    assert prep_a.bounds_ == prep_b.bounds_
    assert prep_a.fill_values_ == prep_b.fill_values_


def test_outliers_are_replaced_by_train_median(splits):
    X_train, _, _, _ = splits
    prep = pl.Preprocessor().fit(X_train)
    sample = X_train.iloc[[0]].copy()
    sample["chol"] = 10_000
    sample_out = prep.transform(sample)
    median_row = sample.copy()
    median_row["chol"] = prep.fill_values_["chol"]
    assert np.allclose(sample_out.values, prep.transform(median_row).values)


# ---------- Models ----------

def test_predictions_are_binary_and_complete(pipeline_output):
    results, _ = pipeline_output
    for metrics in results.values():
        assert metrics["confusion_matrix"].shape == (2, 2)
        assert metrics["confusion_matrix"].sum() == 61


def test_metrics_are_consistent_with_confusion_matrix(pipeline_output):
    results, _ = pipeline_output
    for metrics in results.values():
        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
        assert metrics["test_acc"] == pytest.approx((tp + tn) / 61)
        assert metrics["precision_high_risk"] == pytest.approx(tp / (tp + fp))
        assert metrics["recall_high_risk"] == pytest.approx(tp / (tp + fn))


def test_models_beat_majority_baseline(pipeline_output, splits):
    results, _ = pipeline_output
    _, _, _, y_test = splits
    baseline = y_test.value_counts(normalize=True).max()
    for name, metrics in results.items():
        assert metrics["test_acc"] > baseline, name


def test_results_reproduce_notebook(pipeline_output):
    """The pipeline must reproduce the results reported in the notebook."""
    results, _ = pipeline_output
    assert results["Logistic Regression"]["test_acc"] == pytest.approx(0.8197, abs=1e-4)
    assert results["Decision Tree"]["test_acc"] == pytest.approx(0.7869, abs=1e-4)
    assert results["Random Forest"]["test_acc"] == pytest.approx(0.8525, abs=1e-4)


def test_pipeline_is_deterministic():
    first, _ = pl.run_pipeline()
    second, _ = pl.run_pipeline()
    for name in first:
        assert np.array_equal(first[name]["confusion_matrix"], second[name]["confusion_matrix"])
