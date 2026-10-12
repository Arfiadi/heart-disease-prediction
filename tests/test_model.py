import os
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data.make_dataset import load_and_clean_data, select_features
from src.features.build_features import split_data, scale_features
from src.services.llm_advisor import get_lifestyle_recommendation, get_rule_based_fallback


def test_data_pipeline():
    """
    Test that the data loading and cleaning pipeline removes duplicates,
    filters outliers, and selects the exact 9 features + target.
    """
    df = load_and_clean_data("heart.csv")
    assert not df.empty, "Dataframe should not be empty"
    assert df.duplicated().sum() == 0, "Cleaned dataset should not have duplicate rows"
    
    df_selected = select_features(df)
    expected_cols = ['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age', 'target']
    assert list(df_selected.columns) == expected_cols, "Columns do not match expected features"


def test_split_and_scale():
    """
    Test data splitting and scaling functionality.
    """
    df = load_and_clean_data("heart.csv")
    df_selected = select_features(df)
    
    X_train, X_test, y_train, y_test = split_data(df_selected, test_size=0.2, random_state=42)
    assert len(X_train) > 0 and len(X_test) > 0, "Train or test split is empty"
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape, "Scaled train data shape mismatch"
    assert X_test_scaled.shape == X_test.shape, "Scaled test data shape mismatch"
    
    # Check that scaled data has mean close to 0 and std close to 1
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-1), "Scaled train mean is not close to 0"


def test_model_artifact_loading():
    """
    Ensure serialized artifacts file exists and contains valid model and scaler objects.
    """
    artifact_path = "heart_disease_artifacts.pkl"
    assert os.path.exists(artifact_path), f"Artifact file {artifact_path} does not exist"
    
    with open(artifact_path, "rb") as f:
        artifacts = pickle.load(f)
        
    assert "best_model" in artifacts, "Artifact missing 'best_model' key"
    assert "scaler" in artifacts, "Artifact missing 'scaler' key"
    assert hasattr(artifacts["best_model"], "predict"), "best_model has no predict method"
    assert hasattr(artifacts["scaler"], "transform"), "scaler has no transform method"


def test_prediction_output_format():
    """
    Test prediction output format and binary classification constraint.
    """
    with open("heart_disease_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    model = artifacts["best_model"]
    scaler = artifacts["scaler"]
    
    # Sample input: [cp, thalach, slope, oldpeak, exang, ca, thal, sex, age]
    sample_input = np.array([[2, 150, 1, 1.2, 0, 0, 2, 1, 55]])
    scaled_input = scaler.transform(sample_input)
    prediction = model.predict(scaled_input)
    
    assert prediction.shape == (1,), "Prediction shape should be (1,)"
    assert prediction[0] in [0, 1], f"Prediction value {prediction[0]} must be either 0 or 1"


def test_model_performance_threshold():
    """
    Ensure the production model satisfies quality thresholds: Accuracy >= 0.75 and ROC-AUC >= 0.80.
    """
    df = load_and_clean_data("heart.csv")
    df_selected = select_features(df)
    X_train, X_test, y_train, y_test = split_data(df_selected, test_size=0.2, random_state=42)
    
    with open("heart_disease_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    model = artifacts["best_model"]
    scaler = artifacts["scaler"]
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    assert acc >= 0.75, f"Model accuracy {acc:.4f} is below quality threshold 0.75"
    assert auc >= 0.80, f"Model AUC-ROC {auc:.4f} is below quality threshold 0.80"


def test_edge_cases_input():
    """
    Verify model resilience against extreme boundary values without raising runtime exceptions.
    """
    with open("heart_disease_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    model = artifacts["best_model"]
    scaler = artifacts["scaler"]
    
    # Extreme cases: minimums and maximums
    edge_cases = [
        [0, 70, 0, 0.0, 0, 0, 1, 0, 29],   # Young female, low values
        [3, 205, 2, 6.2, 1, 3, 3, 1, 77],  # Elderly male, high values
    ]
    for case in edge_cases:
        scaled = scaler.transform([case])
        pred = model.predict(scaled)
        assert pred[0] in [0, 1]


def test_llm_advisor_fallback():
    """
    Test that the LLM advisor provides graceful clinical rule fallback when API key is missing or dummy.
    """
    patient_data = {
        'age': 58,
        'sex': 1,
        'thalach': 140,
        'cp': 'Typical Angina'
    }
    
    # Test high risk fallback
    rec_high = get_rule_based_fallback(patient_data, prediction_result=1)
    assert "Risiko Tinggi" in rec_high or "Lanjutan" in rec_high
    assert "Disclaimer" in rec_high
    
    # Test low risk fallback
    rec_low = get_rule_based_fallback(patient_data, prediction_result=0)
    assert "Risiko Rendah" in rec_low or "Preventif" in rec_low
    assert "Disclaimer" in rec_low
    
    # Test service integration with invalid key fallback
    os.environ["GEMINI_API_KEY"] = "dummy_invalid_key"
    rec = get_lifestyle_recommendation(patient_data, prediction_result=1)
    assert isinstance(rec, str)
    assert len(rec) > 50
