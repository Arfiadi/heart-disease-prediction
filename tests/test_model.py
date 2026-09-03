import pandas as pd
from src.data.make_dataset import load_and_clean_data, select_features
from src.features.build_features import split_data, scale_features

def test_data_pipeline():
    """
    Test that the data loading and prep pipeline works and outputs the expected shape and columns.
    """
    df = load_and_clean_data("heart.csv")
    assert not df.empty, "Dataframe should not be empty"
    
    df_selected = select_features(df)
    expected_cols = ['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age', 'target']
    assert list(df_selected.columns) == expected_cols, "Columns do not match expected features"

def test_split_and_scale():
    """
    Test data splitting and scaling functionality.
    """
    df = load_and_clean_data("heart.csv")
    df_selected = select_features(df)
    
    X_train, X_test, y_train, y_test = split_data(df_selected)
    assert len(X_train) > 0 and len(X_test) > 0, "Train or test split is empty"
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape, "Scaled data shape mismatch"
