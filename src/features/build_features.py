import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Tuple, Any, Optional


def split_data(df: pd.DataFrame, target_col: str = 'target', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into train and test sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def create_model_pipeline(classifier: Any, scaler: Optional[Any] = None) -> Pipeline:
    """
    Constructs an atomic Scikit-Learn Pipeline combining feature scaling and estimator.
    Prevents data leakage during cross-validation and preserves feature names.
    """
    if scaler is None:
        scaler = StandardScaler()
    if hasattr(scaler, "set_output"):
        scaler.set_output(transform="pandas")
    return Pipeline([
        ('scaler', scaler),
        ('classifier', classifier)
    ])


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Legacy helper: Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled, scaler
