import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV, remove duplicates and handle outliers using IQR.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    initial_shape = df.shape
    
    # Menghapus duplikat
    df = df.drop_duplicates()
    
    # Menangani nilai anomali/outlier pada trestbps dan chol
    for col in ['trestbps', 'chol']:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
    logger.info("Loaded and cleaned %s: %s -> %s rows", filepath, initial_shape[0], df.shape[0])
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the 9 most relevant medical features and the target.
    Features: cp, thalach, slope, oldpeak, exang, ca, thal, sex, age, target
    """
    selected_cols = ['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age', 'target']
    cols_to_keep = [col for col in selected_cols if col in df.columns]
    return df[cols_to_keep]

if __name__ == "__main__":
    path = "heart.csv" if os.path.exists("heart.csv") else "../../heart.csv"
    data = load_and_clean_data(path)
    data = select_features(data)
    logger.info("Data cleaned and selected. Final shape: %s", data.shape)
