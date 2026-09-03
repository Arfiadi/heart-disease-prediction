import pandas as pd
import numpy as np

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV, remove duplicates and handle outliers.
    """
    df = pd.read_csv(filepath)
    
    # Menghapus duplikat
    df = df.drop_duplicates()
    
    # Menangani nilai anomali/outlier
    # Sesuai deskripsi: menghapus pencilan pada trestbps dan chol
    for col in ['trestbps', 'chol']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
    return df

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the 9 most relevant features and the target.
    Features: cp, thalach, slope, oldpeak, exang, ca, thal, sex, age
    """
    selected_cols = ['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age', 'target']
    cols_to_keep = [col for col in selected_cols if col in df.columns]
    return df[cols_to_keep]

if __name__ == "__main__":
    df = load_and_clean_data("../../heart.csv")
    df = select_features(df)
    print(f"Data cleaned and selected. Shape: {df.shape}")
