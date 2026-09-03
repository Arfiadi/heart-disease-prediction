import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.data.make_dataset import load_and_clean_data, select_features
from src.features.build_features import split_data, scale_features

def train_model(data_path: str, model_save_path: str):
    """
    Train a Random Forest model, evaluate it, and log metrics to MLflow.
    """
    # 1. Load & Prep Data
    df = load_and_clean_data(data_path)
    df = select_features(df)
    
    # 2. Split & Scale
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # MLflow tracking
    mlflow.set_experiment("Heart_Disease_Prediction")
    
    with mlflow.start_run():
        # Hyperparameters (bisa disesuaikan dari hasil GridSearchCV sebelumnya)
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
        mlflow.log_params(params)
        
        # 3. Train Model
        model = RandomForestClassifier(**params)
        model.fit(X_train_scaled, y_train)
        
        # 4. Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # 5. Save Artifacts locally for Streamlit
        artifacts = {
            'best_model': model,
            'scaler': scaler
        }
        with open(model_save_path, 'wb') as f:
            pickle.dump(artifacts, f)
            
        print(f"Model trained successfully.")
        print(f"Accuracy: {accuracy:.4f}, AUC-ROC: {auc:.4f}")
        print(f"Artifacts saved to {model_save_path}")

if __name__ == "__main__":
    # Ensure working directory is correct when running as script
    # Run from root directory of project
    data_path = "heart.csv"
    model_save_path = "heart_disease_artifacts.pkl"
    if os.path.exists(data_path):
        train_model(data_path, model_save_path)
    else:
        print(f"Data not found at {data_path}. Please run this script from the project root.")
