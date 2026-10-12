import os
import pickle
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from src.data.make_dataset import load_and_clean_data, select_features
from src.features.build_features import split_data, scale_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_compare_models(data_path: str = "heart.csv", model_save_path: str = "heart_disease_artifacts.pkl"):
    """
    Train and evaluate multiple models (Logistic Regression, Decision Tree, Random Forest)
    using Stratified 5-Fold Cross-Validation and holdout test set.
    Logs all experiments to MLflow and serializes the best model along with rich evaluation artifacts.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    logger.info("Loading and preprocessing data from %s...", data_path)
    df = load_and_clean_data(data_path)
    df = select_features(df)
    
    feature_names = [c for c in df.columns if c != 'target']
    
    # Holdout Split & Scaling
    X_train, X_test, y_train, y_test = split_data(df, target_col='target', test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    candidate_models = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"solver": "lbfgs", "max_iter": 1000}
        },
        "Decision_Tree": {
            "model": DecisionTreeClassifier(max_depth=4, random_state=42),
            "params": {"max_depth": 4, "criterion": "gini"}
        },
        "Random_Forest_Tuned": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 4}
        }
    }
    
    mlflow.set_experiment("Heart_Disease_Prediction")
    comparison_records = []
    trained_instances = {}
    
    for name, config in candidate_models.items():
        clf = config["model"]
        params = config["params"]
        
        # 5-Fold Stratified Cross-Validation on training set
        cv_acc = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="accuracy")
        cv_auc = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
        
        # Train on full train set
        clf.fit(X_train_scaled, y_train)
        trained_instances[name] = clf
        
        # Evaluate on test set
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        comparison_records.append({
            "Model": name.replace("_", " "),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "AUC-ROC": round(auc, 4),
            "CV_Acc_Mean": round(cv_acc.mean(), 4),
            "CV_Acc_Std": round(cv_acc.std(), 4),
            "CV_AUC_Mean": round(cv_auc.mean(), 4)
        })
        
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall", rec)
            mlflow.log_metric("test_f1", f1)
            mlflow.log_metric("test_auc_roc", auc)
            mlflow.log_metric("cv_accuracy_mean", cv_acc.mean())
            mlflow.log_metric("cv_accuracy_std", cv_acc.std())
            mlflow.log_metric("cv_auc_mean", cv_auc.mean())
            mlflow.sklearn.log_model(clf, artifact_path=f"model_{name.lower()}")
            
        logger.info("[%s] Test Acc: %.4f | AUC: %.4f | CV Acc: %.4f (±%.4f)",
                    name, acc, auc, cv_acc.mean(), cv_acc.std())

    df_comparison = pd.DataFrame(comparison_records)
    logger.info("Model Comparison Summary:\n%s", df_comparison.to_string(index=False))
    
    # Best model selection (Random Forest Tuned)
    best_clf = trained_instances["Random_Forest_Tuned"]
    y_pred_best = best_clf.predict(X_test_scaled)
    y_prob_best = best_clf.predict_proba(X_test_scaled)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred_best)
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    
    # Feature importance from Random Forest
    rf_feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": best_clf.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    artifacts = {
        "best_model": best_clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_comparison": df_comparison,
        "feature_importance": rf_feat_imp,
        "test_eval": {
            "y_test": y_test.values,
            "y_pred": y_pred_best,
            "y_prob": y_prob_best,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc_score(y_test, y_prob_best)
        }
    }
    
    with open(model_save_path, "wb") as f:
        pickle.dump(artifacts, f)
        
    logger.info("Artifacts saved successfully to %s", model_save_path)
    return artifacts


if __name__ == "__main__":
    train_and_compare_models()
