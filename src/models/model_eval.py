import os
import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import dagshub

# Initialize DagsHub for experiment tracking
dagshub.init(
    repo_owner='sivakumar026',
    repo_name='mlops_pipeline',
    mlflow=True
)

# Set MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/sivakumar026/mlops_pipeline.mlflow"
)

# Set experiment name
mlflow.set_experiment("DVC pipeline")


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")


def prepare_data(data: pd.DataFrame):
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")


def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}: {e}")


def evaluation_model(model, X_test, y_test, model_name):
    try:
        # Load parameters
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)

        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log params
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("n_estimators", n_estimators)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")

        cm_path = "reports/confusion_matrix.png"

        plt.savefig(cm_path)
        plt.close()

        # Log confusion matrix
        mlflow.log_artifact(cm_path)

        metrics_dict = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        return metrics_dict

    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")


def save_metrics(metrics: dict, metrics_path: str):
    try:
        with open(metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)

    except Exception as e:
        raise Exception(f"Error saving metrics: {e}")


def main():

    try:
        # Create reports directory
        os.makedirs("reports", exist_ok=True)

        # Paths
        test_data_path = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"

        model_name = "Best Model"

        # Load data and model
        test_data = load_data(test_data_path)

        X_test, y_test = prepare_data(test_data)

        model = load_model(model_path)

        # Start MLflow run
        with mlflow.start_run() as run:

            # Evaluate model
            metrics = evaluation_model(
                model,
                X_test,
                y_test,
                model_name
            )

            # Save metrics
            save_metrics(metrics, metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
            mlflow.log_artifact("src/models/model_eval.py")

            # Save run info
            run_info = {
                "run_id": run.info.run_id,
                "model_name": model_name
            }

            with open("reports/run_info.json", "w") as file:
                json.dump(run_info, file, indent=4)

            print("Model evaluation completed successfully!")

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()