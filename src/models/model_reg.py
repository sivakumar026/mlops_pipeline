import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import dagshub


# Initialize DagsHub
dagshub.init(
    repo_owner="sivakumar026",
    repo_name="mlops_pipeline",
    mlflow=True
)

# Set MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/sivakumar026/mlops_pipeline.mlflow"
)

# Set experiment
mlflow.set_experiment("Final_Model")


def load_run_info(filepath: str) -> dict:
    """
    Load run_id and model_name from JSON file
    """
    try:
        with open(filepath, "r") as file:
            run_info = json.load(file)

        return run_info

    except Exception as e:
        raise Exception(f"Error loading run info: {e}")


def register_model(run_id: str, model_name: str):

    try:

        # Create MLflow client
        client = MlflowClient()

        # Correct MLflow model URI
        model_uri = f"runs:/{run_id}/Best Model"

        print(f"Model URI: {model_uri}")

        # Register model
        reg_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        # Get version
        model_version = reg_model.version

        print(
            f"Successfully registered model "
            f"'{model_name}' version {model_version}"
        )

        # Transition stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True
        )

        print(
            f"Model '{model_name}' "
            f"version {model_version} "
            f"moved to Staging."
        )

    except Exception as e:
        raise Exception(f"Error registering model: {e}")


def main():

    try:

        # Path to run info JSON
        run_info_path = "reports/run_info.json"

        # Load run info
        run_info = load_run_info(run_info_path)

        run_id = run_info["run_id"]
        model_name = run_info["model_name"]

        print(f"Run ID: {run_id}")
        print(f"Model Name: {model_name}")

        # Register model
        register_model(run_id, model_name)

    except Exception as e:
        raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()