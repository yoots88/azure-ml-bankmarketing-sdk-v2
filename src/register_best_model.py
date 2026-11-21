
"""Register the winning AutoML child run as an Azure ML MLflow model."""

import posixpath

import mlflow
from mlflow.tracking import MlflowClient

from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes


def find_model_by_source_job(ml_client, model_name, source_job):
    """Find an existing registered model associated with the winning run."""

    try:
        for summary in ml_client.models.list(name=model_name):
            model = ml_client.models.get(
                name=model_name,
                version=summary.version,
            )

            if (model.tags or {}).get("source_job") == source_job:
                return model

    except ResourceNotFoundError:
        return None

    return None


def find_mlflow_model_path(run_id):
    """Find the folder containing MLmodel in the run's artifacts."""

    tracking_client = MlflowClient()

    def search_artifacts(path=""):
        artifacts = tracking_client.list_artifacts(run_id, path)

        for artifact in artifacts:
            if artifact.is_dir:
                result = search_artifacts(artifact.path)

                if result is not None:
                    return result

            elif artifact.path.split("/")[-1] == "MLmodel":
                return posixpath.dirname(artifact.path)

        return None

    model_folder = search_artifacts()

    if model_folder is None:
        raise FileNotFoundError(
            f"No MLmodel file found in artifacts for run: {run_id}"
        )

    return model_folder


def get_or_register_model(
    ml_client,
    model_name,
    best_child_job,
    allow_registration=False,
):
    """Reuse or register the winning AutoML MLflow model."""

    existing = find_model_by_source_job(
        ml_client,
        model_name,
        best_child_job.name,
    )

    if existing:
        print("Existing registered model found.")
        print("Model:", existing.name)
        print("Version:", existing.version)
        return existing, False

    if not allow_registration:
        raise RuntimeError(
            "Model registration disabled. "
            "Set allow_registration=True to proceed."
        )

    run_id = best_child_job.name

    print("Finding MLflow model artifacts...")
    model_folder = find_mlflow_model_path(run_id)

    model_uri = (
        f"runs:/{run_id}/{model_folder}"
        if model_folder
        else f"runs:/{run_id}/"
    )

    print("Winning run:", run_id)
    print("Model artifact folder:", model_folder)
    print("Model URI:", model_uri)

    candidate = Model(
        name=model_name,
        path=model_uri,
        type=AssetTypes.MLFLOW_MODEL,
        description=(
            "Best Bank Marketing AutoML model "
            "for Udacity Project 2"
        ),
        tags={
            "source_job": run_id,
            "project": "udacity-project-2",
            "sdk": "v2",
            "display_name": best_child_job.display_name or "",
        },
    )

    registered = ml_client.models.create_or_update(candidate)

    print("\nModel registered successfully!")
    print("Name:", registered.name)
    print("Version:", registered.version)

    return registered, True
