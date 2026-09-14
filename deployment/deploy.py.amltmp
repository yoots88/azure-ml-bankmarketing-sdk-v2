"""
deployment/deploy.py

Deploys the winning AutoML trial to a Managed Online Endpoint using the
"no-code" MLflow deployment path.

WHY YOUR OLD DEPLOYMENT WAS FAILING
------------------------------------
A hand-written `conda_inference.yaml` + custom `score.py` almost never
matches the exact `azureml-automl-runtime` / `azureml-training-tabular`
/ scikit-learn build that a specific AutoML trial was actually fit
with. When the container starts, `init()` calls `pickle.load(...)` (or
`joblib.load`) against a model that was pickled under a different
library version than what's installed in your custom environment, and
unpickling fails silently or raises inside the container - which is
what shows up in Studio as a generic "deployment failed" / unhealthy
container error.

THE FIX
-------
Every AutoML trial already logs an MLflow-format model under
`outputs/mlflow-model` on its own run, packaged together with the
*exact* `conda.yaml` it was trained with. If you register that
artifact as an `MLFLOW_MODEL` and deploy it WITHOUT passing
`environment=` or `code_configuration=`, Azure ML builds the inference
image itself from the model's own conda file - so there is no version
mismatch to debug. This is the same mechanism the Studio "Deploy"
button uses under the hood.

Usage:
    python deploy.py --best_run_id musing_wire_ldcscg5qht_8
    # or, to auto-resolve the best child of a parent AutoML job:
    python deploy.py --parent_job_name musing_wire_ldcscg5qht
"""

import argparse

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "6214e780-2ea7-42fc-965e-dae0d86631fc"
RESOURCE_GROUP = "udacity-practice"
WORKSPACE_NAME = "UdacityWorkspace"

MODEL_NAME = "bankmarketing-automl-mlflow-model"
ENDPOINT_NAME = "bankmarketing-sdkv2-endpoint"
DEPLOYMENT_NAME = "blue"
INSTANCE_TYPE = "Standard_D2_v2"


def get_ml_client() -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def resolve_best_run_id(ml_client: MLClient, best_run_id: str, parent_job_name: str) -> str:
    if best_run_id:
        return best_run_id

    if not parent_job_name:
        raise ValueError("Pass either --best_run_id or --parent_job_name.")

    print(f"Looking up the best child run of parent job: {parent_job_name}")

    import mlflow

    tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()

    parent_run = mlflow_client.get_run(parent_job_name)
    best_child_run_id = parent_run.data.tags.get("automl_best_child_run_id")

    if not best_child_run_id:
        raise RuntimeError(
            f"Could not find an 'automl_best_child_run_id' tag on parent run {parent_job_name}. "
            "Pass --best_run_id explicitly instead."
        )

    print("Resolved best child run:", best_child_run_id)
    return best_child_run_id


def register_model(ml_client: MLClient, best_run_id: str) -> Model:
    model = Model(
        path=f"azureml://jobs/{best_run_id}/outputs/artifacts/outputs/mlflow-model",
        name=MODEL_NAME,
        type=AssetTypes.MLFLOW_MODEL,
        description=f"Best AutoML child run ({best_run_id}) MLflow model for bank marketing classification.",
    )

    print("Registering MLflow model from run:", best_run_id)
    registered_model = ml_client.models.create_or_update(model)
    print("Registered model:", registered_model.name, "version:", registered_model.version)

    return registered_model


def create_or_update_endpoint(ml_client: MLClient) -> None:
    try:
        endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
        print("Endpoint already exists:", ENDPOINT_NAME)
        print("Provisioning state:", endpoint.provisioning_state)

    except Exception:
        endpoint = ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            description="Bank marketing classification endpoint (SDK v2, no-code MLflow deployment).",
            auth_mode="key",
        )

        print("Endpoint does not exist. Creating:", ENDPOINT_NAME)
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        print("Endpoint created successfully.")    


def create_or_update_deployment(ml_client: MLClient, model: Model) -> None:
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=model,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        # Deliberately no `environment=` / `code_configuration=` here:
        # Azure ML derives both from the MLflow model's own conda.yaml,
        # which guarantees the versions match what the trial was fit with.
    )

    print("Creating/updating deployment:", DEPLOYMENT_NAME)
    print("(First-time creation typically takes 10-15 minutes.)")
    ml_client.online_deployments.begin_create_or_update(deployment).result()


def route_all_traffic(ml_client: MLClient) -> None:
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--best_run_id",
        type=str,
        default=None,
        help="Run ID of the winning AutoML child trial (e.g. musing_wire_ldcscg5qht_8).",
    )
    parser.add_argument(
        "--parent_job_name",
        type=str,
        default=None,
        help="Parent AutoML job name; the script will auto-resolve its best child run.",
    )
    args = parser.parse_args()

    ml_client = get_ml_client()

    best_run_id = resolve_best_run_id(ml_client, args.best_run_id, args.parent_job_name)

    model = register_model(ml_client, best_run_id)

    create_or_update_endpoint(ml_client)
    create_or_update_deployment(ml_client, model)
    route_all_traffic(ml_client)

    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)

    print()
    print("=" * 60)
    print("DEPLOYMENT SUCCESSFUL")
    print("=" * 60)
    print("Endpoint:", endpoint.name)
    print("Scoring URI:", endpoint.scoring_uri)
    print("Primary key:", keys.primary_key)
    print("Traffic:", endpoint.traffic)
    print("=" * 60)


if __name__ == "__main__":
    main()
