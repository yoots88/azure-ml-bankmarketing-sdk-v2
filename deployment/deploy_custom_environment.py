"""
deployment/deploy_custom_environment.py

FALLBACK deployment path - only use this if your rubric specifically
requires a custom score.py / custom environment instead of the no-code
MLflow deployment in deploy.py (which is the one you should try first).

The failure mode with a hand-written conda_inference.yaml is almost
always a package-version mismatch against whatever the AutoML trial
was actually fit with. This script removes the guesswork by
downloading the *actual* conda.yaml the winning run was trained with
(logged alongside its MLflow model) and building the inference
environment from that file, instead of a manually typed one.

Usage:
    python deploy_custom_environment.py --best_run_id musing_wire_ldcscg5qht_8
"""

import argparse
import tempfile
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "6214e780-2ea7-42fc-965e-dae0d86631fc"
RESOURCE_GROUP = "udacity-practice"
WORKSPACE_NAME = "UdacityWorkspace"

ENDPOINT_NAME = "bankmarketing-sdkv2-endpoint"
DEPLOYMENT_NAME = "blue-custom"
INSTANCE_TYPE = "Standard_D2_v2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"


def get_ml_client() -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def download_run_conda_file(ml_client: MLClient, best_run_id: str) -> Path:
    """Pull the exact conda.yaml the winning trial was fit with, instead
    of guessing package versions by hand."""

    download_dir = Path(tempfile.mkdtemp(prefix="automl_env_"))

    print("Downloading the winning run's own conda.yaml ...")
    ml_client.jobs.download(
        name=best_run_id,
        download_path=str(download_dir),
        output_name="mlflow-model",
    )

    candidates = list(download_dir.rglob("conda.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No conda.yaml found under the downloaded artifacts for run {best_run_id}. "
            "Check that the run actually logged an MLflow model."
        )

    conda_path = candidates[0]
    print("Found run environment file:", conda_path)
    return conda_path


def build_environment(conda_path: Path) -> Environment:
    return Environment(
        name="bankmarketing-inference-env-exact",
        description="Inference environment built from the winning AutoML run's own conda.yaml.",
        conda_file=str(conda_path),
        image="mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:20250620.v2",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_run_id", type=str, required=True)
    args = parser.parse_args()

    ml_client = get_ml_client()

    conda_path = download_run_conda_file(ml_client, args.best_run_id)
    environment = build_environment(conda_path)

    model = ml_client.jobs.get(args.best_run_id)  # ensure the run exists / is reachable

    endpoint = ManagedOnlineEndpoint(name=ENDPOINT_NAME, auth_mode="key")
    print("Creating/updating endpoint:", ENDPOINT_NAME)
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=f"azureml://jobs/{args.best_run_id}/outputs/artifacts/outputs/model.pkl",
        environment=environment,
        code_configuration=CodeConfiguration(
            code=str(DEPLOYMENT_DIR),
            scoring_script="score.py",
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=1,
    )

    print("Creating/updating deployment:", DEPLOYMENT_NAME)
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print("Custom-environment deployment finished. Endpoint:", endpoint.name)


if __name__ == "__main__":
    main()
