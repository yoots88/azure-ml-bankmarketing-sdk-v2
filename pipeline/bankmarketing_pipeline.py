"""
pipeline/bankmarketing_pipeline.py

SDK v2 pipeline that:
  1. Runs `src/prepare_data.py` as a command component to clean the raw
     Bank Marketing data.
  2. Feeds the cleaned output directly into an AutoML classification
     node (no separate "download -> re-upload" step needed - v2 lets
     an AutoML job be a node inside a regular pipeline).

Submit with:
    python pipeline/bankmarketing_pipeline.py --raw_data azureml:bankmarketing-raw:1

Adjust `--raw_data` to whatever data asset (or local/blob path) you
registered the original UCI Bank Marketing CSV under.
"""

import argparse
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, automl, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = "6214e780-2ea7-42fc-965e-dae0d86631fc"
RESOURCE_GROUP = "udacity-practice"
WORKSPACE_NAME = "UdacityWorkspace"

COMPUTE_NAME = "udacity-project2-cluster"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ENV_FILE = PROJECT_ROOT / "environment" / "conda.yaml"


def get_ml_client() -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )


def build_environment() -> Environment:
    return Environment(
        name="bankmarketing-pipeline-env",
        description="Environment for the bank marketing prep step.",
        conda_file=str(ENV_FILE),
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )


def build_prepare_data_component(environment: Environment):
    return command(
        name="prepare_bankmarketing_data",
        display_name="Clean bank marketing data",
        description="Cleans the raw bank marketing CSV into the format the rest of the pipeline expects.",
        inputs={"input_data": Input(type=AssetTypes.URI_FILE)},
        outputs={"output_data": Output(type=AssetTypes.URI_FILE)},
        code=str(SRC_DIR),
        command=(
            "python prepare_data.py "
            "--input_data ${{inputs.input_data}} "
            "--output_data ${{outputs.output_data}}"
        ),
        environment=environment,
        compute=COMPUTE_NAME,
    )


def build_pipeline(prepare_data_component):
    @pipeline(
        name="bankmarketing_sdk_v2_pipeline",
        description="Clean data, then run AutoML classification on the bank marketing dataset.",
        compute=COMPUTE_NAME,
    )
    def bankmarketing_pipeline(raw_data: Input):
        prep_step = prepare_data_component(input_data=raw_data)

        classification_node = automl.classification(
            training_data=prep_step.outputs.output_data,
            target_column_name="y",
            primary_metric="accuracy",
            experiment_name="bankmarketing-sdk-v2-pipeline",
            enable_model_explainability=False,
        )
        classification_node.set_limits(
            max_trials=10,
            max_concurrent_trials=1,
            timeout_minutes=20,
            trial_timeout_minutes=5,
        )
        classification_node.outputs.best_model = Output(type=AssetTypes.MLFLOW_MODEL)

        return {"best_model": classification_node.outputs.best_model}

    return bankmarketing_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data",
        type=str,
        default="azureml:bankmarketing-raw:1",
        help="azureml:// data asset reference (or path) to the raw bankmarketing CSV.",
    )
    args = parser.parse_args()

    ml_client = get_ml_client()

    environment = build_environment()
    prepare_data_component = build_prepare_data_component(environment)
    bankmarketing_pipeline = build_pipeline(prepare_data_component)

    pipeline_job = bankmarketing_pipeline(
        raw_data=Input(type=AssetTypes.URI_FILE, path=args.raw_data)
    )

    submitted = ml_client.jobs.create_or_update(pipeline_job)

    print("Pipeline job submitted.")
    print("Name:", submitted.name)
    print("Studio URL:", submitted.studio_url)


if __name__ == "__main__":
    main()
