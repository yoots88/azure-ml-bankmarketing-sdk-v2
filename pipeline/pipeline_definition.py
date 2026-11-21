
"""
Azure ML SDK v2 pipeline definition for the Bank Marketing project.

Pipeline:
    1. Validate input data.
    2. Train a classification model.
    3. Produce model and metrics outputs.
"""

from pathlib import Path

from azure.ai.ml import Input, Output, command, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment


# ============================================================
# 0. SHARED HELPERS
# ============================================================

def _normalize_status(status):
    """
    Normalize Azure ML job statuses.

    Handles:
        JobStatus.COMPLETED
        Completed
        completed
        jobstatus.completed
    """

    raw_status = getattr(status, "value", status)

    return (
        str(raw_status)
        .strip()
        .lower()
        .rsplit(".", 1)[-1]
    )


def _normalize_type(job_type):
    """Normalize Azure ML job type."""

    raw_type = getattr(job_type, "value", job_type)

    return (
        str(raw_type)
        .strip()
        .lower()
        .rsplit(".", 1)[-1]
    )


# ============================================================
# 1. COMPONENT DEFINITIONS
# ============================================================


def _components():
    """Define the validation and training components."""

    repo_root = Path(__file__).resolve().parents[1]
    steps_dir = repo_root / "pipeline" / "steps"
    conda_file = repo_root / "environment" / "conda.yaml"

    required_files = [
        steps_dir / "validate_data.py",
        steps_dir / "train_model.py",
        conda_file,
    ]

    for required_file in required_files:
        if not required_file.is_file():
            raise FileNotFoundError(
                f"Required project file not found: {required_file}"
            )

    component_environment = Environment(
        name="bankmarketing-training-env",
        description="Environment for Bank Marketing pipeline components",
        image=(
            "mcr.microsoft.com/azureml/"
            "openmpi4.1.0-ubuntu22.04:latest"
        ),
        conda_file=str(conda_file),
    )

    # --------------------------------------------------------
    # Validation component
    # --------------------------------------------------------

    validate = command(
        name="validate_bankmarketing_data",
        display_name="Validate Bank Marketing data",
        code=str(steps_dir),
        command=(
            "python validate_data.py "
            "--input_data ${{inputs.input_data}} "
            "--target_column ${{inputs.target_column}} "
            "--validated_data ${{outputs.validated_data}}"
        ),
        inputs={
            "input_data": Input(
                type=AssetTypes.MLTABLE
            ),
            "target_column": Input(
                type="string"
            ),
        },
        outputs={
            "validated_data": Output(
                type=AssetTypes.URI_FOLDER
            ),
        },
        environment=component_environment,
    )

    # --------------------------------------------------------
    # Training component
    # --------------------------------------------------------

    train = command(
        name="train_bankmarketing_model",
        display_name="Train Bank Marketing model",
        code=str(steps_dir),
        command=(
            "python train_model.py "
            "--validated_data ${{inputs.validated_data}} "
            "--target_column ${{inputs.target_column}} "
            "--model_output ${{outputs.model_output}} "
            "--metrics_output ${{outputs.metrics_output}}"
        ),
        inputs={
            "validated_data": Input(
                type=AssetTypes.URI_FOLDER
            ),
            "target_column": Input(
                type="string"
            ),
        },
        outputs={
            "model_output": Output(
                type=AssetTypes.URI_FOLDER
            ),
            "metrics_output": Output(
                type=AssetTypes.URI_FOLDER
            ),
        },
        environment=component_environment,
    )

    return validate, train



# ============================================================
# 2. PIPELINE DEFINITION
# ============================================================

def build_training_pipeline(
    training_data_uri,
    compute_name,
    target_column,
):
    """
    Build the Bank Marketing training pipeline.

    Parameters
    ----------
    training_data_uri : str
        Azure ML data asset URI.

    compute_name : str
        Azure ML compute target.

    target_column : str
        Name of the prediction target.

    Returns
    -------
    PipelineJob
        Configured Azure ML pipeline job.
    """

    validate_component, train_component = _components()

    @dsl.pipeline(
        name="bankmarketing_training_pipeline",
        display_name="Bank Marketing SDK v2 training pipeline",
        description=(
            "Validate data, train a classifier, "
            "and produce metrics."
        ),
    )
    def pipeline_factory(
        training_data_uri: Input,
        target_column: str,
    ):

        # Step 1: Validate data.
        validate_step = validate_component(
            input_data=training_data_uri,
            target_column=target_column,
        )

        # Step 2: Train model using validated data.
        train_step = train_component(
            validated_data=validate_step.outputs.validated_data,
            target_column=target_column,
        )

        return {
            "model_output": train_step.outputs.model_output,
            "metrics_output": train_step.outputs.metrics_output,
        }

    pipeline_job = pipeline_factory(
        training_data_uri=Input(
            type=AssetTypes.MLTABLE,
            path=training_data_uri,
        ),
        target_column=target_column,
    )

    pipeline_job.settings.default_compute = compute_name

    # Extract data asset name for tracking.
    data_asset_tag = str(training_data_uri)

    if data_asset_tag.startswith("azureml:"):
        data_asset_tag = data_asset_tag.replace(
            "azureml:", "", 1
        )

    # Experiment tracking tags.
    pipeline_job.tags = {
        "project": "udacity-project-2",
        "sdk": "v2",
        "data_asset": data_asset_tag,
        "target_column": target_column,
        "compute": compute_name,
        "purpose": "pipeline-operationalization",
    }

    return pipeline_job


# ============================================================
# 3. FIND A COMPLETED PIPELINE JOB
# ============================================================

def find_completed_pipeline_job(
    ml_client,
    experiment_name,
    expected_tags=None,
    max_results=100,
):
    """
    Find the newest completed pipeline job matching
    the specified experiment and optional tags.

    Parameters
    ----------
    ml_client : MLClient
        Authenticated Azure ML client.

    experiment_name : str
        Target experiment name.

    expected_tags : dict, optional
        Tags that must match the existing pipeline job.

    max_results : int
        Maximum number of jobs to inspect.

    Returns
    -------
    PipelineJob or None
        Newest matching completed pipeline job.
    """

    expected_tags = expected_tags or {}

    jobs = ml_client.jobs.list(
        max_results=max_results,
    )

    matching_jobs = []

    for job in jobs:

        # ----------------------------------------------------
        # Filter by experiment.
        # ----------------------------------------------------

        job_experiment = getattr(
            job,
            "experiment_name",
            None,
        )

        if job_experiment != experiment_name:
            continue

        # ----------------------------------------------------
        # Only pipeline jobs.
        # ----------------------------------------------------

        job_type = _normalize_type(
            getattr(job, "type", None)
        )

        if job_type != "pipeline":
            continue

        # ----------------------------------------------------
        # Only successfully completed jobs.
        # ----------------------------------------------------

        job_status = _normalize_status(
            getattr(job, "status", None)
        )

        if job_status != "completed":
            continue

        # ----------------------------------------------------
        # Match expected tags.
        # ----------------------------------------------------

        job_tags = getattr(
            job,
            "tags",
            None,
        ) or {}

        tags_match = all(
            str(job_tags.get(key)) == str(value)
            for key, value in expected_tags.items()
        )

        if not tags_match:
            continue

        matching_jobs.append(job)

    if not matching_jobs:
        return None

    # --------------------------------------------------------
    # Sort newest job first.
    # --------------------------------------------------------

    matching_jobs.sort(
        key=lambda job: str(
            getattr(
                getattr(job, "creation_context", None),
                "created_at",
                None,
            ) or ""
        ),
        reverse=True,
    )

    return matching_jobs[0]


# ============================================================
# 4. VALIDATE PIPELINE COMPLETION
# ============================================================

def validate_completed_pipeline_job(
    ml_client,
    pipeline_job,
):
    """
    Verify that the parent pipeline and all its child jobs
    completed successfully.

    Parameters
    ----------
    ml_client : MLClient
        Authenticated Azure ML client.

    pipeline_job : PipelineJob or str
        Submitted pipeline job object or job name.

    Returns
    -------
    list
        Child jobs belonging to the pipeline.
    """

    # --------------------------------------------------------
    # Accept either a job object or a job name.
    # --------------------------------------------------------

    if isinstance(pipeline_job, str):
        pipeline_job = ml_client.jobs.get(
            name=pipeline_job
        )

    pipeline_name = pipeline_job.name

    # --------------------------------------------------------
    # Normalize parent pipeline status.
    # --------------------------------------------------------

    pipeline_status = _normalize_status(
        getattr(pipeline_job, "status", None)
    )

    print("Pipeline job:", pipeline_name)
    print("Pipeline status:", pipeline_status)

    if pipeline_status != "completed":
        raise RuntimeError(
            "Pipeline did not complete successfully. "
            f"Status: {pipeline_status}"
        )

    # --------------------------------------------------------
    # Retrieve child jobs.
    # --------------------------------------------------------

    child_jobs = list(
        ml_client.jobs.list(
            parent_job_name=pipeline_name
        )
    )

    if not child_jobs:
        raise RuntimeError(
            f"No child jobs were found for pipeline "
            f"{pipeline_name}."
        )

    print("\n========== CHILD JOB STATUS ==========")

    failed_children = []

    for child in child_jobs:

        child_status = _normalize_status(
            getattr(child, "status", None)
        )

        print(
            f"Name: {child.name} | "
            f"Status: {child_status}"
        )

        if child_status != "completed":
            failed_children.append(
                (child.name, child_status)
            )

    # --------------------------------------------------------
    # Fail validation if any child job did not complete.
    # --------------------------------------------------------

    if failed_children:
        raise RuntimeError(
            "One or more pipeline child jobs did not "
            "complete successfully: "
            f"{failed_children}"
        )

    print(
        "\nAll pipeline child jobs completed successfully."
    )

    return child_jobs


# ============================================================
# 5. MODULE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print(
        "Bank Marketing Azure ML SDK v2 "
        "pipeline definition module."
    )

    print(
        "Import build_training_pipeline() "
        "from your notebook to construct the pipeline."
    )
