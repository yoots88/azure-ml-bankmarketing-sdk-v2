
"""Inspect Azure ML AutoML child jobs and MLflow model artifacts."""

from pathlib import Path
from typing import Any


def _normalized_status(value: Any) -> str:
    """Normalize Azure ML job status strings and enums."""

    if value is None:
        return ""

    normalized = getattr(value, "value", None)

    if normalized is None:
        normalized = getattr(value, "name", None)

    if normalized is None:
        normalized = value

    return str(normalized).rsplit(".", 1)[-1].strip().lower()


def inspect_parent_properties(parent_job):
    """Display parent AutoML job properties and tags."""

    print("=" * 60)
    print("PARENT JOB PROPERTIES")
    print("=" * 60)

    properties = getattr(parent_job, "properties", {}) or {}

    if properties:
        for key, value in properties.items():
            print(f"{key}: {value}")
    else:
        print("No parent properties exposed.")

    print("\n" + "=" * 60)
    print("PARENT JOB TAGS")
    print("=" * 60)

    tags = getattr(parent_job, "tags", {}) or {}

    if tags:
        for key, value in tags.items():
            print(f"{key}: {value}")
    else:
        print("No parent tags exposed.")

    print("\n" + "=" * 60)


def list_child_jobs(ml_client, parent_job_name):
    """Retrieve all child jobs associated with an AutoML parent."""

    return list(
        ml_client.jobs.list(
            parent_job_name=parent_job_name
        )
    )


def print_child_job_summary(child_jobs):
    """Print child job names, statuses, and display names."""

    print("\n========== AUTOML CHILD JOB SUMMARY ==========")

    if not child_jobs:
        print("No child jobs found.")
        return

    for job in child_jobs:
        print(
            job.name,
            "|",
            job.status,
            "|",
            getattr(job, "display_name", None)
        )

    print("Total child jobs:", len(child_jobs))


def get_best_child_job(ml_client, parent_job, child_jobs):
    """Identify the winning AutoML child job.

    First checks parent properties for a best-child identifier.
    If unavailable, requests an explicit manual override.
    """

    properties = getattr(parent_job, "properties", {}) or {}

    candidate_keys = [
        "best_child_run_id",
        "best_child_job_name",
        "best_run_id",
        "azureml.best_child_run_id",
        "azureml.best_child_job_name",
    ]

    # Attempt automatic identification from parent properties
    for key in candidate_keys:
        value = properties.get(key)

        if value:
            print(
                f"Found potential winning child ID "
                f"in parent property '{key}': {value}"
            )

            return ml_client.jobs.get(name=value)

    # Check whether completed child jobs exist
    completed = [
        job for job in child_jobs
        if _normalized_status(job.status) == "completed"
    ]

    if not completed:
        raise RuntimeError(
            "No completed AutoML child jobs were found."
        )

    print(
        f"Found {len(completed)} completed child jobs, "
        "but the winning child ID was not exposed "
        "in the parent properties."
    )

    raise RuntimeError(
        "Unable to automatically identify the winning model. "
        "Open the AutoML parent job Models tab, identify the "
        "winning child job name, and provide it using "
        "BEST_CHILD_JOB_NAME_OVERRIDE in the notebook."
    )


def download_and_find_mlmodel(
    ml_client,
    child_job_name,
    destination
):
    """Download child job artifacts and locate the MLmodel file."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    print("\n========== MODEL ARTIFACT DOWNLOAD ==========")
    print("Child job:", child_job_name)
    print("Destination:", destination)

    ml_client.jobs.download(
        name=child_job_name,
        download_path=str(destination),
        all=True,
    )

    matches = sorted(destination.rglob("MLmodel"))

    if not matches:
        raise FileNotFoundError(
            f"No MLmodel file was found in downloaded "
            f"output for {child_job_name}."
        )

    mlmodel_file = matches[0]

    print("MLmodel found:", mlmodel_file)

    if len(matches) > 1:
        print(
            f"Warning: Found {len(matches)} MLmodel files. "
            "Returning the first discovered file."
        )

    return mlmodel_file


if __name__ == "__main__":
    print(
        "Import this module from "
        "Project_2_Master_Notebook_SDK_v2_Working.ipynb."
    )
