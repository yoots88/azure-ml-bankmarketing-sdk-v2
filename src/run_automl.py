"""Rerun-safe AutoML job search, validation, and submission helpers."""
from __future__ import annotations
from typing import Any
from azure.ai.ml import Input, automl
from azure.ai.ml.constants import AssetTypes


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _normalized_status(value: Any) -> str:
    """Normalize Azure ML job status strings and enums."""

    if value is None:
        return ""

    # Handle enum values when available
    normalized = getattr(value, "value", None)

    if normalized is None:
        normalized = getattr(value, "name", None)

    if normalized is None:
        normalized = value

    # Handle representations such as JobStatus.COMPLETED
    return str(normalized).rsplit(".", 1)[-1].strip().lower()


def _job_uses_expected_configuration(job, data_asset, compute_name, target_column):
    dump = job._to_dict() if hasattr(job, "_to_dict") else {}
    blob = str(dump).lower()
    expected_asset = f"{data_asset.name}:{data_asset.version}".lower()
    checks = {
        "completed": _normalized_status(job.status).lower() == "completed",
        "classification": "classification" in blob,
        "compute": compute_name.lower() in blob,
        "target": target_column.lower() in blob,
        "data_name": data_asset.name.lower() in blob,
        "data_version": str(data_asset.version).lower() in blob or expected_asset in blob,
    }
    return all(checks.values()), checks


def find_suitable_completed_jobs(ml_client, experiment_name, data_asset, compute_name, target_column):
    candidates = []
    for job in ml_client.jobs.list(parent_job_name=None):
        if getattr(job, "experiment_name", None) != experiment_name:
            continue
        full_job = ml_client.jobs.get(job.name)
        valid, checks = _job_uses_expected_configuration(
            full_job, data_asset, compute_name, target_column
        )
        if valid:
            candidates.append(full_job)
        else:
            print("Rejected job", full_job.name, checks)
    candidates.sort(
        key=lambda x: getattr(x, "creation_context", None).created_at
        if getattr(x, "creation_context", None) else "",
        reverse=True,
    )
    return candidates


def build_automl_job(training_data_asset, compute_name, target_column,
                     max_trials=5, max_concurrent_trials=1):
    training_input = Input(
        type=AssetTypes.MLTABLE,
        path=f"azureml:{training_data_asset.name}:{training_data_asset.version}",
    )
    job = automl.classification(
        compute=compute_name,
        experiment_timeout_minutes=60,
        training_data=training_input,
        target_column_name=target_column,
        primary_metric="AUC_weighted",
        n_cross_validations=5,
        enable_model_explainability=True,
        tags={
            "project": "udacity-project-2",
            "sdk": "v2",
            "data_asset": f"{training_data_asset.name}:{training_data_asset.version}",
            "target_column": target_column,
            "task": "classification",
        },
    )
    job.set_limits(
        max_trials=max_trials,
        max_concurrent_trials=max_concurrent_trials,
        timeout_minutes=60,
        trial_timeout_minutes=20,
    )
    job.set_training(
        enable_onnx_compatible_models=False,
        enable_stack_ensemble=True,
        enable_vote_ensemble=True,
    )
    return job


def get_or_submit_automl_job(ml_client, experiment_name, training_data_asset,
                             compute_name, target_column, allow_new_job=False,
                             max_trials=5, max_concurrent_trials=1):
    candidates = find_suitable_completed_jobs(
        ml_client, experiment_name, training_data_asset,
        compute_name, target_column
    )
    if candidates:
        return candidates[0], False
    if not allow_new_job:
        raise RuntimeError(
            "No suitable completed AutoML job was found. Set "
            "ALLOW_NEW_AUTOML_JOB=True only to permit a billable submission."
        )
    definition = build_automl_job(
        training_data_asset, compute_name, target_column,
        max_trials, max_concurrent_trials
    )
    submitted = ml_client.jobs.create_or_update(
        definition, experiment_name=experiment_name
    )
    return submitted, True


if __name__ == "__main__":
    print("Import this module from Project_2_Master_Notebook_SDK_v2_Working.ipynb.")