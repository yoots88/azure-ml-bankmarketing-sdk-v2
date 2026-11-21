"""Azure Machine Learning workspace connection helpers.

Importing this module does not create resources or submit jobs.
"""

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def get_ml_client() -> MLClient:
    """Return an authenticated MLClient using the workspace config file."""
    credential = DefaultAzureCredential()
    return MLClient.from_config(credential=credential)


def validate_workspace(
    ml_client: MLClient,
    expected_workspace: str,
    expected_resource_group: str,
) -> None:
    """Raise a clear error when MLClient points to the wrong workspace."""
    actual_workspace = ml_client.workspace_name
    actual_resource_group = ml_client.resource_group_name

    if actual_workspace != expected_workspace:
        raise ValueError(
            f"Expected workspace {expected_workspace!r}, "
            f"but connected to {actual_workspace!r}."
        )
    if actual_resource_group != expected_resource_group:
        raise ValueError(
            f"Expected resource group {expected_resource_group!r}, "
            f"but connected to {actual_resource_group!r}."
        )


if __name__ == "__main__":
    print(
        "Import this module from nb_setup.ipynb. "
        "No Azure resources are created by this diagnostic."
    )