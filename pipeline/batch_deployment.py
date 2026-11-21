
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.entities import (
    BatchEndpoint,
    PipelineComponentBatchDeployment,
)


# ============================================================
# GET OR CREATE BATCH ENDPOINT
# ============================================================

def get_or_create_batch_endpoint(
    ml_client,
    endpoint_name,
    allow_creation=False,
):
    """
    Retrieve an existing Batch Endpoint.
    Create one only when explicitly permitted.
    """

    try:
        endpoint = ml_client.batch_endpoints.get(
            name=endpoint_name
        )

        print(f"Reusing existing Batch Endpoint: {endpoint.name}")
        return endpoint

    except ResourceNotFoundError:

        if not allow_creation:
            raise RuntimeError(
                f"Batch Endpoint '{endpoint_name}' does not exist. "
                "Endpoint creation is disabled."
            )

        endpoint = BatchEndpoint(
            name=endpoint_name,
            description="Bank Marketing pipeline endpoint",
        )

        endpoint = ml_client.batch_endpoints.begin_create_or_update(
            endpoint
        ).result()

        print(f"Created Batch Endpoint: {endpoint.name}")
        return endpoint


# ============================================================
# GET OR CREATE PIPELINE DEPLOYMENT
# ============================================================

def get_or_create_pipeline_deployment(
    ml_client,
    endpoint_name,
    deployment_name,
    component,
    allow_creation=False,
):
    """
    Retrieve an existing pipeline deployment and verify that
    it references the intended registered pipeline component.

    Create a deployment only when explicitly permitted.
    """

    try:

        existing = ml_client.batch_deployments.get(
            name=deployment_name,
            endpoint_name=endpoint_name,
        )

        # ----------------------------------------------------
        # VERIFY EXISTING DEPLOYMENT COMPONENT ID
        # ----------------------------------------------------

        existing_tags = existing.tags or {}

        actual_component_id = existing_tags.get(
            "PipelineDeployment.ComponentId"
        )

        expected_component_id = getattr(
            component,
            "id",
            None,
        )

        if not actual_component_id:
            raise RuntimeError(
                "Existing deployment does not expose its pipeline "
                "component ID. Cannot safely verify deployment identity."
            )

        if expected_component_id:

            matches = (
                actual_component_id.rstrip("/").lower()
                == expected_component_id.rstrip("/").lower()
            )

        else:

            expected_suffix = (
                f"/components/{component.name}/versions/"
                f"{component.version}"
            ).lower()

            matches = actual_component_id.lower().endswith(
                expected_suffix
            )

        if not matches:
            raise RuntimeError(
                "Existing deployment references a different "
                "pipeline component.\n"
                f"Expected: {expected_component_id}\n"
                f"Actual: {actual_component_id}"
            )

        print("Reusing existing pipeline deployment.")
        print("Deployment:", existing.name)
        print("Verified component ID:", actual_component_id)

        return existing

    except ResourceNotFoundError:

        if not allow_creation:
            raise RuntimeError(
                f"Batch deployment '{deployment_name}' does not exist "
                f"under endpoint '{endpoint_name}'. "
                "Deployment creation is disabled."
            )

        # ----------------------------------------------------
        # CREATE NEW PIPELINE DEPLOYMENT
        # ----------------------------------------------------

        deployment = PipelineComponentBatchDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            component=component,
            description="Bank Marketing reusable pipeline deployment",
            tags={
                "project": "udacity-project-2",
                "sdk": "v2",
                "PipelineDeployment.ComponentId": component.id,
            },
        )

        deployment = (
            ml_client.batch_deployments.begin_create_or_update(
                deployment
            ).result()
        )

        print(f"Created pipeline deployment: {deployment.name}")

        return deployment


# ============================================================
# ENSURE DEFAULT DEPLOYMENT
# ============================================================

def ensure_default_deployment(
    ml_client,
    endpoint,
    deployment_name,
    allow_update=False,
):
    """
    Ensure the requested deployment is the endpoint default.

    Update the endpoint only when explicitly permitted.
    """

    current_default = (
        endpoint.defaults.deployment_name
        if endpoint.defaults
        else None
    )

    if current_default == deployment_name:

        print(
            f"Deployment '{deployment_name}' is already "
            "the default deployment."
        )

        return endpoint

    if not allow_update:

        raise RuntimeError(
            "The requested deployment is not the current default.\n"
            f"Current default: {current_default}\n"
            f"Requested default: {deployment_name}\n"
            "Default deployment updates are disabled."
        )

    endpoint.defaults = {
        "deployment_name": deployment_name
    }

    endpoint = ml_client.batch_endpoints.begin_create_or_update(
        endpoint
    ).result()

    print(f"Updated default deployment: {deployment_name}")

    return endpoint
